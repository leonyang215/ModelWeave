package mockengine

import (
	"encoding/json"
	"strings"
	"unicode/utf8"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// Reasoning markup shared by DeepSeek V4.1 and Qwen3.
const (
	thinkStart = "<think>"
	thinkEnd   = "</think>"
)

// DeepSeek V4.1 tool-call markup (DSML).
const (
	callsOpen   = "<｜DSML｜ calls>"
	callsClose  = "</｜DSML｜ calls>"
	invokeOpen  = "<｜DSML｜ invoke name=\""
	invokeClose = "</｜DSML｜ invoke>"
	paramOpen   = "<｜DSML｜ parameter name=\""
	paramClose  = "</｜DSML｜ parameter>"
)

// Hermes tool-call markup (Qwen2.5, Qwen3).
const (
	toolCallOpen  = "<tool_call>"
	toolCallClose = "</tool_call>"
)

type parseState int

const (
	stReasoning parseState = iota
	stContent
	stCalls
	stDone
)

// parsed is the delta produced by one parser step.
type parsed struct {
	reasoning string
	content   string
	calls     []mwip.DeltaToolCall
}

func (p parsed) empty() bool { return p.reasoning == "" && p.content == "" && len(p.calls) == 0 }

// parserOpts configure an output parser.
type parserOpts struct {
	reasoning   bool   // output starts inside the reasoning
	tools       bool   // tool calls are parsed only when the request has tools
	callsOpened bool   // the prompt opened the DSML calls block (tool_choice required)
	toolParser  string // mwip.ParserDeepseekV4 (DSML) or mwip.ParserHermes
	newID       func() string
}

// parser is the engine-side output parser that mwip asks engines to run
// (ParsingOptions: reasoning and tool-call parsers): it turns the
// detokenized stream into reasoning, content and tool-call deltas. Markup
// split across tokens is held back until it is complete.
//
// Reasoning ends at </think>; a leading <think> (Qwen3 opens the block
// itself) is dropped. Tool calls are DSML blocks (DeepSeek) or hermes
// <tool_call> JSON objects (Qwen).
type parser struct {
	state      parseState
	tools      bool
	hermes     bool
	buf        string
	calls      int // tool calls started
	params     int // parameters of the open DSML invoke
	newID      func() string
	toolSeen   bool
	openWait   bool // a leading <think> may still arrive
	afterOpen  bool // drop the newline after <think>
	afterThink bool // drop newlines between </think> and the content
}

func newParser(o parserOpts) *parser {
	p := &parser{state: stContent, tools: o.tools, newID: o.newID, hermes: o.toolParser == mwip.ParserHermes}
	switch {
	case o.reasoning:
		p.state = stReasoning
		p.openWait = true
	case o.tools && o.callsOpened && !p.hermes:
		// tool_choice "required": the prompt already opened the calls block.
		p.state = stCalls
	}
	if p.newID == nil {
		p.newID = func() string { return "call_0" }
	}
	return p
}

// holdBack returns how much of buf can be emitted: everything except a
// suffix that may be the start of marker.
func holdBack(buf, marker string) int {
	for k := min(len(marker)-1, len(buf)); k > 0; k-- {
		if strings.HasSuffix(buf, marker[:k]) {
			return len(buf) - k
		}
	}
	return len(buf)
}

var dsmlTags = []string{callsClose, invokeOpen, invokeClose, paramOpen}

// feed parses the next piece of text; final flushes held-back text.
func (p *parser) feed(text string, final bool) parsed {
	var out parsed
	p.buf += text
	for {
		switch p.state {
		case stReasoning:
			if p.openWait {
				b := strings.TrimLeft(p.buf, "\n")
				if len(b) < len(thinkStart) && strings.HasPrefix(thinkStart, b) && !final {
					return out
				}
				p.openWait = false
				if strings.HasPrefix(b, thinkStart) {
					p.buf = b[len(thinkStart):]
					p.afterOpen = true
				}
			}
			if p.afterOpen {
				if p.buf == "" && !final {
					return out
				}
				p.buf = strings.TrimPrefix(p.buf, "\n")
				p.afterOpen = false
			}
			if i := strings.Index(p.buf, thinkEnd); i >= 0 {
				out.reasoning += strings.TrimRight(p.buf[:i], "\n")
				p.buf = p.buf[i+len(thinkEnd):]
				p.state = stContent
				p.afterThink = true
				continue
			}
			n := len(p.buf)
			if !final {
				n = holdBack(p.buf, thinkEnd)
				for n > 0 && p.buf[n-1] == '\n' { // may precede </think>
					n--
				}
			}
			out.reasoning += p.buf[:n]
			p.buf = p.buf[n:]
			return out
		case stContent:
			if p.afterThink {
				p.buf = strings.TrimLeft(p.buf, "\n")
				if p.buf == "" && !final {
					return out
				}
				p.afterThink = false
			}
			if !p.tools {
				out.content += p.buf
				p.buf = ""
				return out
			}
			open := callsOpen
			if p.hermes {
				open = toolCallOpen
			}
			if i := strings.Index(p.buf, open); i >= 0 {
				out.content += strings.TrimRight(p.buf[:i], "\n ")
				p.buf = p.buf[i:]
				if !p.hermes {
					p.buf = p.buf[len(callsOpen):]
				}
				p.state = stCalls
				continue
			}
			n := len(p.buf)
			if !final {
				n = holdBack(p.buf, open)
				for n > 0 && (p.buf[n-1] == '\n' || p.buf[n-1] == ' ') { // may precede the calls
					n--
				}
			}
			out.content += p.buf[:n]
			p.buf = p.buf[n:]
			return out
		case stCalls:
			var more bool
			if p.hermes {
				more = p.hermesStep(&out, final)
			} else {
				more = p.dsmlStep(&out, final)
			}
			if !more {
				return out
			}
		default:
			p.buf = ""
			return out
		}
	}
}

// hermesStep parses <tool_call>{"name": ..., "arguments": {...}}</tool_call>
// blocks; it reports whether parsing can continue.
func (p *parser) hermesStep(out *parsed, final bool) bool {
	b := strings.TrimLeft(p.buf, " \n")
	if !strings.HasPrefix(b, toolCallOpen) {
		if len(b) < len(toolCallOpen) && strings.HasPrefix(toolCallOpen, b) && !final {
			p.buf = b
			return false
		}
		// Text after the tool calls is not content.
		p.buf = ""
		return false
	}
	end := strings.Index(b, toolCallClose)
	if end < 0 {
		p.buf = b
		return false
	}
	body := strings.TrimSpace(b[len(toolCallOpen):end])
	p.buf = b[end+len(toolCallClose):]
	var call struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if json.Unmarshal([]byte(body), &call) != nil || call.Name == "" {
		return true // not a valid call: skip it
	}
	args := "{}"
	if len(call.Arguments) > 0 && string(call.Arguments) != "null" {
		args = string(call.Arguments)
		var s string
		if json.Unmarshal(call.Arguments, &s) == nil {
			args = s // arguments sent as a JSON string
		}
	}
	out.calls = append(out.calls, mwip.DeltaToolCall{Index: p.calls, ID: p.newID(), Type: "function",
		Function: &mwip.DeltaFunction{Name: call.Name, Arguments: args}})
	p.calls++
	p.toolSeen = true
	return true
}

// dsmlStep parses DSML invoke blocks; it reports whether parsing can
// continue.
func (p *parser) dsmlStep(out *parsed, final bool) bool {
	b := strings.TrimLeft(p.buf, " \n")
	switch {
	case strings.HasPrefix(b, callsClose):
		p.buf, p.state = b[len(callsClose):], stDone
	case strings.HasPrefix(b, invokeClose):
		args := "}"
		if p.params == 0 {
			args = "{}"
		}
		out.calls = append(out.calls, mwip.DeltaToolCall{Index: p.calls - 1, Function: &mwip.DeltaFunction{Arguments: args}})
		p.buf = b[len(invokeClose):]
	case strings.HasPrefix(b, invokeOpen):
		end := strings.Index(b[len(invokeOpen):], "\">")
		if end < 0 {
			p.buf = b
			return false
		}
		name := b[len(invokeOpen) : len(invokeOpen)+end]
		out.calls = append(out.calls, mwip.DeltaToolCall{Index: p.calls, ID: p.newID(), Type: "function", Function: &mwip.DeltaFunction{Name: name}})
		p.calls++
		p.params = 0
		p.toolSeen = true
		p.buf = b[len(invokeOpen)+end+2:]
	case strings.HasPrefix(b, paramOpen):
		hdr := strings.Index(b, "\">")
		end := strings.Index(b, paramClose)
		if hdr < 0 || end < 0 || end < hdr {
			p.buf = b
			return false
		}
		name, isString := paramHeader(b[len(paramOpen):hdr])
		value := b[hdr+2 : end]
		var v []byte
		if !isString && json.Valid([]byte(value)) {
			v = []byte(value)
		} else {
			v, _ = json.Marshal(value)
		}
		k, _ := json.Marshal(name)
		frag := ", "
		if p.params == 0 {
			frag = "{"
		}
		p.params++
		out.calls = append(out.calls, mwip.DeltaToolCall{Index: p.calls - 1, Function: &mwip.DeltaFunction{Arguments: frag + string(k) + ": " + string(v)}})
		p.buf = b[end+len(paramClose):]
	default:
		partial := false
		for _, t := range dsmlTags {
			if len(b) < len(t) && strings.HasPrefix(t, b) {
				partial = true
			}
		}
		if partial && !final {
			p.buf = b
			return false
		}
		// Unexpected text inside the calls block: skip to the next tag.
		if i := strings.Index(b[min(1, len(b)):], "<"); i >= 0 && !final {
			p.buf = b[i+1:]
			return true
		}
		p.buf = ""
		return false
	}
	return true
}

// paramHeader parses `name" string="true` of a parameter tag (the text
// between `name="` and the closing `">`).
func paramHeader(h string) (string, bool) {
	name, rest, _ := strings.Cut(h, "\"")
	return name, !strings.Contains(rest, `string="false`)
}

// stopper applies stop strings to the detokenized output like an engine's
// detokenizer: text that may begin a stop string is held back, and output
// ends before the first match.
type stopper struct {
	stops   []string
	maxLen  int
	held    string
	matched string
}

func newStopper(stops []string) *stopper {
	s := &stopper{}
	for _, st := range stops {
		if st != "" {
			s.stops = append(s.stops, st)
			s.maxLen = max(s.maxLen, len(st))
		}
	}
	return s
}

// push adds a piece and returns the text that is safe to emit; stopped
// reports a match (the returned text ends before it).
func (s *stopper) push(piece string) (string, bool) {
	if len(s.stops) == 0 {
		return piece, false
	}
	s.held += piece
	best := -1
	for _, st := range s.stops {
		if i := strings.Index(s.held, st); i >= 0 && (best < 0 || i < best) {
			best, s.matched = i, st
		}
	}
	if best >= 0 {
		out := s.held[:best]
		s.held = ""
		return out, true
	}
	cut := max(len(s.held)-(s.maxLen-1), 0)
	for cut > 0 && cut < len(s.held) && !utf8.RuneStart(s.held[cut]) {
		cut--
	}
	out := s.held[:cut]
	s.held = s.held[cut:]
	return out, false
}

func (s *stopper) flush() string {
	out := s.held
	s.held = ""
	return out
}
