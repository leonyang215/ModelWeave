// Package format renders engine output as client responses. Engines parse
// their own output (vLLM output_mode "derender"): what reaches the gateway
// are content, reasoning and tool-call deltas. A Formatter turns them into
// OpenAI Chat Completions, OpenAI Responses or Anthropic Messages, streamed
// as server-sent events or aggregated into one JSON body. Engines are
// always called streaming; only the formatter knows whether the client
// asked for a stream.
package format

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"strings"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// Meta describes the response being produced.
type Meta struct {
	ID      string // response id: chatcmpl-..., resp_..., msg_...
	Model   string // model name as the client sent it
	Created int64  // unix seconds
	// Stream: the client asked for SSE; otherwise one JSON body.
	Stream       bool
	IncludeUsage bool // stream_options.include_usage
	Logprobs     bool
	// Request is the client's request body; the Responses API echoes some
	// of its fields.
	Request json.RawMessage
	// PromptTokens is the rendered prompt length, used until the engine
	// reports usage.
	PromptTokens int
}

// Summary is what a finished response reported, for logs and metrics.
type Summary struct {
	FinishReason     string // as sent to the client
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
	ReasoningTokens  int
	ToolCalls        int
}

// Formatter produces the client response. Stream formatters return SSE
// bytes from every call; JSON formatters return nothing until Finish, which
// returns the whole body.
type Formatter interface {
	ContentType() string
	// Start opens the response when the first output is about to be sent.
	Start() []byte
	// Delta consumes one parsed output delta.
	Delta(ev *mwip.GenerateEvent) []byte
	// Finish consumes the finish event and closes the response.
	Finish(ev *mwip.GenerateEvent) []byte
	// Fail ends a started stream with an error event.
	Fail(kind, message string) []byte
	Summary() Summary
}

// New returns the formatter of a protocol.
func New(protocol string, m Meta) Formatter {
	st := &state{meta: m, stream: m.Stream}
	switch protocol {
	case mwip.ProtocolMessages:
		return &messages{state: st}
	case mwip.ProtocolResponses:
		return newResponses(st)
	default:
		return &chat{state: st}
	}
}

// ResponseID returns the response id of a request in a protocol.
func ResponseID(protocol, requestID string) string {
	id := strings.TrimPrefix(requestID, "mw-")
	switch protocol {
	case mwip.ProtocolMessages:
		return "msg_" + id
	case mwip.ProtocolResponses:
		return "resp_" + id
	}
	return "chatcmpl-" + id
}

// toolCall accumulates one tool call.
type toolCall struct {
	index int
	id    string
	name  string
	args  strings.Builder
}

// state accumulates output shared by every protocol.
type state struct {
	meta       Meta
	stream     bool
	content    strings.Builder
	reasoning  strings.Builder
	calls      []*toolCall
	byIndex    map[int]*toolCall
	logprobs   []mwip.LogprobContent
	tokens     int // output token ids seen
	reasonToks int // token ids of reasoning-only deltas
	finish     *mwip.GenerateEvent
	sum        Summary
}

// toolDelta is one tool call fragment after accumulation: new reports the
// first fragment of a call.
type toolDelta struct {
	call *toolCall
	new  bool
	args string
}

// absorb records a delta and returns its tool call fragments.
func (s *state) absorb(ev *mwip.GenerateEvent, idPrefix string) []toolDelta {
	s.content.WriteString(ev.Content)
	s.reasoning.WriteString(ev.Reasoning)
	s.tokens += len(ev.TokenIDs)
	if ev.Reasoning != "" && ev.Content == "" && len(ev.ToolCalls) == 0 {
		s.reasonToks += len(ev.TokenIDs)
	}
	if s.meta.Logprobs {
		s.logprobs = append(s.logprobs, ev.Logprobs...)
	}
	var out []toolDelta
	for _, tc := range ev.ToolCalls {
		if s.byIndex == nil {
			s.byIndex = map[int]*toolCall{}
		}
		c, ok := s.byIndex[tc.Index]
		if !ok {
			c = &toolCall{index: len(s.calls), id: tc.ID}
			if c.id == "" {
				c.id = idPrefix + randomID(12)
			}
			s.byIndex[tc.Index] = c
			s.calls = append(s.calls, c)
		}
		d := toolDelta{call: c, new: !ok}
		if tc.Function != nil {
			if tc.Function.Name != "" && c.name == "" {
				c.name = tc.Function.Name
			}
			c.args.WriteString(tc.Function.Arguments)
			d.args = tc.Function.Arguments
		}
		out = append(out, d)
	}
	return out
}

// close records the finish event and computes the summary. The finish
// reason is normalized: a stop after tool calls is "tool_calls".
func (s *state) close(ev *mwip.GenerateEvent) {
	s.finish = ev
	reason := ev.Reason
	switch {
	case reason == "" || reason == "stop":
		reason = "stop"
		if len(s.calls) > 0 {
			reason = "tool_calls"
		}
	case reason == "length", reason == "tool_calls", reason == "content_filter":
	default:
		reason = "stop"
	}
	sum := Summary{FinishReason: reason, PromptTokens: s.meta.PromptTokens, CompletionTokens: s.tokens, ReasoningTokens: s.reasonToks, ToolCalls: len(s.calls)}
	if u := ev.Usage; u != nil {
		if u.PromptTokens > 0 {
			sum.PromptTokens = u.PromptTokens
		}
		if u.CompletionTokens > 0 || s.tokens == 0 {
			sum.CompletionTokens = u.CompletionTokens
		}
		sum.CachedTokens = u.Cached()
		if u.CompletionTokensDetails != nil {
			sum.ReasoningTokens = u.Reasoning()
		}
	}
	s.sum = sum
}

func (s *state) Summary() Summary { return s.sum }

func (s *state) ContentType() string {
	if s.stream {
		return "text/event-stream; charset=utf-8"
	}
	return "application/json"
}

// marshal encodes JSON without HTML escaping (model output often holds
// '<' and '>').
func marshal(v any) []byte {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
	return bytes.TrimRight(b.Bytes(), "\n")
}

func sseData(v any) []byte {
	b := marshal(v)
	out := make([]byte, 0, len(b)+8)
	out = append(out, "data: "...)
	out = append(out, b...)
	return append(out, "\n\n"...)
}

func sseEvent(name string, v any) []byte {
	b := marshal(v)
	out := make([]byte, 0, len(b)+len(name)+16)
	out = append(out, "event: "...)
	out = append(out, name...)
	out = append(out, "\ndata: "...)
	out = append(out, b...)
	return append(out, "\n\n"...)
}

func randomID(n int) string {
	b := make([]byte, (n+1)/2)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)[:n]
}

// obj is a JSON object that keeps its keys in order.
type obj []any

// o builds an obj from key, value pairs.
func o(kv ...any) obj { return obj(kv) }

// MarshalJSON implements json.Marshaler.
func (m obj) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i := 0; i+1 < len(m); i += 2 {
		if i > 0 {
			b.WriteByte(',')
		}
		b.Write(marshal(m[i].(string)))
		b.WriteByte(':')
		b.Write(marshal(m[i+1]))
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// jsonObject returns args when it is a JSON object, else {}.
func jsonObject(args string) json.RawMessage {
	t := strings.TrimSpace(args)
	if strings.HasPrefix(t, "{") && json.Valid([]byte(t)) {
		return json.RawMessage(t)
	}
	return json.RawMessage("{}")
}
