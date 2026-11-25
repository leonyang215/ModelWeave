package mockengine

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

const sample = "Plan the search.</think>Sure.\n\n<｜DSML｜ calls>\n" +
	"<｜DSML｜ invoke name=\"search_docs\">\n" +
	"<｜DSML｜ parameter name=\"query\" string=\"true\">Model<Weave></｜DSML｜ parameter>\n" +
	"<｜DSML｜ parameter name=\"limit\" string=\"false\">3</｜DSML｜ parameter>\n" +
	"</｜DSML｜ invoke>\n" +
	"<｜DSML｜ invoke name=\"noop\">\n</｜DSML｜ invoke>\n" +
	"</｜DSML｜ calls>"

type collected struct {
	reasoning, content string
	names              []string
	args               map[int]*strings.Builder
}

func collect(p *parser, pieces []string) collected {
	c := collected{args: map[int]*strings.Builder{}}
	for i, piece := range pieces {
		out := p.feed(piece, i == len(pieces)-1)
		c.reasoning += out.reasoning
		c.content += out.content
		for _, tc := range out.calls {
			if tc.Function.Name != "" {
				c.names = append(c.names, tc.Function.Name)
				if tc.ID == "" || tc.Type != "function" {
					panic("first fragment without id/type")
				}
			}
			if c.args[tc.Index] == nil {
				c.args[tc.Index] = &strings.Builder{}
			}
			c.args[tc.Index].WriteString(tc.Function.Arguments)
		}
	}
	return c
}

func dsml(reasoning, tools, opened bool) *parser {
	return newParser(parserOpts{reasoning: reasoning, tools: tools, callsOpened: opened, toolParser: mwip.ParserDeepseekV4,
		newID: func() string { return "call-1" }})
}

// The parser output does not depend on how the text is split into tokens.
func TestParserIsSplitInvariant(t *testing.T) {
	for cut := 0; cut <= len(sample); cut++ {
		if !utf8.RuneStart(sample[min(cut, len(sample)-1)]) {
			continue
		}
		p := dsml(true, true, false)
		c := collect(p, []string{sample[:cut], sample[cut:]})
		if c.reasoning != "Plan the search." || c.content != "Sure." {
			t.Fatalf("cut %d: reasoning %q content %q", cut, c.reasoning, c.content)
		}
		if strings.Join(c.names, ",") != "search_docs,noop" || !p.toolSeen {
			t.Fatalf("cut %d: names %v", cut, c.names)
		}
		var a map[string]any
		if err := json.Unmarshal([]byte(c.args[0].String()), &a); err != nil || a["query"] != "Model<Weave>" || a["limit"] != float64(3) {
			t.Fatalf("cut %d: args %q (%v)", cut, c.args[0].String(), err)
		}
		if c.args[1].String() != "{}" {
			t.Fatalf("cut %d: empty args %q", cut, c.args[1].String())
		}
	}
	// Token-sized pieces.
	p := dsml(true, true, false)
	c := collect(p, fallbackTokenize(sample).pieces)
	if c.content != "Sure." || len(c.names) != 2 {
		t.Fatalf("token pieces: %+v", c)
	}
}

func TestParserModes(t *testing.T) {
	// No tools: DSML-looking text stays content; no reasoning without <think>.
	p := dsml(false, false, false)
	if out := p.feed("a <｜DSML｜ calls> b", true); out.content != "a <｜DSML｜ calls> b" || out.reasoning != "" {
		t.Fatalf("%+v", out)
	}
	// tool_choice required: the prompt opened the calls block.
	p = dsml(false, true, true)
	c := collect(p, []string{"<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"})
	if strings.Join(c.names, ",") != "f" || c.content != "" {
		t.Fatalf("%+v", c)
	}
	// Unterminated reasoning is flushed at the end.
	p = dsml(true, false, false)
	if out := p.feed("still thinking</th", true); out.reasoning != "still thinking</th" {
		t.Fatalf("%+v", out)
	}
}

const qwenSample = "<think>\nPlan the search.\n</think>\n\nSure.\n<tool_call>\n" +
	`{"name": "search_docs", "arguments": {"query": "Model<Weave>", "limit": 3}}` + "\n</tool_call>\n<tool_call>\n" +
	`{"name": "noop", "arguments": {}}` + "\n</tool_call>"

func hermes(reasoning bool) *parser {
	return newParser(parserOpts{reasoning: reasoning, tools: true, toolParser: mwip.ParserHermes,
		newID: func() string { return "call-1" }})
}

// Qwen3 output: the model opens its reasoning with <think>, and tool calls
// are hermes JSON objects; any split of the text parses the same.
func TestHermesParserIsSplitInvariant(t *testing.T) {
	for cut := 0; cut <= len(qwenSample); cut++ {
		p := hermes(true)
		c := collect(p, []string{qwenSample[:cut], qwenSample[cut:]})
		if c.reasoning != "Plan the search." || c.content != "Sure." {
			t.Fatalf("cut %d: reasoning %q content %q", cut, c.reasoning, c.content)
		}
		if strings.Join(c.names, ",") != "search_docs,noop" || !p.toolSeen {
			t.Fatalf("cut %d: names %v", cut, c.names)
		}
		var a map[string]any
		if err := json.Unmarshal([]byte(c.args[0].String()), &a); err != nil || a["query"] != "Model<Weave>" || a["limit"] != float64(3) {
			t.Fatalf("cut %d: args %q (%v)", cut, c.args[0].String(), err)
		}
		if c.args[1].String() != "{}" {
			t.Fatalf("cut %d: empty args %q", cut, c.args[1].String())
		}
	}
	c := collect(hermes(true), fallbackTokenize(qwenSample).pieces)
	if c.reasoning != "Plan the search." || c.content != "Sure." || len(c.names) != 2 {
		t.Fatalf("token pieces: %+v", c)
	}
	// A prompt that already opened the reasoning (no <think> in the output),
	// and a non-thinking answer with tool-call lookalikes but no tools.
	c = collect(hermes(true), []string{"Plan.\n</think>\n\nDone."})
	if c.reasoning != "Plan." || c.content != "Done." {
		t.Fatalf("opened reasoning: %+v", c)
	}
	p := newParser(parserOpts{toolParser: mwip.ParserHermes})
	if out := p.feed("a <tool_call> b", true); out.content != "a <tool_call> b" {
		t.Fatalf("%+v", out)
	}
	// Invalid JSON inside a tool call is skipped.
	c = collect(hermes(false), []string{"<tool_call>{oops}</tool_call><tool_call>{\"name\": \"f\", \"arguments\": \"{\\\"x\\\": 1}\"}</tool_call>"})
	if strings.Join(c.names, ",") != "f" || c.args[0].String() != `{"x": 1}` {
		t.Fatalf("invalid call: %+v %q", c.names, c.args[0])
	}
}

func TestStopper(t *testing.T) {
	s := newStopper([]string{"four", "xyz"})
	var out strings.Builder
	stopped := false
	for _, piece := range []string{"one", " two", " three", " fo", "ur five"} {
		txt, done := s.push(piece)
		out.WriteString(txt)
		if done {
			stopped = true
			break
		}
	}
	if !stopped || out.String() != "one two three " || s.matched != "four" {
		t.Fatalf("got %q stopped=%v matched=%q", out.String(), stopped, s.matched)
	}
	s = newStopper([]string{"zz"})
	a, _ := s.push("abz")
	if a != "ab" || s.flush() != "z" {
		t.Fatalf("hold back: %q", a)
	}
	if txt, _ := newStopper(nil).push("abc"); txt != "abc" {
		t.Fatal(txt)
	}
}
