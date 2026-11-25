package format

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

func meta(stream bool) Meta {
	return Meta{ID: "id-1", Model: "deepseek-v4.1", Created: 1700000000, PromptTokens: 10,
		Stream: stream, IncludeUsage: true}
}

func usage(prompt, completion, cached int) *mwip.Usage {
	return &mwip.Usage{PromptTokens: prompt, CompletionTokens: completion, TotalTokens: prompt + completion,
		PromptTokensDetails: &mwip.PromptTokensDetails{CachedTokens: cached}}
}

// run feeds deltas and a finish into a formatter and returns its output.
func run(f Formatter, deltas []*mwip.GenerateEvent, fin *mwip.GenerateEvent) string {
	var b strings.Builder
	b.Write(f.Start())
	for _, d := range deltas {
		b.Write(f.Delta(d))
	}
	b.Write(f.Finish(fin))
	return b.String()
}

func reasoningAnswer() []*mwip.GenerateEvent {
	return []*mwip.GenerateEvent{
		{Type: mwip.EventDelta, Reasoning: "Think", TokenIDs: []uint32{1}},
		{Type: mwip.EventDelta, Reasoning: "ing.", TokenIDs: []uint32{2}},
		{Type: mwip.EventDelta, Content: "Hello <b>", TokenIDs: []uint32{3}},
		{Type: mwip.EventDelta, Content: " world", TokenIDs: []uint32{4}},
		{Type: mwip.EventDelta, TokenIDs: []uint32{5}}, // held back by the engine's parser
	}
}

func toolCallDeltas() []*mwip.GenerateEvent {
	return []*mwip.GenerateEvent{
		{Type: mwip.EventDelta, Content: "Let me search.", TokenIDs: []uint32{1}},
		{Type: mwip.EventDelta, TokenIDs: []uint32{2}, ToolCalls: []mwip.DeltaToolCall{{Index: 0, ID: "call_abc", Type: "function", Function: &mwip.DeltaFunction{Name: "search"}}}},
		{Type: mwip.EventDelta, TokenIDs: []uint32{3}, ToolCalls: []mwip.DeltaToolCall{{Index: 0, Function: &mwip.DeltaFunction{Arguments: `{"q": `}}}},
		{Type: mwip.EventDelta, TokenIDs: []uint32{4}, ToolCalls: []mwip.DeltaToolCall{{Index: 0, Function: &mwip.DeltaFunction{Arguments: `"x"}`}}}},
	}
}

func TestChatJSON(t *testing.T) {
	f := New(mwip.ProtocolChatCompletions, meta(false))
	out := run(f, reasoningAnswer(), &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "stop", Usage: usage(10, 5, 4)})
	var r struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content          *string `json:"content"`
				ReasoningContent string  `json:"reasoning_content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
			CompletionTokensDetails struct {
				ReasoningTokens int `json:"reasoning_tokens"`
			} `json:"completion_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		t.Fatalf("%v: %s", err, out)
	}
	ch := r.Choices[0]
	if r.Object != "chat.completion" || *ch.Message.Content != "Hello <b> world" || ch.Message.ReasoningContent != "Thinking." || ch.FinishReason != "stop" {
		t.Fatalf("unexpected response %s", out)
	}
	if !strings.Contains(out, "Hello <b>") {
		t.Fatalf("HTML-escaped output: %s", out)
	}
	u := r.Usage
	if u.PromptTokens != 10 || u.CompletionTokens != 5 || u.PromptTokensDetails.CachedTokens != 4 || u.CompletionTokensDetails.ReasoningTokens != 2 {
		t.Fatalf("usage %+v", u)
	}
	if s := f.Summary(); s.FinishReason != "stop" || s.ReasoningTokens != 2 || s.CachedTokens != 4 {
		t.Fatalf("summary %+v", s)
	}
}

func TestChatToolCallsJSON(t *testing.T) {
	f := New(mwip.ProtocolChatCompletions, meta(false))
	out := run(f, toolCallDeltas()[1:], &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "stop"})
	var r struct {
		Choices []struct {
			Message struct {
				Content   *string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct{ Name, Arguments string }
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		t.Fatal(err)
	}
	ch := r.Choices[0]
	if ch.FinishReason != "tool_calls" || ch.Message.Content != nil || len(ch.Message.ToolCalls) != 1 {
		t.Fatalf("unexpected %s", out)
	}
	tc := ch.Message.ToolCalls[0]
	if tc.ID != "call_abc" || tc.Type != "function" || tc.Function.Name != "search" || tc.Function.Arguments != `{"q": "x"}` {
		t.Fatalf("tool call %+v", tc)
	}
}

func sseDataLines(out string) []string {
	var lines []string
	for _, l := range strings.Split(out, "\n") {
		if strings.HasPrefix(l, "data: ") {
			lines = append(lines, strings.TrimPrefix(l, "data: "))
		}
	}
	return lines
}

func TestChatStream(t *testing.T) {
	f := New(mwip.ProtocolChatCompletions, meta(true))
	if f.ContentType() != "text/event-stream; charset=utf-8" {
		t.Fatal(f.ContentType())
	}
	out := run(f, append(reasoningAnswer(), toolCallDeltas()[1:]...), &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "stop", Usage: usage(10, 9, 0)})
	lines := sseDataLines(out)
	// role, 2 reasoning, 2 content, (held-back delta: nothing), 3 tool call fragments, finish, usage, [DONE]
	if len(lines) != 11 || lines[len(lines)-1] != "[DONE]" {
		t.Fatalf("%d lines:\n%s", len(lines), out)
	}
	var first, tool0, tool1, fin, use map[string]any
	_ = json.Unmarshal([]byte(lines[0]), &first)
	_ = json.Unmarshal([]byte(lines[5]), &tool0)
	_ = json.Unmarshal([]byte(lines[6]), &tool1)
	_ = json.Unmarshal([]byte(lines[8]), &fin)
	_ = json.Unmarshal([]byte(lines[9]), &use)
	delta := func(m map[string]any) map[string]any {
		return m["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	}
	if first["object"] != "chat.completion.chunk" || delta(first)["role"] != "assistant" {
		t.Fatalf("first chunk %s", lines[0])
	}
	tc0 := delta(tool0)["tool_calls"].([]any)[0].(map[string]any)
	tc1 := delta(tool1)["tool_calls"].([]any)[0].(map[string]any)
	if tc0["id"] != "call_abc" || tc0["function"].(map[string]any)["name"] != "search" || tc1["id"] != nil || tc1["function"].(map[string]any)["arguments"] != `{"q": ` {
		t.Fatalf("tool call chunks %s / %s", lines[5], lines[6])
	}
	if fin["choices"].([]any)[0].(map[string]any)["finish_reason"] != "tool_calls" {
		t.Fatalf("finish chunk %s", lines[8])
	}
	if len(use["choices"].([]any)) != 0 || use["usage"].(map[string]any)["completion_tokens"] != float64(9) {
		t.Fatalf("usage chunk %s", lines[9])
	}
}

func events(out string) []string {
	var names []string
	for _, l := range strings.Split(out, "\n") {
		if strings.HasPrefix(l, "event: ") {
			names = append(names, strings.TrimPrefix(l, "event: "))
		}
	}
	return names
}

func TestMessagesJSON(t *testing.T) {
	f := New(mwip.ProtocolMessages, meta(false))
	out := run(f, append(reasoningAnswer(), toolCallDeltas()[1:]...), &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "stop", Usage: usage(10, 9, 4)})
	var m struct {
		Type    string           `json:"type"`
		Content []map[string]any `json:"content"`
		Stop    string           `json:"stop_reason"`
		Usage   anthropicUsage   `json:"usage"`
	}
	if err := json.Unmarshal([]byte(out), &m); err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, c := range m.Content {
		kinds = append(kinds, c["type"].(string))
	}
	if m.Type != "message" || strings.Join(kinds, ",") != "thinking,text,tool_use" || m.Stop != "tool_use" {
		t.Fatalf("unexpected %s", out)
	}
	if m.Content[0]["thinking"] != "Thinking." || m.Content[2]["input"].(map[string]any)["q"] != "x" {
		t.Fatalf("blocks %s", out)
	}
	if m.Usage.InputTokens != 6 || m.Usage.CacheReadInputTokens != 4 || m.Usage.OutputTokens != 9 {
		t.Fatalf("usage %+v", m.Usage)
	}
}

func TestMessagesStopSequenceAndStream(t *testing.T) {
	f := New(mwip.ProtocolMessages, meta(true))
	out := run(f, reasoningAnswer(), &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "stop", StopReason: "four"})
	got := strings.Join(events(out), ",")
	want := "message_start,content_block_start,content_block_delta,content_block_delta,content_block_stop," +
		"content_block_start,content_block_delta,content_block_delta,content_block_stop,message_delta,message_stop"
	if got != want {
		t.Fatalf("events\n got %s\nwant %s", got, want)
	}
	if !strings.Contains(out, `"stop_reason":"stop_sequence","stop_sequence":"four"`) || !strings.Contains(out, `"type":"thinking_delta","thinking":"Think"`) {
		t.Fatalf("unexpected stream %s", out)
	}
	// max_tokens
	g := New(mwip.ProtocolMessages, meta(false))
	if out := run(g, reasoningAnswer(), &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "length"}); !strings.Contains(out, `"stop_reason":"max_tokens"`) {
		t.Fatalf("length: %s", out)
	}
}

func TestResponsesJSONEchoesRequest(t *testing.T) {
	m := meta(false)
	m.Request = json.RawMessage(`{"model":"deepseek-v4.1","input":"hi","instructions":"be brief","max_output_tokens":64,"metadata":{"k":"v"}}`)
	f := New(mwip.ProtocolResponses, m)
	out := run(f, append(reasoningAnswer(), toolCallDeltas()[1:]...), &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "stop", Usage: usage(10, 9, 0)})
	var r map[string]any
	if err := json.Unmarshal([]byte(out), &r); err != nil {
		t.Fatal(err)
	}
	if r["object"] != "response" || r["status"] != "completed" || r["instructions"] != "be brief" || r["max_output_tokens"] != float64(64) || r["tool_choice"] != "auto" {
		t.Fatalf("unexpected %s", out)
	}
	var kinds []string
	for _, it := range r["output"].([]any) {
		kinds = append(kinds, it.(map[string]any)["type"].(string))
	}
	if strings.Join(kinds, ",") != "reasoning,message,function_call" {
		t.Fatalf("output items %v", kinds)
	}
	fc := r["output"].([]any)[2].(map[string]any)
	if fc["call_id"] != "call_abc" || fc["arguments"] != `{"q": "x"}` || fc["status"] != "completed" {
		t.Fatalf("function call %v", fc)
	}
	msg := r["output"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if msg["type"] != "output_text" || msg["text"] != "Hello <b> world" {
		t.Fatalf("message %v", msg)
	}
}

func TestResponsesStream(t *testing.T) {
	f := New(mwip.ProtocolResponses, meta(true))
	out := run(f, reasoningAnswer(), &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "length"})
	got := strings.Join(events(out), ",")
	want := "response.created,response.in_progress," +
		"response.output_item.added,response.content_part.added,response.reasoning_text.delta,response.reasoning_text.delta," +
		"response.reasoning_text.done,response.content_part.done,response.output_item.done," +
		"response.output_item.added,response.content_part.added,response.output_text.delta,response.output_text.delta," +
		"response.output_text.done,response.content_part.done,response.output_item.done,response.incomplete"
	if got != want {
		t.Fatalf("events\n got %s\nwant %s", got, want)
	}
	lines := sseDataLines(out)
	var last map[string]any
	_ = json.Unmarshal([]byte(lines[len(lines)-1]), &last)
	resp := last["response"].(map[string]any)
	if resp["status"] != "incomplete" || resp["incomplete_details"].(map[string]any)["reason"] != "max_output_tokens" || last["sequence_number"] != float64(16) {
		t.Fatalf("final event %s", lines[len(lines)-1])
	}
}

func TestFailEvents(t *testing.T) {
	if out := string(New(mwip.ProtocolChatCompletions, meta(true)).Fail("upstream_error", "boom")); !strings.HasPrefix(out, `data: {"error":`) {
		t.Fatal(out)
	}
	if out := string(New(mwip.ProtocolMessages, meta(true)).Fail("overloaded", "busy")); !strings.HasPrefix(out, "event: error\ndata: ") || !strings.Contains(out, "overloaded_error") {
		t.Fatal(out)
	}
	if out := string(New(mwip.ProtocolResponses, meta(true)).Fail("timeout", "slow")); !strings.HasPrefix(out, "event: error\ndata: ") || !strings.Contains(out, `"code":"timeout"`) {
		t.Fatal(out)
	}
}
