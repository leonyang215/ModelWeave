package mockengine_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mockengine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

func start(t *testing.T) registry.Spec {
	t.Helper()
	eng := mockengine.New(mockengine.Config{
		ID: "w0", Model: "m", Role: "unified", MaxOutputTokens: 256, MaxRunning: 8,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	srv := httptest.NewServer(eng.Handler())
	t.Cleanup(srv.Close)
	return registry.Spec{ID: "w0", Model: "m", Role: registry.RoleUnified, URL: srv.URL}
}

func f64(v float64) *float64 { return &v }

// render is a thinking-mode intermediate state with one tool.
func render(tools bool, thinking bool) *mwip.RenderResult {
	r := &mwip.RenderResult{
		RequestID: "r", TokenIDs: []uint32{0, 42, 128822}, PromptTokens: 3,
		Conversation:     mwip.Conversation{ToolChoice: "auto", ResponseFormat: "text"},
		InferenceOptions: mwip.InferenceOptions{MaxTokens: 200, StopTokenIDs: []uint32{1}, Temperature: f64(1)},
	}
	if thinking {
		r.TokenIDs[2] = 128821
		r.Conversation.ThinkingMode = true
		r.ParsingOptions.ThinkingInitialStage = "reasoning_start"
	}
	if tools {
		r.Conversation.Tools = []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"search_docs","parameters":{"type":"object"}}}`)}
		r.ParsingOptions.ParseToolCalls = true
	}
	return r
}

type result struct {
	reasoning, content string
	calls              map[int]*mwip.DeltaFunction
	finish             *mwip.GenerateEvent
}

func run(t *testing.T, protocol string, spec registry.Spec, r *mwip.RenderResult) result {
	t.Helper()
	s, err := engine.New().Generate(context.Background(), protocol, spec, &engine.Stage{Render: r, AttemptID: "r-" + protocol, Disagg: mwip.Disagg{Mode: mwip.ModeNormal}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	out := result{calls: map[int]*mwip.DeltaFunction{}}
	for {
		ev, err := s.Next()
		if errors.Is(err, io.EOF) {
			return out
		}
		if err != nil {
			t.Fatalf("%s stream: %v", protocol, err)
		}
		switch ev.Type {
		case mwip.EventDelta:
			out.reasoning += ev.Reasoning
			out.content += ev.Content
			for _, tc := range ev.ToolCalls {
				if out.calls[tc.Index] == nil {
					out.calls[tc.Index] = &mwip.DeltaFunction{}
				}
				out.calls[tc.Index].Name += tc.Function.Name
				out.calls[tc.Index].Arguments += tc.Function.Arguments
			}
		case mwip.EventFinish:
			out.finish = ev
		}
	}
}

// The native protocol and vLLM derender both parse on the engine side:
// reasoning and DSML tool calls arrive as deltas.
func TestEngineSideParsing(t *testing.T) {
	spec := start(t)
	for _, protocol := range []string{mwip.EngineMWIP, mwip.EngineVLLM} {
		r := run(t, protocol, spec, render(true, true))
		if !strings.Contains(r.reasoning, "I will answer briefly.") || strings.Contains(r.reasoning, "</think>") {
			t.Fatalf("%s reasoning %q", protocol, r.reasoning)
		}
		if r.content != "" || len(r.calls) != 1 || r.calls[0].Name != "search_docs" {
			t.Fatalf("%s content %q calls %+v", protocol, r.content, r.calls)
		}
		var args map[string]string
		if err := json.Unmarshal([]byte(r.calls[0].Arguments), &args); err != nil || args["query"] != "ModelWeave" || args["worker"] != "w0" {
			t.Fatalf("%s arguments %q: %v", protocol, r.calls[0].Arguments, err)
		}
		f := r.finish
		if f == nil || f.Reason != "tool_calls" || f.Usage == nil || f.Usage.PromptTokens != 3 || f.Usage.Reasoning() == 0 {
			t.Fatalf("%s finish %+v", protocol, f)
		}
	}
	st := getStats(t, spec.URL)
	byProto := st["by_protocol"].(map[string]any)
	if byProto["mwip"] != float64(1) || byProto["vllm"] != float64(1) || st["non_streaming_requests"] != float64(0) {
		t.Fatalf("stats %v", st)
	}
}

func TestStopStringsInTheEngine(t *testing.T) {
	spec := start(t)
	for _, protocol := range []string{mwip.EngineMWIP, mwip.EngineVLLM, mwip.EngineSGLang} {
		r := render(false, false)
		r.ParsingOptions.StopSequences = []string{"four"}
		out := run(t, protocol, spec, r)
		if !strings.HasSuffix(out.content, "one two three ") || out.finish.Reason != "stop" || out.finish.StopReason != "four" {
			t.Fatalf("%s: content %q finish %+v", protocol, out.content, out.finish)
		}
	}
}

// SGLang's /generate returns raw text: no engine-side parsing.
func TestSGLangReturnsRawText(t *testing.T) {
	spec := start(t)
	out := run(t, mwip.EngineSGLang, spec, render(false, true))
	if !strings.Contains(out.content, "</think>Hello from ModelWeave mock worker w0") || out.reasoning != "" {
		t.Fatalf("content %q", out.content)
	}
	if out.finish == nil || out.finish.Reason != "stop" || out.finish.Usage.PromptTokens != 3 {
		t.Fatalf("finish %+v", out.finish)
	}
}

// Engines also answer non-streaming calls; the mock counts them so tests can
// assert the gateway never makes one.
func TestNonStreamingCallsAreCounted(t *testing.T) {
	spec := start(t)
	body := `{"request_id":"x","token_ids":[0,42,128821],"sampling_params":{"max_tokens":50,"stop_token_ids":[1]},"output_mode":"derender","stream":false,"reasoning_ended":false}`
	resp, err := http.Post(spec.URL+"/inference/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || len(out.Choices) != 1 {
		t.Fatalf("decode: %v %+v", err, out)
	}
	if m := out.Choices[0].Message; !strings.HasPrefix(m.Content, "Hello from ModelWeave") || m.Reasoning == "" {
		t.Fatalf("message %+v", m)
	}
	if st := getStats(t, spec.URL); st["non_streaming_requests"] != float64(1) {
		t.Fatalf("stats %v", st)
	}
}

func getStats(t *testing.T, base string) map[string]any {
	t.Helper()
	resp, err := http.Get(base + "/mock/stats")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var m map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&m)
	return m
}

// A Qwen3 request: the parsing options name Qwen's markup; the model opens
// its reasoning with <think> and calls tools with hermes JSON.
func TestQwenMarkupOverMWIP(t *testing.T) {
	spec := start(t)
	r := render(true, true)
	r.TokenIDs = []uint32{151644, 77091, 198} // <|im_start|>assistant\n: Qwen3 opens <think> itself
	r.ParsingOptions.ReasoningParser, r.ParsingOptions.ToolCallParser = mwip.ParserQwen3, mwip.ParserHermes
	out := run(t, mwip.EngineMWIP, spec, r)
	if out.reasoning != "The user is talking to the ModelWeave mock engine; I will answer briefly." {
		t.Fatalf("reasoning %q", out.reasoning)
	}
	if out.content != "" || len(out.calls) != 1 || out.calls[0].Name != "search_docs" {
		t.Fatalf("content %q calls %+v", out.content, out.calls)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(out.calls[0].Arguments), &args); err != nil || args["query"] != "ModelWeave" {
		t.Fatalf("arguments %q: %v", out.calls[0].Arguments, err)
	}
	if out.finish.Reason != "tool_calls" || out.finish.Usage.Reasoning() == 0 {
		t.Fatalf("finish %+v", out.finish)
	}
	// Non-thinking: plain content.
	r = render(false, false)
	r.ParsingOptions.ReasoningParser = mwip.ParserQwen3
	if out := run(t, mwip.EngineMWIP, spec, r); !strings.HasPrefix(out.content, "Hello from ModelWeave mock worker w0") || out.reasoning != "" {
		t.Fatalf("non-thinking %+v", out)
	}
}

// vLLM and SGLang prompts hold collapsed placeholders; the mock rebuilds the
// expanded prompt from the gateway's media hints, so its prompt length and
// block hashes match the gateway's.
func TestCollapsedMediaAreExpandedFromHints(t *testing.T) {
	spec := start(t)
	const vs, ve, img, f0 = 151652, 151653, 1<<30 + 1, 1<<30 + 2
	toks := []uint32{0, vs, img, img, img, ve, 27, 15, 29, vs, f0, f0, ve, 42, 128822}
	r := render(false, false)
	r.TokenIDs, r.PromptTokens = toks, len(toks)
	r.Multimodal = &mwip.Multimodal{PadShift: 1 << 30, Items: []mwip.MediaItem{
		{Index: 0, Modality: "image", Digest: "i", Span: mwip.Span{Offset: 2, Length: 3},
			Placements: []mwip.Placement{{Offset: 2, Length: 3, PadValue: img}}, Placeholder: []uint32{151655}},
		{Index: 1, Modality: "video", Digest: "v", Span: mwip.Span{Offset: 6, Length: 7},
			Placements: []mwip.Placement{{Offset: 10, Length: 2, PadValue: f0}}, Placeholder: []uint32{vs, 151656, ve}},
	}}
	for _, protocol := range []string{mwip.EngineVLLM, mwip.EngineSGLang} {
		out := run(t, protocol, spec, r)
		if out.finish == nil || out.finish.Usage.PromptTokens != len(toks) {
			t.Fatalf("%s: engine prompt tokens %+v, want %d", protocol, out.finish, len(toks))
		}
	}
	if st := getStats(t, spec.URL); st["media_items"] != float64(4) {
		t.Fatalf("media items %v", st["media_items"])
	}
}
