package mockengine_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mockengine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// worker is a running mock instance: its HTTP endpoints (stats) and its
// mwip Engine service (gRPC).
type worker struct {
	http string
	grpc string
}

// spec is the instance as the gateway registers it.
func (w worker) spec() registry.Spec {
	return registry.Spec{ID: "w0", Model: "m", Role: registry.RoleUnified, Target: w.grpc}
}

func serveGRPC(t *testing.T, e *mockengine.Engine) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := mockengine.NewGRPCServer(e)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return lis.Addr().String()
}

func start(t *testing.T) worker {
	t.Helper()
	eng := mockengine.New(mockengine.Config{
		ID: "w0", Model: "m", Role: "unified", MaxOutputTokens: 256, MaxRunning: 8,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	srv := httptest.NewServer(eng.Handler())
	t.Cleanup(srv.Close)
	return worker{http: srv.URL, grpc: serveGRPC(t, eng)}
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

func run(t *testing.T, w worker, r *mwip.RenderResult) result {
	t.Helper()
	s, err := engine.New().Generate(context.Background(), w.spec(), &engine.Stage{Render: r, AttemptID: "r-1", Disagg: mwip.Disagg{Mode: mwip.ModeNormal}})
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
			t.Fatalf("stream: %v", err)
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

// The engine parses its own output: reasoning and DSML tool calls arrive
// as deltas.
func TestEngineSideParsing(t *testing.T) {
	w := start(t)
	r := run(t, w, render(true, true))
	st := getStats(t, w.http)
	// Requests are recorded in their JSON form, with proto field names.
	req, _ := st["last_request"].(map[string]any)
	if po, _ := req["parsing_options"].(map[string]any); req == nil || req["request_id"] != "r-1" ||
		po["parse_tool_calls"] != true || po["thinking_initial_stage"] != "THINKING_STAGE_REASONING_START" {
		t.Fatalf("recorded request %v", req)
	}
	if !strings.Contains(r.reasoning, "I will answer briefly.") || strings.Contains(r.reasoning, "</think>") {
		t.Fatalf("reasoning %q", r.reasoning)
	}
	if r.content != "" || len(r.calls) != 1 || r.calls[0].Name != "search_docs" {
		t.Fatalf("content %q calls %+v", r.content, r.calls)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(r.calls[0].Arguments), &args); err != nil || args["query"] != "ModelWeave" || args["worker"] != "w0" {
		t.Fatalf("arguments %q: %v", r.calls[0].Arguments, err)
	}
	f := r.finish
	if f == nil || f.Reason != "tool_calls" || f.Usage == nil || f.Usage.PromptTokens != 3 || f.Usage.Reasoning() == 0 {
		t.Fatalf("finish %+v", f)
	}
	if st["generate_requests"] != float64(1) {
		t.Fatalf("stats %v", st)
	}
}

func TestStopStringsInTheEngine(t *testing.T) {
	w := start(t)
	r := render(false, false)
	r.ParsingOptions.StopSequences = []string{"four"}
	out := run(t, w, r)
	if !strings.HasSuffix(out.content, "one two three ") || out.finish.Reason != "stop" || out.finish.StopReason != "four" {
		t.Fatalf("content %q finish %+v", out.content, out.finish)
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
	w := start(t)
	r := render(true, true)
	r.TokenIDs = []uint32{151644, 77091, 198} // <|im_start|>assistant\n: Qwen3 opens <think> itself
	r.ParsingOptions.ReasoningParser, r.ParsingOptions.ToolCallParser = mwip.ParserQwen3, mwip.ParserHermes
	out := run(t, w, r)
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
	if out := run(t, w, r); !strings.HasPrefix(out.content, "Hello from ModelWeave mock worker w0") || out.reasoning != "" {
		t.Fatalf("non-thinking %+v", out)
	}
}

// Media arrive expanded: every position of an item holds its pad value (a
// video also its timestamp text), so the prompt is the token list as is.
// Without embedding artifacts the instance encodes the items itself.
func TestMediaArriveExpanded(t *testing.T) {
	w := start(t)
	const vs, ve = 151652, 151653
	ip := mwip.MediaPlan{Processor: "qwen_vl.v1", Width: 96, Height: 32, Tokens: 3, NFrames: 1, PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2}
	vp := mwip.MediaPlan{Processor: "qwen_vl.v1", Width: 64, Height: 32, Tokens: 2, NFrames: 2, PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2, FrameIndices: []uint64{0, 1}, Timestamps: []float64{0.5}}
	di, dv := strings.Repeat("a", 64), strings.Repeat("b", 64)
	pi, pv := mwip.PlanID("image", di, ip), mwip.PlanID("video", dv, vp)
	ref := mwip.PlacementReference(pv, "video", 0)
	img, f0 := mwip.PadValue(pi), mwip.PadValue(ref)
	toks := []uint32{0, vs, img, img, img, ve, 27, 15, 29, vs, f0, f0, ve, 42, 128822}
	r := render(false, false)
	r.TokenIDs, r.PromptTokens = toks, len(toks)
	r.Multimodal = &mwip.Multimodal{PadShift: 1 << 30, Items: []mwip.MediaItem{
		{Index: 0, Modality: "image", Digest: di, Source: mwip.MediaSource{Kind: "url", URL: "https://img/i"}, Plan: ip, PlanID: pi,
			Placements: []mwip.Placement{{Offset: 2, Length: 3, PadValue: img, Reference: pi, Modality: "image"}}},
		{Index: 1, Modality: "video", Digest: dv, Source: mwip.MediaSource{Kind: "url", URL: "https://img/v"}, Plan: vp, PlanID: pv,
			Placements: []mwip.Placement{{Offset: 10, Length: 2, PadValue: f0, Reference: ref, Modality: "frame", Timestamp: &vp.Timestamps[0]}}},
	}}
	out := run(t, w, r)
	if out.finish == nil || out.finish.Usage.PromptTokens != len(toks) {
		t.Fatalf("engine prompt tokens %+v, want %d", out.finish, len(toks))
	}
	if st := getStats(t, w.http); st["media_items"] != float64(2) || st["local_encoded"] != float64(2) {
		t.Fatalf("media %v %v", st["media_items"], st["local_encoded"])
	}
}
