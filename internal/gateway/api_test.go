package gateway

import (
	"bufio"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/mockengine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// fakeTokenizer answers /v1/render with a fixed intermediate state: a
// DeepSeek-style prompt that opens a reasoning block.
func fakeTokenizer(t *testing.T) string {
	t.Helper()
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req mwip.RenderRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		var body struct {
			Stream bool `json:"stream"`
		}
		_ = json.Unmarshal(req.Body, &body)
		one := 1.0
		rr := mwip.RenderResult{
			RequestID: req.RequestID, Protocol: req.Protocol, Model: "m", ModelVersion: "v4.1",
			PromptTokens: 4, TokenIDs: []uint32{0, 11, 12, 128821},
			Conversation:     mwip.Conversation{ThinkingMode: true, ToolChoice: "auto", ResponseFormat: "text"},
			InferenceOptions: mwip.InferenceOptions{MaxTokens: 100, StopTokenIDs: []uint32{1}, Temperature: &one},
			ParsingOptions:   mwip.ParsingOptions{ThinkingInitialStage: "reasoning_start"},
			Stream:           body.Stream, ResponseOptions: mwip.ResponseOptions{IncludeUsage: true},
			RenderArtifact: mwip.RenderArtifact{Version: 2, Encoding: "deepseek_v4_1"},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(rr)
	}))
	srv.Config.Protocols = new(http.Protocols)
	srv.Config.Protocols.SetHTTP1(true)
	srv.Config.Protocols.SetUnencryptedHTTP2(true)
	srv.Start()
	t.Cleanup(srv.Close)
	return srv.URL
}

// mockWorker starts a mock instance and returns its HTTP base (counters)
// and its mwip target.
func mockWorker(t *testing.T) (string, string) {
	t.Helper()
	eng := mockengine.New(mockengine.Config{ID: "w0", Model: "m", Role: "unified", MaxOutputTokens: 256, MaxRunning: 8,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil))})
	srv := httptest.NewServer(eng.Handler())
	t.Cleanup(srv.Close)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := mockengine.NewGRPCServer(eng)
	go func() { _ = gs.Serve(lis) }()
	t.Cleanup(gs.Stop)
	return srv.URL, lis.Addr().String()
}

func engineStats(t *testing.T, base string) map[string]any {
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

func newGateway(t *testing.T, target string) *httptest.Server {
	t.Helper()
	cfg, err := config.Parse([]byte(`
tokenizer: {endpoints: ["` + fakeTokenizer(t) + `"]}
engine_state: {poll: {enabled: false}}
discovery:
  workers: [{id: w0, model: m, role: unified, target: "` + target + `"}]
models: [{name: m, mode: normal}]
`))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "test")
	if err != nil {
		t.Fatal(err)
	}
	gw := httptest.NewServer(s.Handler())
	t.Cleanup(gw.Close)
	return gw
}

func post(t *testing.T, gw *httptest.Server, path, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(gw.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func sse(resp *http.Response) (events, data []string) {
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		switch l := sc.Text(); {
		case strings.HasPrefix(l, "event: "):
			events = append(events, strings.TrimPrefix(l, "event: "))
		case strings.HasPrefix(l, "data: "):
			data = append(data, strings.TrimPrefix(l, "data: "))
		}
	}
	return events, data
}

type chatResponse struct {
	Object  string `json:"object"`
	Choices []struct {
		Message struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens            int `json:"prompt_tokens"`
		CompletionTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

// Clients may stream or not; the engine is always called streaming and
// parses its own output. The gateway formats every client protocol.
func TestStreamAndNonStreamClientsOverStreamingEngine(t *testing.T) {
	worker, target := mockWorker(t)
	gw := newGateway(t, target)

	// Chat, non-streaming: one JSON body with reasoning and content.
	resp := post(t, gw, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	var chat chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chat); err != nil || resp.StatusCode != 200 {
		t.Fatalf("chat: %d %v", resp.StatusCode, err)
	}
	resp.Body.Close()
	m := chat.Choices[0].Message
	if chat.Object != "chat.completion" || !strings.HasPrefix(m.Content, "Hello from ModelWeave mock worker w0") ||
		!strings.Contains(m.ReasoningContent, "answer briefly") || chat.Choices[0].FinishReason != "stop" ||
		chat.Usage.PromptTokens != 4 || chat.Usage.CompletionTokensDetails.ReasoningTokens == 0 {
		t.Fatalf("chat response %+v", chat)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content type %q", ct)
	}

	// Chat, streaming: SSE chunks ending with [DONE].
	resp = post(t, gw, "/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("stream content type %q", ct)
	}
	_, data := sse(resp)
	if len(data) < 5 || data[len(data)-1] != "[DONE]" || !strings.Contains(strings.Join(data, ""), `"reasoning_content"`) {
		t.Fatalf("chat stream %v", data)
	}

	// Anthropic Messages, non-streaming: thinking and text blocks.
	resp = post(t, gw, "/v1/messages", `{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	var msg struct {
		Type    string           `json:"type"`
		Content []map[string]any `json:"content"`
		Stop    string           `json:"stop_reason"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&msg)
	resp.Body.Close()
	if msg.Type != "message" || len(msg.Content) != 2 || msg.Content[0]["type"] != "thinking" || msg.Content[1]["type"] != "text" || msg.Stop != "end_turn" {
		t.Fatalf("messages %+v", msg)
	}

	// Responses, streaming: lifecycle events.
	events, _ := sse(post(t, gw, "/v1/responses", `{"model":"m","stream":true,"input":"hi"}`))
	joined := strings.Join(events, ",")
	if !strings.HasPrefix(joined, "response.created,response.in_progress,") || !strings.HasSuffix(joined, ",response.completed") ||
		!strings.Contains(joined, "response.reasoning_text.delta") || !strings.Contains(joined, "response.output_text.delta") {
		t.Fatalf("responses events %s", joined)
	}

	st := engineStats(t, worker)
	if st["generate_requests"] != float64(4) {
		t.Fatalf("engine calls: %v", st)
	}
	// The intermediate state as the tokenizer produced it.
	last := st["last_request"].(map[string]any)
	if len(last["token_ids"].([]any)) != 4 ||
		last["parsing_options"].(map[string]any)["thinking_initial_stage"] != "THINKING_STAGE_REASONING_START" ||
		last["disagg"].(map[string]any)["mode"] != "DISAGG_MODE_NORMAL" {
		t.Fatalf("engine request %v", last)
	}
}
