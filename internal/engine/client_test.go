package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

func f64(v float64) *float64 { return &v }

// render is an intermediate state: "<bos> a b [image x4] c [image x3] <think>".
func render() *mwip.RenderResult {
	return &mwip.RenderResult{
		RequestID: "r", PromptTokens: 11,
		TokenIDs: []uint32{0, 10, 11, 1 << 30, 1 << 30, 1 << 30, 1 << 30, 12, 1<<30 + 5, 1<<30 + 5, 1<<30 + 5},
		Multimodal: &mwip.Multimodal{PadShift: 1 << 30, PlaceholderTokenID: 129264, Items: []mwip.MediaItem{
			{Index: 0, Modality: "image", Digest: "d0", Source: mwip.MediaSource{Kind: "url", URL: "https://a/0.png"},
				Span: mwip.Span{Offset: 3, Length: 4}, Placements: []mwip.Placement{{Offset: 3, Length: 4, PadValue: 1 << 30, Reference: "d0", Modality: "image"}}},
			{Index: 1, Modality: "image", Digest: "d1", Source: mwip.MediaSource{Kind: "url", URL: "https://a/1.png"},
				Span: mwip.Span{Offset: 8, Length: 3}, Placements: []mwip.Placement{{Offset: 8, Length: 3, PadValue: 1<<30 + 5, Reference: "d1", Modality: "image"}}},
		}},
		Conversation: mwip.Conversation{ThinkingMode: true, ReasoningEffort: "high", ToolChoice: "auto", ResponseFormat: "text",
			Tools: []json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"search"}}`)}},
		InferenceOptions: mwip.InferenceOptions{MaxTokens: 64, Temperature: f64(1), TopP: f64(0.95), StopTokenIDs: []uint32{1},
			OutputConstraint: &mwip.OutputConstraint{Kind: "json", Schema: `{"type":"object"}`}},
		ParsingOptions: mwip.ParsingOptions{ParseToolCalls: true, ThinkingInitialStage: "reasoning_start", StopSequences: []string{"END"}},
	}
}

func TestCollapseRestoresOnePlaceholderPerImage(t *testing.T) {
	r := render()
	tokens, items, err := collapse(r, r.Multimodal)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint32{0, 10, 11, 129264, 12, 129264}
	if len(tokens) != len(want) || len(items) != 2 {
		t.Fatalf("collapsed %v items %d", tokens, len(items))
	}
	for i := range want {
		if tokens[i] != want[i] {
			t.Fatalf("collapsed %v, want %v", tokens, want)
		}
	}
	r.Multimodal.Items[1].Modality = "video"
	if _, _, err := collapse(r, r.Multimodal); err == nil || !strings.Contains(err.Error(), "mwip engine protocol") {
		t.Fatalf("video must need the native protocol: %v", err)
	}
}

func TestAdaptersCarryTheIntermediateState(t *testing.T) {
	st := &Stage{Render: render(), AttemptID: "r-1", ECParams: map[string]any{"d0": map[string]any{"uri": "x"}},
		Disagg: mwip.Disagg{Mode: mwip.ModeSplit, Role: "decode", Handshake: mwip.HandshakeBootstrap, Room: "r-1", BootstrapRoom: 42,
			Prefill: &mwip.Peer{WorkerID: "p0", URL: "http://p0:9000", BootstrapAddr: "http://p0:8998", EngineID: "p0"}}}
	v, err := toVLLM(st)
	if err != nil {
		t.Fatal(err)
	}
	if len(v.TokenIDs) != 6 || len(v.ContentParts) != 2 || v.ContentParts[1].UUID != "d1" || v.OutputMode != "derender" || !v.Stream {
		t.Fatalf("vllm request %+v", v)
	}
	if v.KVTransferParams["remote_bootstrap_addr"] != "http://p0:8998" || v.KVTransferParams["transfer_id"] != "xfer-r-1" || v.ECTransferParams["d0"] == nil {
		t.Fatalf("vllm pd/epd fields %v %v", v.KVTransferParams, v.ECTransferParams)
	}
	if string(v.SamplingParams.StructuredOutputs) != `{"json":{"type":"object"}}` || *v.ReasoningEnded || v.ParseContext.ChatTemplateKwargs["thinking"] != true {
		t.Fatalf("vllm sampling/parse %s %+v", v.SamplingParams.StructuredOutputs, v.ParseContext)
	}
	if h := v.ModelWeave.Media; len(h) != 2 || h[0].Tokens != 4 || h[1].PadValue != 1<<30+5 {
		t.Fatalf("media hints %+v", h)
	}
	s, err := toSGLang(st)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.InputIDs) != 6 || len(s.ImageData) != 2 || s.BootstrapHost != "p0" || *s.BootstrapPort != 8998 || *s.BootstrapRoom != 42 {
		t.Fatalf("sglang request %+v", s)
	}
	if s.SamplingParams["max_new_tokens"] != 64 || s.SamplingParams["json_schema"] != `{"type":"object"}` {
		t.Fatalf("sglang sampling %v", s.SamplingParams)
	}
	m := toMWIP(st)
	if len(m.TokenIDs) != 11 || m.Multimodal == nil || !m.ParsingOptions.ParseToolCalls || m.Disagg.Room != "r-1" || !m.Stream {
		t.Fatalf("mwip request %+v", m)
	}
}

// qwenVL is a Qwen3-VL intermediate state: an image whose pads sit between
// the template's <|vision_start|> / <|vision_end|>, then a video of two kept
// frames, each "<t seconds>" + <|vision_start|> + pads + <|vision_end|>.
func qwenVL() *mwip.RenderResult {
	const vs, ve, img, f0, f1 = 151652, 151653, 1<<30 + 1, 1<<30 + 2, 1<<30 + 3
	toks := []uint32{151644, 872, 198, vs, img, img, ve, 27, 15, 13, 20, 6486, 29, vs, f0, f0, ve, 27, 16, 13, 20, 6486, 29, vs, f1, f1, ve, 151645}
	return &mwip.RenderResult{
		RequestID: "q", PromptTokens: len(toks), TokenIDs: toks,
		Multimodal: &mwip.Multimodal{PadShift: 1 << 30, PlaceholderTokenID: 151655, Items: []mwip.MediaItem{
			{Index: 0, Modality: "image", Digest: "i", Source: mwip.MediaSource{URL: "https://a/i.png"}, Span: mwip.Span{Offset: 4, Length: 2},
				Placements: []mwip.Placement{{Offset: 4, Length: 2, PadValue: img}}, Placeholder: []uint32{151655}},
			{Index: 1, Modality: "video", Digest: "v", Source: mwip.MediaSource{URL: "https://a/v.mp4"}, Span: mwip.Span{Offset: 7, Length: 20},
				Placements:  []mwip.Placement{{Offset: 14, Length: 2, PadValue: f0}, {Offset: 24, Length: 2, PadValue: f1}},
				Placeholder: []uint32{vs, 151656, ve}},
		}},
		Conversation:     mwip.Conversation{ToolChoice: "auto", ResponseFormat: "text", ChatTemplateKwargs: map[string]any{"enable_thinking": false}},
		InferenceOptions: mwip.InferenceOptions{MaxTokens: 16, StopTokenIDs: []uint32{151645, 151643}},
		ParsingOptions:   mwip.ParsingOptions{ToolCallParser: mwip.ParserHermes},
	}
}

// Engines that expand media themselves get each item as the model's own
// placeholder: Qwen-VL images and videos both have one.
func TestCollapseUsesPerItemPlaceholders(t *testing.T) {
	r := qwenVL()
	tokens, items, err := collapse(r, r.Multimodal)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint32{151644, 872, 198, 151652, 151655, 151653, 151652, 151656, 151653, 151645}
	if fmt.Sprint(tokens) != fmt.Sprint(want) || len(items) != 2 {
		t.Fatalf("collapsed %v, want %v", tokens, want)
	}
	hints := mediaHints(r.TokenIDs, items)
	if hints[0].Layout != nil || fmt.Sprint(hints[0].Placeholder) != "[151655]" {
		t.Fatalf("image hint %+v", hints[0])
	}
	// The video's expanded span, run-length encoded.
	if l := hints[1].Layout; len(l) != 18 || l[7] != [2]uint32{1<<30 + 2, 2} || l[17] != [2]uint32{151653, 1} {
		t.Fatalf("video layout %v", l)
	}
	st := &Stage{Render: r, AttemptID: "q-1", Disagg: mwip.Disagg{Mode: mwip.ModeNormal}}
	v, err := toVLLM(st)
	if err != nil {
		t.Fatal(err)
	}
	if v.ContentParts[0].Type != "image_url" || v.ContentParts[1].Type != "video_url" || v.ContentParts[1].UUID != "v" {
		t.Fatalf("content parts %+v", v.ContentParts)
	}
	// The tokenizer's template variables reach vLLM's parsers as they are.
	if kw := v.ParseContext.ChatTemplateKwargs; len(kw) != 1 || kw["enable_thinking"] != false || !*v.ReasoningEnded {
		t.Fatalf("parse context %+v", v.ParseContext)
	}
	s, err := toSGLang(st)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(s.ImageData) != "[https://a/i.png]" || fmt.Sprint(s.VideoData) != "[https://a/v.mp4]" || len(s.InputIDs) != 10 {
		t.Fatalf("sglang media %+v", s)
	}
	enc, err := vllmEncodeRequest(&EncodeStage{AttemptID: "e", Items: r.Multimodal.Items})
	if err != nil || fmt.Sprint(enc.TokenIDs) != "[151655 151652 151656 151653]" {
		t.Fatalf("encode leg %v %v", enc, err)
	}
	// Without a placeholder (DeepSeek V4.1 videos) only mwip can carry it.
	r.Multimodal.Items[1].Placeholder = nil
	if err := Collapsible(r.Multimodal); err == nil || !strings.Contains(err.Error(), "no placeholder for video") {
		t.Fatalf("collapsible: %v", err)
	}
}

func serve(t *testing.T, path string, lines []string, sse bool, seen *map[string]any) registry.Spec {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != path {
			http.NotFound(w, r)
			return
		}
		_ = json.NewDecoder(r.Body).Decode(seen)
		for _, l := range lines {
			if sse {
				_, _ = io.WriteString(w, "data: "+l+"\n\n")
			} else {
				_, _ = io.WriteString(w, l+"\n")
			}
		}
	}))
	t.Cleanup(srv.Close)
	return registry.Spec{ID: "w", Role: registry.RoleDecode, URL: srv.URL}
}

func collect(t *testing.T, s *Stream) (string, *mwip.GenerateEvent, error) {
	t.Helper()
	var text strings.Builder
	for {
		ev, err := s.Next()
		if errors.Is(err, io.EOF) {
			return text.String(), nil, nil
		}
		if err != nil {
			return text.String(), ev, err
		}
		text.WriteString(ev.Content)
		if ev.Type == mwip.EventFinish || ev.Type == mwip.EventPrefillDone {
			return text.String(), ev, nil
		}
	}
}

func TestMWIPEvents(t *testing.T) {
	var seen map[string]any
	spec := serve(t, "/mwip/v1/generate", []string{
		`{"type":"delta","reasoning":"think","token_ids":[5]}`,
		`{"type":"delta","content":"Hi","token_ids":[6]}`,
		`{"type":"finish","reason":"stop","usage":{"prompt_tokens":11,"completion_tokens":2}}`,
	}, false, &seen)
	s, err := New().Generate(context.Background(), mwip.EngineMWIP, spec, &Stage{Render: render(), AttemptID: "r-1", Disagg: mwip.Disagg{Mode: mwip.ModeNormal}})
	if err != nil {
		t.Fatal(err)
	}
	text, fin, err := collect(t, s)
	s.Close()
	if err != nil || text != "Hi" || fin.Reason != "stop" || fin.Usage.PromptTokens != 11 {
		t.Fatalf("text %q finish %+v err %v", text, fin, err)
	}
	if seen["stream"] != true || len(seen["token_ids"].([]any)) != 11 || seen["parsing_options"].(map[string]any)["thinking_initial_stage"] != "reasoning_start" {
		t.Fatalf("native request %v", seen)
	}

	spec = serve(t, "/mwip/v1/generate", []string{
		`{"type":"prefill_done","prompt_tokens":11,"kv_handle":{"worker_id":"p","url":"http://p","room":"r-1","tokens":11}}`,
	}, false, &seen)
	s, err = New().Generate(context.Background(), mwip.EngineMWIP, spec, &Stage{Render: render(), AttemptID: "r-1",
		Disagg: mwip.Disagg{Mode: mwip.ModeSplit, Role: "prefill", Handshake: mwip.HandshakePull}})
	if err != nil {
		t.Fatal(err)
	}
	_, done, err := collect(t, s)
	s.Close()
	if err != nil || done.Type != mwip.EventPrefillDone || done.Handle == nil || done.Handle.Room != "r-1" {
		t.Fatalf("prefill leg %+v %v", done, err)
	}
}

func TestSGLangCumulativeAndIncrementalStreams(t *testing.T) {
	cumulative := []string{
		`{"text":"Hel","output_ids":[1],"meta_info":{"id":"r","completion_tokens":1}}`,
		`{"text":"Hello","output_ids":[1,2],"meta_info":{"id":"r","completion_tokens":2}}`,
		`{"text":"Hello world","output_ids":[1,2,3],"meta_info":{"id":"r","completion_tokens":3,"prompt_tokens":5,"cached_tokens":2,"finish_reason":{"type":"stop","matched":"four"}}}`,
		`[DONE]`,
	}
	incremental := []string{
		`{"text":"Hel","output_ids":[1],"meta_info":{"id":"r","completion_tokens":1}}`,
		`{"text":"lo","output_ids":[2],"meta_info":{"id":"r","completion_tokens":2}}`,
		`{"text":" world","output_ids":[3],"meta_info":{"id":"r","completion_tokens":3,"prompt_tokens":5,"cached_tokens":2,"finish_reason":{"type":"stop","matched":"four"}}}`,
		`[DONE]`,
	}
	for name, chunks := range map[string][]string{"cumulative": cumulative, "incremental": incremental} {
		var seen map[string]any
		spec := serve(t, "/generate", chunks, true, &seen)
		s, err := New().Generate(context.Background(), mwip.EngineSGLang, spec, &Stage{Render: render(), AttemptID: "r"})
		if err != nil {
			t.Fatal(err)
		}
		text, fin, err := collect(t, s)
		s.Close()
		if err != nil || text != "Hello world" || fin == nil || fin.Reason != "stop" || fin.StopReason != "four" || fin.Usage.Cached() != 2 {
			t.Fatalf("%s: text %q finish %+v err %v", name, text, fin, err)
		}
		if seen["stream"] != true || seen["rid"] != "r" || len(seen["image_data"].([]any)) != 2 {
			t.Fatalf("%s: SGLang request %v", name, seen)
		}
	}
}

func TestVLLMPrefillReturnsKVParamsAndErrors(t *testing.T) {
	var seen map[string]any
	spec := serve(t, "/inference/v1/generate", []string{
		`{"request_id":"r","output_mode":"derender","choices":[{"index":0,"delta":{"role":"assistant","content":"x"},"finish_reason":"length","token_ids":[5]}]}`,
		`{"request_id":"r","output_mode":"derender","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1},"kv_transfer_params":{"remote_host":"p","remote_port":1}}`,
		`[DONE]`,
	}, true, &seen)
	s, err := New().Generate(context.Background(), mwip.EngineVLLM, spec, &Stage{Render: render(), AttemptID: "r",
		Disagg: mwip.Disagg{Mode: mwip.ModeSplit, Role: "prefill", Handshake: mwip.HandshakePull, Room: "a-1"}})
	if err != nil {
		t.Fatal(err)
	}
	text, done, err := collect(t, s)
	s.Close()
	if err != nil || text != "" || done.Type != mwip.EventPrefillDone || done.KVTransferParams["remote_host"] != "p" || done.Usage.PromptTokens != 9 {
		t.Fatalf("prefill leg: text %q done %+v err %v", text, done, err)
	}
	kv := seen["kv_transfer_params"].(map[string]any)
	if seen["stream"] != true || seen["sampling_params"].(map[string]any)["max_tokens"] != float64(1) || kv["do_remote_decode"] != true {
		t.Fatalf("prefill request %v", seen)
	}

	spec = serve(t, "/inference/v1/generate", []string{
		`{"error":{"message":"KV load failed","type":"kv_transfer_failed","code":502}}`, `[DONE]`,
	}, true, &seen)
	s, err = New().Generate(context.Background(), mwip.EngineVLLM, spec, &Stage{Render: render(), AttemptID: "r"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = collect(t, s)
	s.Close()
	if !IsCode(err, mwip.ErrKVTransfer) {
		t.Fatalf("want kv_transfer_failed, got %v", err)
	}
}

func TestVLLMEncodeLeg(t *testing.T) {
	var seen map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&seen)
		_, _ = io.WriteString(w, `{"request_id":"e","choices":[{"index":0,"finish_reason":"length"}],"ec_transfer_params":{"d0":{"uri":"mock://e/x"}}}`)
	}))
	defer srv.Close()
	r := render()
	res, err := New().Encode(context.Background(), mwip.EngineVLLM, registry.Spec{ID: "e", URL: srv.URL},
		&EncodeStage{AttemptID: "e-1", Items: r.Multimodal.Items[:1], PlaceholderTokenID: 129264})
	if err != nil || res.ECParams["d0"] == nil {
		t.Fatalf("encode: %+v %v", res, err)
	}
	if seen["stream"] != false || seen["token_ids"].([]any)[0] != float64(129264) || seen["content_parts"].([]any)[0].(map[string]any)["uuid"] != "d0" {
		t.Fatalf("encode request %v", seen)
	}
	if _, err := New().Encode(context.Background(), mwip.EngineSGLang, registry.Spec{ID: "e", URL: srv.URL}, &EncodeStage{}); err == nil {
		t.Fatal("sglang encodes inside the engine")
	}
}
