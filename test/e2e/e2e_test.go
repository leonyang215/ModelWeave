//go:build e2e

package e2e

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	// Wait for every worker to register (heartbeats) with both gateways and
	// for state polling.
	deadline := time.Now().Add(90 * time.Second)
	all := append(append([]string{}, encodeWorkers...), generateWorkers...)
	if !waitWorkers(gateway, all, deadline) || !waitWorkers(gatewayEv, secondGatewayWorkers, deadline) {
		println("workers did not register in time")
	}
	m.Run()
}

// waitWorkers waits until the gateway lists every worker as healthy with
// engine state.
func waitWorkers(gw string, want []string, deadline time.Time) bool {
	for {
		req, _ := http.NewRequest(http.MethodGet, gw+"/admin/workers", nil)
		req.Header.Set("X-ModelWeave-Token", ctlToken)
		if r, err := client.Do(req); err == nil {
			var out struct {
				Workers []worker `json:"workers"`
			}
			_ = json.NewDecoder(r.Body).Decode(&out)
			r.Body.Close()
			seen := map[string]bool{}
			for _, w := range out.Workers {
				if w.Healthy && w.State != nil {
					seen[w.ID] = true
				}
			}
			ok := true
			for _, id := range want {
				ok = ok && seen[id]
			}
			if ok {
				return true
			}
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(500 * time.Millisecond)
	}
}

// Workers that only encode, and workers that generate (every mock worker
// serves all three engine protocols).
var (
	encodeWorkers   = []string{"enc-0", "ev-0", "es-0", "qvl-enc-0"}
	generateWorkers = []string{"prefill-0", "prefill-1", "decode-0", "decode-1", "pv-0", "dv-0", "pp-0", "dp-0",
		"ps-0", "ds-0", "pf-0", "df-0", "normal-0", "normal-1", "normal-bad", "tiny-0",
		"qwen-p0", "qwen-d0", "qvl-p0", "qvl-d0", "qvl-sg-0"}
	// Workers that also heartbeat to the second gateway.
	secondGatewayWorkers = []string{"prefill-0", "prefill-1", "decode-0", "decode-1",
		"qwen-p0", "qwen-d0", "qvl-enc-0", "qvl-p0", "qvl-d0"}
)

func TestWorkersRegisteredWithState(t *testing.T) {
	ws := workers(t, gateway)
	// The protocol is the model's engine protocol: the gateway decides it,
	// the workers serve all of them.
	for id, want := range map[string][2]string{
		"enc-0": {"encode", "mwip"}, "prefill-0": {"prefill", "mwip"}, "decode-0": {"decode", "mwip"},
		"ev-0": {"encode", "vllm"}, "pv-0": {"prefill", "vllm"}, "dv-0": {"decode", "vllm"},
		"es-0": {"encode", "sglang"}, "ds-0": {"decode", "sglang"},
		"normal-0": {"unified", "mwip"}, "tiny-0": {"unified", "mwip"},
	} {
		w, ok := ws[id]
		if !ok || w.Role != want[0] || w.Protocol != want[1] || !w.Healthy {
			t.Fatalf("worker %s: %+v, want role %s protocol %s", id, w, want[0], want[1])
		}
		if w.State == nil {
			t.Fatalf("worker %s has no engine state", id)
		}
	}
	if !ws["decode-0"].Normal || !ws["dv-0"].Normal || ws["ds-0"].Normal || !ws["normal-0"].Normal {
		t.Fatalf("normal flags: decode-0 %v dv-0 %v ds-0 %v normal-0 %v", ws["decode-0"].Normal, ws["dv-0"].Normal, ws["ds-0"].Normal, ws["normal-0"].Normal)
	}
	// tiny-0 is static and only polled through its Prometheus endpoint.
	if c := ws["tiny-0"].State["kv_capacity_tokens"]; c != float64(262144) {
		t.Fatalf("tiny-0 state not polled: %v", ws["tiny-0"].State)
	}
	evWorkers := workers(t, gatewayEv)
	for _, id := range secondGatewayWorkers {
		if w, ok := evWorkers[id]; !ok || !w.Healthy || w.State == nil {
			t.Fatalf("second gateway misses %s: %+v", id, w)
		}
	}
}

func TestControlPlaneToken(t *testing.T) {
	r, err := client.Get(gateway + "/admin/workers")
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != http.StatusUnauthorized {
		t.Fatalf("admin API without token: %d", r.StatusCode)
	}
	fake := post(t, gateway, "/mwip/v1/heartbeat", "", map[string]any{"worker": map[string]any{
		"id": "intruder", "model": "deepseek-v4.1", "role": "decode", "url": "http://evil:9000"}}, nil)
	if fake.status != http.StatusUnauthorized {
		t.Fatalf("heartbeat without token: %d", fake.status)
	}
	if _, ok := workers(t, gateway)["intruder"]; ok {
		t.Fatal("unauthenticated worker registered")
	}
}

func TestAuthAndModels(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", "", chatBody("deepseek-v4.1", "hi", nil), nil)
	if r.status != 401 || r.json(t)["error"].(map[string]any)["type"] != "invalid_api_key" {
		t.Fatalf("missing key: %d %s", r.status, r.body)
	}
	r = post(t, gateway, "/v1/messages", "nope", map[string]any{"model": "deepseek-v4.1", "max_tokens": 5, "messages": []any{}}, nil)
	if m := r.json(t); r.status != 401 || m["type"] != "error" || m["error"].(map[string]any)["type"] != "authentication_error" {
		t.Fatalf("anthropic-shaped 401 expected: %d %s", r.status, r.body)
	}
	r = post(t, gateway, "/v1/chat/completions", keyA, chatBody("gpt-4", "hi", nil), nil)
	if r.status != 404 {
		t.Fatalf("unknown model: %d %s", r.status, r.body)
	}
	var models struct {
		Data []struct{ ID string } `json:"data"`
	}
	_ = json.Unmarshal(get(t, gateway+"/v1/models"), &models)
	if len(models.Data) < 6 {
		t.Fatalf("models: %+v", models)
	}
}

// The tokenizer converts and tokenizes only: its result is deepseek-recipe's
// ConversationRequest with the conversation rendered into one token list
// (the intermediate state), not an engine request.
func TestTokenizerRender(t *testing.T) {
	status, m := render(t, "chat_completions", chatBody("deepseek-v4.1", "Hello", map[string]any{"thinking": map[string]any{"type": "disabled"}, "stop": []string{"END"}}))
	ids, _ := m["token_ids"].([]any)
	if status != 200 || len(ids) == 0 || ids[0] != float64(0) || ids[len(ids)-1] != float64(128822) || m["prompt_tokens"] != float64(len(ids)) {
		t.Fatalf("render: %d %v", status, m)
	}
	if path(m, "inference_options", "stop_token_ids").([]any)[0] != float64(1) || path(m, "parsing_options", "stop_sequences").([]any)[0] != "END" ||
		path(m, "conversation", "thinking_mode") != false || path(m, "inference_options", "max_tokens").(float64) <= 0 {
		t.Fatalf("options: %v %v %v", m["conversation"], m["inference_options"], m["parsing_options"])
	}
	if path(m, "render_artifact", "encoding") != "deepseek-v4.1" || path(m, "render_artifact", "version") != float64(2) ||
		m["protocol"] != "chat_completions" || m["stream"] != false || m["multimodal"] != nil {
		t.Fatalf("artifact/protocol: %v", m)
	}
	for _, gone := range []string{"generate", "output", "sampling_params"} {
		if _, ok := m[gone]; ok {
			t.Fatalf("intermediate state carries engine field %q", gone)
		}
	}
	if gone := post(t, tokURL, "/v1/sessions", "", map[string]any{}, nil); gone.status != http.StatusNotFound {
		t.Fatalf("tokenizer still serves output sessions: %d", gone.status)
	}
}

// normal = no PD split: a short prompt runs prefill and decode on one
// decode worker, which received the intermediate state as is (mwip).
func TestChatShortPromptRunsNormal(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if r.status != 200 {
		t.Fatalf("status %d: %s", r.status, r.body)
	}
	if r.header.Get("X-ModelWeave-Mode") != "normal" || r.header.Get("X-ModelWeave-Reason") != "short_prompt" {
		t.Fatalf("routing headers: %v", r.header)
	}
	msg := message(t, r.json(t))
	content := msg["message"].(map[string]any)["content"].(string)
	if !strings.HasPrefix(content, "Hello from ModelWeave mock worker decode-") || msg["finish_reason"] != "stop" {
		t.Fatalf("unexpected answer %q (%v)", content, msg["finish_reason"])
	}
	d := r.header.Get("X-ModelWeave-Decode-Worker")
	st := workerStats(t, d)
	req := st["last_request"].(map[string]any)
	if st["last_protocol"] != "mwip" || path(req, "disagg", "mode") != "normal" || req["stream"] != true ||
		len(req["token_ids"].([]any)) == 0 || path(req, "conversation", "thinking_mode") != false {
		t.Fatalf("%s received %v: %v", d, st["last_protocol"], req)
	}
}

// A long uncached prompt runs prefill/decode split (push handshake); the
// same prompt again is a prefix hit on the decode worker that received the
// KV, which then runs it normal.
func TestLongPromptPDThenPrefixHit(t *testing.T) {
	prompt := longPrompt()
	body := chatBody("deepseek-v4.1", prompt, map[string]any{"stream": true, "stream_options": map[string]any{"include_usage": true}})
	h, events := stream(t, gateway, "/v1/chat/completions", keyA, body, nil)
	if h.Get("X-ModelWeave-Mode") != "pd" || h.Get("X-ModelWeave-Prefill-Worker") == "" {
		t.Fatalf("first request should run pd: %v", h)
	}
	decodeWorker := h.Get("X-ModelWeave-Decode-Worker")
	if len(events) < 4 || events[len(events)-1].data != "[DONE]" {
		t.Fatalf("bad SSE stream (%d events)", len(events))
	}
	var reasoning, content strings.Builder
	var usage map[string]any
	for _, ev := range events[:len(events)-1] {
		var c map[string]any
		if err := json.Unmarshal([]byte(ev.data), &c); err != nil {
			t.Fatalf("bad chunk %q", ev.data)
		}
		if u, ok := c["usage"].(map[string]any); ok {
			usage = u
		}
		for _, ch := range c["choices"].([]any) {
			d := ch.(map[string]any)["delta"].(map[string]any)
			if s, ok := d["reasoning_content"].(string); ok {
				reasoning.WriteString(s)
			}
			if s, ok := d["content"].(string); ok {
				content.WriteString(s)
			}
		}
	}
	if !strings.Contains(content.String(), "Hello from ModelWeave mock worker "+decodeWorker) || reasoning.Len() == 0 {
		t.Fatalf("content %q reasoning %q", content.String(), reasoning.String())
	}
	if usage == nil || usage["prompt_tokens"].(float64) < 800 {
		t.Fatalf("usage %v", usage)
	}
	// Push: D registered a receiver, P pushed the KV to it.
	p, d := lastRequest(t, h.Get("X-ModelWeave-Prefill-Worker")), lastRequest(t, decodeWorker)
	if path(p, "disagg", "role") != "prefill" || path(p, "disagg", "kv_receiver", "url") != "http://"+decodeWorker+":9000" ||
		path(d, "disagg", "role") != "decode" || path(d, "disagg", "handshake") != "push" {
		t.Fatalf("push fields: P %v D %v", p["disagg"], d["disagg"])
	}

	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", prompt, nil), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "normal" || r.header.Get("X-ModelWeave-Reason") != "prefix_hit" {
		t.Fatalf("second request: %d %v", r.status, r.header)
	}
	if r.header.Get("X-ModelWeave-Decode-Worker") != decodeWorker {
		t.Fatalf("cache-aware routing picked %s, KV lives on %s", r.header.Get("X-ModelWeave-Decode-Worker"), decodeWorker)
	}
	cached := r.json(t)["usage"].(map[string]any)["prompt_tokens_details"].(map[string]any)["cached_tokens"].(float64)
	if cached < 768 {
		t.Fatalf("engine reported %v cached tokens", cached)
	}
}

func TestMultiTurnSessionAffinity(t *testing.T) {
	session := "sess-" + nonce()
	hdr := map[string]string{"X-Session-Id": session}
	first := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", longPrompt(), nil), hdr)
	if first.status != 200 {
		t.Fatalf("turn 1: %d %s", first.status, first.body)
	}
	d := first.header.Get("X-ModelWeave-Decode-Worker")
	answer := message(t, first.json(t))["message"].(map[string]any)["content"].(string)
	body := map[string]any{"model": "deepseek-v4.1", "messages": []map[string]any{
		{"role": "user", "content": "A completely different long question " + longPrompt()},
		{"role": "assistant", "content": answer},
		{"role": "user", "content": "And a follow-up?"},
	}}
	second := post(t, gateway, "/v1/chat/completions", keyA, body, hdr)
	if second.status != 200 || second.header.Get("X-ModelWeave-Reason") != "session" || second.header.Get("X-ModelWeave-Decode-Worker") != d {
		t.Fatalf("turn 2 should stick to %s via session affinity: %v", d, second.header)
	}
}

func TestAnthropicMessages(t *testing.T) {
	r := post(t, gateway, "/v1/messages", keyA, map[string]any{
		"model": "deepseek-v4.1", "max_tokens": 256, "messages": []any{map[string]any{"role": "user", "content": "Hello"}},
	}, map[string]string{"anthropic-version": "2023-06-01"})
	m := r.json(t)
	if r.status != 200 || m["type"] != "message" || m["role"] != "assistant" || m["stop_reason"] != "end_turn" {
		t.Fatalf("messages: %d %s", r.status, r.body)
	}
	kinds := []string{}
	for _, c := range m["content"].([]any) {
		kinds = append(kinds, c.(map[string]any)["type"].(string))
	}
	if strings.Join(kinds, ",") != "thinking,text" {
		t.Fatalf("content blocks %v", kinds)
	}
	_, events := stream(t, gateway, "/v1/messages", keyA, map[string]any{
		"model": "deepseek-v4.1", "max_tokens": 256, "stream": true, "thinking": map[string]any{"type": "disabled"},
		"messages": []any{map[string]any{"role": "user", "content": "Hello"}},
	}, nil)
	seq := []string{}
	for _, ev := range events {
		if len(seq) == 0 || seq[len(seq)-1] != ev.event {
			seq = append(seq, ev.event)
		}
	}
	got := strings.Join(seq, ",")
	if !strings.HasPrefix(got, "message_start,") || !strings.Contains(got, "content_block_delta") || !strings.HasSuffix(got, "message_delta,message_stop") {
		t.Fatalf("event sequence %s", got)
	}
}

func TestResponsesAPI(t *testing.T) {
	r := post(t, gateway, "/v1/responses", keyA, map[string]any{"model": "deepseek-v4.1", "input": "Hello", "reasoning": map[string]any{"effort": "none"}}, nil)
	m := r.json(t)
	if r.status != 200 || m["object"] != "response" || m["status"] != "completed" {
		t.Fatalf("responses: %d %s", r.status, r.body)
	}
	if !strings.Contains(string(r.body), "Hello from ModelWeave mock worker") {
		t.Fatalf("output: %s", r.body)
	}
}

func TestLogprobs(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", "Hi", map[string]any{
		"logprobs": true, "top_logprobs": 2, "thinking": map[string]any{"type": "disabled"},
	}), nil)
	lp, _ := message(t, r.json(t))["logprobs"].(map[string]any)
	content, _ := lp["content"].([]any)
	if r.status != 200 || len(content) == 0 {
		t.Fatalf("logprobs: %d %s", r.status, r.body)
	}
	first := content[0].(map[string]any)
	if _, ok := first["logprob"].(float64); !ok || len(first["top_logprobs"].([]any)) != 2 || first["token"] == "" {
		t.Fatalf("logprob entry %v", first)
	}
}

func TestResponsesStream(t *testing.T) {
	_, events := stream(t, gateway, "/v1/responses", keyA, map[string]any{"model": "deepseek-v4.1", "input": "Hello", "stream": true}, nil)
	types := map[string]bool{}
	for _, ev := range events {
		types[ev.event] = true
	}
	for _, want := range []string{"response.created", "response.output_text.delta", "response.completed"} {
		if !types[want] {
			t.Fatalf("missing %s in %v", want, types)
		}
	}
}

func TestToolCalls(t *testing.T) {
	tools := []any{map[string]any{"type": "function", "function": map[string]any{
		"name": "search_docs", "parameters": map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}, "worker": map[string]any{"type": "string"}}},
	}}}
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", "Search the docs", map[string]any{"tools": tools, "thinking": map[string]any{"type": "disabled"}}), nil)
	ch := message(t, r.json(t))
	calls, _ := ch["message"].(map[string]any)["tool_calls"].([]any)
	if r.status != 200 || ch["finish_reason"] != "tool_calls" || len(calls) != 1 {
		t.Fatalf("tool calls: %d %s", r.status, r.body)
	}
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	var args map[string]any
	if fn["name"] != "search_docs" || json.Unmarshal([]byte(fn["arguments"].(string)), &args) != nil || args["query"] != "ModelWeave" {
		t.Fatalf("tool call %v", fn)
	}
	// Streaming: the engine's tool-call deltas become chat chunks.
	_, events := stream(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", "Search the docs", map[string]any{
		"tools": tools, "thinking": map[string]any{"type": "disabled"}, "stream": true}), nil)
	var name, argsText, finish string
	for _, ev := range events[:len(events)-1] {
		var c struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						ID       string `json:"id"`
						Function struct{ Name, Arguments string }
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		_ = json.Unmarshal([]byte(ev.data), &c)
		for _, ch := range c.Choices {
			for _, tc := range ch.Delta.ToolCalls {
				name += tc.Function.Name
				argsText += tc.Function.Arguments
			}
			if ch.FinishReason != nil {
				finish = *ch.FinishReason
			}
		}
	}
	if name != "search_docs" || finish != "tool_calls" || json.Unmarshal([]byte(argsText), &args) != nil || args["query"] != "ModelWeave" {
		t.Fatalf("streamed tool call: name %q args %q finish %q", name, argsText, finish)
	}
	// Anthropic tool_use.
	ar := post(t, gateway, "/v1/messages", keyA, map[string]any{
		"model": "deepseek-v4.1", "max_tokens": 256, "thinking": map[string]any{"type": "disabled"},
		"tools":    []any{map[string]any{"name": "search_docs", "input_schema": map[string]any{"type": "object"}}},
		"messages": []any{map[string]any{"role": "user", "content": "Search"}},
	}, nil)
	am := ar.json(t)
	if ar.status != 200 || am["stop_reason"] != "tool_use" || !strings.Contains(string(ar.body), `"type":"tool_use"`) {
		t.Fatalf("anthropic tool_use: %d %s", ar.status, ar.body)
	}
}

func TestJSONResponseFormat(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", "Reply in json please", map[string]any{
		"response_format": map[string]any{"type": "json_object"}, "thinking": map[string]any{"type": "disabled"},
	}), nil)
	content := message(t, r.json(t))["message"].(map[string]any)["content"].(string)
	var v map[string]any
	if r.status != 200 || json.Unmarshal([]byte(content), &v) != nil || v["answer"] != "ModelWeave mock" {
		t.Fatalf("json output %q (%d)", content, r.status)
	}
}

// Stop strings travel in parsing_options and the engine stops itself; the
// gateway neither matches nor truncates.
func TestStopSequenceInEngine(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", "Count", map[string]any{"stop": []string{"four"}, "thinking": map[string]any{"type": "disabled"}}), nil)
	ch := message(t, r.json(t))
	content := ch["message"].(map[string]any)["content"].(string)
	if r.status != 200 || !strings.HasSuffix(content, "one two three ") || ch["finish_reason"] != "stop" {
		t.Fatalf("stop sequence: %q %v", content, ch["finish_reason"])
	}
	w := r.header.Get("X-ModelWeave-Decode-Worker")
	st := workerStats(t, w)
	req := st["last_request"].(map[string]any)
	if st["last_finish"] != "stop" || st["stop_string_matches"].(float64) < 1 || path(req, "parsing_options", "stop_sequences").([]any)[0] != "four" {
		t.Fatalf("engine %s did not stop on the string itself: %v", w, st)
	}
	// Anthropic reports the matched sequence the engine returned.
	am := post(t, gateway, "/v1/messages", keyA, map[string]any{
		"model": "deepseek-v4.1", "max_tokens": 256, "thinking": map[string]any{"type": "disabled"}, "stop_sequences": []string{"four"},
		"messages": []any{map[string]any{"role": "user", "content": "Count"}},
	}, nil).json(t)
	if am["stop_reason"] != "stop_sequence" || am["stop_sequence"] != "four" {
		t.Fatalf("anthropic stop: %v", am)
	}
}

func TestLengthLimit(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", "Hi", map[string]any{"max_tokens": 3, "thinking": map[string]any{"type": "disabled"}}), nil)
	m := r.json(t)
	if r.status != 200 || message(t, m)["finish_reason"] != "length" || m["usage"].(map[string]any)["completion_tokens"] != float64(3) {
		t.Fatalf("length: %d %s", r.status, r.body)
	}
	tooLong := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-tiny", "Hi", map[string]any{"max_tokens": 131070}), nil)
	if tooLong.status != 400 || !strings.Contains(string(tooLong.body), "maximum context length") {
		t.Fatalf("context limit: %d %s", tooLong.status, tooLong.body)
	}
}

// df-0 declines normal requests: the attempt falls back to pd before any
// output.
func TestNormalFallbackToPD(t *testing.T) {
	before := metric(t, gateway, "modelweave_fallbacks_total", `from="normal"`, `to="pd"`, `reason="normal_rejected"`)
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-fallback", "Hi", nil), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" || r.header.Get("X-ModelWeave-Attempt") != "2" || r.header.Get("X-ModelWeave-Reason") != "fallback" {
		t.Fatalf("fallback: %d %v", r.status, r.header)
	}
	if after := metric(t, gateway, "modelweave_fallbacks_total", `from="normal"`, `to="pd"`, `reason="normal_rejected"`); after != before+1 {
		t.Fatalf("fallback metric %v -> %v", before, after)
	}
}

// One of three unified workers fails every request: requests are retried
// elsewhere, and the worker is ejected after consecutive failures.
func TestRetryAndEjectBrokenWorker(t *testing.T) {
	for i := 0; i < 40; i++ {
		r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-normal", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}}),
			map[string]string{"X-Session-Id": "normal-" + nonce()})
		if r.status != 200 || r.header.Get("X-ModelWeave-Decode-Worker") == "normal-bad" {
			t.Fatalf("request %d: %d %v %s", i, r.status, r.header, r.body)
		}
		if metric(t, gateway, "modelweave_worker_up", `worker="normal-bad"`) == 0 {
			break
		}
	}
	if metric(t, gateway, "modelweave_retries_total", `model="deepseek-v4.1-normal"`) < 2 {
		t.Fatal("retries not recorded")
	}
	if up := metric(t, gateway, "modelweave_worker_up", `worker="normal-bad"`); up != 0 {
		t.Fatalf("normal-bad should be ejected, worker_up=%v", up)
	}
	if n := metric(t, gateway, "modelweave_upstream_errors_total", `model="deepseek-v4.1-normal"`, `code="internal"`); n < 2 {
		t.Fatalf("upstream errors not recorded: %v", n)
	}
}

func TestNormalSessionStickiness(t *testing.T) {
	hdr := map[string]string{"X-Session-Id": "sticky-" + nonce()}
	var first string
	for i := 0; i < 4; i++ {
		r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-normal", "Hi "+strconv.Itoa(i), nil), hdr)
		if r.status != 200 {
			t.Fatalf("%d %s", r.status, r.body)
		}
		w := r.header.Get("X-ModelWeave-Decode-Worker")
		if first == "" {
			first = w
		} else if w != first {
			t.Fatalf("session moved from %s to %s", first, w)
		}
	}
}

func TestRateLimit(t *testing.T) {
	limited, ok := 0, 0
	for i := 0; i < 5; i++ {
		r := post(t, gateway, "/v1/chat/completions", "sk-limited", chatBody("deepseek-v4.1", "Hi", map[string]any{"max_tokens": 4}), nil)
		if r.status == 429 {
			limited++
			if r.header.Get("Retry-After") == "" || r.json(t)["error"].(map[string]any)["type"] != "rate_limit_exceeded" {
				t.Fatalf("429 without Retry-After / error type: %v %s", r.header, r.body)
			}
		} else if r.status == 200 {
			ok++
		} else {
			t.Fatalf("status %d: %s", r.status, r.body)
		}
	}
	// rpm=3: at most 3 of 5 back-to-back requests pass (fewer when an
	// earlier run drained the bucket).
	if ok > 3 || limited < 2 {
		t.Fatalf("rpm=3: %d passed, %d limited", ok, limited)
	}
	if metric(t, gateway, "modelweave_rate_limited_total", `tenant="limited"`, `reason="rpm"`) < 2 {
		t.Fatal("rate limit metric missing")
	}
}

// deepseek-v4.1-tiny has one slot and a queue of two: of four concurrent
// requests one runs, two wait for capacity and one is rejected.
func TestAdmissionQueue(t *testing.T) {
	queued0 := metric(t, gateway, "modelweave_admission_total", `model="deepseek-v4.1-tiny"`, `result="queued"`)
	var wg sync.WaitGroup
	codes := make([]int, 4)
	for i := range codes {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			time.Sleep(time.Duration(i) * 50 * time.Millisecond)
			r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-tiny", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
			codes[i] = r.status
			if r.status == 503 && r.header.Get("Retry-After") == "" {
				t.Errorf("503 without Retry-After")
			}
		}(i)
	}
	wg.Wait()
	ok, rejected := 0, 0
	for _, c := range codes {
		switch c {
		case 200:
			ok++
		case 503:
			rejected++
		}
	}
	if ok != 3 || rejected != 1 {
		t.Fatalf("codes %v: want 3x200 and 1x503", codes)
	}
	if d := metric(t, gateway, "modelweave_admission_total", `model="deepseek-v4.1-tiny"`, `result="queued"`) - queued0; d != 2 {
		t.Fatalf("want 2 queued admissions, got %v", d)
	}
}

func TestClientDisconnectAbortsEngine(t *testing.T) {
	before := metric(t, gateway, "modelweave_aborts_total", `model="deepseek-v4.1-tiny"`, `reason="client_gone"`)
	b, _ := json.Marshal(chatBody("deepseek-v4.1-tiny", "Tell me everything", map[string]any{"stream": true, "thinking": map[string]any{"type": "disabled"}}))
	req, _ := http.NewRequest(http.MethodPost, gateway+"/v1/chat/completions", strings.NewReader(string(b)))
	req.Header.Set("Authorization", "Bearer "+keyA)
	r, err := client.Do(req)
	if err != nil || r.StatusCode != 200 {
		t.Fatalf("stream: %v %v", err, r)
	}
	buf := make([]byte, 256)
	if _, err := r.Body.Read(buf); err != nil {
		t.Fatal(err)
	}
	r.Body.Close() // hang up mid-stream
	deadline := time.Now().Add(10 * time.Second)
	for metric(t, gateway, "modelweave_aborts_total", `model="deepseek-v4.1-tiny"`, `reason="client_gone"`) <= before {
		if time.Now().After(deadline) {
			t.Fatal("abort not recorded")
		}
		time.Sleep(100 * time.Millisecond)
	}
	for {
		w := workers(t, gateway)["tiny-0"]
		if w.State != nil && w.State["running_requests"] == float64(0) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("tiny-0 still running: %v", w.State)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

// The second gateway learns cache contents only from engine KV events.
func TestKVEventsDrivenRouting(t *testing.T) {
	prompt := longPrompt()
	first := post(t, gatewayEv, "/v1/chat/completions", "", chatBody("deepseek-v4.1", prompt, nil), nil)
	if first.status != 200 || first.header.Get("X-ModelWeave-Mode") != "pd" {
		t.Fatalf("first: %d %v", first.status, first.header)
	}
	time.Sleep(2500 * time.Millisecond) // engines publish KV events with heartbeats
	second := post(t, gatewayEv, "/v1/chat/completions", "", chatBody("deepseek-v4.1", prompt, nil), nil)
	if second.status != 200 || second.header.Get("X-ModelWeave-Reason") != "prefix_hit" ||
		second.header.Get("X-ModelWeave-Decode-Worker") != first.header.Get("X-ModelWeave-Decode-Worker") {
		t.Fatalf("second: %d %v", second.status, second.header)
	}
	if metric(t, gatewayEv, "modelweave_kv_events_total", `kind="stored"`) == 0 {
		t.Fatal("no KV events ingested")
	}
}
