//go:build e2e

package e2e

import (
	"fmt"
	"strings"
	"testing"
)

// The vLLM adapter in normal mode: one call to the decode worker, token in,
// output_mode derender (vLLM parses its own output), no kv_transfer_params.
func TestVLLMNormal(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-vllm", "Hi", nil), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "normal" || r.header.Get("X-ModelWeave-Decode-Worker") != "dv-0" {
		t.Fatalf("vllm normal: %d %v %s", r.status, r.header, r.body)
	}
	msg := message(t, r.json(t))["message"].(map[string]any)
	if !strings.HasPrefix(msg["content"].(string), "Hello from ModelWeave mock worker dv-0") || !strings.Contains(fmt.Sprint(msg["reasoning_content"]), "answer briefly") {
		t.Fatalf("engine-parsed answer: %v", msg)
	}
	st := workerStats(t, "dv-0")
	req := st["last_request"].(map[string]any)
	if st["last_protocol"] != "vllm" || req["output_mode"] != "derender" || req["stream"] != true || req["kv_transfer_params"] != nil ||
		path(req, "parse_context", "chat_template_kwargs", "thinking") != true || req["reasoning_ended"] != false {
		t.Fatalf("dv-0 request: %v", req)
	}
}

// The vLLM adapter with EPD and PD: the image goes to the EC producer ev-0
// (unary), its ec_transfer_params to P and D; P and D pair through a
// MooncakeConnector bootstrap (shared transfer_id, P's bootstrap address).
func TestVLLMEPDAndPD(t *testing.T) {
	digest := sha256Hex([]byte("e2e-vllm-" + nonce()))
	url := "https://img.example/" + digest + ".png"
	body := userBlocks("deepseek-v4.1-vllm", text(longPrompt()), remoteImage(url, digest, 1024, 768, nil))
	_, rr := render(t, "chat_completions", body)
	ec0, enc0 := num(workerStats(t, "pv-0"), "ec_loaded"), num(workerStats(t, "ev-0"), "encoded")
	epd0 := metric(t, gateway, "modelweave_requests_total", `model="deepseek-v4.1-vllm"`, `mode="epd_pd"`)

	r := post(t, gateway, "/v1/chat/completions", keyA, body, nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" || r.header.Get("X-ModelWeave-Prefill-Worker") != "pv-0" || r.header.Get("X-ModelWeave-Decode-Worker") != "dv-0" {
		t.Fatalf("vllm epd: %d %v %s", r.status, r.header, r.body)
	}
	if !strings.HasPrefix(message(t, r.json(t))["message"].(map[string]any)["content"].(string), "Hello from ModelWeave mock worker dv-0") {
		t.Fatalf("answer: %s", r.body)
	}
	// vLLM expands the collapsed placeholder itself: the prompt length the
	// engine reports equals the tokenizer's expanded count.
	if pt := path(r.json(t), "usage", "prompt_tokens"); pt != rr["prompt_tokens"] {
		t.Fatalf("engine prompt_tokens %v, tokenizer %v", pt, rr["prompt_tokens"])
	}
	planTokens := path(mediaItem(t, rr, 0), "plan", "tokens")

	e := lastRequest(t, "ev-0")
	if e["stream"] != false || fmt.Sprint(e["token_ids"]) != "[129264]" || path(e["content_parts"].([]any)[0], "uuid") != digest {
		t.Fatalf("encode leg: %v", e)
	}
	p, d := lastRequest(t, "pv-0"), lastRequest(t, "dv-0")
	for name, leg := range map[string]map[string]any{"P": p, "D": d} {
		part := leg["content_parts"].([]any)[0].(map[string]any)
		if leg["stream"] != true || countToken(leg["token_ids"], 129264) != 1 || part["uuid"] != digest || part["url"] != url ||
			path(leg, "ec_transfer_params", digest) == nil || path(leg["modelweave"].(map[string]any)["media"].([]any)[0], "tokens") != planTokens {
			t.Fatalf("%s leg: %v", name, leg)
		}
	}
	pkv, dkv := p["kv_transfer_params"].(map[string]any), d["kv_transfer_params"].(map[string]any)
	if pkv["do_remote_decode"] != true || path(p, "sampling_params", "max_tokens") != float64(1) ||
		dkv["do_remote_prefill"] != true || dkv["remote_bootstrap_addr"] != "http://pv-0:9000" || dkv["transfer_id"] != pkv["transfer_id"] || dkv["remote_engine_id"] != "pv-0" {
		t.Fatalf("kv_transfer_params: P %v D %v", pkv, dkv)
	}
	if d := num(workerStats(t, "pv-0"), "ec_loaded") - ec0; d != 1 {
		t.Fatalf("pv-0 loaded %v embeddings through ec_transfer_params, want 1", d)
	}
	if d := num(workerStats(t, "ev-0"), "encoded") - enc0; d != 1 {
		t.Fatalf("ev-0 encoded %v items, want 1", d)
	}
	if d := metric(t, gateway, "modelweave_requests_total", `model="deepseek-v4.1-vllm"`, `mode="epd_pd"`) - epd0; d != 1 {
		t.Fatalf("epd_pd requests: %v", d)
	}
	if metric(t, gateway, "modelweave_engine_requests_total", `model="deepseek-v4.1-vllm"`, `role="encode"`, `protocol="vllm"`) == 0 {
		t.Fatal("encode leg not counted")
	}
}

// The SGLang adapter: P and D pair through bootstrap_host/port/room. Its
// native /generate returns raw text, so there is no engine-side parsing:
// with thinking on, the reasoning and </think> arrive as content.
func TestSGLangPDAndRawText(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-sglang", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" || r.header.Get("X-ModelWeave-Prefill-Worker") != "ps-0" || r.header.Get("X-ModelWeave-Decode-Worker") != "ds-0" {
		t.Fatalf("sglang: %d %v %s", r.status, r.header, r.body)
	}
	if c := message(t, r.json(t))["message"].(map[string]any)["content"].(string); !strings.HasPrefix(c, "Hello from ModelWeave mock worker ds-0") {
		t.Fatalf("answer %q", c)
	}
	p, d := lastRequest(t, "ps-0"), lastRequest(t, "ds-0")
	if p["stream"] != true || d["stream"] != true || d["bootstrap_host"] != "ps-0" || d["bootstrap_port"] != float64(9000) ||
		p["bootstrap_room"] == nil || p["bootstrap_room"] != d["bootstrap_room"] || d["input_ids"] == nil {
		t.Fatalf("sglang bootstrap fields: P %v D %v", p, d)
	}
	raw := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-sglang", "Hi", nil), nil)
	m := message(t, raw.json(t))["message"].(map[string]any)
	if raw.status != 200 || !strings.Contains(m["content"].(string), "</think>Hello from ModelWeave mock worker ds-0") || m["reasoning_content"] != nil {
		t.Fatalf("sglang raw text: %d %v", raw.status, m)
	}
}

// SGLang encodes media itself: the gateway sends the raw image (image_data)
// to the prefill server and never calls the registered encoder es-0.
func TestSGLangEngineManagedEncoding(t *testing.T) {
	digest := sha256Hex([]byte("e2e-sglang-" + nonce()))
	url := "https://img.example/" + digest + ".png"
	local0 := num(workerStats(t, "ps-0"), "local_encoded")
	r := post(t, gateway, "/v1/chat/completions", keyA, userBlocks("deepseek-v4.1-sglang", text("Describe"), remoteImage(url, digest, 800, 600, nil)), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" {
		t.Fatalf("sglang image: %d %v %s", r.status, r.header, r.body)
	}
	p := lastRequest(t, "ps-0")
	if fmt.Sprint(p["image_data"]) != "["+url+"]" || countToken(p["input_ids"], 129264) != 1 || path(p["modelweave"].(map[string]any)["media"].([]any)[0], "digest") != digest {
		t.Fatalf("ps-0 request: %v", p)
	}
	if d := num(workerStats(t, "ps-0"), "local_encoded") - local0; d != 1 {
		t.Fatalf("ps-0 encoded %v images itself, want 1", d)
	}
	es := workerStats(t, "es-0")
	if num(es, "encoded") != 0 || len(es["encode_by_protocol"].(map[string]any)) != 0 ||
		metric(t, gateway, "modelweave_engine_requests_total", `model="deepseek-v4.1-sglang"`, `role="encode"`) != 0 {
		t.Fatalf("the gateway called SGLang's encoder: %v", es)
	}
	if metric(t, gateway, "modelweave_model_info", `model="deepseek-v4.1-sglang"`, `engine_protocol="sglang"`, `gateway_encode="false"`) != 1 {
		t.Fatal("model_info does not show engine-managed encoding")
	}
}

// mwip pull: P keeps the KV and returns a kv_handle; D fetches it.
func TestMWIPPullHandshake(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-pull", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" || r.header.Get("X-ModelWeave-Prefill-Worker") != "pp-0" || r.header.Get("X-ModelWeave-Decode-Worker") != "dp-0" {
		t.Fatalf("pull: %d %v %s", r.status, r.header, r.body)
	}
	p, d := lastRequest(t, "pp-0"), lastRequest(t, "dp-0")
	if path(p, "disagg", "handshake") != "pull" || path(p, "disagg", "role") != "prefill" ||
		path(d, "disagg", "kv_handle", "url") != "http://pp-0:9000" || path(d, "disagg", "kv_handle", "room") != path(p, "disagg", "room") {
		t.Fatalf("pull fields: P %v D %v", p["disagg"], d["disagg"])
	}
}

// The startup option picks the engine protocol: gateway-events runs the
// same deepseek-v4.1 workers with -engine-protocol vllm (pull, NixlConnector
// style), the main gateway with engine.protocol mwip.
func TestStartupOptionSelectsEngineProtocol(t *testing.T) {
	if metric(t, gatewayEv, "modelweave_model_info", `model="deepseek-v4.1"`, `engine_protocol="vllm"`, `handshake="pull"`) != 1 ||
		metric(t, gateway, "modelweave_model_info", `model="deepseek-v4.1"`, `engine_protocol="mwip"`, `handshake="push"`) != 1 {
		t.Fatal("model_info does not reflect the engine protocols")
	}
	if w := workers(t, gatewayEv)["decode-0"]; w.Protocol != "vllm" {
		t.Fatalf("gateway-events sees decode-0 as %q", w.Protocol)
	}
	r := post(t, gatewayEv, "/v1/chat/completions", "", chatBody("deepseek-v4.1", longPrompt(), map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" {
		t.Fatalf("gateway-events: %d %v %s", r.status, r.header, r.body)
	}
	pw, dw := r.header.Get("X-ModelWeave-Prefill-Worker"), r.header.Get("X-ModelWeave-Decode-Worker")
	if c := message(t, r.json(t))["message"].(map[string]any)["content"].(string); !strings.HasPrefix(c, "Hello from ModelWeave mock worker "+dw) {
		t.Fatalf("answer %q", c)
	}
	ps, ds := workerStats(t, pw), workerStats(t, dw)
	pkv := path(ps["last_request"], "kv_transfer_params").(map[string]any)
	dkv := path(ds["last_request"], "kv_transfer_params").(map[string]any)
	if ps["last_protocol"] != "vllm" || ds["last_protocol"] != "vllm" || pkv["do_remote_decode"] != true || pkv["transfer_id"] != nil ||
		dkv["remote_host"] != pw || dkv["remote_request_id"] == nil || dkv["do_remote_prefill"] != true {
		t.Fatalf("vllm pull: P(%s) %v %v D(%s) %v %v", pw, ps["last_protocol"], pkv, dw, ds["last_protocol"], dkv)
	}
	// The same workers through the main gateway speak mwip.
	r = post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if st := workerStats(t, r.header.Get("X-ModelWeave-Decode-Worker")); r.status != 200 || st["last_protocol"] != "mwip" {
		t.Fatalf("main gateway: %d %v", r.status, st["last_protocol"])
	}
}
