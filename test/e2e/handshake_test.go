//go:build e2e

package e2e

import (
	"strings"
	"testing"
)

// mwip pull: P keeps the KV and returns a kv_handle; D fetches it.
func TestMWIPPullHandshake(t *testing.T) {
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1-pull", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" || r.header.Get("X-ModelWeave-Prefill-Worker") != "pp-0" || r.header.Get("X-ModelWeave-Decode-Worker") != "dp-0" {
		t.Fatalf("pull: %d %v %s", r.status, r.header, r.body)
	}
	p, d := lastRequest(t, "pp-0"), lastRequest(t, "dp-0")
	if path(p, "disagg", "handshake") != "HANDSHAKE_PULL" || path(p, "disagg", "role") != "ROLE_PREFILL" ||
		path(d, "disagg", "kv_handle", "url") != "http://pp-0:9000" || path(d, "disagg", "kv_handle", "room") != path(p, "disagg", "room") {
		t.Fatalf("pull fields: P %v D %v", p["disagg"], d["disagg"])
	}
}

// The deepseek-v4.1 instances register with both gateways; each gateway
// runs the model with its own PD handshake: pull on gateway-events, push on
// the main gateway.
func TestSecondGatewayPullsFromTheSameInstances(t *testing.T) {
	if metric(t, gatewayEv, "modelweave_model_info", `model="deepseek-v4.1"`, `handshake="pull"`) != 1 ||
		metric(t, gateway, "modelweave_model_info", `model="deepseek-v4.1"`, `handshake="push"`) != 1 {
		t.Fatal("model_info does not reflect the handshakes")
	}
	if w := workers(t, gatewayEv)["decode-0"]; w.Target != "decode-0:9001" || w.Source != "heartbeat" {
		t.Fatalf("gateway-events sees decode-0 as %+v", w)
	}
	r := post(t, gatewayEv, "/v1/chat/completions", "", chatBody("deepseek-v4.1", longPrompt(), map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" {
		t.Fatalf("gateway-events: %d %v %s", r.status, r.header, r.body)
	}
	pw, dw := r.header.Get("X-ModelWeave-Prefill-Worker"), r.header.Get("X-ModelWeave-Decode-Worker")
	if c := message(t, r.json(t))["message"].(map[string]any)["content"].(string); !strings.HasPrefix(c, "Hello from ModelWeave mock worker "+dw) {
		t.Fatalf("answer %q", c)
	}
	p, d := lastRequest(t, pw), lastRequest(t, dw)
	if path(p, "disagg", "handshake") != "HANDSHAKE_PULL" || path(d, "disagg", "kv_handle", "url") != "http://"+pw+":9000" ||
		path(d, "disagg", "kv_handle", "room") != path(p, "disagg", "room") {
		t.Fatalf("pull on gateway-events: P(%s) %v D(%s) %v", pw, p["disagg"], dw, d["disagg"])
	}
	// The same instances through the main gateway: push.
	r = post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", longPrompt(), map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if r.status != 200 || r.header.Get("X-ModelWeave-Mode") != "pd" {
		t.Fatalf("main gateway: %d %v", r.status, r.header)
	}
	if p := lastRequest(t, r.header.Get("X-ModelWeave-Prefill-Worker")); path(p, "disagg", "handshake") != "HANDSHAKE_PUSH" {
		t.Fatalf("push on the main gateway: %v", p["disagg"])
	}
}
