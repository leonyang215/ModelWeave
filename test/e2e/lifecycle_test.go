//go:build e2e

package e2e

import (
	"net/http"
	"testing"
	"time"
)

// registration steers the registration of a mock instance: pause (stop
// heartbeating), leave (deregister now), resume, drain, undrain.
func registration(t *testing.T, id, action string) map[string]any {
	t.Helper()
	return postEmpty(t, instanceURL(id)+"/mock/registration/"+action)
}

// admin calls the gateway's admin API.
func admin(t *testing.T, path string) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, gateway+path, nil)
	req.Header.Set("X-ModelWeave-Token", ctlToken)
	r, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	r.Body.Close()
	if r.StatusCode != 200 {
		t.Fatalf("POST %s: %d", path, r.StatusCode)
	}
}

// route sends a deepseek-v4.1 request, short (normal on a decode instance)
// or long (prefill and decode instances), and returns the instances that
// ran it.
func route(t *testing.T, long bool) (prefill, decode string) {
	t.Helper()
	prompt := "Hi " + nonce()
	if long {
		prompt = longPrompt()
	}
	r := post(t, gateway, "/v1/chat/completions", keyA, chatBody("deepseek-v4.1", prompt, map[string]any{"thinking": map[string]any{"type": "disabled"}}), nil)
	if r.status != 200 {
		t.Fatalf("deepseek-v4.1: %d %s", r.status, r.body)
	}
	return r.header.Get("X-ModelWeave-Prefill-Worker"), r.header.Get("X-ModelWeave-Decode-Worker")
}

// The instances of a model are maintained one by one, per role. On
// deepseek-v4.1 (prefill-0/1, decode-0/1, registered with both gateways):
// an engine that drains gets no new requests; one whose heartbeats stop
// stays registered until its TTL passes, then expires, while its sibling of
// the same role serves alone; one that leaves is gone at once; both come
// back with their next heartbeat, with their whole KV cache in the
// KV-event index; and an operator's drain holds across heartbeats.
func TestInstancesMaintainedPerRole(t *testing.T) {
	t.Cleanup(func() {
		// Leave the instances as the suite found them, even after a failure.
		for _, u := range []string{
			instanceURL("decode-0") + "/mock/registration/undrain", instanceURL("decode-1") + "/mock/registration/resume",
			instanceURL("prefill-1") + "/mock/registration/resume",
		} {
			if r, err := client.Post(u, "application/json", nil); err == nil {
				r.Body.Close()
			}
		}
		req, _ := http.NewRequest(http.MethodPost, gateway+"/admin/workers/decode-0/drain?undo=1", nil)
		req.Header.Set("X-ModelWeave-Token", ctlToken)
		if r, err := client.Do(req); err == nil {
			r.Body.Close()
		}
	})
	registered := func(base, id string) bool {
		w, ok := workers(t, base)[id]
		return ok && w.Healthy && w.State != nil
	}
	for _, base := range []string{gateway, gatewayEv} {
		for _, id := range []string{"prefill-0", "prefill-1", "decode-0", "decode-1"} {
			if !registered(base, id) {
				t.Fatalf("%s is not registered on %s", id, base)
			}
		}
	}

	// 1. decode-0 reports draining: new requests, normal or PD, go to
	// decode-1 (which caches the PD prompt's KV).
	registration(t, "decode-0", "drain")
	if w := workers(t, gateway)["decode-0"]; !w.Draining || w.AdminDrained {
		t.Fatalf("decode-0 after its drain report: %+v", w)
	}
	for i := 0; i < 3; i++ {
		if _, d := route(t, false); d != "decode-1" {
			t.Fatalf("normal request %d ran on %s while decode-0 drains", i, d)
		}
	}
	if _, d := route(t, true); d != "decode-1" {
		t.Fatalf("PD request decoded on %s while decode-0 drains", d)
	}
	registration(t, "decode-0", "undrain")
	if workers(t, gateway)["decode-0"].Draining {
		t.Fatal("decode-0 still draining")
	}

	// 2. decode-1 stops heartbeating, like a crashed engine: it stays
	// registered until the TTL (6s) passes, then expires on both gateways.
	expired := func() float64 {
		return metric(t, gateway, "modelweave_instance_removals_total", `model="deepseek-v4.1"`, `role="decode"`, `reason="expired"`)
	}
	expired0 := expired()
	registration(t, "decode-1", "pause")
	paused := time.Now()
	time.Sleep(2 * time.Second)
	if w, ok := workers(t, gateway)["decode-1"]; !ok || w.HeartbeatAgeMS < 1500 {
		t.Fatalf("decode-1 must stay registered until its TTL passes: %+v", w)
	}
	waitFor(t, "decode-1 to expire", 15*time.Second, func() bool { _, ok := workers(t, gateway)["decode-1"]; return !ok })
	if d := time.Since(paused); d < 5*time.Second {
		t.Fatalf("decode-1 expired %v after its last heartbeat, before the 6s TTL", d)
	}
	waitFor(t, "decode-1 to expire on gateway-events", 5*time.Second, func() bool { _, ok := workers(t, gatewayEv)["decode-1"]; return !ok })
	if d := expired() - expired0; d != 1 {
		t.Fatalf("expirations counted: %v", d)
	}
	for i := 0; i < 3; i++ {
		if _, d := route(t, false); d != "decode-0" {
			t.Fatalf("normal request %d ran on %s after decode-1 expired", i, d)
		}
	}
	if p, d := route(t, true); p == "" || d != "decode-0" {
		t.Fatalf("PD request on %s -> %s after decode-1 expired", p, d)
	}

	// 3. prefill-1 shuts down gracefully: deregistered from both gateways at
	// once, prefill-0 takes every PD request.
	left := func() float64 {
		return metric(t, gateway, "modelweave_instance_removals_total", `model="deepseek-v4.1"`, `role="prefill"`, `reason="deregistered"`)
	}
	left0 := left()
	if m := registration(t, "prefill-1", "leave"); m["removed"] != float64(2) {
		t.Fatalf("prefill-1 leave: %v", m)
	}
	for _, base := range []string{gateway, gatewayEv} {
		if _, ok := workers(t, base)["prefill-1"]; ok {
			t.Fatalf("prefill-1 still registered on %s", base)
		}
	}
	if d := left() - left0; d != 1 {
		t.Fatalf("deregistrations counted: %v", d)
	}
	for i := 0; i < 2; i++ {
		if p, d := route(t, true); p != "prefill-0" || d != "decode-0" {
			t.Fatalf("PD request %d on %s -> %s without prefill-1 and decode-1", i, p, d)
		}
	}

	// 4. Both come back: registered again with their first heartbeat, and
	// decode-1 resends its whole KV cache, which the KV-event index of
	// gateway-events (empty for it after the expiry) takes in.
	registration(t, "decode-1", "resume")
	registration(t, "prefill-1", "resume")
	for _, base := range []string{gateway, gatewayEv} {
		for _, id := range []string{"decode-1", "prefill-1"} {
			if !registered(base, id) {
				t.Fatalf("%s did not register again on %s", id, base)
			}
		}
	}
	if num(workerStats(t, "decode-1"), "cached_blocks") == 0 {
		t.Fatal("decode-1 holds no KV")
	}
	waitFor(t, "decode-1's KV cache in the index of gateway-events", 5*time.Second, func() bool {
		return float64(workers(t, gatewayEv)["decode-1"].IndexBlocks) == num(workerStats(t, "decode-1"), "cached_blocks")
	})

	// 5. An operator drains decode-0 through the admin API: the drain holds
	// across its heartbeats (which report it not draining), and decode-1
	// serves again.
	admin(t, "/admin/workers/decode-0/drain")
	time.Sleep(1500 * time.Millisecond)
	if w := workers(t, gateway)["decode-0"]; !w.Draining || !w.AdminDrained {
		t.Fatalf("decode-0 after the operator's drain and a heartbeat: %+v", w)
	}
	for i := 0; i < 3; i++ {
		if _, d := route(t, false); d != "decode-1" {
			t.Fatalf("normal request %d ran on %s while decode-0 is drained", i, d)
		}
	}
	admin(t, "/admin/workers/decode-0/drain?undo=1")
	if w := workers(t, gateway)["decode-0"]; w.Draining || w.AdminDrained {
		t.Fatalf("decode-0 after undo: %+v", w)
	}
}
