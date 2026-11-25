//go:build e2e

package e2e

import (
	"encoding/json"
	"strings"
	"testing"
)

// Checks over the whole suite. Go runs test files in name order, so these
// run after every other test.

func TestObservability(t *testing.T) {
	gw := string(get(t, gateway+"/metrics"))
	for _, name := range []string{
		"modelweave_requests_total", "modelweave_request_duration_seconds_bucket", "modelweave_ttft_seconds_bucket",
		"modelweave_ttft_segment_seconds_bucket", "modelweave_time_per_output_token_seconds_bucket", "modelweave_tokens_total",
		"modelweave_stage_duration_seconds_bucket", "modelweave_schedule_decisions_total", "modelweave_prefix_cache_match_ratio_bucket",
		"modelweave_execution_plans_total", "modelweave_admission_total", "modelweave_queue_wait_seconds_bucket",
		"modelweave_worker_up", "modelweave_worker_kv_utilization", "modelweave_worker_inflight_requests",
		"modelweave_heartbeats_total", "modelweave_engine_state_polls_total", "modelweave_tokenizer_client_duration_seconds_bucket",
		"modelweave_overload_level", "modelweave_pool_utilization", "modelweave_finish_reasons_total",
		"modelweave_engine_requests_total", "modelweave_model_info", "modelweave_media_items_total", "modelweave_media_tokens_total",
		"modelweave_execution_plans_total", "modelweave_fallbacks_total", "modelweave_embedding_artifact_lookups_total",
	} {
		if !strings.Contains(gw, name) {
			t.Errorf("gateway metric %s missing", name)
		}
	}
	if metric(t, gateway, "modelweave_schedule_decisions_total", `reason="cache_hit"`) == 0 {
		t.Error("no cache-hit scheduling decisions recorded")
	}
	if metric(t, gateway, "modelweave_tokens_total", `kind="cached"`) == 0 {
		t.Error("no cached tokens recorded")
	}
	tok := string(get(t, tokURL+"/metrics"))
	for _, name := range []string{
		"modelweave_tokenizer_requests_total", "modelweave_tokenizer_render_duration_seconds_bucket",
		"modelweave_tokenizer_prompt_tokens_bucket", "modelweave_tokenizer_build_info", "modelweave_tokenizer_media_total",
	} {
		if !strings.Contains(tok, name) {
			t.Errorf("tokenizer metric %s missing", name)
		}
	}
	var stats map[string]any
	if err := json.Unmarshal(get(t, gateway+"/admin/stats"), &stats); err != nil || stats["kv_index"].(map[string]any)["kind"] != "approx" {
		t.Errorf("admin stats: %v %v", err, stats)
	}
}

// Clients stream or not; every generate call the engines received during the
// whole suite, over every engine protocol, was a streaming one. Encoders
// only receive encode calls (mwip /mwip/v1/encode, or the vLLM EC producer
// leg), which are unary.
func TestEnginesOnlyReceiveStreaming(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		body := chatBody("deepseek-v4.1-normal", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}, "stream": streaming})
		if streaming {
			stream(t, gateway, "/v1/chat/completions", keyA, body, nil)
		} else if r := post(t, gateway, "/v1/chat/completions", keyA, body, nil); r.status != 200 {
			t.Fatalf("non-streaming request: %d %s", r.status, r.body)
		}
	}
	served := 0
	protocols := map[string]float64{}
	for _, id := range generateWorkers {
		st := workerStats(t, id)
		if st["non_streaming_requests"] != float64(0) {
			t.Fatalf("%s received %v non-streaming generate calls", id, st["non_streaming_requests"])
		}
		if st["streaming_requests"].(float64) > 0 {
			served++
		}
		for p, n := range st["by_protocol"].(map[string]any) {
			protocols[p] += n.(float64)
		}
	}
	if served < 14 || protocols["mwip"] == 0 || protocols["vllm"] == 0 || protocols["sglang"] == 0 {
		t.Fatalf("%d workers served traffic, generate calls by protocol %v", served, protocols)
	}
	for _, id := range encodeWorkers {
		if st := workerStats(t, id); st["streaming_requests"] != float64(0) || st["non_streaming_requests"] != float64(0) {
			t.Fatalf("encoder %s received generate calls: %v", id, st)
		}
	}
	for _, label := range []string{`stream="true"`, `stream="false"`} {
		if metric(t, gateway, "modelweave_requests_total", label, `code="200"`) == 0 {
			t.Fatalf("no successful client requests with %s", label)
		}
	}
	for _, proto := range []string{`protocol="mwip"`, `protocol="vllm"`, `protocol="sglang"`} {
		if metric(t, gateway, "modelweave_engine_requests_total", proto) == 0 {
			t.Fatalf("no engine calls with %s", proto)
		}
	}
	if metric(t, gatewayEv, "modelweave_engine_requests_total", `protocol="vllm"`) == 0 || metric(t, gatewayEv, "modelweave_engine_requests_total", `protocol="mwip"`) != 0 {
		t.Fatal("gateway-events (-engine-protocol vllm) must only speak vllm")
	}
}
