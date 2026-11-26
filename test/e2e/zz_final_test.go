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
		"modelweave_heartbeats_total", "modelweave_instance_removals_total", "modelweave_engine_state_polls_total", "modelweave_tokenizer_client_duration_seconds_bucket",
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

// Clients stream or not; engines are always called the same way. Over the
// whole suite every instance received the calls of its role only: Generate
// streams for prefill, decode and unified instances, Encode calls for
// encode instances.
func TestEveryInstanceServedItsRole(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		body := chatBody("deepseek-v4.1-normal", "Hi", map[string]any{"thinking": map[string]any{"type": "disabled"}, "stream": streaming})
		if streaming {
			stream(t, gateway, "/v1/chat/completions", keyA, body, nil)
		} else if r := post(t, gateway, "/v1/chat/completions", keyA, body, nil); r.status != 200 {
			t.Fatalf("non-streaming request: %d %s", r.status, r.body)
		}
	}
	served := 0
	for _, id := range generateWorkers {
		st := workerStats(t, id)
		if st["encode_requests"] != float64(0) {
			t.Fatalf("%s (%s) received %v Encode calls", id, st["role"], st["encode_requests"])
		}
		if st["generate_requests"].(float64) > 0 {
			served++
		}
	}
	if served < 14 {
		t.Fatalf("only %d of %d generating instances served traffic", served, len(generateWorkers))
	}
	for _, id := range encodeWorkers {
		if st := workerStats(t, id); st["generate_requests"] != float64(0) || st["encode_requests"].(float64) == 0 {
			t.Fatalf("encoder %s: %v Generate and %v Encode calls", id, st["generate_requests"], st["encode_requests"])
		}
	}
	for _, label := range []string{`stream="true"`, `stream="false"`} {
		if metric(t, gateway, "modelweave_requests_total", label, `code="200"`) == 0 {
			t.Fatalf("no successful client requests with %s", label)
		}
	}
	for _, role := range []string{"encode", "prefill", "decode", "unified"} {
		if metric(t, gateway, "modelweave_engine_requests_total", `role="`+role+`"`) == 0 {
			t.Fatalf("no engine calls to %s instances", role)
		}
	}
}
