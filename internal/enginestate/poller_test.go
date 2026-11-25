package enginestate

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"github.com/leonyang215/ModelWeave/internal/registry"
)

const vllmMetrics = `# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3
vllm:num_requests_running{engine="1",model_name="m"} 2
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 4
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.5
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.7
`

func TestVLLMPreset(t *testing.T) {
	p := expfmt.NewTextParser(model.UTF8Validation)
	parsed, err := p.TextToMetricFamilies(strings.NewReader(vllmMetrics))
	if err != nil {
		t.Fatal(err)
	}
	names := map[string][]string{}
	for k, v := range Presets["vllm"] {
		names[k] = strings.Split(v, "|")
	}
	st := StateFromFamilies(parsed, names)
	if st.RunningRequests != 5 || st.WaitingRequests != 4 {
		t.Fatalf("running/waiting: %+v", st)
	}
	if st.KVUsage < 0.59 || st.KVUsage > 0.61 {
		t.Fatalf("kv usage should average to 0.6: %+v", st)
	}
}

func TestPollerUpdatesRegistry(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/mwip/v1/state":
			_, _ = w.Write([]byte(`{"running_requests":7,"waiting_requests":1,"kv_used_tokens":100,"kv_capacity_tokens":1000}`))
		case "/metrics":
			_, _ = w.Write([]byte("sglang:num_running_reqs 9\nsglang:num_queue_reqs 2\nsglang:token_usage 0.25\n"))
		}
	}))
	defer srv.Close()
	reg := registry.New(time.Minute)
	w1, _, _ := reg.Upsert(registry.Spec{ID: "a", Model: "m", Role: registry.RoleDecode, URL: srv.URL}, "static")
	NewPoller(reg, srv.Client(), Options{Format: "mwip"}).PollAll(context.Background())
	if s := w1.Snapshot(time.Now(), time.Minute); s.State.RunningRequests != 7 || s.KVCapacity != 1000 || s.EffectiveKVUsed != 100 {
		t.Fatalf("mwip poll: %+v", s)
	}
	NewPoller(reg, srv.Client(), Options{Format: "prometheus", Preset: "sglang"}).PollAll(context.Background())
	if s := w1.Snapshot(time.Now(), time.Minute); s.State.RunningRequests != 9 || s.State.WaitingRequests != 2 || s.State.KVUsage != 0.25 {
		t.Fatalf("prometheus poll: %+v", s.State)
	}
}
