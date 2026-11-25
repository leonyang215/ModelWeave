// Package enginestate collects engine load by polling workers. It is the
// pull side of the engine-state adapter layer; the push side is the MWIP
// heartbeat handled by the gateway. Both end up in registry.Worker.
//
// Formats:
//
//	mwip        GET {url}/mwip/v1/state returning registry.EngineState JSON
//	prometheus  GET {url}/metrics (or the worker's state_url), mapped with a
//	            preset (vllm, sglang) or explicit metric names
package enginestate

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"github.com/leonyang215/ModelWeave/internal/registry"
)

// Field names usable in Options.Metrics.
const (
	FieldRunning      = "running_requests"
	FieldWaiting      = "waiting_requests"
	FieldKVUsage      = "kv_usage"
	FieldKVUsed       = "kv_used_tokens"
	FieldKVCapacity   = "kv_capacity_tokens"
	FieldPrefillQueue = "prefill_queue_tokens"
)

// Presets map engine-native Prometheus metrics to engine state fields.
// Several candidate names may be listed per field (separated by '|'); the
// first present one is used. Values are summed over label sets (DP ranks).
var Presets = map[string]map[string]string{
	"vllm": {
		FieldRunning: "vllm:num_requests_running",
		FieldWaiting: "vllm:num_requests_waiting",
		FieldKVUsage: "vllm:kv_cache_usage_perc|vllm:gpu_cache_usage_perc",
	},
	"sglang": {
		FieldRunning: "sglang:num_running_reqs",
		FieldWaiting: "sglang:num_queue_reqs",
		FieldKVUsage: "sglang:token_usage",
		FieldKVUsed:  "sglang:num_used_tokens",
	},
	// The mock engine and MWIP-native engines can also be scraped.
	"modelweave": {
		FieldRunning:      "mwip_engine_running_requests",
		FieldWaiting:      "mwip_engine_waiting_requests",
		FieldKVUsed:       "mwip_engine_kv_used_tokens",
		FieldKVCapacity:   "mwip_engine_kv_capacity_tokens",
		FieldPrefillQueue: "mwip_engine_prefill_queue_tokens",
	},
}

// Options configure a poller.
type Options struct {
	Format   string
	Preset   string
	Metrics  map[string]string
	Timeout  time.Duration
	OnResult func(format, outcome string)
}

// Poller fetches engine state for every registered worker.
type Poller struct {
	reg    *registry.Registry
	client *http.Client
	opts   Options
	names  map[string][]string
}

// NewPoller returns a poller.
func NewPoller(reg *registry.Registry, client *http.Client, opts Options) *Poller {
	if opts.Timeout <= 0 {
		opts.Timeout = time.Second
	}
	names := map[string][]string{}
	for k, v := range Presets[opts.Preset] {
		names[k] = strings.Split(v, "|")
	}
	for k, v := range opts.Metrics {
		names[k] = strings.Split(v, "|")
	}
	return &Poller{reg: reg, client: client, opts: opts, names: names}
}

// PollAll polls every worker concurrently (at most 32 at a time).
func (p *Poller) PollAll(ctx context.Context) {
	sem := make(chan struct{}, 32)
	var wg sync.WaitGroup
	for _, w := range p.reg.All() {
		wg.Add(1)
		sem <- struct{}{}
		go func(w *registry.Worker) {
			defer wg.Done()
			defer func() { <-sem }()
			st, err := p.Fetch(ctx, w.Spec())
			outcome := "ok"
			if err != nil {
				outcome = "error"
			} else {
				w.UpdateState(st, time.Now())
			}
			if p.opts.OnResult != nil {
				p.opts.OnResult(p.opts.Format, outcome)
			}
		}(w)
	}
	wg.Wait()
}

// Fetch retrieves the state of one worker.
func (p *Poller) Fetch(ctx context.Context, spec registry.Spec) (registry.EngineState, error) {
	ctx, cancel := context.WithTimeout(ctx, p.opts.Timeout)
	defer cancel()
	url := spec.StateURL
	if url == "" {
		if p.opts.Format == "prometheus" {
			url = spec.URL + "/metrics"
		} else {
			url = spec.URL + "/mwip/v1/state"
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return registry.EngineState{}, err
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return registry.EngineState{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return registry.EngineState{}, fmt.Errorf("state %s: status %d", url, resp.StatusCode)
	}
	body := io.LimitReader(resp.Body, 8<<20)
	if p.opts.Format == "prometheus" {
		return p.parsePrometheus(body)
	}
	var st registry.EngineState
	if err := json.NewDecoder(body).Decode(&st); err != nil {
		return st, fmt.Errorf("state %s: %w", url, err)
	}
	return st, nil
}

func (p *Poller) parsePrometheus(r io.Reader) (registry.EngineState, error) {
	parser := expfmt.NewTextParser(model.UTF8Validation)
	fams, err := parser.TextToMetricFamilies(r)
	if err != nil {
		return registry.EngineState{}, err
	}
	return StateFromFamilies(fams, p.names), nil
}

// StateFromFamilies maps parsed metric families to engine state.
func StateFromFamilies(fams map[string]*dto.MetricFamily, names map[string][]string) registry.EngineState {
	get := func(field string) (float64, bool) {
		for _, n := range names[field] {
			if f, ok := fams[n]; ok {
				return sum(f), true
			}
		}
		return 0, false
	}
	var st registry.EngineState
	if v, ok := get(FieldRunning); ok {
		st.RunningRequests = int(v)
	}
	if v, ok := get(FieldWaiting); ok {
		st.WaitingRequests = int(v)
	}
	if v, ok := get(FieldKVUsed); ok {
		st.KVUsedTokens = int64(v)
	}
	if v, ok := get(FieldKVCapacity); ok {
		st.KVCapacityTokens = int64(v)
	}
	if v, ok := get(FieldPrefillQueue); ok {
		st.PrefillQueueTokens = int64(v)
	}
	if v, ok := get(FieldKVUsage); ok {
		if n := countSeries(fams, names[FieldKVUsage]); n > 1 {
			v /= float64(n) // utilization is averaged, not summed
		}
		if v > 1 {
			v /= 100 // some engines report percent
		}
		st.KVUsage = v
	}
	return st
}

func countSeries(fams map[string]*dto.MetricFamily, names []string) int {
	for _, n := range names {
		if f, ok := fams[n]; ok {
			return len(f.GetMetric())
		}
	}
	return 0
}

func sum(f *dto.MetricFamily) float64 {
	total := 0.0
	for _, m := range f.GetMetric() {
		switch {
		case m.Gauge != nil:
			total += m.GetGauge().GetValue()
		case m.Counter != nil:
			total += m.GetCounter().GetValue()
		case m.Untyped != nil:
			total += m.GetUntyped().GetValue()
		}
	}
	return total
}
