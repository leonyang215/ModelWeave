// Package obs holds the gateway's logging setup and Prometheus metrics.
package obs

import (
	"log/slog"
	"os"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
)

// NewLogger returns a slog logger writing JSON (or text) to stderr.
func NewLogger(level, format string) *slog.Logger {
	var lv slog.Level
	switch strings.ToLower(level) {
	case "debug":
		lv = slog.LevelDebug
	case "warn", "warning":
		lv = slog.LevelWarn
	case "error":
		lv = slog.LevelError
	default:
		lv = slog.LevelInfo
	}
	opts := &slog.HandlerOptions{Level: lv}
	if format == "text" {
		return slog.New(slog.NewTextHandler(os.Stderr, opts))
	}
	return slog.New(slog.NewJSONHandler(os.Stderr, opts))
}

var (
	latencyBuckets = []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120, 300}
	fastBuckets    = []float64{.00005, .0001, .00025, .0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1}
	ratioBuckets   = []float64{0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1}
)

// Metrics is the gateway metric set.
type Metrics struct {
	Registry *prometheus.Registry

	Requests        *prometheus.CounterVec
	Finishes        *prometheus.CounterVec
	RequestDuration *prometheus.HistogramVec
	TTFT            *prometheus.HistogramVec
	TTFTSegment     *prometheus.HistogramVec
	TPOT            *prometheus.HistogramVec
	Tokens          *prometheus.CounterVec
	Inflight        *prometheus.GaugeVec
	StageDuration   *prometheus.HistogramVec

	TokenizerDuration *prometheus.HistogramVec
	TokenizerDown     *prometheus.CounterVec

	ScheduleDecisions *prometheus.CounterVec
	ScheduleFailures  *prometheus.CounterVec
	ScheduleDuration  *prometheus.HistogramVec
	CacheMatchRatio   *prometheus.HistogramVec
	CacheLookup       *prometheus.HistogramVec

	ExecutionMode  *prometheus.CounterVec
	Fallbacks      *prometheus.CounterVec
	Retries        *prometheus.CounterVec
	UpstreamErrors *prometheus.CounterVec
	EngineRequests *prometheus.CounterVec
	Aborts         *prometheus.CounterVec
	ArtifactLookup *prometheus.CounterVec

	Admission     *prometheus.CounterVec
	RateLimited   *prometheus.CounterVec
	QueueDepth    *prometheus.GaugeVec
	QueueWait     *prometheus.HistogramVec
	OverloadLevel *prometheus.GaugeVec
	Utilization   *prometheus.GaugeVec

	Heartbeats *prometheus.CounterVec
	// InstanceRemovals counts instances leaving the registry (deregistered,
	// expired).
	InstanceRemovals *prometheus.CounterVec
	KVEvents         *prometheus.CounterVec
	StatePolls       *prometheus.CounterVec
	BuildInfo        *prometheus.GaugeVec
	ModelInfo        *prometheus.GaugeVec

	MediaItems  *prometheus.CounterVec
	MediaTokens *prometheus.CounterVec
}

// NewMetrics creates and registers the gateway metrics.
func NewMetrics() *Metrics {
	r := prometheus.NewRegistry()
	r.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	m := &Metrics{Registry: r}
	f := func(c prometheus.Collector) { r.MustRegister(c) }

	m.Requests = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_requests_total",
		Help: "Client requests by model, API, client streaming (true/false), executed mode and HTTP status code.",
	}, []string{"model", "api", "stream", "mode", "code"})
	m.Finishes = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_finish_reasons_total",
		Help: "Completed generations by the finish reason sent to the client (stop, length, tool_calls).",
	}, []string{"model", "api", "reason"})
	m.RequestDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "modelweave_request_duration_seconds", Help: "End-to-end request latency.", Buckets: latencyBuckets,
	}, []string{"model", "api", "mode"})
	m.TTFT = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "modelweave_ttft_seconds", Help: "Time from request arrival to the first output token written to the client.", Buckets: latencyBuckets,
	}, []string{"model", "mode"})
	m.TTFTSegment = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "modelweave_ttft_segment_seconds",
		Help:    "TTFT split into segments: gateway (tokenize+admission+queue+schedule), encode, engine (dispatch to first token).",
		Buckets: latencyBuckets,
	}, []string{"model", "mode", "segment"})
	m.TPOT = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "modelweave_time_per_output_token_seconds", Help: "Mean inter-token latency of a request after its first token.",
		Buckets: []float64{.001, .0025, .005, .01, .02, .03, .05, .075, .1, .15, .25, .5, 1},
	}, []string{"model", "mode"})
	m.Tokens = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_tokens_total", Help: "Tokens by kind: prompt, cached (prompt tokens served from KV cache), completion, reasoning.",
	}, []string{"model", "tenant", "kind"})
	m.Inflight = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "modelweave_inflight_requests", Help: "Requests being processed (including queued).",
	}, []string{"model"})
	m.StageDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "modelweave_stage_duration_seconds",
		Help:    "Duration of request stages: tokenize, admission_queue, schedule, encode, decode_register, prefill, first_token, decode.",
		Buckets: latencyBuckets,
	}, []string{"model", "stage"})
	m.TokenizerDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "modelweave_tokenizer_client_duration_seconds", Help: "Tokenizer render latency seen by the gateway.", Buckets: latencyBuckets,
	}, []string{"outcome"})
	m.TokenizerDown = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_tokenizer_unreachable_total", Help: "Tokenizer replicas that could not be reached, by endpoint; the render moved on to the next replica.",
	}, []string{"endpoint"})
	m.ScheduleDecisions = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_schedule_decisions_total", Help: "Worker selections by role, policy and decision reason (cache_hit, least_load, rebalance, affinity...).",
	}, []string{"model", "role", "policy", "reason"})
	m.ScheduleFailures = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_schedule_failures_total", Help: "Scheduling passes without a worker (no_workers, no_capacity).",
	}, []string{"model", "role", "error"})
	m.ScheduleDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "modelweave_schedule_duration_seconds", Help: "Latency of one scheduling pass.", Buckets: fastBuckets,
	}, []string{"model", "role"})
	m.CacheMatchRatio = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "modelweave_prefix_cache_match_ratio", Help: "Predicted prefix-cache hit ratio of the selected worker.", Buckets: ratioBuckets,
	}, []string{"model", "role"})
	m.CacheLookup = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "modelweave_kv_index_lookup_seconds", Help: "Prefix-cache index lookups by index kind and outcome.", Buckets: fastBuckets,
	}, []string{"kind", "outcome"})
	m.ExecutionMode = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_execution_plans_total", Help: "Execution plans by mode (normal, pd; epd_ prefix when a gateway encode stage ran) and reason.",
	}, []string{"model", "mode", "reason"})
	m.Fallbacks = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_fallbacks_total", Help: "Execution fallbacks (normal to pd).",
	}, []string{"model", "from", "to", "reason"})
	m.Retries = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_retries_total", Help: "Retries before the first output token by stage and reason.",
	}, []string{"model", "stage", "reason"})
	m.UpstreamErrors = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_upstream_errors_total", Help: "Errors returned by workers by role and MWIP error code.",
	}, []string{"model", "role", "code"})
	m.EngineRequests = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_engine_requests_total", Help: "mwip calls sent to instances by role (encode, prefill, decode, unified). Generate calls always stream; Encode calls are unary.",
	}, []string{"model", "role"})
	m.Aborts = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_aborts_total", Help: "Engine requests aborted by the gateway by reason (client_gone, stopped, timeout, error).",
	}, []string{"model", "reason"})
	m.ArtifactLookup = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_embedding_artifact_lookups_total", Help: "Embedding artifact cache lookups (hit, miss, invalidated).",
	}, []string{"model", "result"})
	m.Admission = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_admission_total", Help: "Admission outcomes: admitted, queued, rejected_rate_limit, rejected_overload, rejected_queue_full, rejected_timeout, shed.",
	}, []string{"model", "tenant", "priority", "result"})
	m.RateLimited = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_rate_limited_total", Help: "Requests rejected by tenant limits (rpm, tpm, concurrency).",
	}, []string{"tenant", "reason"})
	m.QueueDepth = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "modelweave_queue_depth", Help: "Requests waiting for capacity.",
	}, []string{"model", "priority"})
	m.QueueWait = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "modelweave_queue_wait_seconds", Help: "Time spent in the admission queue by outcome.", Buckets: latencyBuckets,
	}, []string{"model", "priority", "outcome"})
	m.OverloadLevel = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "modelweave_overload_level", Help: "Load level (0-4) of the overload controller.",
	}, []string{"model"})
	m.Utilization = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "modelweave_pool_utilization", Help: "Pool utilization used by the overload controller.",
	}, []string{"model", "role"})
	m.Heartbeats = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_heartbeats_total", Help: "mwip Registry heartbeats by result (registered, updated, rejected).",
	}, []string{"result"})
	m.InstanceRemovals = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_instance_removals_total", Help: "Instances removed from the registry by reason: deregistered (Registry.Deregister) or expired (no heartbeat for discovery.heartbeat_ttl).",
	}, []string{"model", "role", "reason"})
	m.KVEvents = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_kv_events_total", Help: "KV cache events received from engines.",
	}, []string{"kind"})
	m.StatePolls = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_engine_state_polls_total", Help: "Engine.GetState polls by outcome (ok, error).",
	}, []string{"outcome"})
	m.ModelInfo = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "modelweave_model_info", Help: "Configuration of a served model (1): mode and PD handshake.",
	}, []string{"model", "mode", "handshake"})
	m.MediaItems = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_media_items_total", Help: "Images and videos in rendered requests by modality.",
	}, []string{"model", "modality"})
	m.MediaTokens = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "modelweave_media_tokens_total", Help: "Prompt positions taken by media (pad-value tokens and video timestamp text) by modality.",
	}, []string{"model", "modality"})
	m.BuildInfo = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "modelweave_build_info", Help: "Build information.",
	}, []string{"version"})

	for _, c := range []prometheus.Collector{
		m.Requests, m.Finishes, m.RequestDuration, m.TTFT, m.TTFTSegment, m.TPOT, m.Tokens, m.Inflight, m.StageDuration,
		m.TokenizerDuration, m.TokenizerDown, m.ScheduleDecisions, m.ScheduleFailures, m.ScheduleDuration, m.CacheMatchRatio,
		m.CacheLookup, m.ExecutionMode, m.Fallbacks, m.Retries, m.UpstreamErrors, m.EngineRequests, m.Aborts, m.ArtifactLookup,
		m.Admission, m.RateLimited, m.QueueDepth, m.QueueWait, m.OverloadLevel, m.Utilization,
		m.Heartbeats, m.InstanceRemovals, m.KVEvents, m.StatePolls, m.BuildInfo, m.ModelInfo, m.MediaItems, m.MediaTokens,
	} {
		f(c)
	}
	return m
}
