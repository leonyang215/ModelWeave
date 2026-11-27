package gateway

import (
	"context"
	"errors"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/leonyang215/ModelWeave/internal/obs"
	"github.com/leonyang215/ModelWeave/internal/qos"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

var workerLabels = []string{"worker", "model", "role"}

var (
	descUp       = prometheus.NewDesc("modelweave_worker_up", "1 when the worker is healthy, not draining and not blocked.", workerLabels, nil)
	descBlocked  = prometheus.NewDesc("modelweave_worker_blocked", "1 when an operator has blocked the instance ID.", workerLabels, nil)
	descInflight = prometheus.NewDesc("modelweave_worker_inflight_requests", "Requests the gateway has assigned to the worker.", workerLabels, nil)
	descReserved = prometheus.NewDesc("modelweave_worker_reserved_kv_tokens", "KV tokens reserved by the gateway on the worker.", workerLabels, nil)
	descKVUsed   = prometheus.NewDesc("modelweave_worker_kv_used_tokens", "Effective KV usage (reported usage merged with unreported reservations).", workerLabels, nil)
	descKVCap    = prometheus.NewDesc("modelweave_worker_kv_capacity_tokens", "KV capacity of the worker.", workerLabels, nil)
	descKVUtil   = prometheus.NewDesc("modelweave_worker_kv_utilization", "Effective KV utilization (0-1).", workerLabels, nil)
	descPrefill  = prometheus.NewDesc("modelweave_worker_queued_prefill_tokens", "Prefill tokens queued on the worker (gateway view merged with reports).", workerLabels, nil)
	descRunning  = prometheus.NewDesc("modelweave_worker_reported_running_requests", "Running requests reported by the engine.", workerLabels, nil)
	descWaiting  = prometheus.NewDesc("modelweave_worker_reported_waiting_requests", "Waiting requests reported by the engine.", workerLabels, nil)
	descAge      = prometheus.NewDesc("modelweave_worker_state_age_seconds", "Age of the last engine state report.", workerLabels, nil)
	descDone     = prometheus.NewDesc("modelweave_worker_completed_total", "Requests completed on the worker.", workerLabels, nil)
	descErrs     = prometheus.NewDesc("modelweave_worker_errors_total", "Transport-level failures of the worker.", workerLabels, nil)
)

// workerCollector exports per-worker state at scrape time.
type workerCollector struct {
	reg        *registry.Registry
	staleAfter time.Duration
}

func (c *workerCollector) Describe(ch chan<- *prometheus.Desc) {
	for _, d := range []*prometheus.Desc{descUp, descBlocked, descInflight, descReserved, descKVUsed, descKVCap, descKVUtil, descPrefill, descRunning, descWaiting, descAge, descDone, descErrs} {
		ch <- d
	}
}

func (c *workerCollector) Collect(ch chan<- prometheus.Metric) {
	now := time.Now()
	ws := c.reg.All()
	snaps, err := c.reg.Snapshots(context.Background(), ws, now, c.staleAfter)
	if err != nil {
		return
	}
	for i, w := range ws {
		s := snaps[i]
		if s.Gone {
			continue
		}
		lv := []string{s.Spec.ID, s.Spec.Model, string(s.Spec.Role)}
		up := 0.0
		if s.Healthy && !s.Draining && !s.Blocked {
			up = 1
		}
		gauge := func(d *prometheus.Desc, v float64) {
			ch <- prometheus.MustNewConstMetric(d, prometheus.GaugeValue, v, lv...)
		}
		gauge(descUp, up)
		blocked := 0.0
		if s.Blocked {
			blocked = 1
		}
		gauge(descBlocked, blocked)
		gauge(descInflight, float64(s.Inflight))
		gauge(descReserved, float64(s.ReservedKV))
		gauge(descKVUsed, float64(s.EffectiveKVUsed))
		gauge(descKVCap, float64(s.KVCapacity))
		gauge(descKVUtil, s.KVUtil())
		gauge(descPrefill, float64(s.QueuedPrefillTokens()))
		if s.HasState {
			gauge(descRunning, float64(s.State.RunningRequests))
			gauge(descWaiting, float64(s.State.WaitingRequests))
			gauge(descAge, s.StateAge.Seconds())
		}
		done, errs := w.Counters()
		ch <- prometheus.MustNewConstMetric(descDone, prometheus.CounterValue, float64(done), lv...)
		ch <- prometheus.MustNewConstMetric(descErrs, prometheus.CounterValue, float64(errs), lv...)
	}
}

type schedObserver struct {
	met   *obs.Metrics
	model string
}

func (o *schedObserver) Scheduled(req *scheduler.Request, d *scheduler.Decision) {
	role := string(req.Role)
	if req.RequireNormal {
		role = "decode_normal"
	}
	o.met.ScheduleDecisions.WithLabelValues(o.model, role, d.Policy, d.Reason).Inc()
	o.met.ScheduleDuration.WithLabelValues(o.model, role).Observe(d.Duration.Seconds())
	if len(req.Blocks) > 0 {
		o.met.CacheMatchRatio.WithLabelValues(o.model, role).Observe(d.CacheRatio)
	}
}

func (o *schedObserver) Failed(req *scheduler.Request, _ string, err error) {
	kind := "error"
	switch {
	case errors.Is(err, scheduler.ErrNoCapacity):
		kind = "no_capacity"
	case errors.Is(err, scheduler.ErrNoWorkers):
		kind = "no_workers"
	}
	o.met.ScheduleFailures.WithLabelValues(o.model, string(req.Role), kind).Inc()
}

func (o *schedObserver) CacheLookup(kind string, d time.Duration, err error) {
	outcome := "ok"
	if err != nil {
		outcome = "error"
	}
	o.met.CacheLookup.WithLabelValues(kind, outcome).Observe(d.Seconds())
}

type queueObserver struct {
	met   *obs.Metrics
	model string
}

func (o *queueObserver) Outcome(p qos.Priority, outcome string, waited time.Duration) {
	o.met.QueueWait.WithLabelValues(o.model, p.String(), outcome).Observe(waited.Seconds())
}

func (o *queueObserver) Depth(p qos.Priority, depth int) {
	o.met.QueueDepth.WithLabelValues(o.model, p.String()).Set(float64(depth))
}
