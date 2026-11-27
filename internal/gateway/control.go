package gateway

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// control guards admin APIs, not the public console shell. It does not
// accept tenant API keys or put the admin token in a URL/cookie.
func (s *Server) control(h http.HandlerFunc) http.HandlerFunc {
	token := s.adminToken()
	var origin http.CrossOriginProtection
	return func(w http.ResponseWriter, r *http.Request) {
		adminHeaders(w)
		if err := origin.Check(r); err != nil {
			controlError(w, http.StatusForbidden, "cross-origin admin mutation is not allowed")
			return
		}
		if token != "" && subtle.ConstantTimeCompare([]byte(r.Header.Get("X-ModelWeave-Token")), []byte(token)) != 1 {
			writeJSON(w, http.StatusUnauthorized, mwip.ErrorBody{Type: "error", Code: "unauthorized", Message: "missing or invalid X-ModelWeave-Token"})
			return
		}
		h(w, r)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func controlError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, mwip.ErrorBody{Type: "error", Code: mwip.ErrBadRequest, Message: msg})
}

// applyKVEvents feeds KV cache events of an instance to the event index.
func (s *Server) applyKVEvents(worker *registry.Worker, ev *mwip.KVEvents) error {
	if s.events == nil {
		return nil
	}
	if err := s.events.ApplyEvents(worker.CacheKey(), kvindex.EventBatch{Stored: ev.Stored, Removed: ev.Removed, Cleared: ev.Cleared}); err != nil {
		return err
	}
	s.met.KVEvents.WithLabelValues("stored").Add(float64(len(ev.Stored)))
	s.met.KVEvents.WithLabelValues("removed").Add(float64(len(ev.Removed)))
	if ev.Cleared {
		s.met.KVEvents.WithLabelValues("cleared").Inc()
	}
	return nil
}

type workerView struct {
	ID            string            `json:"id"`
	Model         string            `json:"model"`
	Role          registry.Role     `json:"role"`
	Target        string            `json:"target"`           // mwip target
	Normal        bool              `json:"normal,omitempty"` // runs normal (non-PD) requests
	Source        string            `json:"source"`           // heartbeat or static
	Healthy       bool              `json:"healthy"`
	Draining      bool              `json:"draining"`                // takes no new requests
	AdminDrained  bool              `json:"admin_drained,omitempty"` // drained through the admin API
	Blocked       bool              `json:"blocked"`
	Block         *registry.Block   `json:"block,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
	EngineID      string            `json:"engine_id,omitempty"`
	BootstrapAddr string            `json:"bootstrap_addr,omitempty"`
	MaxRunning    int               `json:"max_running_requests"`
	MaxPrefill    int64             `json:"max_prefill_queue_tokens"`
	// HeartbeatAgeMS is the time since the last heartbeat; the instance
	// expires when it passes the heartbeat TTL (heartbeat instances).
	HeartbeatAgeMS  int64                 `json:"heartbeat_age_ms,omitempty"`
	Inflight        int64                 `json:"inflight"`
	ReservedKV      int64                 `json:"reserved_kv_tokens"`
	InflightPrefill int64                 `json:"inflight_prefill_tokens"`
	EffectiveKVUsed int64                 `json:"effective_kv_used_tokens"`
	KVCapacity      int64                 `json:"kv_capacity_tokens"`
	StateAgeMS      int64                 `json:"state_age_ms,omitempty"`
	State           *registry.EngineState `json:"state,omitempty"`
	StateStatus     string                `json:"state_status"` // fresh | stale | unknown
	Completed       int64                 `json:"completed"`
	Errors          int64                 `json:"errors"`
	CachedBlocks    int                   `json:"index_blocks,omitempty"`
}

func (s *Server) handleWorkers(w http.ResponseWriter, r *http.Request) {
	out, err := s.workerViews(r.Context())
	if err != nil {
		controlError(w, http.StatusServiceUnavailable, "admin state unavailable")
		return
	}
	model := r.URL.Query().Get("model")
	if model != "" {
		m := s.byName[model]
		if m == nil {
			controlError(w, http.StatusNotFound, "unknown model")
			return
		}
		model = m.cfg.Name
	}
	role := r.URL.Query().Get("role")
	if role != "" {
		if _, err := registry.ParseRole(role); err != nil {
			controlError(w, http.StatusBadRequest, err.Error())
			return
		}
	}
	status := r.URL.Query().Get("status")
	switch status {
	case "", "active", "blocked", "draining", "unhealthy":
	default:
		controlError(w, http.StatusBadRequest, "status must be active, blocked, draining or unhealthy")
		return
	}
	filtered := make([]workerView, 0, len(out))
	for _, v := range out {
		if (model == "" || v.Model == model) && (role == "" || string(v.Role) == role) && (status == "" || v.status() == status) {
			filtered = append(filtered, v)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"workers": filtered})
}

func (v workerView) status() string {
	switch {
	case v.Blocked:
		return "blocked"
	case v.Draining:
		return "draining"
	case !v.Healthy:
		return "unhealthy"
	default:
		return "active"
	}
}

func (s *Server) workerViews(ctx context.Context) ([]workerView, error) {
	if err := s.reg.Sync(ctx); err != nil {
		return nil, err
	}
	now := time.Now()
	ws := s.reg.All()
	_, perWorker, err := indexStats(ctx, s.index, ws)
	if err != nil {
		return nil, fmt.Errorf("prefix index unavailable: %w", err)
	}
	snaps, err := s.reg.Snapshots(ctx, ws, now, s.cfg.EngineState.StaleAfter.D())
	if err != nil {
		return nil, err
	}
	out := make([]workerView, 0, len(ws))
	for i, wk := range ws {
		snap := snaps[i]
		if snap.Gone {
			continue
		}
		done, errs := wk.Counters()
		v := workerView{
			ID: snap.Spec.ID, Model: snap.Spec.Model, Role: snap.Spec.Role, Target: snap.Spec.Target, Normal: snap.Spec.ServesNormal(),
			Source: wk.Source(), Healthy: snap.Healthy, Draining: snap.Draining, AdminDrained: snap.AdminDrained, Inflight: snap.Inflight,
			Blocked: snap.Blocked, Block: snap.Block, Labels: snap.Spec.Labels,
			EngineID: snap.Spec.EngineID, BootstrapAddr: snap.Spec.BootstrapAddr, MaxRunning: snap.MaxRunning, MaxPrefill: snap.MaxPrefillTokens,
			ReservedKV: snap.ReservedKV, InflightPrefill: snap.PrefillTok, EffectiveKVUsed: snap.EffectiveKVUsed,
			KVCapacity: snap.KVCapacity, Completed: done, Errors: errs, CachedBlocks: perWorker[wk.CacheKey()],
			StateStatus: "unknown",
		}
		if wk.Source() == "heartbeat" {
			v.HeartbeatAgeMS = max(now.Sub(snap.LastSeen).Milliseconds(), 1)
		}
		if snap.StateReported {
			v.StateStatus = "stale"
			v.StateAgeMS = snap.StateAge.Milliseconds()
		}
		if snap.HasState {
			st := snap.State
			v.State = &st
			v.StateStatus = "fresh"
		}
		out = append(out, v)
	}
	return out, nil
}

func (s *Server) handleDrain(w http.ResponseWriter, r *http.Request) {
	s.setDraining(w, r, r.URL.Query().Get("undo") == "")
}

func (s *Server) handleResume(w http.ResponseWriter, r *http.Request) {
	s.setDraining(w, r, false)
}

func (s *Server) setDraining(w http.ResponseWriter, r *http.Request, drain bool) {
	if err := s.reg.Sync(r.Context()); err != nil {
		controlError(w, http.StatusServiceUnavailable, "state store unavailable")
		return
	}
	wk := s.reg.Get(r.PathValue("id"))
	if wk == nil {
		controlError(w, http.StatusNotFound, "unknown worker")
		return
	}
	if err := wk.SetDrainingContext(r.Context(), drain); err != nil {
		if errors.Is(err, registry.ErrGone) {
			controlError(w, http.StatusConflict, "instance registration changed; refresh and retry")
			return
		}
		controlError(w, http.StatusServiceUnavailable, "state store unavailable")
		return
	}
	if m := s.models[wk.Spec().Model]; m != nil {
		m.queue.Notify()
	}
	s.log.Info("admin action", "action", "set_drain", "worker", wk.ID(), "draining", drain, "remote_addr", r.RemoteAddr)
	writeJSON(w, http.StatusOK, map[string]any{"worker": wk.ID(), "draining": drain})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if err := s.reg.Sync(r.Context()); err != nil {
		controlError(w, http.StatusServiceUnavailable, "state store unavailable")
		return
	}
	models := map[string]any{}
	names := make([]string, 0, len(s.models))
	for n := range s.models {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		m := s.models[n]
		queueSize, err := m.queue.LenContext(r.Context())
		if err != nil {
			controlError(w, http.StatusServiceUnavailable, "QoS state store unavailable")
			return
		}
		level, util := m.overload.Level()
		models[n] = map[string]any{
			"mode":             m.cfg.Mode,
			"handshake":        m.cfg.PD.Handshake,
			"queue_len":        queueSize,
			"overload_level":   level,
			"utilization":      util,
			"sessions":         m.rt.Sessions.Len(),
			"artifacts_cached": m.rt.Artifacts.Len(),
			"policies": map[string]string{
				"prefill": m.rt.Prefill.Policy(), "decode": m.rt.Decode.Policy(),
				"encode": m.rt.Encode.Policy(), "unified": m.rt.Unified.Policy(),
			},
			"scorers": map[string]any{"prefill": m.rt.Prefill.Scorers(), "decode": m.rt.Decode.Scorers(), "encode": m.rt.Encode.Scorers(), "unified": m.rt.Unified.Scorers()},
		}
	}
	idx := map[string]any{"kind": s.index.Kind()}
	b, _, err := indexStats(r.Context(), s.index, s.reg.All())
	if err != nil {
		controlError(w, http.StatusServiceUnavailable, "prefix index unavailable")
		return
	}
	idx["blocks"] = b
	tenants, err := s.limiter.StatsContext(r.Context())
	if err != nil {
		controlError(w, http.StatusServiceUnavailable, "QoS state store unavailable")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"version": s.version, "models": models, "kv_index": idx, "tenants": tenants, "state_backend": s.reg.Store().Kind(), "qos_backend": s.qosBackend.Kind(),
		"ready": s.ready.Load(), "uptime_seconds": int64(time.Since(s.startedAt).Seconds()), "admin_auth_required": s.adminToken() != "",
	})
}

func indexStats(ctx context.Context, index kvindex.Index, ws []*registry.Worker) (int, map[string]int, error) {
	switch ix := index.(type) {
	case *kvindex.Approx:
		b, p := ix.Blocks().Stats()
		return b, p, nil
	case *kvindex.Events:
		b, p := ix.Exact().Stats()
		return b, p, nil
	case *kvindex.Redis:
		keys := make([]string, len(ws))
		for i, w := range ws {
			keys[i] = w.CacheKey()
		}
		return ix.Stats(ctx, keys)
	case *kvindex.Remote:
		return indexStats(ctx, ix.Fallback, ws)
	}
	return 0, nil, nil
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	var data []model
	for _, m := range s.cfg.Models {
		for _, n := range append([]string{m.Name}, m.Aliases...) {
			data = append(data, model{ID: n, Object: "model", Created: 0, OwnedBy: "modelweave"})
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}
