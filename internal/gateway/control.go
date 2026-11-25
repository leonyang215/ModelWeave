package gateway

import (
	"crypto/subtle"
	"encoding/json"
	"io"
	"net/http"
	"sort"
	"time"

	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// control guards control-plane and admin endpoints with the optional shared
// token (constant-time comparison).
func (s *Server) control(h http.HandlerFunc) http.HandlerFunc {
	token := s.cfg.Server.ControlToken
	if token == "" {
		return h
	}
	return func(w http.ResponseWriter, r *http.Request) {
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("X-ModelWeave-Token")), []byte(token)) != 1 {
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

// handleHeartbeat registers or refreshes a worker and records its state.
func (s *Server) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	var hb mwip.Heartbeat
	if err := json.NewDecoder(io.LimitReader(r.Body, 8<<20)).Decode(&hb); err != nil {
		s.met.Heartbeats.WithLabelValues("rejected").Inc()
		controlError(w, http.StatusBadRequest, "invalid heartbeat: "+err.Error())
		return
	}
	m := s.byName[hb.Worker.Model]
	if m == nil {
		s.met.Heartbeats.WithLabelValues("rejected").Inc()
		controlError(w, http.StatusBadRequest, "heartbeat for unknown model "+hb.Worker.Model)
		return
	}
	role, err := registry.ParseRole(hb.Worker.Role)
	if err != nil {
		s.met.Heartbeats.WithLabelValues("rejected").Inc()
		controlError(w, http.StatusBadRequest, err.Error())
		return
	}
	spec := registry.Spec{
		ID: hb.Worker.ID, Model: m.cfg.Name, Role: role, URL: hb.Worker.URL, Normal: hb.Worker.Normal,
		BootstrapAddr: hb.Worker.BootstrapAddr, EngineID: hb.Worker.EngineID, KVCapacityTokens: hb.Worker.KVCapacityTokens,
		MaxRunningRequests: hb.Worker.MaxRunningRequests, MaxPrefillTokens: hb.Worker.MaxPrefillTokens,
		Labels: hb.Worker.Labels,
	}
	if existing := s.reg.Get(spec.ID); existing != nil && existing.Source == "static" {
		// Static workers keep their configured identity; heartbeats only
		// contribute state.
		spec = existing.Spec()
	}
	wk, created, err := s.reg.Upsert(spec, "heartbeat")
	if err != nil {
		s.met.Heartbeats.WithLabelValues("rejected").Inc()
		controlError(w, http.StatusBadRequest, err.Error())
		return
	}
	now := time.Now()
	if len(hb.State) > 0 && string(hb.State) != "null" {
		var st registry.EngineState
		if err := json.Unmarshal(hb.State, &st); err != nil {
			s.met.Heartbeats.WithLabelValues("rejected").Inc()
			controlError(w, http.StatusBadRequest, "invalid state: "+err.Error())
			return
		}
		wk.UpdateState(st, now)
	} else {
		wk.Touch(now)
	}
	wk.SetDraining(hb.Draining)
	if hb.KVEvents != nil {
		s.applyKVEvents(spec.ID, hb.KVEvents)
	}
	if created {
		s.met.Heartbeats.WithLabelValues("registered").Inc()
	} else {
		s.met.Heartbeats.WithLabelValues("updated").Inc()
	}
	m.queue.Notify()
	writeJSON(w, http.StatusOK, mwip.HeartbeatAck{OK: true, TTLMS: s.reg.TTL().Milliseconds()})
}

func (s *Server) applyKVEvents(worker string, ev *mwip.KVEvents) {
	if s.events == nil {
		return
	}
	s.events.ApplyEvents(worker, kvindex.EventBatch{Stored: ev.Stored, Removed: ev.Removed, Cleared: ev.Cleared})
	s.met.KVEvents.WithLabelValues("stored").Add(float64(len(ev.Stored)))
	s.met.KVEvents.WithLabelValues("removed").Add(float64(len(ev.Removed)))
	if ev.Cleared {
		s.met.KVEvents.WithLabelValues("cleared").Inc()
	}
}

// handleKVEvents ingests KV cache events: {"worker_id": "...", "stored": [...], ...}.
func (s *Server) handleKVEvents(w http.ResponseWriter, r *http.Request) {
	var req struct {
		WorkerID string `json:"worker_id"`
		mwip.KVEvents
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 32<<20)).Decode(&req); err != nil || req.WorkerID == "" {
		controlError(w, http.StatusBadRequest, "invalid kv events")
		return
	}
	if s.reg.Get(req.WorkerID) == nil {
		controlError(w, http.StatusNotFound, "unknown worker "+req.WorkerID)
		return
	}
	s.applyKVEvents(req.WorkerID, &req.KVEvents)
	writeJSON(w, http.StatusOK, map[string]bool{"ok": s.events != nil})
}

func (s *Server) handleDeregister(w http.ResponseWriter, r *http.Request) {
	var req struct {
		WorkerID string `json:"worker_id"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil || req.WorkerID == "" {
		controlError(w, http.StatusBadRequest, "worker_id required")
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"removed": s.reg.Remove(req.WorkerID)})
}

type workerView struct {
	ID              string                `json:"id"`
	Model           string                `json:"model"`
	Role            registry.Role         `json:"role"`
	URL             string                `json:"url"`
	Protocol        string                `json:"protocol"`         // engine protocol the gateway speaks to it
	Normal          bool                  `json:"normal,omitempty"` // runs normal (non-PD) requests
	Source          string                `json:"source"`
	Healthy         bool                  `json:"healthy"`
	Draining        bool                  `json:"draining"`
	Inflight        int64                 `json:"inflight"`
	ReservedKV      int64                 `json:"reserved_kv_tokens"`
	InflightPrefill int64                 `json:"inflight_prefill_tokens"`
	EffectiveKVUsed int64                 `json:"effective_kv_used_tokens"`
	KVCapacity      int64                 `json:"kv_capacity_tokens"`
	StateAgeMS      int64                 `json:"state_age_ms,omitempty"`
	State           *registry.EngineState `json:"state,omitempty"`
	Completed       int64                 `json:"completed"`
	Errors          int64                 `json:"errors"`
	CachedBlocks    int                   `json:"index_blocks,omitempty"`
}

func (s *Server) protocolOf(model string) string {
	if m := s.models[model]; m != nil {
		return m.rt.Protocol
	}
	return ""
}

func (s *Server) handleWorkers(w http.ResponseWriter, _ *http.Request) {
	now := time.Now()
	var perWorker map[string]int
	switch ix := s.index.(type) {
	case *kvindex.Approx:
		_, perWorker = ix.Blocks().Stats()
	case *kvindex.Events:
		_, perWorker = ix.Exact().Stats()
	}
	var out []workerView
	for _, wk := range s.reg.All() {
		snap := wk.Snapshot(now, s.cfg.EngineState.StaleAfter.D())
		done, errs := wk.Counters()
		v := workerView{
			ID: snap.Spec.ID, Model: snap.Spec.Model, Role: snap.Spec.Role, URL: snap.Spec.URL, Protocol: s.protocolOf(snap.Spec.Model), Normal: snap.Spec.ServesNormal(),
			Source: wk.Source, Healthy: snap.Healthy, Draining: snap.Draining, Inflight: snap.Inflight,
			ReservedKV: snap.ReservedKV, InflightPrefill: snap.PrefillTok, EffectiveKVUsed: snap.EffectiveKVUsed,
			KVCapacity: snap.KVCapacity, Completed: done, Errors: errs, CachedBlocks: perWorker[snap.Spec.ID],
		}
		if snap.HasState {
			st := snap.State
			v.State = &st
			v.StateAgeMS = snap.StateAge.Milliseconds()
		}
		out = append(out, v)
	}
	writeJSON(w, http.StatusOK, map[string]any{"workers": out})
}

func (s *Server) handleDrain(w http.ResponseWriter, r *http.Request) {
	wk := s.reg.Get(r.PathValue("id"))
	if wk == nil {
		controlError(w, http.StatusNotFound, "unknown worker")
		return
	}
	drain := r.URL.Query().Get("undo") == ""
	wk.SetDraining(drain)
	writeJSON(w, http.StatusOK, map[string]any{"worker": wk.ID(), "draining": drain})
}

func (s *Server) handleStats(w http.ResponseWriter, _ *http.Request) {
	models := map[string]any{}
	names := make([]string, 0, len(s.models))
	for n := range s.models {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		m := s.models[n]
		level, util := m.overload.Level()
		models[n] = map[string]any{
			"mode":             m.cfg.Mode,
			"engine_protocol":  m.rt.Protocol,
			"handshake":        m.cfg.PD.Handshake,
			"queue_len":        m.queue.Len(),
			"overload_level":   level,
			"utilization":      util,
			"sessions":         m.rt.Sessions.Len(),
			"artifacts_cached": m.rt.Artifacts.Len(),
			"policies": map[string]string{
				"prefill": m.rt.Prefill.Policy(), "decode": m.rt.Decode.Policy(),
				"encode": m.rt.Encode.Policy(), "unified": m.rt.Unified.Policy(),
			},
		}
	}
	idx := map[string]any{"kind": s.index.Kind()}
	switch ix := s.index.(type) {
	case *kvindex.Approx:
		b, _ := ix.Blocks().Stats()
		idx["blocks"] = b
	case *kvindex.Events:
		b, _ := ix.Exact().Stats()
		idx["blocks"] = b
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"version": s.version, "models": models, "kv_index": idx, "tenants": s.limiter.Stats(),
	})
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
