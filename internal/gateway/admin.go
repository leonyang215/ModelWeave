package gateway

import (
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

func (s *Server) adminToken() string {
	if s.cfg.Server.AdminToken != "" {
		return s.cfg.Server.AdminToken
	}
	return s.cfg.Server.ControlToken
}

func (s *Server) mountAdmin(mux *http.ServeMux) {
	mux.HandleFunc("GET /admin/models", s.control(s.handleAdminModels))
	mux.HandleFunc("GET /admin/models/{name}", s.control(s.handleAdminModels))
	mux.HandleFunc("GET /admin/workers", s.control(s.handleWorkers))
	mux.HandleFunc("GET /admin/workers/{id}", s.control(s.handleWorker))
	mux.HandleFunc("POST /admin/workers/{id}/drain", s.control(s.handleDrain))
	mux.HandleFunc("POST /admin/workers/{id}/resume", s.control(s.handleResume))
	mux.HandleFunc("GET /admin/stats", s.control(s.handleStats))
	mux.HandleFunc("GET /admin/blacklist", s.control(s.handleBlacklist))
	mux.HandleFunc("PUT /admin/blacklist/{id}", s.control(s.handleBlock))
	mux.HandleFunc("DELETE /admin/blacklist/{id}", s.control(s.handleUnblock))
	if !s.cfg.Server.DisableAdminUI {
		mux.HandleFunc("GET /admin", s.handleAdminUI)
		mux.HandleFunc("GET /admin/{$}", s.handleAdminUI)
		mux.HandleFunc("GET /admin/assets/{name}", s.handleAdminUI)
	}
}

type instanceCounts struct {
	Total     int `json:"total"`
	Available int `json:"available"` // health/admin gates only, not request-specific capacity
	Blocked   int `json:"blocked"`
	Draining  int `json:"draining"`
	Unhealthy int `json:"unhealthy"`
}

func (c *instanceCounts) add(w workerView) {
	c.Total++
	switch w.status() {
	case "active":
		c.Available++
	case "blocked":
		c.Blocked++
	case "draining":
		c.Draining++
	case "unhealthy":
		c.Unhealthy++
	}
}

type policyView struct {
	Name    string                   `json:"name"`
	Scorers []scheduler.ScorerConfig `json:"scorers"`
}

type adminModelView struct {
	Name           string                            `json:"name"`
	Aliases        []string                          `json:"aliases"`
	TokenizerModel string                            `json:"tokenizer_model"`
	EngineModel    string                            `json:"engine_model,omitempty"`
	ContextLength  int                               `json:"context_length"`
	Mode           string                            `json:"mode"`
	Handshake      string                            `json:"handshake"`
	EncodePolicy   string                            `json:"encode_policy"`
	Instances      instanceCounts                    `json:"instances"`
	Roles          map[registry.Role]*instanceCounts `json:"roles"`
	Policies       map[registry.Role]policyView      `json:"policies"`
	QueueLength    int                               `json:"queue_len"`
	QueueCapacity  int                               `json:"queue_capacity"`
	MaxWaitMS      int64                             `json:"max_wait_ms"`
	OverloadLevel  int                               `json:"overload_level"`
	Utilization    float64                           `json:"utilization"`
}

func (s *Server) handleAdminModels(w http.ResponseWriter, r *http.Request) {
	selected := r.PathValue("name")
	if selected != "" {
		m := s.byName[selected]
		if m == nil {
			controlError(w, http.StatusNotFound, "unknown model")
			return
		}
		selected = m.cfg.Name
	}
	workers, err := s.workerViews(r.Context())
	if err != nil {
		controlError(w, http.StatusServiceUnavailable, "admin state unavailable")
		return
	}
	out := make([]adminModelView, 0, len(s.models))
	for name, m := range s.models {
		if selected != "" && name != selected {
			continue
		}
		n, err := m.queue.LenContext(r.Context())
		if err != nil {
			controlError(w, http.StatusServiceUnavailable, "QoS state store unavailable")
			return
		}
		level, util := m.overload.Level()
		v := adminModelView{
			Name: name, Aliases: append([]string{}, m.cfg.Aliases...), TokenizerModel: m.cfg.TokenizerModel,
			EngineModel: m.cfg.EngineModel, ContextLength: m.cfg.ContextLength,
			Mode: m.cfg.Mode, Handshake: m.cfg.PD.Handshake, EncodePolicy: m.cfg.Encode.Policy,
			Roles: make(map[registry.Role]*instanceCounts), Policies: make(map[registry.Role]policyView),
			QueueLength: n, QueueCapacity: m.cfg.Admission.QueueSize, MaxWaitMS: m.cfg.Admission.MaxWait.D().Milliseconds(),
			OverloadLevel: int(level), Utilization: util,
		}
		for role, sched := range map[registry.Role]*scheduler.Scheduler{
			registry.RolePrefill: m.rt.Prefill, registry.RoleDecode: m.rt.Decode,
			registry.RoleEncode: m.rt.Encode, registry.RoleUnified: m.rt.Unified,
		} {
			v.Roles[role] = &instanceCounts{}
			v.Policies[role] = policyView{Name: sched.Policy(), Scorers: append([]scheduler.ScorerConfig{}, sched.Scorers()...)}
		}
		for _, worker := range workers {
			if worker.Model == name {
				v.Instances.add(worker)
				v.Roles[worker.Role].add(worker)
			}
		}
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	if selected != "" {
		writeJSON(w, http.StatusOK, out[0])
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": out})
}

func (s *Server) handleWorker(w http.ResponseWriter, r *http.Request) {
	workers, err := s.workerViews(r.Context())
	if err != nil {
		controlError(w, http.StatusServiceUnavailable, "admin state unavailable")
		return
	}
	for _, worker := range workers {
		if worker.ID == r.PathValue("id") {
			writeJSON(w, http.StatusOK, worker)
			return
		}
	}
	controlError(w, http.StatusNotFound, "unknown worker")
}

func (s *Server) handleBlacklist(w http.ResponseWriter, r *http.Request) {
	blocks, err := s.reg.Store().Blocks(r.Context())
	if err != nil {
		controlError(w, http.StatusServiceUnavailable, "state store unavailable")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"blacklist": blocks})
}

func validBlockID(w http.ResponseWriter, r *http.Request) bool {
	id := r.PathValue("id")
	if strings.TrimSpace(id) == "" || len(id) > 256 {
		controlError(w, http.StatusBadRequest, "instance id must contain 1 to 256 bytes")
		return false
	}
	return true
}

func (s *Server) handleBlock(w http.ResponseWriter, r *http.Request) {
	if !validBlockID(w, r) {
		return
	}
	media, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if media != "application/json" {
		controlError(w, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 8<<10)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	var body struct {
		Reason string `json:"reason"`
	}
	if err := dec.Decode(&body); err != nil {
		adminBodyError(w, err)
		return
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		adminBodyError(w, err)
		return
	}
	body.Reason = strings.TrimSpace(body.Reason)
	if n := utf8.RuneCountInString(body.Reason); n == 0 || n > 512 {
		controlError(w, http.StatusBadRequest, "reason must contain 1 to 512 characters")
		return
	}
	b, err := s.reg.Store().Block(r.Context(), r.PathValue("id"), body.Reason, time.Now())
	if err != nil {
		controlError(w, http.StatusServiceUnavailable, "state store unavailable")
		return
	}
	s.notifyAdminChange()
	s.log.Info("admin action", "action", "block", "worker", b.ID, "reason", b.Reason, "remote_addr", r.RemoteAddr)
	writeJSON(w, http.StatusOK, b)
}

func adminBodyError(w http.ResponseWriter, err error) {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		controlError(w, http.StatusRequestEntityTooLarge, "admin request body exceeds 8 KiB")
		return
	}
	controlError(w, http.StatusBadRequest, "expected one JSON object containing only a reason string")
}

func (s *Server) handleUnblock(w http.ResponseWriter, r *http.Request) {
	if !validBlockID(w, r) {
		return
	}
	id := r.PathValue("id")
	if err := s.reg.Store().Unblock(r.Context(), id); err != nil {
		controlError(w, http.StatusServiceUnavailable, "state store unavailable")
		return
	}
	s.notifyAdminChange()
	s.log.Info("admin action", "action", "unblock", "worker", id, "remote_addr", r.RemoteAddr)
	writeJSON(w, http.StatusOK, map[string]any{"worker": id, "blocked": false})
}

func (s *Server) notifyAdminChange() {
	for _, m := range s.models {
		m.queue.Notify()
	}
}
