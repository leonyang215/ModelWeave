// Package gateway is the HTTP front of ModelWeave: OpenAI / Anthropic APIs,
// the MWIP control plane for workers, admin endpoints and metrics.
package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/enginestate"
	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/obs"
	"github.com/leonyang215/ModelWeave/internal/orchestrator"
	"github.com/leonyang215/ModelWeave/internal/qos"
	"github.com/leonyang215/ModelWeave/internal/redisstate"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
	"github.com/leonyang215/ModelWeave/internal/tokenizer"
)

type modelRT struct {
	cfg      *config.Model
	rt       *orchestrator.Model
	queue    qos.AdmissionQueue
	overload *qos.Overload
}

// Server is the gateway.
type Server struct {
	cfg        *config.Config
	log        *slog.Logger
	met        *obs.Metrics
	reg        *registry.Registry
	tok        *tokenizer.Client
	eng        *engine.Client
	orch       *orchestrator.Orchestrator
	limiter    qos.TenantLimiter
	qosBackend qos.Backend
	keys       map[string]string // api key -> tenant
	tenants    map[string]config.Tenant
	index      kvindex.Index
	events     kvindex.EventSink
	models     map[string]*modelRT // canonical name
	byName     map[string]*modelRT // name or alias
	ready      atomic.Bool
	version    string
	startedAt  time.Time
	wg         sync.WaitGroup
	closeOnce  sync.Once
}

// Default absolute load-imbalance thresholds per role, in the role's load
// unit (queued prefill tokens, or running requests).
var defaultBalanceAbs = map[registry.Role]float64{
	registry.RolePrefill: 32768,
	registry.RoleDecode:  32,
	registry.RoleUnified: 32,
	registry.RoleEncode:  8,
}

// New builds a gateway from configuration.
func New(cfg *config.Config, log *slog.Logger, version string) (*Server, error) {
	var store registry.Store = registry.NewMemoryStore()
	var shared *redisstate.Store
	if cfg.State.Backend == "redis" {
		var err error
		shared, err = redisstate.New(context.Background(), redisstate.Options{
			URL: cfg.State.Redis.URL, Namespace: cfg.State.Redis.Namespace,
			Timeout: cfg.State.Redis.Timeout.D(), LeaseTTL: cfg.State.Redis.LeaseTTL.D(),
		})
		if err != nil {
			return nil, err
		}
		store = shared
	}
	met := obs.NewMetrics()
	met.BuildInfo.WithLabelValues(version).Set(1)
	s := &Server{
		cfg:       cfg,
		log:       log,
		met:       met,
		reg:       registry.NewWithStore(cfg.Discovery.HeartbeatTTL.D(), store),
		tok:       tokenizer.New(cfg.Tokenizer.Endpoints, cfg.Tokenizer.Timeout.D()),
		eng:       engine.New(),
		keys:      map[string]string{},
		tenants:   map[string]config.Tenant{},
		models:    map[string]*modelRT{},
		byName:    map[string]*modelRT{},
		version:   version,
		startedAt: time.Now(),
	}
	constructed := false
	defer func() {
		if !constructed {
			s.Close()
		}
	}()
	if shared != nil {
		if err := shared.CheckSettings(context.Background(), cfg.Discovery.HeartbeatTTL.D(), cfg.KVIndex.BlockSize); err != nil {
			return nil, err
		}
	}
	s.orch = orchestrator.New(s.reg, s.eng, met, log, cfg.Health, cfg.Server)
	s.tok.OnUnreachable = func(endpoint string, err error) {
		met.TokenizerDown.WithLabelValues(endpoint).Inc()
		log.Warn("tokenizer replica unreachable, trying the next one", "endpoint", endpoint, "error", err)
	}

	var policies []qos.TenantPolicy
	for _, t := range cfg.Tenants {
		p, _ := qos.ParsePriority(t.Priority)
		policies = append(policies, qos.TenantPolicy{Name: t.Name, Priority: p, Weight: t.Weight, RPM: t.RPM, TPM: t.TPM, MaxConcurrency: t.MaxConcurrency})
		s.tenants[t.Name] = t
		for _, k := range t.APIKeys {
			s.keys[k] = t.Name
		}
	}
	s.qosBackend = qos.NewMemoryBackend(policies)
	if shared != nil {
		var err error
		s.qosBackend, err = qos.NewRedisBackend(context.Background(), shared, policies)
		if err != nil {
			return nil, err
		}
	}
	s.limiter = s.qosBackend.Limiter()

	ki := cfg.KVIndex
	capacity := ki.CapacityBlocks
	approx := func() kvindex.Index {
		if shared != nil {
			return kvindex.NewRedis(shared.Client, shared.Key("prefix:"), kvindex.RedisOptions{Type: "approx", TTL: ki.TTL.D(), CapacityBlocks: capacity, Timeout: cfg.State.Redis.Timeout.D(), OnError: func(err error) { log.Warn("shared prefix index mutation failed", "error", err) }})
		}
		return kvindex.NewApprox(ki.TTL.D(), capacity)
	}
	switch ki.Type {
	case "approx":
		s.index = approx()
	case "events":
		if shared != nil {
			ev := kvindex.NewRedis(shared.Client, shared.Key("prefix:"), kvindex.RedisOptions{Type: "events", OptimisticTTL: ki.OptimisticTTL.D(), Timeout: cfg.State.Redis.Timeout.D(), OnError: func(err error) { log.Warn("shared KV index mutation failed", "error", err) }})
			s.index, s.events = ev, ev
		} else {
			ev := kvindex.NewEvents(ki.OptimisticTTL.D())
			s.index, s.events = ev, ev
		}
	case "remote":
		s.index = &kvindex.Remote{
			URL: ki.RemoteURL, Timeout: ki.RemoteTimeout.D(),
			Client:   &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 64, IdleConnTimeout: 90 * time.Second}},
			Fallback: approx(),
			OnResult: func(outcome string, d time.Duration) {
				met.CacheLookup.WithLabelValues("remote_"+outcome, outcome).Observe(d.Seconds())
			},
		}
	default:
		s.index = kvindex.None{}
	}

	for i := range cfg.Models {
		mc := &cfg.Models[i]
		build := func(role registry.Role, pc scheduler.PolicyConfig) (*scheduler.Scheduler, error) {
			pol, err := scheduler.BuildPolicy(pc, defaultBalanceAbs[role])
			if err != nil {
				return nil, err
			}
			return scheduler.New(s.reg, scheduler.Config{
				Policy: pol, Index: s.index, StaleAfter: cfg.EngineState.StaleAfter.D(),
				Observer:         &schedObserver{met: met, model: mc.Name},
				RegistrationKeys: true,
			}), nil
		}
		rt := &orchestrator.Model{
			Cfg:       mc,
			Name:      mc.Name,
			BlockSize: ki.BlockSize,
			Seed:      kvindex.ModelSeed(mc.Name),
			Artifacts: orchestrator.NewArtifactCache(mc.Encode.ArtifactCacheTTL.D(), mc.Encode.ArtifactCacheMax),
			Sessions:  orchestrator.NewSessionMap(mc.Normal.SessionTTL.D()),
		}
		if shared != nil {
			onError := func(err error) { log.Warn("shared advisory cache unavailable", "model", mc.Name, "error", err) }
			rt.Sessions = orchestrator.NewRedisSessionMap(shared, mc.Name, mc.Normal.SessionTTL.D(), onError)
			rt.Artifacts = orchestrator.NewRedisArtifactCache(shared, mc.Name, mc.Encode.ArtifactCacheTTL.D(), mc.Encode.ArtifactCacheMax, onError)
		}
		log.Info("model configured", "model", mc.Name, "mode", mc.Mode, "pd_handshake", mc.PD.Handshake)
		met.ModelInfo.WithLabelValues(mc.Name, mc.Mode, mc.PD.Handshake).Set(1)
		var err error
		if rt.Prefill, err = build(registry.RolePrefill, mc.Routing.Policy(registry.RolePrefill)); err != nil {
			return nil, err
		}
		if rt.Decode, err = build(registry.RoleDecode, mc.Routing.Policy(registry.RoleDecode)); err != nil {
			return nil, err
		}
		if rt.Encode, err = build(registry.RoleEncode, mc.Routing.Policy(registry.RoleEncode)); err != nil {
			return nil, err
		}
		if rt.Unified, err = build(registry.RoleUnified, mc.Routing.Policy(registry.RoleUnified)); err != nil {
			return nil, err
		}
		queue, err := s.qosBackend.NewQueue(context.Background(), mc.Name, qos.QueueConfig{
			MaxSize: mc.Admission.QueueSize, MaxWait: mc.Admission.MaxWait.D(), BackfillDepth: mc.Admission.BackfillDepth,
		}, &queueObserver{met: met, model: mc.Name})
		if err != nil {
			return nil, err
		}
		m := &modelRT{
			cfg:      mc,
			rt:       rt,
			queue:    queue,
			overload: qos.NewOverload(mc.Admission.Overload.Thresholds, mc.Admission.Overload.Hysteresis, mc.Admission.Overload.CacheExemptRatio),
		}
		s.models[mc.Name] = m
		s.byName[mc.Name] = m
		for _, a := range mc.Aliases {
			s.byName[a] = m
		}
	}

	s.reg.OnChange(func(ev registry.Event) {
		spec := ev.Worker.Spec()
		forget := func(w *registry.Worker) {
			s.index.ForgetWorker(w.CacheKey())
			if m := s.models[w.Spec().Model]; m != nil {
				m.rt.Sessions.ForgetWorker(w.CacheKey())
				m.queue.Notify()
			}
		}
		if ev.Previous != nil && ev.Previous.Ref() != ev.Worker.Ref() {
			forget(ev.Previous)
		}
		switch ev.Kind {
		case "removed":
			forget(ev.Worker)
			s.log.Info("worker removed", "worker", spec.ID, "model", spec.Model, "role", spec.Role)
		case "added", "updated":
			if spec.KVCapacityTokens > 0 && ki.BlockSize > 0 && capacity == 0 {
				s.index.SetWorkerCapacity(ev.Worker.CacheKey(), int(spec.KVCapacityTokens)/ki.BlockSize)
			}
			s.log.Info("worker "+ev.Kind, "worker", spec.ID, "model", spec.Model, "role", spec.Role, "target", spec.Target, "source", ev.Worker.Source())
		}
		if m := s.models[spec.Model]; m != nil {
			m.queue.Notify()
		}
	})
	if err := s.reg.Sync(context.Background()); err != nil {
		return nil, err
	}
	for _, w := range cfg.Discovery.Workers {
		w.Model = s.byName[w.Model].cfg.Name
		if _, _, err := s.reg.Upsert(w, "static"); err != nil {
			return nil, err
		}
	}
	met.Registry.MustRegister(&workerCollector{reg: s.reg, staleAfter: cfg.EngineState.StaleAfter.D()})
	if s.adminToken() == "" {
		log.Warn("admin API has no authentication token; restrict network access or set server.admin_token")
	}
	s.ready.Store(true)
	constructed = true
	return s, nil
}

// Close releases resources of a stopped server (also useful with Handler in
// embedded deployments/tests). Run calls it after draining and stopping its
// background tasks. Closing one Redis gateway never clears shared state.
func (s *Server) Close() {
	s.closeOnce.Do(func() {
		s.ready.Store(false)
		for _, m := range s.models {
			m.queue.Close()
		}
		if s.qosBackend != nil {
			s.qosBackend.Close()
		}
		if s.eng != nil {
			s.eng.Close()
		}
		if s.reg != nil {
			_ = s.reg.Store().Close()
		}
	})
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleAPI("chat_completions"))
	mux.HandleFunc("POST /v1/responses", s.handleAPI("responses"))
	mux.HandleFunc("POST /v1/messages", s.handleAPI("messages"))
	mux.HandleFunc("GET /v1/models", s.handleModels)
	s.mountAdmin(mux)
	mux.Handle("GET /metrics", promhttp.HandlerFor(s.met.Registry, promhttp.HandlerOpts{}))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) })
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		if !s.ready.Load() {
			http.Error(w, "draining", http.StatusServiceUnavailable)
			return
		}
		if s.reg.Store().Shared() {
			if err := s.reg.Store().Maintain(r.Context()); err != nil {
				http.Error(w, "state store unavailable", http.StatusServiceUnavailable)
				return
			}
			if err := s.qosBackend.Maintain(r.Context()); err != nil {
				http.Error(w, "QoS state store unavailable", http.StatusServiceUnavailable)
				return
			}
		}
		_, _ = w.Write([]byte("ready"))
	})
	return mux
}

// Run serves the API (HTTP) and the mwip Registry (gRPC) until ctx is
// canceled, then drains.
func (s *Server) Run(ctx context.Context) error {
	bg, cancel := context.WithCancel(context.Background())
	defer func() { cancel(); s.wg.Wait(); s.Close() }()
	s.startBackground(bg)
	srv := &http.Server{
		Addr:              s.cfg.Server.Listen,
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	lis, err := net.Listen("tcp", s.cfg.Server.GRPCListen)
	if err != nil {
		return fmt.Errorf("mwip registry listener: %w", err)
	}
	gsrv, hs := s.newGRPCServer()
	errc := make(chan error, 2)
	go func() { errc <- srv.ListenAndServe() }()
	go func() { errc <- gsrv.Serve(lis) }()
	s.log.Info("modelweave gateway listening", "addr", s.cfg.Server.Listen, "mwip_registry", lis.Addr().String(), "version", s.version,
		"models", len(s.models), "static_workers", len(s.cfg.Discovery.Workers), "kv_index", s.index.Kind())
	select {
	case err := <-errc:
		gsrv.Stop()
		_ = srv.Close()
		return err
	case <-ctx.Done():
	}
	s.ready.Store(false)
	hs.Shutdown()
	s.log.Info("shutting down, draining requests", "grace", s.cfg.Server.ShutdownGrace.D().String())
	sctx, scancel := context.WithTimeout(context.Background(), s.cfg.Server.ShutdownGrace.D())
	defer scancel()
	stopped := make(chan struct{})
	go func() { gsrv.GracefulStop(); close(stopped) }()
	err = srv.Shutdown(sctx)
	select {
	case <-stopped:
	case <-sctx.Done():
		gsrv.Stop()
	}
	cancel()
	for _, m := range s.models {
		m.queue.Close()
	}
	s.wg.Wait()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func (s *Server) startBackground(ctx context.Context) {
	every := func(d time.Duration, fn func()) {
		s.wg.Add(1)
		go func() {
			defer s.wg.Done()
			t := time.NewTicker(d)
			defer t.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-t.C:
					fn()
				}
			}
		}()
	}
	every(time.Second, func() {
		specs, err := s.reg.SweepContext(ctx)
		if err != nil {
			s.log.Warn("registry expiry failed", "error", err)
			return
		}
		for _, spec := range specs {
			s.met.InstanceRemovals.WithLabelValues(spec.Model, string(spec.Role), "expired").Inc()
			s.log.Warn("instance expired: no heartbeat", "worker", spec.ID, "model", spec.Model, "role", spec.Role, "ttl", s.reg.TTL().String())
		}
	})
	if s.reg.Store().Shared() {
		every(s.cfg.State.SyncInterval.D(), func() {
			if err := s.reg.Sync(ctx); err != nil {
				s.log.Warn("registry sync failed", "error", err)
			}
		})
		every(min(time.Second, s.cfg.State.Redis.LeaseTTL.D()/3), func() {
			if err := s.reg.Store().Maintain(ctx); err != nil {
				s.log.Error("shared state maintenance failed", "error", err)
				return
			}
			if err := s.qosBackend.Maintain(ctx); err != nil {
				s.log.Error("QoS state maintenance failed", "error", err)
			}
		})
	}
	every(30*time.Second, func() {
		switch ix := s.index.(type) {
		case *kvindex.Approx:
			ix.Blocks().Sweep()
		case *kvindex.Events:
			ix.Sweep()
		}
	})
	every(time.Second, s.updateOverload)
	if p := s.cfg.EngineState.Poll; p.Enabled {
		poller := enginestate.NewPoller(s.reg, enginestate.Options{
			Timeout: p.Timeout.D(), Interval: p.Interval.D(), State: s.eng.State,
			OnResult: func(outcome string) { s.met.StatePolls.WithLabelValues(outcome).Inc() },
		})
		every(p.Interval.D(), func() { poller.PollAll(ctx) })
	}
}

// updateOverload computes pool utilization per model and feeds the
// overload controllers.
func (s *Server) updateOverload() {
	now := time.Now()
	stale := s.cfg.EngineState.StaleAfter.D()
	for name, m := range s.models {
		util := 0.0
		for _, role := range []registry.Role{registry.RoleDecode, registry.RoleUnified, registry.RolePrefill} {
			ws := s.reg.Candidates(name, role)
			if len(ws) == 0 {
				continue
			}
			var used, capTot, load float64
			n := 0
			for _, w := range ws {
				snap := w.Snapshot(now, stale)
				if !snap.Healthy || snap.Draining || snap.Blocked {
					continue
				}
				n++
				switch {
				case role != registry.RolePrefill && snap.KVCapacity > 0:
					used += float64(snap.EffectiveKVUsed)
					capTot += float64(snap.KVCapacity)
				case role == registry.RolePrefill && snap.MaxPrefillTokens > 0:
					used += float64(snap.QueuedPrefillTokens())
					capTot += float64(snap.MaxPrefillTokens)
				case snap.MaxRunning > 0:
					load += float64(snap.Running()) / float64(snap.MaxRunning)
				}
			}
			u := 0.0
			if capTot > 0 {
				u = used / capTot
			} else if n > 0 {
				u = load / float64(n)
			}
			s.met.Utilization.WithLabelValues(name, string(role)).Set(u)
			util = max(util, u)
		}
		if qmax := m.cfg.Admission.QueueSize; qmax > 0 {
			if size, err := m.queue.LenContext(context.Background()); err == nil {
				util = max(util, float64(size)/float64(qmax))
			}
		}
		level := m.overload.Update(util)
		s.met.OverloadLevel.WithLabelValues(name).Set(float64(level))
	}
}
