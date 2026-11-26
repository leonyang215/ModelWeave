package mockengine

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// ServeConfig configures how an instance is served and registered.
type ServeConfig struct {
	// Listen serves the instance's HTTP endpoints: KV hand-off, artifact
	// checks, metrics, counters and registration controls.
	Listen string
	// GRPCListen serves the mwip Engine service.
	GRPCListen string
	// Target is the mwip target registered for the instance.
	Target string
	// Registry lists the gateways' mwip Registry targets; empty: the
	// instance is configured statically in the gateways.
	Registry          []string
	ControlToken      string // x-modelweave-token of Registry calls
	HeartbeatInterval time.Duration
	Log               *slog.Logger
}

// Registration states of an instance.
const (
	stateHeartbeating = "heartbeating" // registered, a heartbeat every interval
	statePaused       = "paused"       // heartbeats stopped: the gateways expire it after their TTL
	stateLeft         = "left"         // deregistered, no heartbeats
)

// Server runs one instance: its HTTP endpoints, the mwip Engine service,
// and its registration with the gateways' mwip Registry. Tests steer the
// registration through POST /mock/registration/{action}.
type Server struct {
	cfg ServeConfig
	e   *Engine
	log *slog.Logger

	hb       sync.Mutex // one heartbeat round at a time
	mu       sync.Mutex
	state    string
	draining bool
	clients  []*registryClient // set by Run
}

type registryClient struct {
	target string
	rc     mwipv1.RegistryClient
	// resync: the next heartbeat carries every resident KV block (cleared,
	// then stored), because the gateway's index of the instance is empty or
	// missed events (it has just registered it, or a heartbeat failed).
	resync     bool
	registered bool   // the last heartbeat was accepted
	lastErr    string // the last heartbeat error, logged once
}

// NewServer returns the server of an instance.
func NewServer(cfg ServeConfig, e *Engine) *Server {
	if cfg.HeartbeatInterval <= 0 {
		cfg.HeartbeatInterval = time.Second
	}
	if cfg.Log == nil {
		cfg.Log = slog.Default()
	}
	return &Server{cfg: cfg, e: e, log: cfg.Log.With("worker", e.cfg.ID), state: stateHeartbeating}
}

// Handler serves the instance's HTTP endpoints and the registration
// controls:
//
//	GET  /mock/registration           the instance's registration
//	POST /mock/registration/pause     stop heartbeating (the gateways expire it)
//	POST /mock/registration/leave     deregister now and stop heartbeating
//	POST /mock/registration/resume    heartbeat again (registers it again)
//	POST /mock/registration/drain     report draining in heartbeats
//	POST /mock/registration/undrain   stop reporting draining
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/", s.e.Handler())
	mux.HandleFunc("GET /mock/registration", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, s.registration()) })
	mux.HandleFunc("POST /mock/registration/{action}", s.handleAction)
	return mux
}

func (s *Server) registration() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	regs := make([]map[string]any, 0, len(s.clients))
	for _, c := range s.clients {
		regs = append(regs, map[string]any{"registry": c.target, "registered": c.registered, "error": c.lastErr})
	}
	return map[string]any{"instance": s.e.cfg.ID, "model": s.e.cfg.Model, "role": s.e.cfg.Role, "target": s.cfg.Target,
		"state": s.state, "draining": s.draining, "registries": regs}
}

func (s *Server) handleAction(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	removed := 0
	s.mu.Lock()
	switch r.PathValue("action") {
	case "pause":
		s.state = statePaused
	case "leave":
		s.state = stateLeft
	case "resume":
		s.state = stateHeartbeating
	case "drain":
		s.draining = true
	case "undrain":
		s.draining = false
	default:
		s.mu.Unlock()
		writeErr(w, 404, mwip.ErrBadRequest, "unknown action "+r.PathValue("action"), false)
		return
	}
	state := s.state
	s.mu.Unlock()
	if state == stateLeft {
		removed = s.deregister(ctx)
	} else {
		// Applied at once rather than at the next tick.
		s.heartbeat(ctx)
	}
	out := s.registration()
	out["removed"] = removed
	writeJSON(w, 200, out)
}

// Run serves until ctx ends, then deregisters the instance and stops.
func (s *Server) Run(ctx context.Context) error {
	srv := &http.Server{Addr: s.cfg.Listen, Handler: s.Handler(), ReadHeaderTimeout: 10 * time.Second}
	lis, err := net.Listen("tcp", s.cfg.GRPCListen)
	if err != nil {
		return err
	}
	gsrv := NewGRPCServer(s.e)
	errc := make(chan error, 2)
	go func() { errc <- srv.ListenAndServe() }()
	go func() { errc <- gsrv.Serve(lis) }()
	s.log.Info("mock engine listening", "http", s.cfg.Listen, "grpc", lis.Addr().String(), "target", s.cfg.Target,
		"model", s.e.cfg.Model, "role", s.e.cfg.Role, "registry", s.cfg.Registry)
	var clients []*registryClient
	for _, t := range s.cfg.Registry {
		addr, creds := engine.DialTarget(t)
		cc, err := grpc.NewClient(addr, grpc.WithTransportCredentials(creds))
		if err != nil {
			return err
		}
		defer cc.Close()
		clients = append(clients, &registryClient{target: t, rc: mwipv1.NewRegistryClient(cc)})
	}
	s.mu.Lock()
	s.clients = clients
	s.mu.Unlock()
	if len(clients) > 0 {
		go s.heartbeatLoop(ctx)
	}
	select {
	case err := <-errc:
		gsrv.Stop()
		return err
	case <-ctx.Done():
	}
	// Leave the gateways first, so that they stop routing here, then stop.
	dctx, dcancel := context.WithTimeout(context.Background(), 2*time.Second)
	s.mu.Lock()
	s.state = stateLeft
	s.mu.Unlock()
	s.deregister(dctx)
	dcancel()
	sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stopped := make(chan struct{})
	go func() { gsrv.GracefulStop(); close(stopped) }()
	err = srv.Shutdown(sctx)
	select {
	case <-stopped:
	case <-sctx.Done():
		gsrv.Stop()
	}
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func (s *Server) outgoing(ctx context.Context) context.Context {
	if s.cfg.ControlToken == "" {
		return ctx
	}
	return metadata.AppendToOutgoingContext(ctx, "x-modelweave-token", s.cfg.ControlToken)
}

func (s *Server) heartbeatLoop(ctx context.Context) {
	t := time.NewTicker(s.cfg.HeartbeatInterval)
	defer t.Stop()
	for {
		s.heartbeat(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// heartbeat registers or refreshes the instance with every gateway: its
// load, the KV events since the last round (or a full resync) and whether
// it drains.
func (s *Server) heartbeat(ctx context.Context) {
	s.hb.Lock()
	defer s.hb.Unlock()
	s.mu.Lock()
	if s.state != stateHeartbeating {
		s.mu.Unlock()
		return
	}
	draining, clients := s.draining, s.clients
	s.mu.Unlock()
	inst := s.e.instance(s.cfg.Target)
	state := registry.StateToProto(s.e.State())
	events := mwip.KVEventsToProto(s.e.kv.drainEvents())
	for _, c := range clients {
		req := &mwipv1.HeartbeatRequest{Instance: inst, State: state, KvEvents: events, Draining: draining}
		s.mu.Lock()
		resync := c.resync
		s.mu.Unlock()
		if resync {
			req.KvEvents = mwip.KVEventsToProto(s.e.kv.snapshot())
		}
		cctx, cancel := context.WithTimeout(s.outgoing(ctx), 2*time.Second)
		resp, err := c.rc.Heartbeat(cctx, req)
		cancel()
		s.mu.Lock()
		if err != nil {
			msg := status.Convert(err).Message()
			if msg != c.lastErr {
				s.log.Warn("heartbeat failed", "registry", c.target, "code", status.Code(err).String(), "error", msg)
			}
			c.registered, c.lastErr, c.resync = false, msg, true
		} else {
			if resp.GetRegistered() {
				s.log.Info("instance registered", "registry", c.target, "ttl_ms", resp.GetTtlMs())
			}
			c.registered, c.lastErr, c.resync = true, "", resp.GetRegistered()
		}
		s.mu.Unlock()
	}
}

// deregister removes the instance from every gateway; it returns how many
// had it registered.
func (s *Server) deregister(ctx context.Context) int {
	s.hb.Lock()
	defer s.hb.Unlock()
	s.mu.Lock()
	clients := s.clients
	s.mu.Unlock()
	removed := 0
	for _, c := range clients {
		resp, err := c.rc.Deregister(s.outgoing(ctx), &mwipv1.DeregisterRequest{InstanceId: s.e.cfg.ID})
		if err != nil {
			s.log.Warn("deregister failed", "registry", c.target, "error", err)
			continue
		}
		if resp.GetRemoved() {
			removed++
		}
		s.mu.Lock()
		c.registered = false
		s.mu.Unlock()
	}
	s.log.Info("instance deregistered", "removed", removed)
	return removed
}

// instance describes the instance in heartbeats.
func (e *Engine) instance(target string) *mwipv1.Instance {
	inst := &mwipv1.Instance{
		Id: e.cfg.ID, Model: e.cfg.Model, Role: mwip.RoleToProto(e.cfg.Role), Target: target,
		Normal: e.cfg.Normal, KvCapacityTokens: e.cfg.KVCapacityTokens, MaxRunningRequests: int32(e.cfg.MaxRunning),
		MaxPrefillQueueTokens: e.cfg.MaxPrefillTokens, BlockSize: int32(e.cfg.BlockSize),
	}
	if e.cfg.Role == "prefill" {
		inst.BootstrapAddr, inst.EngineId = e.cfg.AdvertiseURL, e.cfg.ID
	}
	return inst
}
