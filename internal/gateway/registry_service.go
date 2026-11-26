package gateway

import (
	"context"
	"crypto/subtle"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// TokenMetadata is the metadata key of the control token on Registry calls.
const TokenMetadata = "x-modelweave-token"

// registryService serves the mwip Registry: engine instances register,
// report their load and KV events, and leave.
type registryService struct {
	mwipv1.UnimplementedRegistryServer
	s *Server
}

// newGRPCServer builds the gateway's gRPC server: the mwip Registry and the
// standard health service, guarded by the control token.
func (s *Server) newGRPCServer() (*grpc.Server, *health.Server) {
	srv := grpc.NewServer(
		grpc.UnaryInterceptor(tokenInterceptor(s.cfg.Server.ControlToken)),
		grpc.MaxRecvMsgSize(engine.MaxMessageBytes),
	)
	mwipv1.RegisterRegistryServer(srv, &registryService{s: s})
	hs := health.NewServer()
	healthpb.RegisterHealthServer(srv, hs)
	return srv, hs
}

func tokenInterceptor(token string) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, h grpc.UnaryHandler) (any, error) {
		if token != "" && info.FullMethod != healthpb.Health_Check_FullMethodName {
			var got string
			if md, ok := metadata.FromIncomingContext(ctx); ok {
				if v := md.Get(TokenMetadata); len(v) > 0 {
					got = v[0]
				}
			}
			if subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
				return nil, status.Error(codes.Unauthenticated, "missing or invalid "+TokenMetadata+" metadata")
			}
		}
		return h(ctx, req)
	}
}

// Heartbeat registers or refreshes an instance: one model in one role, at
// its mwip target. It records the instance's load, KV events and whether
// it drains. Invalid instances are rejected with the reason.
func (r *registryService) Heartbeat(ctx context.Context, req *mwipv1.HeartbeatRequest) (*mwipv1.HeartbeatResponse, error) {
	s := r.s
	inst := req.GetInstance()
	reject := func(code codes.Code, format string, args ...any) error {
		reason := fmt.Sprintf(format, args...)
		s.met.Heartbeats.WithLabelValues("rejected").Inc()
		s.log.Debug("heartbeat rejected", "instance", inst.GetId(), "model", inst.GetModel(), "reason", reason)
		return status.Error(code, reason)
	}
	id := inst.GetId()
	if id == "" {
		return nil, reject(codes.InvalidArgument, "instance.id is required")
	}
	m := s.byName[inst.GetModel()]
	if m == nil {
		return nil, reject(codes.NotFound, "instance %s: the gateway serves no model %q", id, inst.GetModel())
	}
	if inst.GetRole() == mwipv1.Role_ROLE_UNSPECIFIED {
		return nil, reject(codes.InvalidArgument, "instance %s: role is required", id)
	}
	role, err := registry.ParseRole(mwip.RoleFromProto(inst.GetRole()))
	if err != nil {
		return nil, reject(codes.InvalidArgument, "instance %s: %v", id, err)
	}
	if inst.GetTarget() == "" {
		return nil, reject(codes.InvalidArgument, "instance %s: target is required (where it serves the Engine service)", id)
	}
	spec := registry.Spec{
		ID: id, Model: m.cfg.Name, Role: role, Target: inst.GetTarget(), Normal: inst.GetNormal(),
		BootstrapAddr: inst.GetBootstrapAddr(), EngineID: inst.GetEngineId(), KVCapacityTokens: inst.GetKvCapacityTokens(),
		MaxRunningRequests: int(inst.GetMaxRunningRequests()), MaxPrefillTokens: inst.GetMaxPrefillQueueTokens(),
		Labels: inst.GetLabels(),
	}
	beat := &registry.Beat{Draining: req.GetDraining()}
	if req.GetState() != nil {
		st := registry.StateFromProto(req.GetState())
		beat.State = &st
	}
	if err := spec.Validate(); err != nil {
		return nil, reject(codes.InvalidArgument, "instance %s: %v", id, err)
	}
	if beat.State != nil {
		if err := beat.State.Validate(); err != nil {
			return nil, reject(codes.InvalidArgument, "instance %s: %v", id, err)
		}
	}
	wk, created, err := s.reg.UpsertContext(ctx, spec, registry.SourceHeartbeat, beat)
	if err != nil {
		return nil, reject(codes.Unavailable, "instance %s: %v", id, err)
	}
	if ev := mwip.KVEventsFromProto(req.GetKvEvents()); ev != nil {
		if err := s.applyKVEvents(wk, ev); err != nil {
			return nil, reject(codes.Unavailable, "instance %s: KV events: %v", id, err)
		}
	}
	if created {
		s.met.Heartbeats.WithLabelValues("registered").Inc()
	} else {
		s.met.Heartbeats.WithLabelValues("updated").Inc()
	}
	m.queue.Notify()
	return &mwipv1.HeartbeatResponse{Registered: created, TtlMs: s.reg.TTL().Milliseconds()}, nil
}

// PublishKVEvents applies KV events of a registered instance.
func (r *registryService) PublishKVEvents(ctx context.Context, req *mwipv1.PublishKVEventsRequest) (*mwipv1.PublishKVEventsResponse, error) {
	id := req.GetInstanceId()
	if err := r.s.reg.Sync(ctx); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	w := r.s.reg.Get(id)
	if w == nil {
		return nil, status.Errorf(codes.NotFound, "instance %q is not registered", id)
	}
	ev := mwip.KVEventsFromProto(req.GetEvents())
	if ev == nil || r.s.events == nil {
		return &mwipv1.PublishKVEventsResponse{}, nil
	}
	if err := r.s.applyKVEvents(w, ev); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &mwipv1.PublishKVEventsResponse{Applied: true}, nil
}

// Deregister removes an instance at once (graceful shutdown).
func (r *registryService) Deregister(ctx context.Context, req *mwipv1.DeregisterRequest) (*mwipv1.DeregisterResponse, error) {
	spec, ok, err := r.s.reg.RemoveContext(ctx, req.GetInstanceId())
	if err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	if ok {
		r.s.met.InstanceRemovals.WithLabelValues(spec.Model, string(spec.Role), "deregistered").Inc()
	}
	return &mwipv1.DeregisterResponse{Removed: ok}, nil
}
