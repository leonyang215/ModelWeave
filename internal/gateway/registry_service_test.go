package gateway

import (
	"context"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// registryClient serves the gateway's gRPC server and dials it.
func registryClient(t *testing.T, s *Server) *grpc.ClientConn {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs, _ := s.newGRPCServer()
	go func() { _ = gs.Serve(lis) }()
	t.Cleanup(gs.Stop)
	cc, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cc.Close() })
	return cc
}

func instance(id, model string, role mwipv1.Role, target string) *mwipv1.HeartbeatRequest {
	return &mwipv1.HeartbeatRequest{Instance: &mwipv1.Instance{Id: id, Model: model, Role: role, Target: target}}
}

// Every heartbeat describes one instance: one model in one role at its mwip
// target. It is registered under the canonical model name with its load
// and drain state; invalid instances are rejected with the reason.
func TestRegistryHeartbeatRegistersAnInstance(t *testing.T) {
	cfg, err := config.Parse([]byte(`
tokenizer: {endpoints: ["http://127.0.0.1:1"]}
server: {control_token: secret}
kv_index: {type: events}
models: [{name: m, aliases: [m-alias]}, {name: n}]
`))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "test")
	if err != nil {
		t.Fatal(err)
	}
	cc := registryClient(t, s)
	rc := mwipv1.NewRegistryClient(cc)
	ctx := context.Background()

	d0 := instance("d0", "m-alias", mwipv1.Role_ROLE_DECODE, "host-a:9001")
	d0.Instance.Normal = true
	d0.State = &mwipv1.EngineState{RunningRequests: 2, KvCapacityTokens: 1000, KvUsedTokens: 100}
	if _, err := rc.Heartbeat(ctx, d0); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("heartbeat without the token: %v", err)
	}
	auth := metadata.AppendToOutgoingContext(ctx, TokenMetadata, "secret")
	resp, err := rc.Heartbeat(auth, d0)
	if err != nil || !resp.GetRegistered() || resp.GetTtlMs() <= 0 {
		t.Fatalf("d0: %v %v", resp, err)
	}
	w := s.reg.Get("d0")
	if w == nil || w.Spec().Model != "m" || w.Spec().Target != "host-a:9001" || !w.Spec().ServesNormal() || w.Source != "heartbeat" {
		t.Fatalf("d0 not registered under the canonical model with its target: %+v", w)
	}
	if snap := w.Snapshot(time.Now(), 0); !snap.HasState || snap.State.RunningRequests != 2 || snap.Draining {
		t.Fatalf("d0 state %+v", snap)
	}
	// Instances of other roles and models register side by side; the
	// scheduler sees each under its model and role.
	for _, hb := range []*mwipv1.HeartbeatRequest{
		instance("p0", "m", mwipv1.Role_ROLE_PREFILL, "host-b:9001"),
		instance("e0", "m", mwipv1.Role_ROLE_ENCODE, "host-c:9001"),
		instance("n0", "n", mwipv1.Role_ROLE_UNIFIED, "host-d:9001"),
	} {
		if _, err := rc.Heartbeat(auth, hb); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct {
		model string
		role  registry.Role
		ids   string
	}{{"m", registry.RoleDecode, "d0"}, {"m", registry.RolePrefill, "p0"}, {"m", registry.RoleEncode, "e0"}, {"n", registry.RoleUnified, "n0"}, {"n", registry.RoleDecode, ""}} {
		var ids []string
		for _, w := range s.reg.Candidates(c.model, c.role) {
			ids = append(ids, w.ID())
		}
		if strings.Join(ids, ",") != c.ids {
			t.Fatalf("%s/%s: %v, want %s", c.model, c.role, ids, c.ids)
		}
	}

	// The next heartbeat refreshes: not newly registered. Draining is the
	// engine's report of each heartbeat.
	d0.Draining = true
	if resp, err := rc.Heartbeat(auth, d0); err != nil || resp.GetRegistered() || !w.Snapshot(time.Now(), 0).Draining {
		t.Fatalf("refresh: %v %v", resp, err)
	}
	d0.Draining = false
	if _, err := rc.Heartbeat(auth, d0); err != nil || w.Snapshot(time.Now(), 0).Draining {
		t.Fatalf("undrain: %v", err)
	}

	for _, c := range []struct {
		hb   *mwipv1.HeartbeatRequest
		code codes.Code
		want string
	}{
		{&mwipv1.HeartbeatRequest{}, codes.InvalidArgument, "instance.id is required"},
		{instance("x", "nope", mwipv1.Role_ROLE_DECODE, "host:9001"), codes.NotFound, `serves no model "nope"`},
		{instance("x", "m", mwipv1.Role_ROLE_UNSPECIFIED, "host:9001"), codes.InvalidArgument, "role is required"},
		{instance("x", "m", mwipv1.Role_ROLE_DECODE, ""), codes.InvalidArgument, "target is required"},
	} {
		_, err := rc.Heartbeat(auth, c.hb)
		if status.Code(err) != c.code || !strings.Contains(status.Convert(err).Message(), c.want) {
			t.Fatalf("%v: %v, want %s %q", c.hb, err, c.code, c.want)
		}
	}
	if s.reg.Get("x") != nil {
		t.Fatal("a rejected instance was registered")
	}

	// KV events of registered instances reach the event index.
	kv, err := rc.PublishKVEvents(auth, &mwipv1.PublishKVEventsRequest{InstanceId: "d0", Events: &mwipv1.KVEvents{Stored: []uint64{1, 2, 3}}})
	if err != nil || !kv.GetApplied() {
		t.Fatalf("kv events %v %v", kv, err)
	}
	if _, err := rc.PublishKVEvents(auth, &mwipv1.PublishKVEventsRequest{InstanceId: "unknown", Events: &mwipv1.KVEvents{Stored: []uint64{4}}}); status.Code(err) != codes.NotFound {
		t.Fatalf("kv events of an unknown instance: %v", err)
	}
	if blocks, per := s.index.(*kvindex.Events).Exact().Stats(); blocks != 3 || per["d0"] != 3 {
		t.Fatalf("event index %d %v", blocks, per)
	}

	// Deregister removes one instance at once; the others stay.
	dr, err := rc.Deregister(auth, &mwipv1.DeregisterRequest{InstanceId: "d0"})
	if err != nil || !dr.GetRemoved() || s.reg.Get("d0") != nil || s.reg.Get("p0") == nil {
		t.Fatalf("deregister %v %v", dr, err)
	}
	if blocks, _ := s.index.(*kvindex.Events).Exact().Stats(); blocks != 0 {
		t.Fatalf("the index still holds the blocks of a deregistered instance: %d", blocks)
	}
	if dr, err := rc.Deregister(auth, &mwipv1.DeregisterRequest{InstanceId: "d0"}); err != nil || dr.GetRemoved() {
		t.Fatalf("deregister twice %v %v", dr, err)
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if want := `modelweave_instance_removals_total{model="m",reason="deregistered",role="decode"} 1`; !strings.Contains(rec.Body.String(), want) {
		t.Fatalf("metrics lack %s", want)
	}

	// The standard health service answers without the token; the API does
	// not need it either.
	hc, err := healthpb.NewHealthClient(cc).Check(ctx, &healthpb.HealthCheckRequest{})
	if err != nil || hc.GetStatus() != healthpb.HealthCheckResponse_SERVING {
		t.Fatalf("health %v %v", hc, err)
	}
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "m-alias") {
		t.Fatalf("models: %d %s", rec.Code, rec.Body)
	}
}

// An operator drain (admin API) holds across the engine's heartbeats, which
// report it not draining, until it is undone.
func TestAdminDrainOutlivesHeartbeats(t *testing.T) {
	s := newTestServer(t, "")
	rc := mwipv1.NewRegistryClient(registryClient(t, s))
	hb := instance("u0", "m", mwipv1.Role_ROLE_UNIFIED, "host:9001")
	if _, err := rc.Heartbeat(context.Background(), hb); err != nil {
		t.Fatal(err)
	}
	admin := func(path string) {
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: %d %s", path, rec.Code, rec.Body)
		}
	}
	admin("/admin/workers/u0/drain")
	if _, err := rc.Heartbeat(context.Background(), hb); err != nil {
		t.Fatal(err)
	}
	w := s.reg.Get("u0")
	if !w.Snapshot(time.Now(), 0).Draining || !w.AdminDrained() {
		t.Fatal("a heartbeat undid the operator's drain")
	}
	admin("/admin/workers/u0/drain?undo=1")
	if w.Snapshot(time.Now(), 0).Draining {
		t.Fatal("still draining after undo")
	}
}
