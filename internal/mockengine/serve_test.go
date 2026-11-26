package mockengine

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
)

// fakeRegistry records Registry calls. Like the gateway, it answers
// registered for an instance it does not know, and rejects unknown models.
type fakeRegistry struct {
	mwipv1.UnimplementedRegistryServer
	mu           sync.Mutex
	known        map[string]bool
	heartbeats   []*mwipv1.HeartbeatRequest
	deregistered []string
	tokens       []string
}

func startRegistry(t *testing.T) (*fakeRegistry, string) {
	t.Helper()
	f := &fakeRegistry{known: map[string]bool{}}
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	mwipv1.RegisterRegistryServer(gs, f)
	go func() { _ = gs.Serve(lis) }()
	t.Cleanup(gs.Stop)
	return f, lis.Addr().String()
}

func (f *fakeRegistry) token(ctx context.Context) {
	md, _ := metadata.FromIncomingContext(ctx)
	f.tokens = append(f.tokens, md.Get("x-modelweave-token")...)
}

func (f *fakeRegistry) Heartbeat(ctx context.Context, req *mwipv1.HeartbeatRequest) (*mwipv1.HeartbeatResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.token(ctx)
	if req.GetInstance().GetModel() != "m" {
		return nil, status.Errorf(codes.NotFound, "instance %s: the gateway serves no model %q", req.GetInstance().GetId(), req.GetInstance().GetModel())
	}
	f.heartbeats = append(f.heartbeats, req)
	id := req.GetInstance().GetId()
	registered := !f.known[id]
	f.known[id] = true
	return &mwipv1.HeartbeatResponse{Registered: registered, TtlMs: 15000}, nil
}

func (f *fakeRegistry) Deregister(ctx context.Context, req *mwipv1.DeregisterRequest) (*mwipv1.DeregisterResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.token(ctx)
	f.deregistered = append(f.deregistered, req.GetInstanceId())
	removed := f.known[req.GetInstanceId()]
	delete(f.known, req.GetInstanceId())
	return &mwipv1.DeregisterResponse{Removed: removed}, nil
}

func (f *fakeRegistry) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.heartbeats)
}

func (f *fakeRegistry) get(i int) *mwipv1.HeartbeatRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.heartbeats[i]
}

func (f *fakeRegistry) firstToken() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.tokens) == 0 {
		return ""
	}
	return f.tokens[0]
}

func (f *fakeRegistry) left() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.deregistered)
}

func waitUntil(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func control(t *testing.T, base, action string) map[string]any {
	t.Helper()
	resp, err := http.Post(base+"/mock/registration/"+action, "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var m map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&m)
	if resp.StatusCode != 200 {
		t.Fatalf("%s: %d %v", action, resp.StatusCode, m)
	}
	return m
}

// One instance, one model in one role: it registers with heartbeats that
// carry its description, load and KV events; after every (re)registration
// it resends its whole KV cache. It drains, stops heartbeating (the gateway
// would expire it), leaves at once, comes back, and leaves on shutdown.
func TestServerRegistrationLifecycle(t *testing.T) {
	reg, target := startRegistry(t)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	e := New(Config{ID: "p0", Model: "m", Role: "prefill", AdvertiseURL: "http://p0:9000", KVCapacityTokens: 4096,
		MaxRunning: 4, BlockSize: 64, KVEvents: true, Log: log})
	e.kv.insert([]uint64{11, 12})
	srv := NewServer(ServeConfig{Listen: "127.0.0.1:0", GRPCListen: "127.0.0.1:0", Target: "p0:9001", Registry: []string{target},
		ControlToken: "secret", HeartbeatInterval: 20 * time.Millisecond, Log: log}, e)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- srv.Run(ctx) }()
	api := httptest.NewServer(srv.Handler())
	defer api.Close()

	waitUntil(t, "three heartbeats", func() bool { return reg.count() >= 3 })
	first, second := reg.get(0), reg.get(1)
	inst := first.GetInstance()
	if inst.GetId() != "p0" || inst.GetModel() != "m" || inst.GetRole() != mwipv1.Role_ROLE_PREFILL || inst.GetTarget() != "p0:9001" ||
		inst.GetBootstrapAddr() != "http://p0:9000" || inst.GetEngineId() != "p0" || inst.GetKvCapacityTokens() != 4096 ||
		inst.GetMaxRunningRequests() != 4 || inst.GetBlockSize() != 64 || first.GetState().GetKvCapacityTokens() != 4096 || first.GetDraining() {
		t.Fatalf("first heartbeat %v", first)
	}
	if tok := reg.firstToken(); tok != "secret" {
		t.Fatalf("token %q", tok)
	}
	// The first heartbeat carries the events recorded so far; the gateway
	// registered the instance, so the next one resends the whole cache.
	if ev := first.GetKvEvents(); ev.GetCleared() || len(ev.GetStored()) != 2 {
		t.Fatalf("first kv events %v", ev)
	}
	if ev := second.GetKvEvents(); !ev.GetCleared() || len(ev.GetStored()) != 2 {
		t.Fatalf("resync after registration %v", ev)
	}
	if ev := reg.get(2).GetKvEvents(); ev != nil {
		t.Fatalf("nothing new to report: %v", ev)
	}

	if m := control(t, api.URL, "drain"); m["draining"] != true || !reg.get(reg.count()-1).GetDraining() {
		t.Fatalf("drain: %v", m)
	}

	// Paused: no heartbeat goes out.
	if m := control(t, api.URL, "pause"); m["state"] != "paused" {
		t.Fatalf("pause: %v", m)
	}
	n := reg.count()
	time.Sleep(120 * time.Millisecond)
	if reg.count() != n {
		t.Fatalf("heartbeats while paused: %d -> %d", n, reg.count())
	}

	// Leave: deregistered at once, still no heartbeats.
	if m := control(t, api.URL, "leave"); m["state"] != "left" || m["removed"] != float64(1) || !slices.Equal(reg.left(), []string{"p0"}) {
		t.Fatalf("leave: %v %v", m, reg.left())
	}
	time.Sleep(60 * time.Millisecond)
	if reg.count() != n {
		t.Fatal("heartbeats after leaving")
	}

	// Resume: registered again at once, then the whole cache follows.
	e.kv.insert([]uint64{13})
	if m := control(t, api.URL, "resume"); m["state"] != "heartbeating" {
		t.Fatalf("resume: %v", m)
	}
	if hb := reg.get(n); slices.Contains(hb.GetKvEvents().GetStored(), 11) || !slices.Equal(hb.GetKvEvents().GetStored(), []uint64{13}) {
		t.Fatalf("first heartbeat after resuming %v", hb.GetKvEvents())
	}
	waitUntil(t, "the resync after re-registration", func() bool { return reg.count() > n+1 })
	if ev := reg.get(n + 1).GetKvEvents(); !ev.GetCleared() || len(ev.GetStored()) != 3 {
		t.Fatalf("resync after re-registration %v", ev)
	}
	if control(t, api.URL, "undrain"); reg.get(reg.count() - 1).GetDraining() {
		t.Fatal("still draining")
	}
	resp, err := http.Post(api.URL+"/mock/registration/explode", "application/json", nil)
	if err != nil || resp.StatusCode != 404 {
		t.Fatalf("unknown action: %v %v", resp, err)
	}
	resp.Body.Close()

	// Shutdown deregisters.
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if got := reg.left(); !slices.Equal(got, []string{"p0", "p0"}) {
		t.Fatalf("deregistered %v", got)
	}
}

// A rejected heartbeat is reported by the registration status with the
// gateway's reason.
func TestServerReportsRejections(t *testing.T) {
	_, target := startRegistry(t)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	e := New(Config{ID: "x0", Model: "nope", Role: "unified", Log: log})
	srv := NewServer(ServeConfig{Listen: "127.0.0.1:0", GRPCListen: "127.0.0.1:0", Target: "x0:9001", Registry: []string{target},
		HeartbeatInterval: 20 * time.Millisecond, Log: log}, e)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = srv.Run(ctx) }()
	waitUntil(t, "the rejection", func() bool {
		regs, _ := srv.registration()["registries"].([]map[string]any)
		return len(regs) == 1 && strings.Contains(regs[0]["error"].(string), `serves no model "nope"`) && regs[0]["registered"] == false
	})
}
