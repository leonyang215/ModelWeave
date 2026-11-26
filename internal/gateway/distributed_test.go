package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/qos"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

func gatewayPair(t *testing.T) (*Server, *Server, *miniredis.Miniredis) {
	return gatewayPairWithConfig(t, nil)
}

func gatewayPairWithConfig(t *testing.T, mut func(*config.Config)) (*Server, *Server, *miniredis.Miniredis) {
	t.Helper()
	url := os.Getenv("MW_TEST_REDIS_URL")
	var mini *miniredis.Miniredis
	if url == "" {
		mini = miniredis.RunT(t)
		url = "redis://" + mini.Addr()
	}
	ns := fmt.Sprintf("%s/%d", t.Name(), time.Now().UnixNano())
	tok := fakeTokenizer(t)
	newServer := func() *Server {
		cfg, err := config.Parse([]byte(fmt.Sprintf(`
server: {expose_routing_headers: true}
tokenizer: {endpoints: [%q]}
state:
  backend: redis
  redis: {url: %q, namespace: %q, timeout: 200ms, lease_ttl: 15s}
kv_index: {type: events, optimistic_ttl: 1ms}
models:
  - name: m
    aliases: [m-alias]
    mode: normal
    normal: {session_affinity: true}
    routing:
      unified:
        policy: weighted
        scorers:
          - {name: prefix-cache, weight: 2}
          - {name: load, weight: 1}
`, tok, url, ns)))
		if err != nil {
			t.Fatal(err)
		}
		if mut != nil {
			mut(cfg)
		}
		s, err := New(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), "test")
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		return s
	}
	return newServer(), newServer(), mini
}

func registerGatewayWorker(t *testing.T, s *Server) {
	t.Helper()
	_, target := mockWorker(t)
	rc := mwipv1.NewRegistryClient(registryClient(t, s))
	hb := instance("w0", "m", mwipv1.Role_ROLE_UNIFIED, target)
	hb.State = &mwipv1.EngineState{KvCapacityTokens: 100000, MaxRunningRequests: 1}
	if _, err := rc.Heartbeat(context.Background(), hb); err != nil {
		t.Fatal(err)
	}
}

func awaitGateway(t *testing.T, what string, fn func() bool) {
	t.Helper()
	until := time.Now().Add(5 * time.Second)
	for !fn() {
		if time.Now().After(until) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestRedisGatewayTenantLimitsAreGlobal(t *testing.T) {
	for _, limit := range []string{"rpm", "tpm", "concurrency"} {
		t.Run(limit, func(t *testing.T) {
			a, b, _ := gatewayPairWithConfig(t, func(c *config.Config) {
				tenant := config.Tenant{Name: c.Auth.AnonymousTenant, Priority: "normal", Weight: 1}
				switch limit {
				case "rpm":
					tenant.RPM = 2
				case "tpm":
					tenant.TPM = 4
				case "concurrency":
					tenant.MaxConcurrency = 1
				}
				c.Tenants = []config.Tenant{tenant}
			})
			registerGatewayWorker(t, a)
			body := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
			if limit == "concurrency" {
				lease, err := a.limiter.AdmitContext(context.Background(), "anonymous", 1)
				if err != nil {
					t.Fatal(err)
				}
				defer lease.Finish(1)
			} else {
				if w := request(a, http.MethodPost, "/v1/chat/completions", body); w.Code != 200 {
					t.Fatalf("first inference: %d %s", w.Code, w.Body)
				}
				if limit == "rpm" {
					if w := request(b, http.MethodPost, "/v1/chat/completions", body); w.Code != 200 {
						t.Fatalf("second inference: %d %s", w.Code, w.Body)
					}
				}
			}
			for _, s := range []*Server{a, b} {
				w := request(s, http.MethodPost, "/v1/chat/completions", body)
				if w.Code != 429 || w.Header().Get("Retry-After") == "" {
					t.Fatalf("%s not global: %d %s", limit, w.Code, w.Body)
				}
			}
		})
	}
}

func TestRedisGatewayAdmissionQueueIsSharedAndCancelsCleanly(t *testing.T) {
	a, b, _ := gatewayPairWithConfig(t, func(c *config.Config) {
		c.Models[0].Admission.QueueSize = 2
		c.Models[0].Admission.MaxWait = config.Duration(5 * time.Second)
		c.Tenants = []config.Tenant{{Name: "anonymous", Priority: "normal", Weight: 1}}
	})
	registerGatewayWorker(t, a)
	blocked, err := a.models["m"].rt.Unified.Schedule(context.Background(), &scheduler.Request{Model: "m", Role: registry.RoleUnified, KVReserve: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer blocked.Reservation.Release(false)
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	ctxA, cancelA := context.WithCancel(context.Background())
	defer cancelA()
	ctxB, cancelB := context.WithCancel(context.Background())
	defer cancelB()
	start := func(s *Server, ctx context.Context) <-chan *httptest.ResponseRecorder {
		done := make(chan *httptest.ResponseRecorder, 1)
		go func() {
			w := httptest.NewRecorder()
			s.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)).WithContext(ctx))
			done <- w
		}()
		return done
	}
	doneA := start(a, ctxA)
	awaitGateway(t, "A queued globally", func() bool {
		n, err := b.models["m"].queue.LenContext(context.Background())
		return err == nil && n == 1
	})
	doneB := start(b, ctxB)
	awaitGateway(t, "both requests queued globally", func() bool {
		n, err := a.models["m"].queue.LenContext(context.Background())
		return err == nil && n == 2
	})
	full := request(a, http.MethodPost, "/v1/chat/completions", body)
	if full.Code != 503 || !strings.Contains(full.Body.String(), "overloaded") {
		t.Fatalf("queue bound multiplied by replicas: %d %s", full.Code, full.Body)
	}
	cancelA()
	<-doneA
	awaitGateway(t, "cancellation shared", func() bool {
		n, err := b.models["m"].queue.LenContext(context.Background())
		return err == nil && n == 1
	})
	stats, err := b.limiter.StatsContext(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(stats) != 1 || stats[0].Inflight != 1 {
		t.Fatalf("waiting/canceled tenant concurrency: %+v", stats)
	}
	if err := blocked.Reservation.Release(true); err != nil {
		t.Fatal(err)
	}
	select {
	case w := <-doneB:
		if w.Code != 200 {
			t.Fatalf("B did not wake after A released capacity: %d %s", w.Code, w.Body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("B remained queued after remote release")
	}
	statsReply := request(a, http.MethodGet, "/admin/stats", "")
	var admin struct {
		QoSBackend string            `json:"qos_backend"`
		Tenants    []qos.TenantStats `json:"tenants"`
	}
	if statsReply.Code != 200 || json.Unmarshal(statsReply.Body.Bytes(), &admin) != nil || admin.QoSBackend != "redis" || len(admin.Tenants) != 1 || admin.Tenants[0].Inflight != 0 {
		t.Fatalf("admin global QoS: %d %s", statsReply.Code, statsReply.Body)
	}
}

func request(s *Server, method, path, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest(method, path, strings.NewReader(body)))
	return w
}

func workerViews(t *testing.T, s *Server) []workerView {
	t.Helper()
	w := request(s, http.MethodGet, "/admin/workers", "")
	var out struct {
		Workers []workerView `json:"workers"`
	}
	if w.Code != 200 {
		t.Fatalf("workers: %d %s", w.Code, w.Body)
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	return out.Workers
}

// Two independently constructed gateways, gRPC control planes, HTTP APIs,
// connection pools and local membership views use one Redis namespace. The
// instance heartbeats to A only; B still serves inference and sees its KV.
func TestRedisGatewaysShareLifecycleSchedulingAndCaches(t *testing.T) {
	a, b, _ := gatewayPair(t)
	_, target := mockWorker(t)
	ca, cb := mwipv1.NewRegistryClient(registryClient(t, a)), mwipv1.NewRegistryClient(registryClient(t, b))
	ctx := context.Background()
	hb := instance("w0", "m-alias", mwipv1.Role_ROLE_UNIFIED, target)
	hb.State = &mwipv1.EngineState{KvCapacityTokens: 10000, MaxRunningRequests: 1}
	hb.KvEvents = &mwipv1.KVEvents{Stored: []uint64{1, 1<<53 + 1, ^uint64(0)}}
	if res, err := ca.Heartbeat(ctx, hb); err != nil || !res.GetRegistered() {
		t.Fatalf("register on A: %v %v", res, err)
	}
	wA := a.reg.Get("w0")
	if ws := workerViews(t, b); len(ws) != 1 || ws[0].ID != "w0" || ws[0].CachedBlocks != 3 || ws[0].State == nil {
		t.Fatalf("B's shared view: %+v", ws)
	}
	q := kvindex.Query{Blocks: hb.KvEvents.Stored, Candidates: []string{"w0"}, WorkerKeys: map[string]string{"w0": wA.CacheKey()}}
	if m, err := b.index.Match(ctx, q); err != nil || m["w0"] != 3 {
		t.Fatalf("B's prefix view: %v %v", m, err)
	}
	resp := request(b, http.MethodPost, "/v1/chat/completions", `{"model":"m","prompt_cache_key":"session-a","messages":[{"role":"user","content":"hi"}]}`)
	if resp.Code != 200 || resp.Header().Get("X-ModelWeave-Decode-Worker") != "w0" || !strings.Contains(resp.Body.String(), "Hello from ModelWeave mock worker w0") {
		t.Fatalf("inference on B: %d %s", resp.Code, resp.Body)
	}
	if cached, ok := a.models["m"].rt.Sessions.Get("session-a"); !ok || cached != wA.CacheKey() {
		t.Fatalf("session not shared back to A: %q %v", cached, ok)
	}
	req := &scheduler.Request{Model: "m", Role: registry.RoleUnified, KVReserve: 256, PrefillCost: func(int) int64 { return 77 }}
	d, err := b.models["m"].rt.Unified.Schedule(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = d.Reservation.Release(false) })
	if _, err := a.models["m"].rt.Unified.Schedule(ctx, req); !errors.Is(err, scheduler.ErrNoCapacity) {
		t.Fatalf("A ignored B's reservation: %v", err)
	}
	if ws := workerViews(t, a); len(ws) != 1 || ws[0].Inflight != 1 || ws[0].ReservedKV != 256 || ws[0].InflightPrefill != 77 {
		t.Fatalf("global accounting: %+v", ws)
	}
	if err := d.Reservation.PrefillDone(); err != nil {
		t.Fatal(err)
	}
	_ = d.Reservation.PrefillDone()
	if ws := workerViews(t, a); ws[0].InflightPrefill != 0 || ws[0].Inflight != 1 {
		t.Fatalf("partial release: %+v", ws)
	}
	if w := request(a, http.MethodPost, "/admin/workers/w0/drain", ""); w.Code != 200 {
		t.Fatalf("drain: %d %s", w.Code, w.Body)
	}
	if res, err := cb.Heartbeat(ctx, hb); err != nil || res.GetRegistered() {
		t.Fatalf("heartbeat to B registered twice: %v %v", res, err)
	}
	if ws := workerViews(t, b); !ws[0].Draining || !ws[0].AdminDrained {
		t.Fatal("heartbeat on B undid drain on A")
	}
	_ = d.Reservation.Release(true)
	if _, err := b.models["m"].rt.Unified.Schedule(ctx, req); !errors.Is(err, scheduler.ErrNoWorkers) {
		t.Fatalf("B scheduled drained instance: %v", err)
	}
	if w := request(b, http.MethodPost, "/admin/workers/w0/drain?undo=1", ""); w.Code != 200 {
		t.Fatalf("undrain: %d %s", w.Code, w.Body)
	}
	if _, err := cb.PublishKVEvents(ctx, &mwipv1.PublishKVEventsRequest{InstanceId: "w0", Events: &mwipv1.KVEvents{Removed: []uint64{1}}}); err != nil {
		t.Fatal(err)
	}
	if m, err := a.index.Match(ctx, q); err != nil || m["w0"] != 0 {
		t.Fatalf("KV eviction on B not seen by A: %v %v", m, err)
	}
	if res, err := cb.Deregister(ctx, &mwipv1.DeregisterRequest{InstanceId: "w0"}); err != nil || !res.GetRemoved() {
		t.Fatalf("leave on B: %v %v", res, err)
	}
	if ws := workerViews(t, a); len(ws) != 0 {
		t.Fatal("A still sees removed instance")
	}
	if _, ok := a.models["m"].rt.Sessions.Get("session-a"); ok {
		t.Fatal("removed registration kept shared sessions")
	}
	hb.KvEvents = nil
	if res, err := cb.Heartbeat(ctx, hb); err != nil || !res.GetRegistered() {
		t.Fatalf("re-register: %v %v", res, err)
	}
	wB := b.reg.Get("w0")
	q.WorkerKeys["w0"] = wB.CacheKey()
	if wA.Ref() == wB.Ref() {
		t.Fatal("epoch reused")
	}
	if m, err := a.index.Match(ctx, q); err != nil || m["w0"] != 0 {
		t.Fatalf("stale prefix inherited by new registration: %v %v", m, err)
	}
	// Shutting down a replica must not deregister engines used by siblings.
	a.Close()
	d2, err := b.models["m"].rt.Unified.Schedule(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	_ = d2.Reservation.Release(true)
	stats := request(b, http.MethodGet, "/admin/stats", "")
	if stats.Code != 200 || !strings.Contains(stats.Body.String(), `"state_backend": "redis"`) || !strings.Contains(stats.Body.String(), `"prefix-cache"`) {
		t.Fatalf("stats: %d %s", stats.Code, stats.Body)
	}
}

func TestRedisOutageFailsClosedAndRecovers(t *testing.T) {
	a, b, mini := gatewayPair(t)
	if mini == nil {
		t.Skip("fault injection uses miniredis")
	}
	ca := mwipv1.NewRegistryClient(registryClient(t, a))
	hb := instance("w0", "m", mwipv1.Role_ROLE_UNIFIED, "host:9001")
	if _, err := ca.Heartbeat(context.Background(), hb); err != nil {
		t.Fatal(err)
	}
	mini.SetError("ERR injected unavailable")
	for _, c := range []struct{ method, path, body string }{
		{http.MethodGet, "/admin/workers", ""},
		{http.MethodGet, "/readyz", ""},
		{http.MethodPost, "/admin/workers/w0/drain", ""},
		{http.MethodPost, "/v1/chat/completions", `{"model":"m","messages":[]}`},
	} {
		if w := request(b, c.method, c.path, c.body); w.Code != 503 {
			t.Errorf("outage %s: %d %s", c.path, w.Code, w.Body)
		}
	}
	if _, err := ca.Heartbeat(context.Background(), hb); status.Code(err) != codes.Unavailable {
		t.Fatalf("heartbeat acknowledged while unavailable: %v", err)
	}
	if _, err := ca.Deregister(context.Background(), &mwipv1.DeregisterRequest{InstanceId: "w0"}); status.Code(err) != codes.Unavailable {
		t.Fatalf("deregister acknowledged while unavailable: %v", err)
	}
	mini.SetError("")
	if w := request(b, http.MethodGet, "/readyz", ""); w.Code != 200 {
		t.Fatalf("recovery readiness: %d %s", w.Code, w.Body)
	}
	if ws := workerViews(t, b); len(ws) != 1 || ws[0].Draining {
		t.Fatalf("failed mutations changed state: %+v", ws)
	}
}
