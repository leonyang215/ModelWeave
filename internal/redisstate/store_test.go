package redisstate_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/leonyang215/ModelWeave/internal/enginestate"
	"github.com/leonyang215/ModelWeave/internal/redisstate"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

// MW_TEST_REDIS_URL runs the same contract tests against a real Redis too.
// Each test uses its own namespace and never flushes a database.
func pair(t *testing.T) (*redisstate.Store, *redisstate.Store, *miniredis.Miniredis) {
	t.Helper()
	url := os.Getenv("MW_TEST_REDIS_URL")
	var mini *miniredis.Miniredis
	if url == "" {
		mini = miniredis.RunT(t)
		url = "redis://" + mini.Addr()
	}
	ns := fmt.Sprintf("%s/%d", t.Name(), time.Now().UnixNano())
	newStore := func() *redisstate.Store {
		s, err := redisstate.New(context.Background(), redisstate.Options{URL: url, Namespace: ns, Timeout: time.Second, LeaseTTL: 15 * time.Second})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = s.Close() })
		return s
	}
	return newStore(), newStore(), mini
}

func register(t *testing.T, r *registry.Registry, id string) *registry.Worker {
	t.Helper()
	w, _, err := r.Upsert(registry.Spec{ID: id, Model: "m", Role: registry.RoleDecode, Target: id + ":9001", Normal: true, KVCapacityTokens: 1000, MaxRunningRequests: 10}, registry.SourceHeartbeat)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

func TestSharedRegistrationStateDrainHealthAndRemoval(t *testing.T) {
	a, b, _ := pair(t)
	ra, rb := registry.NewWithStore(time.Minute, a), registry.NewWithStore(time.Minute, b)
	w := register(t, ra, "d0")
	other := rb.Get("d0")
	if other == nil || other.Source() != registry.SourceHeartbeat {
		t.Fatal("second gateway cannot discover registration")
	}
	if err := w.UpdateState(registry.EngineState{RunningRequests: 2, KVUsedTokens: 200, KVCapacityTokens: 1000}, time.Now()); err != nil {
		t.Fatal(err)
	}
	if s := other.Snapshot(time.Now(), time.Minute); !s.HasState || s.State.RunningRequests != 2 || s.EffectiveKVUsed != 200 {
		t.Fatalf("shared report: %+v", s)
	}
	if err := other.SetDraining(true); err != nil {
		t.Fatal(err)
	}
	beat := &registry.Beat{State: &registry.EngineState{KVUsedTokens: 300}}
	if _, _, err := ra.UpsertContext(context.Background(), w.Spec(), registry.SourceHeartbeat, beat); err != nil {
		t.Fatal(err)
	}
	if s := w.Snapshot(time.Now(), 0); !s.AdminDrained || !s.Draining {
		t.Fatal("heartbeat undid shared operator drain")
	}
	if err := other.SetDraining(false); err != nil {
		t.Fatal(err)
	}
	w.ReportFailure(time.Now(), 2, time.Minute)
	if !other.ReportFailure(time.Now(), 2, time.Minute) || w.Snapshot(time.Now(), 0).Healthy {
		t.Fatal("health counter not shared")
	}
	if _, ok, err := rb.RemoveContext(context.Background(), "d0"); err != nil || !ok {
		t.Fatalf("remove: %v %v", ok, err)
	}
	if ra.Get("d0") != nil || len(ra.Candidates("m", registry.RoleDecode)) != 0 {
		t.Fatal("removal not visible on first gateway")
	}
	newW := register(t, rb, "d0")
	if newW.Ref() == w.Ref() {
		t.Fatal("re-registration reused epoch")
	}
	if err := w.UpdateState(registry.EngineState{RunningRequests: 999}, time.Now()); !errors.Is(err, registry.ErrGone) {
		t.Fatalf("old poll not fenced: %v", err)
	}
	if s := newW.Snapshot(time.Now(), time.Minute); s.HasState || s.Draining || !s.Healthy {
		t.Fatalf("new epoch inherited state: %+v", s)
	}
}

func TestDistributedConditionalReservations(t *testing.T) {
	a, b, _ := pair(t)
	ra, rb := registry.NewWithStore(time.Minute, a), registry.NewWithStore(time.Minute, b)
	register(t, ra, "d0")
	for _, r := range []*registry.Registry{ra, rb} {
		if err := r.Sync(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	sa, sb := scheduler.New(ra, scheduler.Config{}), scheduler.New(rb, scheduler.Config{})
	var mu sync.Mutex
	var got []*scheduler.Decision
	var wg sync.WaitGroup
	for i := 0; i < 80; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			s := sa
			if i%2 == 1 {
				s = sb
			}
			d, err := s.Schedule(context.Background(), &scheduler.Request{Model: "m", Role: registry.RoleDecode, KVReserve: 100})
			if err != nil && !errors.Is(err, scheduler.ErrNoCapacity) {
				t.Errorf("schedule: %v", err)
				return
			}
			if err == nil {
				mu.Lock()
				got = append(got, d)
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	if len(got) != 10 {
		t.Fatalf("admitted %d, want exactly 10 across both gateways", len(got))
	}
	if s := rb.Get("d0").Snapshot(time.Now(), time.Minute); s.Inflight != 10 || s.ReservedKV != 1000 {
		t.Fatalf("global reservations: %+v", s)
	}
	for _, d := range got {
		if err := d.Reservation.Release(true); err != nil {
			t.Fatal(err)
		}
		_ = d.Reservation.Release(true)
	}
	if s := ra.Get("d0").Snapshot(time.Now(), time.Minute); s.Inflight != 0 || s.ReservedKV != 0 {
		t.Fatalf("release leaked/doubled: %+v", s)
	}
}

func TestHeartbeatExpiryUsesServerClockAndStaticDoesNotExpire(t *testing.T) {
	a, b, mini := pair(t)
	if mini == nil {
		t.Skip("clock control uses miniredis; other tests also run on real Redis")
	}
	now := time.Unix(1700000000, 0)
	mini.SetTime(now)
	ra, rb := registry.NewWithStore(10*time.Second, a), registry.NewWithStore(10*time.Second, b)
	register(t, ra, "d0")
	spec := registry.Spec{ID: "static", Model: "m", Role: registry.RoleUnified, Target: "s:9001"}
	if _, _, err := ra.Upsert(spec, registry.SourceStatic); err != nil {
		t.Fatal(err)
	}
	mini.SetTime(now.Add(9 * time.Second))
	if es, err := rb.SweepContext(context.Background()); err != nil || len(es) != 0 {
		t.Fatalf("early expiry: %v %v", es, err)
	}
	mini.SetTime(now.Add(11 * time.Second))
	es, err := rb.SweepContext(context.Background())
	if err != nil || len(es) != 1 || es[0].ID != "d0" {
		t.Fatalf("expiry: %v %v", es, err)
	}
	if es, err := ra.SweepContext(context.Background()); err != nil || len(es) != 0 {
		t.Fatalf("expiry committed twice: %v %v", es, err)
	}
	if ra.Get("d0") != nil || ra.Get("static") == nil {
		t.Fatal("shared expiry/static semantics")
	}
}

func TestCrashedGatewayLeaseReclaimsReservationsAndCannotResurrect(t *testing.T) {
	a, b, mini := pair(t)
	if mini == nil {
		t.Skip("clock control uses miniredis")
	}
	now := time.Now()
	mini.SetTime(now)
	ra, rb := registry.NewWithStore(time.Minute, a), registry.NewWithStore(time.Minute, b)
	w := register(t, ra, "d0")
	res := w.Reserve(900, 77, nil)
	if res == nil {
		t.Fatal("reserve")
	}
	mini.SetTime(now.Add(10 * time.Second))
	if err := b.Maintain(context.Background()); err != nil {
		t.Fatal(err)
	}
	mini.SetTime(now.Add(16 * time.Second))
	if s := rb.Get("d0").Snapshot(time.Now(), time.Minute); s.Inflight != 0 || s.ReservedKV != 0 || s.PrefillTok != 0 {
		t.Fatalf("crashed gateway claims not reclaimed: %+v", s)
	}
	if err := a.Maintain(context.Background()); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("expired owner resurrected: %v", err)
	}
	if _, err := w.TryReserve(context.Background(), 1, 0, registry.Unlimited(), nil); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("expired owner can reserve: %v", err)
	}
	if err := res.Release(false); err != nil {
		t.Fatal(err)
	}
}

func TestFailedReleaseIsReconciledIdempotently(t *testing.T) {
	a, b, mini := pair(t)
	if mini == nil {
		t.Skip("fault injection uses miniredis")
	}
	ra, rb := registry.NewWithStore(time.Minute, a), registry.NewWithStore(time.Minute, b)
	w := register(t, ra, "d0")
	res := w.Reserve(400, 77, nil)
	if res == nil {
		t.Fatal("reserve")
	}
	mini.SetError("ERR injected unavailable")
	if err := res.PrefillDone(); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("partial release failure: %v", err)
	}
	if err := res.Release(true); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("release failure: %v", err)
	}
	mini.SetError("")
	if s := rb.Get("d0").Snapshot(time.Now(), 0); s.Inflight != 1 || s.ReservedKV != 400 {
		t.Fatal("ambiguous release bypassed global capacity")
	}
	if err := a.Maintain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := a.Maintain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if s := rb.Get("d0").Snapshot(time.Now(), 0); s.Inflight != 0 || s.ReservedKV != 0 || s.PrefillTok != 0 {
		t.Fatalf("release reconciliation: %+v", s)
	}
}

func TestReportRevisionFencesSlowPollAndSettingsAreIsolated(t *testing.T) {
	a, b, mini := pair(t)
	ctx := context.Background()
	if err := a.CheckSettings(ctx, time.Minute, 64); err != nil {
		t.Fatal(err)
	}
	if err := b.CheckSettings(ctx, time.Minute, 64); err != nil {
		t.Fatal(err)
	}
	if err := b.CheckSettings(ctx, time.Second, 64); err == nil {
		t.Fatal("incompatible heartbeat TTL accepted")
	}
	if err := b.CheckSettings(ctx, time.Minute, 32); err == nil {
		t.Fatal("incompatible block size accepted")
	}
	ra, rb := registry.NewWithStore(time.Minute, a), registry.NewWithStore(time.Minute, b)
	w := register(t, ra, "d0")
	snap := w.Snapshot(time.Now(), 0)
	if err := rb.Get("d0").UpdateState(registry.EngineState{RunningRequests: 5}, time.Now()); err != nil {
		t.Fatal(err)
	}
	if err := w.UpdateStateIfRevision(ctx, registry.EngineState{RunningRequests: 1}, time.Now(), snap.Revision); !errors.Is(err, registry.ErrConflict) {
		t.Fatalf("slow poll overwrote heartbeat: %v", err)
	}
	if s := w.Snapshot(time.Now(), 0); s.State.RunningRequests != 5 {
		t.Fatalf("newer report lost: %+v", s)
	}
	// This global server operation is confined to this test's own miniredis,
	// never performed against a caller-provided real Redis URL.
	if mini != nil {
		if err := a.Client.ScriptFlush(ctx).Err(); err != nil {
			t.Fatal(err)
		}
		if s := w.Snapshot(time.Now(), 0); !s.HasState || s.State.RunningRequests != 5 {
			t.Fatalf("script cache recovery: %+v", s)
		}
	}
}

func TestReplicasCoordinateEngineStatePolls(t *testing.T) {
	a, b, _ := pair(t)
	ra, rb := registry.NewWithStore(time.Minute, a), registry.NewWithStore(time.Minute, b)
	register(t, ra, "d0")
	var calls atomic.Int64
	state := func(context.Context, registry.Spec) (registry.EngineState, error) {
		calls.Add(1)
		return registry.EngineState{RunningRequests: 3}, nil
	}
	pa, pb := enginestate.NewPoller(ra, enginestate.Options{State: state}), enginestate.NewPoller(rb, enginestate.Options{State: state})
	var wg sync.WaitGroup
	for _, p := range []*enginestate.Poller{pa, pb} {
		wg.Add(1)
		go func() { defer wg.Done(); p.PollAll(context.Background()) }()
	}
	wg.Wait()
	if n := calls.Load(); n != 1 {
		t.Fatalf("polled %d times for one shared instance/window", n)
	}
	if s := rb.Get("d0").Snapshot(time.Now(), time.Minute); !s.HasState || s.State.RunningRequests != 3 {
		t.Fatalf("shared poll report: %+v", s)
	}
}
