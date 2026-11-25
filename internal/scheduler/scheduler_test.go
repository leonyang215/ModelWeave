package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

func newReg(t *testing.T, role registry.Role, n int, mut func(*registry.Spec)) *registry.Registry {
	t.Helper()
	reg := registry.New(time.Minute)
	for i := 0; i < n; i++ {
		s := registry.Spec{ID: fmt.Sprintf("w%d", i), Model: "m", Role: role, URL: fmt.Sprintf("http://w%d", i)}
		if mut != nil {
			mut(&s)
		}
		if _, _, err := reg.Upsert(s, "static"); err != nil {
			t.Fatal(err)
		}
	}
	return reg
}

func TestCacheAwarePrefersCachedWorker(t *testing.T) {
	reg := newReg(t, registry.RolePrefill, 3, nil)
	idx := kvindex.NewApprox(0, 0)
	pol, _ := BuildPolicy(PolicyConfig{Name: "cache_aware"}, 1e9)
	s := New(reg, Config{Policy: pol, Index: idx})
	blocks := []uint64{1, 2, 3, 4}
	idx.Observe("w2", blocks[:3])
	d, err := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RolePrefill, PromptTokens: 256, Blocks: blocks, BlockSize: 64, PrefillCost: func(c int) int64 { return int64(256 - c) }})
	if err != nil {
		t.Fatal(err)
	}
	if d.Worker.ID() != "w2" || d.Reason != "cache_hit" || d.CachedTokens != 192 {
		t.Fatalf("got %s reason=%s cached=%d", d.Worker.ID(), d.Reason, d.CachedTokens)
	}
	snap := d.Worker.Snapshot(time.Now(), time.Minute)
	if snap.PrefillTok != 64 {
		t.Fatalf("only uncached tokens should be accounted, got %d", snap.PrefillTok)
	}
	d.Reservation.Release(true)
	if s := d.Worker.Snapshot(time.Now(), time.Minute); s.PrefillTok != 0 || s.Inflight != 0 {
		t.Fatalf("reservation leaked: %+v", s)
	}
}

func TestCacheAwareMissSeedsAffinity(t *testing.T) {
	reg := newReg(t, registry.RoleDecode, 4, nil)
	idx := kvindex.NewApprox(0, 0)
	pol, _ := BuildPolicy(PolicyConfig{Name: "cache_aware"}, 1e9)
	s := New(reg, Config{Policy: pol, Index: idx})
	blocks := []uint64{9, 8, 7}
	first, err := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RoleDecode, PromptTokens: 200, Blocks: blocks, BlockSize: 64})
	if err != nil || first.Reason != "least_load" {
		t.Fatalf("first: %v %v", first, err)
	}
	first.Reservation.Release(true)
	for i := 0; i < 5; i++ {
		d, err := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RoleDecode, PromptTokens: 200, Blocks: blocks, BlockSize: 64})
		if err != nil {
			t.Fatal(err)
		}
		if d.Worker != first.Worker || d.Reason != "cache_hit" {
			t.Fatalf("request %d went to %s (%s), want %s", i, d.Worker.ID(), d.Reason, first.Worker.ID())
		}
		d.Reservation.Release(true)
	}
}

func TestCacheAwareRebalancesUnderImbalance(t *testing.T) {
	reg := newReg(t, registry.RoleDecode, 2, nil)
	idx := kvindex.NewApprox(0, 0)
	pol, _ := BuildPolicy(PolicyConfig{Name: "cache_aware", BalanceAbs: 4, BalanceRel: 1.5}, 0)
	s := New(reg, Config{Policy: pol, Index: idx})
	blocks := []uint64{5}
	idx.Observe("w0", blocks)
	// Load w0 with 10 running requests as reported by its engine.
	reg.Get("w0").UpdateState(registry.EngineState{RunningRequests: 10}, time.Now())
	d, err := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RoleDecode, PromptTokens: 64, Blocks: blocks, BlockSize: 64})
	if err != nil {
		t.Fatal(err)
	}
	if d.Worker.ID() != "w1" || d.Reason != "rebalance" {
		t.Fatalf("got %s/%s, want w1/rebalance", d.Worker.ID(), d.Reason)
	}
}

func TestCapacityFiltersAndErrors(t *testing.T) {
	reg := newReg(t, registry.RoleDecode, 2, func(s *registry.Spec) { s.KVCapacityTokens = 1000 })
	s := New(reg, Config{Policy: LeastLoad{}})
	ctx := context.Background()
	var held []*Decision
	for i := 0; i < 2; i++ {
		d, err := s.Schedule(ctx, &Request{Model: "m", Role: registry.RoleDecode, KVReserve: 800})
		if err != nil {
			t.Fatal(err)
		}
		held = append(held, d)
	}
	if held[0].Worker == held[1].Worker {
		t.Fatal("both reservations landed on one worker")
	}
	if _, err := s.Schedule(ctx, &Request{Model: "m", Role: registry.RoleDecode, KVReserve: 800}); !errors.Is(err, ErrNoCapacity) {
		t.Fatalf("want ErrNoCapacity, got %v", err)
	}
	// Engine-reported usage counts too.
	held[0].Reservation.Release(true)
	held[0].Worker.UpdateState(registry.EngineState{KVUsedTokens: 900, KVCapacityTokens: 1000}, time.Now())
	if _, err := s.Schedule(ctx, &Request{Model: "m", Role: registry.RoleDecode, KVReserve: 800}); !errors.Is(err, ErrNoCapacity) {
		t.Fatalf("reported usage ignored: %v", err)
	}
	// Ejected workers do not count as healthy.
	for _, w := range reg.All() {
		w.ReportFailure(time.Now(), 1, time.Minute)
	}
	if _, err := s.Schedule(ctx, &Request{Model: "m", Role: registry.RoleDecode, KVReserve: 1}); !errors.Is(err, ErrNoWorkers) {
		t.Fatalf("want ErrNoWorkers, got %v", err)
	}
	if _, err := s.Schedule(ctx, &Request{Model: "other", Role: registry.RoleDecode}); !errors.Is(err, ErrNoWorkers) {
		t.Fatalf("unknown model: %v", err)
	}
}

func TestRequireNormalAndExclude(t *testing.T) {
	reg := newReg(t, registry.RoleDecode, 3, func(s *registry.Spec) { s.Normal = s.ID == "w1" })
	s := New(reg, Config{Policy: &RoundRobin{}})
	for i := 0; i < 3; i++ {
		d, err := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RoleDecode, RequireNormal: true})
		if err != nil || d.Worker.ID() != "w1" {
			t.Fatalf("normal pick: %v %v", d, err)
		}
		d.Reservation.Release(true)
	}
	d, err := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RoleDecode, Exclude: map[string]bool{"w0": true, "w1": true}})
	if err != nil || d.Worker.ID() != "w2" {
		t.Fatalf("exclude: %v %v", d, err)
	}
}

func TestSessionAffinityIsStableAndBounded(t *testing.T) {
	reg := newReg(t, registry.RoleDecode, 5, nil)
	pol, _ := BuildPolicy(PolicyConfig{Name: "session_affinity", Fallback: "least_load"}, 0)
	s := New(reg, Config{Policy: pol})
	var first string
	for i := 0; i < 10; i++ {
		d, err := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RoleDecode, SessionKey: "user-42"})
		if err != nil {
			t.Fatal(err)
		}
		if first == "" {
			first = d.Worker.ID()
		} else if d.Worker.ID() != first {
			t.Fatalf("session moved from %s to %s", first, d.Worker.ID())
		}
		d.Reservation.Release(true)
	}
	reg.Get(first).UpdateState(registry.EngineState{RunningRequests: 100}, time.Now())
	d, _ := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RoleDecode, SessionKey: "user-42"})
	if d.Worker.ID() == first || d.Reason != "affinity_spill" {
		t.Fatalf("overloaded affinity target kept: %s %s", d.Worker.ID(), d.Reason)
	}
}

func TestConcurrentSchedulingNeverOvercommits(t *testing.T) {
	reg := newReg(t, registry.RoleDecode, 4, func(s *registry.Spec) { s.KVCapacityTokens = 10_000 })
	pol, _ := BuildPolicy(PolicyConfig{Name: "cache_aware"}, 8)
	s := New(reg, Config{Policy: pol, Index: kvindex.NewApprox(0, 0)})
	var mu sync.Mutex
	var got []*Decision
	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			d, err := s.Schedule(context.Background(), &Request{Model: "m", Role: registry.RoleDecode, KVReserve: 1000, Blocks: []uint64{uint64(i % 7)}, BlockSize: 64, PromptTokens: 64})
			if err == nil {
				mu.Lock()
				got = append(got, d)
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	if len(got) != 40 {
		t.Fatalf("admitted %d, capacity allows exactly 40", len(got))
	}
	for _, w := range reg.All() {
		if snap := w.Snapshot(time.Now(), time.Minute); snap.ReservedKV > 10_000 {
			t.Fatalf("%s overcommitted: %d", w.ID(), snap.ReservedKV)
		}
	}
}
