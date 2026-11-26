package kvindex

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/leonyang215/ModelWeave/internal/redisstate"
)

func redisIndexes(t *testing.T, kind string, capacity int) (*Redis, *Redis, *miniredis.Miniredis) {
	t.Helper()
	url := os.Getenv("MW_TEST_REDIS_URL")
	var mini *miniredis.Miniredis
	if url == "" {
		mini = miniredis.RunT(t)
		mini.ClockTTL(true)
		url = "redis://" + mini.Addr()
	}
	ns := fmt.Sprintf("%s/%d", t.Name(), time.Now().UnixNano())
	newIndex := func() *Redis {
		s, err := redisstate.New(context.Background(), redisstate.Options{URL: url, Namespace: ns})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = s.Close() })
		return NewRedis(s.Client, s.Key("prefix:"), RedisOptions{Type: kind, TTL: time.Minute, OptimisticTTL: time.Second, CapacityBlocks: capacity, OnError: func(err error) { t.Error(err) }})
	}
	return newIndex(), newIndex(), mini
}

func TestRedisApproxSharedPrefixesCapacityAndEpochKeys(t *testing.T) {
	a, b, _ := redisIndexes(t, "approx", 2)
	// Adjacent hashes above 2^53 must remain distinct (no Lua float rounding).
	h := []uint64{1<<53 + 1, 1<<53 + 2, ^uint64(0)}
	a.Observe("w@1", h)
	q := Query{Blocks: h, Candidates: []string{"w", "missing"}, WorkerKeys: map[string]string{"w": "w@1"}}
	m, err := b.Match(context.Background(), q)
	if err != nil || m["w"] != 2 || m["missing"] != 0 {
		t.Fatalf("shared bounded prefix: %v %v", m, err)
	}
	b.SetWorkerCapacity("w@1", 3)
	b.Observe("w@1", h)
	m, err = a.Match(context.Background(), q)
	if err != nil || m["w"] != 3 {
		t.Fatalf("capacity update: %v %v", m, err)
	}
	q.Blocks = []uint64{1<<53 + 2}
	if m, err := a.Match(context.Background(), q); err != nil || m["w"] != 1 {
		t.Fatalf("uint64 precision: %v %v", m, err)
	}
	q.Blocks = []uint64{1<<53 + 3}
	if m, err := a.Match(context.Background(), q); err != nil || m["w"] != 0 {
		t.Fatalf("rounded distinct hash into hit: %v %v", m, err)
	}
	q.Blocks = h
	q.WorkerKeys["w"] = "w@2"
	if m, err := a.Match(context.Background(), q); err != nil || m["w"] != 0 {
		t.Fatalf("re-registration inherited prefixes: %v %v", m, err)
	}
	b.Observe("w@2", h)
	a.ForgetWorker("w@1")
	if m, err := a.Match(context.Background(), q); err != nil || m["w"] != 2 {
		t.Fatalf("old epoch invalidation erased new epoch: %v %v", m, err)
	}
}

func TestRedisEventsSharedInvalidationAndOptimisticTTL(t *testing.T) {
	a, b, mini := redisIndexes(t, "events", 0)
	h := []uint64{11, 1<<53 + 1, ^uint64(0)}
	if err := a.ApplyEvents("w", EventBatch{Stored: h}); err != nil {
		t.Fatal(err)
	}
	q := Query{Blocks: h, Candidates: []string{"w", "optimistic"}}
	b.Observe("optimistic", h)
	if m, err := b.Match(context.Background(), q); err != nil || m["w"] != 3 || m["optimistic"] != 3 {
		t.Fatalf("events: %v %v", m, err)
	}
	if err := b.ApplyEvents("w", EventBatch{Removed: h[1:2]}); err != nil {
		t.Fatal(err)
	}
	if m, err := a.Match(context.Background(), q); err != nil || m["w"] != 1 {
		t.Fatalf("shared eviction: %v %v", m, err)
	}
	if err := a.ApplyEvents("w", EventBatch{Cleared: true, Stored: h[:2]}); err != nil {
		t.Fatal(err)
	}
	if m, err := b.Match(context.Background(), q); err != nil || m["w"] != 2 {
		t.Fatalf("clear + resync: %v %v", m, err)
	}
	if mini != nil {
		mini.SetTime(time.Now().Add(2 * time.Second))
		if m, err := a.Match(context.Background(), q); err != nil || m["w"] != 2 || m["optimistic"] != 0 {
			t.Fatalf("optimistic expiry: %v %v", m, err)
		}
	}
	a.ForgetWorker("w")
	if m, err := b.Match(context.Background(), q); err != nil || m["w"] != 0 {
		t.Fatalf("shared removal: %v %v", m, err)
	}
}

func TestRedisApproxTTLRetainsConfiguredCapacity(t *testing.T) {
	a, b, mini := redisIndexes(t, "approx", 0)
	if mini == nil {
		t.Skip("virtual clock uses miniredis")
	}
	now := time.Now()
	mini.SetTime(now)
	a.SetWorkerCapacity("w", 2)
	a.Observe("w", []uint64{1, 2, 3})
	mini.SetTime(now.Add(3 * time.Minute))
	q := Query{Blocks: []uint64{1, 2, 3}, Candidates: []string{"w"}}
	if m, err := b.Match(context.Background(), q); err != nil || m["w"] != 0 {
		t.Fatalf("TTL: %v %v", m, err)
	}
	b.Observe("w", q.Blocks)
	if m, err := a.Match(context.Background(), q); err != nil || m["w"] != 2 {
		t.Fatalf("idle index lost capacity hint: %v %v", m, err)
	}
}
