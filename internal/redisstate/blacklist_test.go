package redisstate_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

func TestSharedBlacklistSurvivesRemovalAndExpiry(t *testing.T) {
	a, b, mini := pair(t)
	ctx := context.Background()
	ra, rb := registry.NewWithStore(time.Second, a), registry.NewWithStore(time.Second, b)
	w := register(t, ra, "d0")
	old := rb.Get("d0").Snapshot(time.Now(), 0)
	held := w.Reserve(20, 0, nil)
	// Passing an incorrect client clock must not affect the Redis timestamp.
	block, err := a.Block(ctx, "d0", "transport errors", time.Unix(0, 0))
	if err != nil || block.ID != "d0" || block.BlockedAtMS < time.Now().Add(-time.Minute).UnixMilli() {
		t.Fatalf("block/timestamp: %+v %v", block, err)
	}
	if again, err := b.Block(ctx, "d0", block.Reason, time.Now()); err != nil || again != block {
		t.Fatalf("idempotent block across replicas: %+v %v", again, err)
	}
	if err := w.UpdateStateIfRevision(ctx, registry.EngineState{}, time.Now(), old.Revision); !errors.Is(err, registry.ErrConflict) {
		t.Fatalf("block did not fence old state decisions: %v", err)
	}
	if _, err := w.TryReserve(ctx, 1, 0, registry.Unlimited(), nil); !errors.Is(err, registry.ErrConflict) {
		t.Fatalf("store admitted blocked instance: %v", err)
	}
	if err := held.Release(true); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ra.UpsertContext(ctx, w.Spec(), registry.SourceHeartbeat, &registry.Beat{State: &registry.EngineState{}}); err != nil {
		t.Fatal(err)
	}
	sched := scheduler.New(rb, scheduler.Config{})
	req := &scheduler.Request{Model: "m", Role: registry.RoleDecode}
	if _, err := sched.Schedule(ctx, req); !errors.Is(err, scheduler.ErrNoWorkers) {
		t.Fatalf("another gateway routed to blocked instance: %v", err)
	}
	if _, ok, err := rb.RemoveContext(ctx, "d0"); err != nil || !ok {
		t.Fatalf("remove: %v %v", ok, err)
	}
	if blocks, err := b.Blocks(ctx); err != nil || len(blocks) != 1 || blocks[0] != block {
		t.Fatalf("block lost at deregistration: %+v %v", blocks, err)
	}
	next := register(t, rb, "d0")
	if next.Ref() == w.Ref() || !next.Snapshot(time.Now(), 0).Blocked {
		t.Fatal("re-registration evaded block")
	}
	if mini != nil {
		mini.SetTime(time.Now().Add(2 * time.Second))
		if expired, err := rb.SweepContext(ctx); err != nil || len(expired) != 1 {
			t.Fatalf("expiry: %+v %v", expired, err)
		}
		next = register(t, ra, "d0")
		if !next.Snapshot(time.Now(), 0).Blocked {
			t.Fatal("expiry evaded block")
		}
	}
	// Closing one gateway leaves the shared rules intact.
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if blocks, err := b.Blocks(ctx); err != nil || len(blocks) != 1 {
		t.Fatalf("closing a gateway deleted rules: %+v %v", blocks, err)
	}
	// A newly constructed registry has no local state to accidentally rely on.
	fresh := registry.NewWithStore(time.Minute, b)
	if !fresh.Get("d0").Snapshot(time.Now(), 0).Blocked {
		t.Fatal("fresh registry missed persistent rule")
	}
	next = fresh.Get("d0")
	next.SetDraining(true)
	next.SetEngineDraining(true)
	next.ReportFailure(time.Now(), 1, time.Minute)
	if err := b.Unblock(ctx, "d0"); err != nil {
		t.Fatal(err)
	}
	if snap := next.Snapshot(time.Now(), 0); snap.Blocked || !snap.AdminDrained || !snap.Draining || snap.Healthy {
		t.Fatalf("unblock overwrote independent guards: %+v", snap)
	}
	if err := b.Unblock(ctx, "d0"); err != nil {
		t.Fatal(err)
	}
}

func TestSharedBlacklistOfflineRulesAndOutage(t *testing.T) {
	a, b, mini := pair(t)
	ctx := context.Background()
	for _, id := range []string{"offline-z", "offline-a"} {
		if _, err := a.Block(ctx, id, "maintenance", time.Now()); err != nil {
			t.Fatal(err)
		}
	}
	if blocks, err := b.Blocks(ctx); err != nil || len(blocks) != 2 || blocks[0].ID != "offline-a" {
		t.Fatalf("shared offline rules: %+v %v", blocks, err)
	}
	if mini == nil {
		return
	}
	mini.SetError("ERR injected outage")
	if _, err := b.Block(ctx, "failed", "reason", time.Now()); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("block acknowledged during outage: %v", err)
	}
	if err := b.Unblock(ctx, "offline-a"); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("unblock acknowledged during outage: %v", err)
	}
	if _, err := b.Blocks(ctx); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("list hid outage: %v", err)
	}
	mini.SetError("")
	if blocks, err := b.Blocks(ctx); err != nil || len(blocks) != 2 {
		t.Fatalf("failed mutations changed rules: %+v %v", blocks, err)
	}
}

func TestBlacklistSchemaRejectsPreBlacklistGateways(t *testing.T) {
	a, _, _ := pair(t)
	ctx := context.Background()
	if err := a.Client.Set(ctx, a.Key("settings"), `{"schema":2,"heartbeat_ttl_ms":60000,"block_size":64}`, 0).Err(); err != nil {
		t.Fatal(err)
	}
	if err := a.CheckSettings(ctx, time.Minute, 64); err == nil {
		t.Fatal("schema-2 gateway could route around the blacklist")
	}
}
