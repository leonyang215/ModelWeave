package registry

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestBlacklistSurvivesRegistrationLifecycle(t *testing.T) {
	r, clock := newTestRegistry(time.Second)
	ctx := context.Background()
	w, _, _ := r.Upsert(spec("d0", "m", RoleDecode), SourceHeartbeat)
	old := w.Snapshot(clock.t, 0)
	held := w.Reserve(100, 10, nil)
	b, err := r.Store().Block(ctx, "d0", "bad outputs", clock.t)
	if err != nil || b.ID != "d0" || b.Reason != "bad outputs" || b.BlockedAtMS != clock.t.UnixMilli() {
		t.Fatalf("block: %+v %v", b, err)
	}
	if again, err := r.Store().Block(ctx, "d0", b.Reason, clock.t.Add(time.Hour)); err != nil || again != b {
		t.Fatalf("PUT is not idempotent: %+v %v", again, err)
	}
	if err := w.UpdateStateIfRevision(ctx, EngineState{}, clock.t, old.Revision); !errors.Is(err, ErrConflict) {
		t.Fatalf("a pre-block poll was not fenced: %v", err)
	}
	if _, err := w.TryReserve(ctx, 1, 0, Unlimited(), nil); !errors.Is(err, ErrConflict) {
		t.Fatalf("new reservation on a blocked ID: %v", err)
	}
	if err := held.Release(true); err != nil {
		t.Fatal(err)
	}
	if _, _, err := r.UpsertContext(ctx, w.Spec(), SourceHeartbeat, &Beat{State: &EngineState{}}); err != nil {
		t.Fatal(err)
	}
	if snap := w.Snapshot(clock.t, 0); !snap.Blocked || snap.Block == nil || snap.Draining || snap.Inflight != 0 {
		t.Fatalf("heartbeat changed the block or in-flight drain semantics: %+v", snap)
	}
	// Expiry must not delete a denylist entry, even though it deletes the
	// registration and all its ephemeral drain/state/accounting fields.
	clock.t = clock.t.Add(2 * time.Second)
	if expired := r.Sweep(); len(expired) != 1 {
		t.Fatalf("expiry: %+v", expired)
	}
	if blocks, err := r.Store().Blocks(ctx); err != nil || len(blocks) != 1 || blocks[0] != b {
		t.Fatalf("offline blacklist: %+v %v", blocks, err)
	}
	next, _, _ := r.Upsert(spec("d0", "m", RoleDecode), SourceHeartbeat)
	if next.Ref() == w.Ref() || !next.Snapshot(clock.t, 0).Blocked {
		t.Fatal("new registration evaded the block")
	}
	r.Remove("d0")
	next, _, _ = r.Upsert(spec("d0", "n", RoleUnified), SourceStatic)
	if !next.Snapshot(clock.t, 0).Blocked {
		t.Fatal("a role/model/target/source change evaded the block")
	}
	next.SetDraining(true)
	next.SetEngineDraining(true)
	next.ReportFailure(clock.t, 1, time.Minute)
	if err := r.Store().Unblock(ctx, "d0"); err != nil {
		t.Fatal(err)
	}
	if snap := next.Snapshot(clock.t, 0); snap.Blocked || !snap.AdminDrained || !snap.Draining || snap.Healthy {
		t.Fatalf("unblock overwrote independent health/drain state: %+v", snap)
	}
	if err := r.Store().Unblock(ctx, "d0"); err != nil {
		t.Fatal("DELETE must be idempotent", err)
	}
}

func TestBlacklistCanPrecedeRegistration(t *testing.T) {
	r, clock := newTestRegistry(time.Minute)
	ctx := context.Background()
	for _, id := range []string{"z", "a"} {
		if _, err := r.Store().Block(ctx, id, "quarantine", clock.t); err != nil {
			t.Fatal(err)
		}
	}
	blocks, err := r.Store().Blocks(ctx)
	if err != nil || len(blocks) != 2 || blocks[0].ID != "a" || blocks[1].ID != "z" {
		t.Fatalf("sorted blacklist: %+v %v", blocks, err)
	}
	w, _, _ := r.Upsert(spec("a", "m", RoleUnified), SourceHeartbeat)
	if !w.Snapshot(clock.t, 0).Blocked {
		t.Fatal("registration evaded a pre-existing rule")
	}
	if _, err := r.Store().Block(ctx, "a", "updated reason", clock.t.Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	if snap := w.Snapshot(clock.t, 0); snap.Block.Reason != "updated reason" {
		t.Fatalf("updated reason: %+v", snap)
	}
	ctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := r.Store().Block(ctx, "cancelled", "reason", clock.t); !errors.Is(err, context.Canceled) {
		t.Fatalf("ignored cancellation: %v", err)
	}
}
