package registry

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
	"time"
)

// clock is a settable time source for the registry.
type clock struct{ t time.Time }

func (c *clock) now() time.Time { return c.t }

func newTestRegistry(ttl time.Duration) (*Registry, *clock) {
	c := &clock{t: time.Unix(1_700_000_000, 0)}
	r := New(ttl)
	r.now = c.now
	return r, c
}

func ids(ws []*Worker) []string {
	out := make([]string, 0, len(ws))
	for _, w := range ws {
		out = append(out, w.ID())
	}
	return out
}

func spec(id, model string, role Role) Spec {
	return Spec{ID: id, Model: model, Role: role, Target: id + ":9001"}
}

// Several models, each with instances in several roles: the scheduler asks
// for the candidates of one model and role. An instance that changes its
// role moves to the other pool; listeners hear every change.
func TestInstancesByModelAndRole(t *testing.T) {
	r, _ := newTestRegistry(time.Minute)
	var events []string
	r.OnChange(func(ev Event) { events = append(events, ev.Kind+":"+ev.Worker.ID()) })
	for _, s := range []Spec{
		spec("ds-d1", "deepseek", RoleDecode), spec("ds-d0", "deepseek", RoleDecode), spec("ds-p0", "deepseek", RolePrefill),
		spec("ds-e0", "deepseek", RoleEncode), spec("qw-u0", "qwen", RoleUnified), spec("qw-d0", "qwen", RoleDecode),
	} {
		if _, created, err := r.Upsert(s, "heartbeat"); err != nil || !created {
			t.Fatalf("%s: created %v err %v", s.ID, created, err)
		}
	}
	for _, c := range []struct {
		model string
		role  Role
		want  []string
	}{
		{"deepseek", RoleDecode, []string{"ds-d0", "ds-d1"}},
		{"deepseek", RolePrefill, []string{"ds-p0"}},
		{"deepseek", RoleEncode, []string{"ds-e0"}},
		{"deepseek", RoleUnified, []string{}},
		{"qwen", RoleUnified, []string{"qw-u0"}},
		{"qwen", RoleDecode, []string{"qw-d0"}},
		{"nope", RoleDecode, []string{}},
	} {
		if got := ids(r.Candidates(c.model, c.role)); !slices.Equal(got, c.want) {
			t.Errorf("%s/%s: %v, want %v", c.model, c.role, got, c.want)
		}
	}
	if got := ids(r.All()); len(got) != 6 || !slices.IsSorted(got) {
		t.Fatalf("all %v", got)
	}

	// The same id again refreshes; a new role moves it.
	if _, created, err := r.Upsert(spec("ds-d1", "deepseek", RolePrefill), "heartbeat"); err != nil || created {
		t.Fatalf("re-upsert: created %v err %v", created, err)
	}
	if got := ids(r.Candidates("deepseek", RolePrefill)); !slices.Equal(got, []string{"ds-d1", "ds-p0"}) {
		t.Fatalf("prefill after the move %v", got)
	}
	if got := ids(r.Candidates("deepseek", RoleDecode)); !slices.Equal(got, []string{"ds-d0"}) {
		t.Fatalf("decode after the move %v", got)
	}
	// A refresh without changes notifies nobody.
	n := len(events)
	_, _, _ = r.Upsert(spec("qw-u0", "qwen", RoleUnified), "heartbeat")
	if len(events) != n {
		t.Fatalf("a plain refresh emitted %v", events[n:])
	}

	got, ok := r.Remove("ds-p0")
	if !ok || got.ID != "ds-p0" || got.Model != "deepseek" || got.Role != RolePrefill {
		t.Fatalf("remove: %+v %v", got, ok)
	}
	if _, ok := r.Remove("ds-p0"); ok {
		t.Fatal("removed twice")
	}
	if got := ids(r.Candidates("deepseek", RolePrefill)); !slices.Equal(got, []string{"ds-d1"}) {
		t.Fatalf("prefill after the removal %v", got)
	}
	want := []string{"added:ds-d1", "added:ds-d0", "added:ds-p0", "added:ds-e0", "added:qw-u0", "added:qw-d0", "updated:ds-d1", "removed:ds-p0"}
	if !slices.Equal(events, want) {
		t.Fatalf("events %v, want %v", events, want)
	}
}

func TestUpsertValidates(t *testing.T) {
	r, _ := newTestRegistry(time.Minute)
	for _, c := range []struct {
		spec Spec
		want string
	}{
		{Spec{Model: "m", Role: RoleDecode, Target: "t"}, "needs id, model and target"},
		{Spec{ID: "a", Role: RoleDecode, Target: "t"}, "needs id, model and target"},
		{Spec{ID: "a", Model: "m", Role: RoleDecode}, "needs id, model and target"},
		{Spec{ID: "a", Model: "m", Role: "leader", Target: "t"}, "unknown worker role"},
	} {
		if _, _, err := r.Upsert(c.spec, "heartbeat"); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%+v: %v, want %q", c.spec, err, c.want)
		}
	}
	if len(r.All()) != 0 {
		t.Fatal("an invalid spec was registered")
	}
}

// Heartbeat instances live while heartbeats arrive: each one expires a TTL
// after its last heartbeat, on its own; its siblings of the same model and
// role stay. Static instances never expire.
func TestHeartbeatInstancesExpire(t *testing.T) {
	r, c := newTestRegistry(10 * time.Second)
	var removed []string
	r.OnChange(func(ev Event) {
		if ev.Kind == "removed" {
			removed = append(removed, ev.Worker.ID())
		}
	})
	_, _, _ = r.Upsert(spec("d0", "m", RoleDecode), "heartbeat")
	_, _, _ = r.Upsert(spec("d1", "m", RoleDecode), "heartbeat")
	_, _, _ = r.Upsert(spec("s0", "m", RoleDecode), "static")

	c.t = c.t.Add(6 * time.Second)
	_, _, _ = r.Upsert(spec("d0", "m", RoleDecode), "heartbeat") // d0 heartbeats, d1 does not
	if got := r.Sweep(); len(got) != 0 {
		t.Fatalf("expired before the TTL: %v", got)
	}
	c.t = c.t.Add(5 * time.Second) // d1: 11s since its last heartbeat, d0: 5s
	got := r.Sweep()
	if len(got) != 1 || got[0].ID != "d1" || got[0].Model != "m" || got[0].Role != RoleDecode {
		t.Fatalf("sweep %+v, want d1", got)
	}
	if c := ids(r.Candidates("m", RoleDecode)); !slices.Equal(c, []string{"d0", "s0"}) {
		t.Fatalf("candidates after d1 expired %v", c)
	}

	// Load reports (polls, heartbeats) do not keep an instance alive:
	// only heartbeats do.
	r.Get("d0").UpdateState(EngineState{RunningRequests: 1}, c.t)
	c.t = c.t.Add(6 * time.Second) // d0: 11s since its last heartbeat
	if got := r.Sweep(); len(got) != 1 || got[0].ID != "d0" {
		t.Fatalf("sweep %+v, want d0", got)
	}
	c.t = c.t.Add(time.Hour)
	if got := r.Sweep(); len(got) != 0 || r.Get("s0") == nil {
		t.Fatalf("a static instance expired: %+v", got)
	}
	if !slices.Equal(removed, []string{"d1", "d0"}) {
		t.Fatalf("removed events %v", removed)
	}

	// An expired instance registers again with its next heartbeat.
	if _, created, err := r.Upsert(spec("d1", "m", RoleDecode), "heartbeat"); err != nil || !created {
		t.Fatalf("re-registration: created %v err %v", created, err)
	}
}

// A static instance that also sends heartbeats stays static: it keeps its
// configured identity and never expires.
func TestStaticInstancesStayStatic(t *testing.T) {
	r, c := newTestRegistry(time.Second)
	_, _, _ = r.Upsert(spec("s0", "m", RoleUnified), "static")
	w, created, _ := r.Upsert(spec("s0", "m", RoleUnified), "heartbeat")
	if created || w.Source() != "static" {
		t.Fatalf("source %s created %v", w.Source(), created)
	}
	c.t = c.t.Add(time.Minute)
	if got := r.Sweep(); len(got) != 0 {
		t.Fatalf("expired %+v", got)
	}
}

// Draining comes from the operator (admin API) or the engine (heartbeats,
// its state report); either keeps new requests away, and the engine's
// reports do not undo the operator's drain.
func TestDraining(t *testing.T) {
	r, c := newTestRegistry(time.Minute)
	w, _, _ := r.Upsert(spec("d0", "m", RoleDecode), "heartbeat")
	draining := func() bool { return w.Snapshot(c.t, time.Minute).Draining }
	if draining() {
		t.Fatal("draining at registration")
	}
	w.SetEngineDraining(true)
	if !draining() || w.AdminDrained() {
		t.Fatal("engine drain")
	}
	w.SetEngineDraining(false)
	w.SetDraining(true)
	w.SetEngineDraining(false) // the next heartbeat
	if !draining() || !w.AdminDrained() {
		t.Fatal("a heartbeat undid the operator's drain")
	}
	w.SetDraining(false)
	w.UpdateState(EngineState{Draining: true}, c.t)
	if !draining() {
		t.Fatal("state report drain")
	}
	w.UpdateState(EngineState{}, c.t)
	if draining() {
		t.Fatal("still draining")
	}
}

// Consecutive failures eject an instance for the cooldown; a success
// resets the count.
func TestFailuresEjectForCooldown(t *testing.T) {
	r, c := newTestRegistry(time.Minute)
	w, _, _ := r.Upsert(spec("d0", "m", RoleDecode), "heartbeat")
	w.ReportFailure(c.t, 3, 10*time.Second)
	w.ReportFailure(c.t, 3, 10*time.Second)
	w.ReportSuccess()
	w.ReportFailure(c.t, 3, 10*time.Second)
	w.ReportFailure(c.t, 3, 10*time.Second)
	if !w.Snapshot(c.t, 0).Healthy {
		t.Fatal("ejected without three consecutive failures")
	}
	if !w.ReportFailure(c.t, 3, 10*time.Second) || w.Snapshot(c.t, 0).Healthy {
		t.Fatal("not ejected after three consecutive failures")
	}
	if !w.Snapshot(c.t.Add(10*time.Second), 0).Healthy {
		t.Fatal("still ejected after the cooldown")
	}
}

func TestStateValidationAndRevisionFencing(t *testing.T) {
	r, c := newTestRegistry(time.Minute)
	w, _, _ := r.Upsert(spec("d0", "m", RoleDecode), SourceHeartbeat)
	old := w.Snapshot(c.t, 0)
	if err := w.UpdateState(EngineState{KVUsage: math.MaxFloat64, KVCapacityTokens: 1000}, c.t); err != nil {
		t.Fatal(err)
	}
	if s := w.Snapshot(c.t, 0); s.EffectiveKVUsed != 1000 {
		t.Fatalf("overflow made full KV appear empty: %+v", s)
	}
	if err := w.UpdateStateIfRevision(context.Background(), EngineState{}, c.t, old.Revision); !errors.Is(err, ErrConflict) {
		t.Fatalf("old poll accepted: %v", err)
	}
	for _, st := range []EngineState{{RunningRequests: -1}, {KVUsedTokens: -1}, {KVUsage: math.NaN()}, {DecodeTokensPerSec: math.Inf(1)}} {
		if err := w.UpdateState(st, c.t); err == nil {
			t.Errorf("invalid state accepted: %+v", st)
		}
	}
}
