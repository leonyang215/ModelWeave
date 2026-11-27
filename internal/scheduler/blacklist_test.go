package scheduler

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

type beforeReservePolicy struct{ before func(*Candidate) }

func (beforeReservePolicy) Name() string    { return "before-reserve-test" }
func (beforeReservePolicy) UsesCache() bool { return false }
func (p beforeReservePolicy) Pick(_ *Request, c []*Candidate) (*Candidate, string) {
	p.before(c[0])
	return c[0], "test"
}

func TestBlacklistOverridesPrefixCacheAndPinnedRouting(t *testing.T) {
	for _, role := range []registry.Role{registry.RoleEncode, registry.RolePrefill, registry.RoleDecode, registry.RoleUnified} {
		t.Run(string(role), func(t *testing.T) {
			r := newReg(t, role, 2, nil)
			ctx := context.Background()
			index := kvindex.NewApprox(time.Minute, 0)
			index.Observe("w0", []uint64{1, 2})
			s := New(r, Config{Index: index})
			if _, err := r.Store().Block(ctx, "w0", "quarantine", time.Now()); err != nil {
				t.Fatal(err)
			}
			req := &Request{Model: "m", Role: role, Blocks: []uint64{1, 2}, BlockSize: 64, PromptTokens: 128}
			d, err := s.Schedule(ctx, req)
			if err != nil {
				t.Fatal(err)
			}
			if d.Worker.ID() != "w1" {
				t.Fatal("a prefix hit bypassed blacklisting")
			}
			if err := d.Reservation.Release(true); err != nil {
				t.Fatal(err)
			}
			req.Pin = "w0"
			if _, err := s.Schedule(ctx, req); !errors.Is(err, ErrNoWorkers) {
				t.Fatalf("a pinned session bypassed blacklisting: %v", err)
			}
			if err := r.Store().Unblock(ctx, "w0"); err != nil {
				t.Fatal(err)
			}
			d, err = s.Schedule(ctx, req)
			if err != nil || d.Worker.ID() != "w0" {
				t.Fatalf("unblocked routing did not recover: %+v %v", d, err)
			}
			if err := d.Reservation.Release(true); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// Deterministically place a block between filtering the candidate and the
// final reservation. No timing/sleep is needed to exercise the race window.
func TestBlacklistFencesAnAlreadyPickedCandidate(t *testing.T) {
	for _, role := range []registry.Role{registry.RoleEncode, registry.RolePrefill, registry.RoleDecode, registry.RoleUnified} {
		t.Run(string(role), func(t *testing.T) {
			r := newReg(t, role, 1, nil)
			ctx := context.Background()
			s := New(r, Config{Policy: beforeReservePolicy{before: func(c *Candidate) {
				if _, err := r.Store().Block(ctx, c.Snap.Spec.ID, "isolated during selection", time.Now()); err != nil {
					t.Fatal(err)
				}
			}}})
			if _, err := s.Schedule(ctx, &Request{Model: "m", Role: role, KVReserve: 1}); !errors.Is(err, ErrNoWorkers) {
				t.Fatalf("a stale pick bypassed blacklisting: %v", err)
			}
			if snap := r.Get("w0").Snapshot(time.Now(), 0); snap.Inflight != 0 || snap.ReservedKV != 0 || !snap.Blocked {
				t.Fatalf("rejected pick leaked a reservation: %+v", snap)
			}
		})
	}
}
