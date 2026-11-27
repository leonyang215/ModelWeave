package scheduler

import (
	"context"

	"github.com/leonyang215/ModelWeave/internal/registry"
)

func reserve(ctx context.Context, req *Request, pick *Candidate) (*registry.Reservation, error) {
	var prefill, kv int64
	if req.PrefillCost != nil {
		prefill = req.PrefillCost(pick.CachedTokens)
	}
	if req.Role == registry.RoleDecode || req.Role == registry.RoleUnified {
		kv = req.KVReserve
	}
	return pick.Snap.Worker.TryReserve(ctx, kv, prefill, reservationLimits(req, &pick.Snap), req.OnFree)
}

// Bounds apply before this claim; the store atomically checks both these
// limits and the snapshot revision (including intervening block/drain changes).
func reservationLimits(req *Request, snapshot *registry.Snapshot) registry.Limits {
	limits := registry.Unlimited()
	limits.Revision = snapshot.Revision
	if snapshot.MaxRunning > 0 {
		limits.MaxInflight = int64(snapshot.MaxRunning) - 1
	}
	if (req.Role == registry.RoleDecode || req.Role == registry.RoleUnified) && snapshot.KVCapacity > 0 {
		limits.MaxReservedKV = snapshot.KVCapacity - req.KVReserve
		if snapshot.HasState {
			limits.MaxPendingKV = snapshot.KVCapacity - snapshot.ReportedKVUsed() - req.KVReserve
		}
	}
	if snapshot.MaxPrefillTokens > 0 && (req.Role == registry.RolePrefill || req.Role == registry.RoleUnified || req.RequireNormal) {
		limits.MaxPrefill = snapshot.MaxPrefillTokens - 1
	}
	return limits
}

// UncachedCost is attention-weighted prefill work; beta=0 counts plain tokens.
func UncachedCost(promptTokens, cached int, beta float64) int64 {
	uncached := promptTokens - cached
	if uncached <= 0 {
		return 0
	}
	if beta <= 0 {
		return int64(uncached)
	}
	return int64(float64(uncached) * (1 + beta*(float64(cached)+float64(uncached)/2)))
}
