package scheduler

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
)

func (s *Scheduler) lookupCache(ctx context.Context, req *Request, workers []*registry.Worker) map[string]int {
	if !s.cfg.Policy.UsesCache() || len(req.Blocks) == 0 {
		return nil
	}
	ids := make([]string, 0, len(workers))
	keys := make(map[string]string, len(workers))
	for _, worker := range workers {
		if !req.Exclude[worker.ID()] && (req.Pin == "" || req.Pin == worker.ID()) {
			ids = append(ids, worker.ID())
			if s.cfg.RegistrationKeys {
				keys[worker.ID()] = worker.CacheKey()
			}
		}
	}
	start := time.Now()
	matches, err := s.cfg.Index.Match(ctx, kvindex.Query{Model: req.Model, Role: string(req.Role), BlockSize: req.BlockSize, Blocks: req.Blocks, Candidates: ids, WorkerKeys: keys})
	if s.cfg.Observer != nil {
		s.cfg.Observer.CacheLookup(s.cfg.Index.Kind(), time.Since(start), err)
	}
	if err != nil {
		return nil
	}
	return matches
}

func (s *Scheduler) filterCandidates(ctx context.Context, req *Request, workers []*registry.Worker, matches map[string]int, now time.Time) ([]*Candidate, int, error) {
	snapshots, err := s.reg.Snapshots(ctx, workers, now, s.cfg.StaleAfter)
	if err != nil {
		return nil, 0, err
	}
	candidates := make([]*Candidate, 0, len(workers))
	healthy := 0
	for i, worker := range workers {
		candidate := &Candidate{Snap: snapshots[i]}
		if blocks := matches[worker.ID()]; blocks > 0 && req.BlockSize > 0 {
			candidate.CachedBlocks = blocks
			candidate.CachedTokens = min(blocks*req.BlockSize, req.PromptTokens)
			if req.PromptTokens > 0 {
				candidate.CacheRatio = float64(candidate.CachedTokens) / float64(req.PromptTokens)
			}
		}
		status := s.framework.RunFilterPlugins(req, candidate)
		switch status.Code {
		case framework.UnschedulableAndUnresolvable:
			continue
		case framework.Unschedulable:
			healthy++
			continue
		case framework.Success:
			healthy++
			candidates = append(candidates, candidate)
		default:
			return nil, 0, fmt.Errorf("invalid filter status: %d", status.Code)
		}
	}
	return candidates, healthy, nil
}

// computeLoad normalizes eligible candidates only, in their role's load unit.
func computeLoad(role registry.Role, candidates []*Candidate) {
	maxLoad := 0.0
	for _, candidate := range candidates {
		if role == registry.RolePrefill {
			candidate.Load = float64(candidate.Snap.QueuedPrefillTokens())
		} else {
			candidate.Load = float64(candidate.Snap.Running())
		}
		maxLoad = math.Max(maxLoad, candidate.Load)
	}
	for _, candidate := range candidates {
		snapshot := &candidate.Snap
		norm := 0.0
		switch {
		case role == registry.RolePrefill && snapshot.MaxPrefillTokens > 0:
			norm = candidate.Load / float64(snapshot.MaxPrefillTokens)
		case role != registry.RolePrefill && snapshot.MaxRunning > 0:
			norm = candidate.Load / float64(snapshot.MaxRunning)
		case maxLoad > 0:
			norm = candidate.Load / maxLoad
		}
		if role == registry.RoleDecode || role == registry.RoleUnified {
			norm = math.Max(norm, snapshot.KVUtil())
		}
		candidate.NormLoad = math.Min(norm, 1)
	}
}
