// Package scheduler picks the worker for one stage of a request. A pass
// snapshots the candidates, filters out unhealthy, draining or full workers,
// looks up prefix-cache matches through a kvindex.Index, and lets a policy
// pick among the rest. The pick is reserved on the worker before the pass
// returns, so concurrent passes never over-commit a worker.
package scheduler

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework/plugins"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework/runtime"
)

// Errors returned by Schedule.
var (
	// ErrNoWorkers means no healthy worker serves the model and role.
	ErrNoWorkers = errors.New("no healthy worker available")
	// ErrNoCapacity means healthy workers exist but none can take the
	// request now; the caller may queue and retry.
	ErrNoCapacity = errors.New("no worker has capacity")
)

// Config tunes a scheduler for one role.
type Config struct {
	Policy     Policy
	Index      kvindex.Index
	StaleAfter time.Duration // engine state older than this is ignored
	Observer   Observer
	// RegistrationKeys isolates prefix entries across removal/re-registration
	// of an instance ID. The gateway enables this for every backend.
	RegistrationKeys bool
}

// Scheduler schedules one role of one model.
type Scheduler struct {
	reg       *registry.Registry
	cfg       Config
	mu        sync.Mutex // serializes this policy instance; the store enforces admission
	now       func() time.Time
	framework *runtime.Framework
}

// New returns a scheduler.
func New(reg *registry.Registry, cfg Config) *Scheduler {
	if cfg.Policy == nil {
		cfg.Policy, _ = BuildPolicy(PolicyConfig{}, 32)
	}
	if cfg.Index == nil {
		cfg.Index = kvindex.None{}
	}
	if cfg.StaleAfter <= 0 {
		cfg.StaleAfter = 10 * time.Second
	}
	return &Scheduler{reg: reg, cfg: cfg, now: time.Now,
		framework: runtime.New(cfg.Policy, plugins.Eligibility{}, plugins.Capacity{}),
	}
}

// Policy returns the policy name.
func (s *Scheduler) Policy() string { return s.cfg.Policy.Name() }

func (s *Scheduler) Scorers() []ScorerConfig {
	if p, ok := s.cfg.Policy.(interface{ Scorers() []ScorerConfig }); ok {
		return p.Scorers()
	}
	return nil
}

// Index returns the cache index.
func (s *Scheduler) Index() kvindex.Index { return s.cfg.Index }

// Schedule picks and reserves a worker.
func (s *Scheduler) Schedule(ctx context.Context, req *Request) (*Decision, error) {
	start := s.now()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := s.reg.Sync(ctx); err != nil {
		s.fail(req, err)
		return nil, err
	}
	workers := s.reg.Candidates(req.Model, req.Role)
	if len(workers) == 0 {
		s.fail(req, ErrNoWorkers)
		return nil, ErrNoWorkers
	}

	// Advisory prefix lookup may do network I/O and stays outside the policy lock.
	matches := s.lookupCache(ctx, req, workers)

	s.mu.Lock()
	defer s.mu.Unlock()
	// Reservations are conditional at the store, so other schedulers and
	// gateway replicas cannot over-commit between snapshot and reserve.
	for attempt := 0; attempt < 32; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		now := s.now()
		cands, healthy, err := s.filterCandidates(ctx, req, workers, matches, now)
		if err != nil {
			s.fail(req, err)
			return nil, err
		}
		if healthy == 0 {
			s.fail(req, ErrNoWorkers)
			return nil, ErrNoWorkers
		}
		if len(cands) == 0 {
			s.fail(req, ErrNoCapacity)
			return nil, ErrNoCapacity
		}
		computeLoad(req.Role, cands)
		pick, reason, err := s.framework.RunScorePlugins(req, cands)
		if err != nil {
			s.fail(req, err)
			return nil, err
		}

		// Reserve is always the store's atomic, revision-fenced operation.
		res, err := reserve(ctx, req, pick)
		if errors.Is(err, registry.ErrConflict) || errors.Is(err, registry.ErrGone) {
			continue
		}
		if err != nil {
			s.fail(req, err)
			return nil, err
		}
		d := &Decision{
			Worker:       pick.Snap.Worker,
			Spec:         pick.Snap.Spec,
			Reservation:  res,
			CachedTokens: pick.CachedTokens,
			CacheRatio:   pick.CacheRatio,
			Policy:       s.cfg.Policy.Name(),
			Reason:       reason,
			Candidates:   len(workers),
			Eligible:     len(cands),
			Duration:     s.now().Sub(start),
			Score:        pick.Score, Scores: pick.Scores,
		}
		if len(req.Blocks) > 0 {
			key := d.Worker.ID()
			if s.cfg.RegistrationKeys {
				key = d.Worker.CacheKey()
			}
			s.cfg.Index.Observe(key, req.Blocks)
		}
		if s.cfg.Observer != nil {
			s.cfg.Observer.Scheduled(req, d)
		}
		return d, nil
	}
	s.fail(req, ErrNoCapacity)
	return nil, ErrNoCapacity
}

func (s *Scheduler) fail(req *Request, err error) {
	if s.cfg.Observer != nil {
		s.cfg.Observer.Failed(req, s.cfg.Policy.Name(), err)
	}
}
