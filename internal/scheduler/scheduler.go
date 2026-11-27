// Package scheduler picks the worker for one stage of a request. A pass
// snapshots the candidates, filters out unhealthy, draining or full workers,
// looks up prefix-cache matches through a kvindex.Index, and lets a policy
// pick among the rest. The pick is reserved on the worker before the pass
// returns, so concurrent passes never over-commit a worker.
package scheduler

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// Errors returned by Schedule.
var (
	// ErrNoWorkers means no healthy worker serves the model and role.
	ErrNoWorkers = errors.New("no healthy worker available")
	// ErrNoCapacity means healthy workers exist but none can take the
	// request now; the caller may queue and retry.
	ErrNoCapacity = errors.New("no worker has capacity")
)

// Request describes the stage being scheduled.
type Request struct {
	Model         string
	Role          registry.Role
	RequireNormal bool // decode candidates must run normal (non-PD) requests
	PromptTokens  int
	Blocks        []uint64 // prefix block hashes
	BlockSize     int
	// KVReserve is reserved on decode/unified workers (prompt + output
	// reservation).
	KVReserve int64
	// PrefillCost returns the prefill work accounted on the picked worker
	// given its cached prefix tokens (nil: no prefill accounting).
	PrefillCost func(cachedTokens int) int64
	SessionKey  string
	Exclude     map[string]bool
	// Pin restricts the pass to one worker (session affinity).
	Pin string
	// OnFree is invoked when the reservation is released (used to wake the
	// admission queue).
	OnFree func()
}

// Candidate is a worker under consideration.
type Candidate struct {
	Snap         registry.Snapshot
	CachedBlocks int
	CachedTokens int
	CacheRatio   float64
	Load         float64 // raw load in the role's unit
	NormLoad     float64 // load normalized to [0, 1]
	Score        float64
	Scores       map[string]float64 // raw per-plugin preferences (0..1)
}

// Decision is the outcome of a scheduling pass.
type Decision struct {
	Worker       *registry.Worker
	Spec         registry.Spec
	Reservation  *registry.Reservation
	CachedTokens int
	CacheRatio   float64
	Policy       string
	Reason       string
	Candidates   int
	Eligible     int
	Duration     time.Duration
	Score        float64
	Scores       map[string]float64
}

// Policy picks one candidate. Candidates are non-empty and eligible.
type Policy interface {
	Name() string
	UsesCache() bool
	Pick(req *Request, cands []*Candidate) (*Candidate, string)
}

// Observer receives scheduling outcomes for metrics.
type Observer interface {
	Scheduled(req *Request, d *Decision)
	Failed(req *Request, policy string, err error)
	CacheLookup(kind string, d time.Duration, err error)
}

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
	reg *registry.Registry
	cfg Config
	mu  sync.Mutex // serializes this policy instance; the store enforces admission
	now func() time.Time
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
	return &Scheduler{reg: reg, cfg: cfg, now: time.Now}
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

	// Cache lookup happens outside the lock: it may be a network call.
	var matches map[string]int
	if s.cfg.Policy.UsesCache() && len(req.Blocks) > 0 {
		ids := make([]string, 0, len(workers))
		keys := make(map[string]string, len(workers))
		for _, w := range workers {
			if !req.Exclude[w.ID()] && (req.Pin == "" || req.Pin == w.ID()) {
				ids = append(ids, w.ID())
				if s.cfg.RegistrationKeys {
					keys[w.ID()] = w.CacheKey()
				}
			}
		}
		t0 := time.Now()
		m, err := s.cfg.Index.Match(ctx, kvindex.Query{
			Model: req.Model, Role: string(req.Role), BlockSize: req.BlockSize, Blocks: req.Blocks, Candidates: ids, WorkerKeys: keys,
		})
		if s.cfg.Observer != nil {
			s.cfg.Observer.CacheLookup(s.cfg.Index.Kind(), time.Since(t0), err)
		}
		if err == nil {
			matches = m
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	// Reservations are conditional at the store, so other schedulers and
	// gateway replicas cannot over-commit between snapshot and reserve.
	for attempt := 0; attempt < 32; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		now := s.now()
		snaps, err := s.reg.Snapshots(ctx, workers, now, s.cfg.StaleAfter)
		if err != nil {
			s.fail(req, err)
			return nil, err
		}
		cands := make([]*Candidate, 0, len(workers))
		healthy := 0
		for i, w := range workers {
			if req.Exclude[w.ID()] || (req.Pin != "" && req.Pin != w.ID()) {
				continue
			}
			snap := snaps[i]
			if snap.Gone || snap.Spec.Model != req.Model || snap.Spec.Role != req.Role || !snap.Healthy || snap.Draining || snap.Blocked {
				continue
			}
			if req.RequireNormal && !snap.Spec.ServesNormal() {
				continue
			}
			healthy++
			c := &Candidate{Snap: snap}
			if n := matches[w.ID()]; n > 0 && req.BlockSize > 0 {
				c.CachedBlocks = n
				c.CachedTokens = min(n*req.BlockSize, req.PromptTokens)
				if req.PromptTokens > 0 {
					c.CacheRatio = float64(c.CachedTokens) / float64(req.PromptTokens)
				}
			}
			if !hasCapacity(req, c) {
				continue
			}
			cands = append(cands, c)
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
		pick, reason := s.cfg.Policy.Pick(req, cands)
		valid := false
		for _, c := range cands {
			if c == pick {
				valid = true
				break
			}
		}
		if !valid {
			err := errors.New("scheduling policy returned an ineligible candidate")
			s.fail(req, err)
			return nil, err
		}

		var prefill int64
		if req.PrefillCost != nil {
			prefill = req.PrefillCost(pick.CachedTokens)
		}
		var kv int64
		if req.Role == registry.RoleDecode || req.Role == registry.RoleUnified {
			kv = req.KVReserve
		}
		res, err := pick.Snap.Worker.TryReserve(ctx, kv, prefill, reservationLimits(req, &pick.Snap), req.OnFree)
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

// These are bounds on the totals BEFORE adding this claim. The state/drain/
// health revision must also still match the snapshot used for filtering.
func reservationLimits(req *Request, s *registry.Snapshot) registry.Limits {
	l := registry.Unlimited()
	l.Revision = s.Revision
	if s.MaxRunning > 0 {
		l.MaxInflight = int64(s.MaxRunning) - 1
	}
	if (req.Role == registry.RoleDecode || req.Role == registry.RoleUnified) && s.KVCapacity > 0 {
		l.MaxReservedKV = s.KVCapacity - req.KVReserve
		if s.HasState {
			reported := s.ReportedKVUsed()
			l.MaxPendingKV = s.KVCapacity - reported - req.KVReserve
		}
	}
	if s.MaxPrefillTokens > 0 && (req.Role == registry.RolePrefill || req.Role == registry.RoleUnified || req.RequireNormal) {
		l.MaxPrefill = s.MaxPrefillTokens - 1
	}
	return l
}

func (s *Scheduler) fail(req *Request, err error) {
	if s.cfg.Observer != nil {
		s.cfg.Observer.Failed(req, s.cfg.Policy.Name(), err)
	}
}

// hasCapacity applies the role's admission limits: running requests, free
// KV for the reservation, and the prefill queue.
func hasCapacity(req *Request, c *Candidate) bool {
	s := &c.Snap
	if s.MaxRunning > 0 && s.Running() >= int64(s.MaxRunning) {
		return false
	}
	switch req.Role {
	case registry.RoleDecode, registry.RoleUnified:
		if s.KVCapacity > 0 && s.KVFree() < req.KVReserve {
			return false
		}
		if req.Role == registry.RoleUnified || req.RequireNormal {
			if s.MaxPrefillTokens > 0 && s.QueuedPrefillTokens() >= s.MaxPrefillTokens {
				return false
			}
		}
	case registry.RolePrefill:
		if s.MaxPrefillTokens > 0 && s.QueuedPrefillTokens() >= s.MaxPrefillTokens {
			return false
		}
	}
	return true
}

// computeLoad fills Load (role unit) and NormLoad ([0,1]).
//
//	prefill:        queued prefill tokens
//	decode/unified: running requests (KV utilization folds into NormLoad)
//	encode:         running requests
func computeLoad(role registry.Role, cands []*Candidate) {
	maxLoad := 0.0
	for _, c := range cands {
		s := &c.Snap
		switch role {
		case registry.RolePrefill:
			c.Load = float64(s.QueuedPrefillTokens())
		default:
			c.Load = float64(s.Running())
		}
		maxLoad = math.Max(maxLoad, c.Load)
	}
	for _, c := range cands {
		s := &c.Snap
		norm := 0.0
		switch {
		case role == registry.RolePrefill && s.MaxPrefillTokens > 0:
			norm = c.Load / float64(s.MaxPrefillTokens)
		case role != registry.RolePrefill && s.MaxRunning > 0:
			norm = c.Load / float64(s.MaxRunning)
		case maxLoad > 0:
			norm = c.Load / maxLoad
		}
		if role == registry.RoleDecode || role == registry.RoleUnified {
			norm = math.Max(norm, s.KVUtil())
		}
		c.NormLoad = math.Min(norm, 1)
	}
}

// pickMin returns the candidate minimizing key; ties are broken randomly.
func pickMin(cands []*Candidate, key func(*Candidate) float64) *Candidate {
	var best []*Candidate
	bestK := math.Inf(1)
	for _, c := range cands {
		k := key(c)
		switch {
		case k < bestK:
			bestK = k
			best = append(best[:0], c)
		case k == bestK:
			best = append(best, c)
		}
	}
	return best[rand.IntN(len(best))]
}

// UncachedCost returns the attention-weighted prefill cost of a prompt with
// cached tokens already in KV: w = u * (1 + beta*(cached + u/2)), u the
// uncached tokens. beta = 0 counts plain uncached tokens.
func UncachedCost(promptTokens, cached int, beta float64) int64 {
	u := promptTokens - cached
	if u <= 0 {
		return 0
	}
	if beta <= 0 {
		return int64(u)
	}
	return int64(float64(u) * (1 + beta*(float64(cached)+float64(u)/2)))
}
