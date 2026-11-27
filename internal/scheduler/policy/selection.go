// Package policy implements standalone selection algorithms. Registration and
// configuration assembly live in profile; cycle execution lives in runtime.
package policy

import (
	"math"
	"math/rand/v2"
	"sort"
	"sync/atomic"

	"github.com/zeebo/xxh3"

	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
)

type Request = framework.Request
type Candidate = framework.Candidate
type Policy = framework.Policy
type ScorerConfig = framework.ScorerConfig

func pickMin(c []*Candidate, key func(*Candidate) float64) *Candidate {
	return framework.PickMin(c, key)
}

// CacheAware is the legacy threshold policy (modeled on the SGLang router's
// cache-aware policy, extended with KV/queue load). When load is balanced it
// routes to the worker with the longest cached prefix if that prefix covers
// at least CacheThreshold of the prompt; otherwise, and whenever load is
// imbalanced, it routes to the least loaded worker. Routing a miss to the
// least loaded worker seeds that worker's cache, so later requests sharing
// the prefix follow it.
type CacheAware struct {
	CacheThreshold float64 // minimum cached fraction to prefer cache (default 0.3)
	// Load is imbalanced when max-min > BalanceAbs and max > BalanceRel*min,
	// in the role's load unit (running requests, or queued prefill tokens).
	BalanceAbs float64
	BalanceRel float64
	// LoadPenalty trades cache for load among cache hits: a candidate's
	// effective ratio is CacheRatio - LoadPenalty*NormLoad (default 0.1).
	LoadPenalty float64
}

func (p *CacheAware) Name() string    { return "cache_aware" }
func (p *CacheAware) UsesCache() bool { return true }

func (p *CacheAware) Pick(_ *Request, cands []*Candidate) (*Candidate, string) {
	if len(cands) == 1 {
		if cands[0].CacheRatio > 0 {
			return cands[0], "only_candidate_cache_hit"
		}
		return cands[0], "only_candidate"
	}
	minL, maxL := math.Inf(1), math.Inf(-1)
	for _, c := range cands {
		minL = math.Min(minL, c.Load)
		maxL = math.Max(maxL, c.Load)
	}
	if maxL-minL > p.BalanceAbs && maxL > p.BalanceRel*minL {
		return pickMin(cands, func(c *Candidate) float64 { return c.Load }), "rebalance"
	}
	thr := p.CacheThreshold
	var best *Candidate
	bestEff := math.Inf(-1)
	for _, c := range cands {
		if c.CacheRatio <= 0 {
			continue
		}
		eff := c.CacheRatio - p.LoadPenalty*c.NormLoad
		if eff > bestEff || (eff == bestEff && c.Load < best.Load) {
			best, bestEff = c, eff
		}
	}
	if best != nil && best.CacheRatio >= thr {
		return best, "cache_hit"
	}
	return pickMin(cands, func(c *Candidate) float64 { return c.Load }), "least_load"
}

// Weighted scores every candidate as
//
//	CacheWeight*CacheRatio - LoadWeight*NormLoad + AffinityWeight*[affinity match]
//
// and picks the maximum (llm-d style scorer composition). Affinity uses
// rendezvous hashing of the session key, so a session sticks to one worker as
// long as its score stays competitive.
type Weighted struct {
	CacheWeight    float64
	LoadWeight     float64
	AffinityWeight float64
}

func (p *Weighted) Name() string    { return "weighted" }
func (p *Weighted) UsesCache() bool { return p.CacheWeight > 0 }

func (p *Weighted) Pick(req *Request, cands []*Candidate) (*Candidate, string) {
	var aff *Candidate
	if p.AffinityWeight > 0 && req.SessionKey != "" {
		aff = rendezvous(req.SessionKey, cands)[0]
	}
	for _, c := range cands {
		c.Score = p.CacheWeight*c.CacheRatio - p.LoadWeight*c.NormLoad
		if c == aff {
			c.Score += p.AffinityWeight
		}
	}
	best := pickMin(cands, func(c *Candidate) float64 { return -c.Score })
	switch {
	case best == aff && aff != nil:
		return best, "affinity"
	case best.CacheRatio > 0:
		return best, "score_cache"
	default:
		return best, "score_load"
	}
}

// LeastLoad picks the least loaded worker (KV utilization breaks ties).
type LeastLoad struct{}

func (LeastLoad) Name() string    { return "least_load" }
func (LeastLoad) UsesCache() bool { return false }
func (LeastLoad) Pick(_ *Request, cands []*Candidate) (*Candidate, string) {
	return pickMin(cands, func(c *Candidate) float64 { return c.Load + c.NormLoad*1e-3 }), "least_load"
}

// PowerOfTwo samples two candidates and keeps the less loaded one.
type PowerOfTwo struct{}

func (PowerOfTwo) Name() string    { return "power_of_two" }
func (PowerOfTwo) UsesCache() bool { return false }
func (PowerOfTwo) Pick(_ *Request, cands []*Candidate) (*Candidate, string) {
	if len(cands) == 1 {
		return cands[0], "only_candidate"
	}
	i := rand.IntN(len(cands))
	j := rand.IntN(len(cands) - 1)
	if j >= i {
		j++
	}
	a, b := cands[i], cands[j]
	if b.Load < a.Load {
		a = b
	}
	return a, "power_of_two"
}

// RoundRobin rotates over the eligible candidates.
type RoundRobin struct{ next atomic.Uint64 }

func (*RoundRobin) Name() string    { return "round_robin" }
func (*RoundRobin) UsesCache() bool { return false }
func (p *RoundRobin) Pick(_ *Request, cands []*Candidate) (*Candidate, string) {
	n := p.next.Add(1) - 1
	return cands[n%uint64(len(cands))], "round_robin"
}

// Random picks uniformly.
type Random struct{}

func (Random) Name() string    { return "random" }
func (Random) UsesCache() bool { return false }
func (Random) Pick(_ *Request, cands []*Candidate) (*Candidate, string) {
	return cands[rand.IntN(len(cands))], "random"
}

// SessionAffinity routes a session to a stable worker with rendezvous
// hashing, bounded by load: when the preferred worker is loaded beyond
// (1+LoadFactor) times the mean load, the next worker in rendezvous order is
// used. Requests without a session key use Fallback.
type SessionAffinity struct {
	LoadFactor float64 // default 0.25
	Fallback   Policy
}

func (p *SessionAffinity) Name() string { return "session_affinity" }
func (p *SessionAffinity) UsesCache() bool {
	return p.Fallback != nil && p.Fallback.UsesCache()
}

func (p *SessionAffinity) Scorers() []ScorerConfig {
	if fallback, ok := p.Fallback.(interface{ Scorers() []ScorerConfig }); ok {
		return fallback.Scorers()
	}
	return nil
}

func (p *SessionAffinity) Pick(req *Request, cands []*Candidate) (*Candidate, string) {
	if req.SessionKey == "" {
		if p.Fallback == nil {
			return pickMin(cands, func(c *Candidate) float64 { return c.Load }), "least_load"
		}
		c, r := p.Fallback.Pick(req, cands)
		return c, "no_session_" + r
	}
	order := rendezvous(req.SessionKey, cands)
	mean := 0.0
	for _, c := range cands {
		mean += c.Load
	}
	mean /= float64(len(cands))
	bound := (1+p.LoadFactor)*mean + 1
	for i, c := range order {
		if c.Load <= bound {
			if i == 0 {
				return c, "affinity"
			}
			return c, "affinity_spill"
		}
	}
	return order[0], "affinity_overloaded"
}

// rendezvous orders candidates by highest random weight for key.
func rendezvous(key string, cands []*Candidate) []*Candidate {
	type kv struct {
		c *Candidate
		w uint64
	}
	ws := make([]kv, len(cands))
	for i, c := range cands {
		ws[i] = kv{c, xxh3.HashString(key + "\x00" + c.Snap.Spec.ID)}
	}
	sort.Slice(ws, func(i, j int) bool { return ws[i].w > ws[j].w })
	out := make([]*Candidate, len(ws))
	for i := range ws {
		out[i] = ws[i].c
	}
	return out
}
