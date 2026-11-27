// Package framework defines the internal scheduling-cycle contracts. Like
// kube-scheduler, it separates extension points and data from their execution
// runtime and in-tree registration. It does not depend on Kubernetes APIs.
package framework

import (
	"math"
	"math/rand/v2"
	"time"

	"github.com/leonyang215/ModelWeave/internal/registry"
	scoring "github.com/leonyang215/ModelWeave/internal/scheduling"
)

// Request describes one stage of an inference request, not a Kubernetes Pod.
type Request struct {
	Model         string
	Role          registry.Role
	RequireNormal bool
	PromptTokens  int
	Blocks        []uint64
	BlockSize     int
	KVReserve     int64
	PrefillCost   func(cachedTokens int) int64
	SessionKey    string
	Exclude       map[string]bool
	Pin           string
	OnFree        func()
}

type Candidate struct {
	Snap         registry.Snapshot
	CachedBlocks int
	CachedTokens int
	CacheRatio   float64
	Load         float64
	NormLoad     float64
	Score        float64
	Scores       map[string]float64
}

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

// Policy selects from a non-empty, already filtered candidate list. The
// runtime validates its selection before the core can reserve capacity.
type Policy interface {
	Name() string
	UsesCache() bool
	Pick(*Request, []*Candidate) (*Candidate, string)
}

type PolicyConfig struct {
	Scorers        []ScorerConfig `yaml:"scorers,omitempty" json:"scorers,omitempty"`
	Name           string         `yaml:"policy" json:"policy"`
	CacheThreshold float64        `yaml:"cache_threshold" json:"cache_threshold"`
	BalanceAbs     float64        `yaml:"balance_abs_threshold" json:"balance_abs_threshold"`
	BalanceRel     float64        `yaml:"balance_rel_threshold" json:"balance_rel_threshold"`
	LoadPenalty    float64        `yaml:"load_penalty" json:"load_penalty"`
	CacheWeight    float64        `yaml:"cache_weight" json:"cache_weight"`
	LoadWeight     float64        `yaml:"load_weight" json:"load_weight"`
	AffinityWeight float64        `yaml:"affinity_weight" json:"affinity_weight"`
	LoadFactor     float64        `yaml:"load_factor" json:"load_factor"`
	Fallback       string         `yaml:"fallback" json:"fallback"`
}

type PolicyFactory func(PolicyConfig, float64) (Policy, error)
type ScorerConfig = scoring.Config

// Scorer is the original extension contract. New in-repository plugins use
// internal/scheduling.Scorer, which has no registry/reservation handles.
type Scorer interface {
	Name() string
	UsesCache() bool
	Score(*Request, *Candidate, []*Candidate) float64
}

type ScorerFactory func(ScorerConfig) (Scorer, error)

type Observer interface {
	Scheduled(*Request, *Decision)
	Failed(*Request, string, error)
	CacheLookup(string, time.Duration, error)
}

// PickMin is shared by independent policies and weighted scoring. Randomized
// ties retain the existing routing behaviour; callers supply a non-empty list.
func PickMin(candidates []*Candidate, key func(*Candidate) float64) *Candidate {
	var best []*Candidate
	bestKey := math.Inf(1)
	for _, candidate := range candidates {
		k := key(candidate)
		switch {
		case k < bestKey:
			bestKey = k
			best = append(best[:0], candidate)
		case k == bestKey:
			best = append(best, candidate)
		}
	}
	return best[rand.IntN(len(best))]
}
