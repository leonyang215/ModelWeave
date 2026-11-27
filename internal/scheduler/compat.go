package scheduler

import (
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework/runtime"
	"github.com/leonyang215/ModelWeave/internal/scheduler/plugins/load"
	"github.com/leonyang215/ModelWeave/internal/scheduler/plugins/prefixcache"
	"github.com/leonyang215/ModelWeave/internal/scheduler/policy"
	_ "github.com/leonyang215/ModelWeave/internal/scheduler/profile"
	scoring "github.com/leonyang215/ModelWeave/internal/scheduling"
)

// Aliases preserve existing gateway/config/in-module plugin integrations.
// New implementation code uses framework contracts and runtime execution.
type Request = framework.Request
type Candidate = framework.Candidate
type Decision = framework.Decision
type Policy = framework.Policy
type PolicyConfig = framework.PolicyConfig
type PolicyFactory = framework.PolicyFactory
type Observer = framework.Observer
type Scorer = framework.Scorer
type ScorerConfig = framework.ScorerConfig
type ScorerFactory = framework.ScorerFactory
type Scored = runtime.Scored
type CacheAware = policy.CacheAware
type Weighted = policy.Weighted
type LeastLoad = policy.LeastLoad
type PowerOfTwo = policy.PowerOfTwo
type RoundRobin = policy.RoundRobin
type Random = policy.Random
type SessionAffinity = policy.SessionAffinity

func BuildPolicy(pc PolicyConfig, abs float64) (Policy, error) { return runtime.BuildPolicy(pc, abs) }
func BuildScored(configs []ScorerConfig) (*Scored, error)      { return runtime.BuildScored(configs) }
func RegisterPolicy(name string, factory PolicyFactory) error {
	return runtime.RegisterPolicy(name, factory)
}
func RegisterScorer(name string, factory ScorerFactory) error {
	return runtime.RegisterScorer(name, factory)
}
func RegisteredPolicies() []string        { return runtime.RegisteredPolicies() }
func RegisteredScorers() []string         { return runtime.RegisteredScorers() }
func ScorerCatalog() []scoring.Descriptor { return runtime.ScorerCatalog() }

type PrefixCacheScorer struct{}

func (PrefixCacheScorer) Name() string    { return "prefix-cache" }
func (PrefixCacheScorer) UsesCache() bool { return true }
func (PrefixCacheScorer) Score(_ *Request, c *Candidate, _ []*Candidate) float64 {
	return (prefixcache.Scorer{}).Score(nil, &scoring.Candidate{CacheRatio: c.CacheRatio}, nil)
}

type LoadScorer struct{}

func (LoadScorer) Name() string    { return "load" }
func (LoadScorer) UsesCache() bool { return false }
func (LoadScorer) Score(_ *Request, c *Candidate, _ []*Candidate) float64 {
	return (load.Scorer{}).Score(nil, &scoring.Candidate{NormLoad: c.NormLoad}, nil)
}
