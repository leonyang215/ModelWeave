// Package profile assembles the in-tree policy registry. A gateway builds one
// configured Policy instance for each model/role; the runtime knows no built-ins.
package profile

import (
	"fmt"

	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework/runtime"
	_ "github.com/leonyang215/ModelWeave/internal/scheduler/plugins"
	"github.com/leonyang215/ModelWeave/internal/scheduler/policy"
)

func init() {
	mustRegister := func(name string, factory framework.PolicyFactory) {
		if err := runtime.RegisterPolicy(name, factory); err != nil {
			panic(err)
		}
	}
	mustRegister("weighted", func(pc framework.PolicyConfig, _ float64) (framework.Policy, error) {
		if pc.Scorers != nil && (pc.CacheWeight != 0 || pc.LoadWeight != 0 || pc.AffinityWeight != 0) {
			return nil, fmt.Errorf("use scorers or legacy cache/load/affinity weights, not both")
		}
		configs := pc.Scorers
		if configs == nil && (pc.CacheWeight != 0 || pc.LoadWeight != 0 || pc.AffinityWeight != 0) {
			configs = []framework.ScorerConfig{{Name: "prefix-cache", Weight: pc.CacheWeight}, {Name: "load", Weight: pc.LoadWeight}, {Name: "affinity", Weight: pc.AffinityWeight}}
		}
		return runtime.BuildScored(configs)
	})
	for _, name := range []string{"prefix-cache", "load"} {
		mustRegister(name, func(pc framework.PolicyConfig, _ float64) (framework.Policy, error) {
			if pc.Scorers != nil {
				return nil, fmt.Errorf("single-scorer policy does not compose scorers; use weighted")
			}
			return runtime.BuildScoredNamed(name, []framework.ScorerConfig{{Name: name, Weight: 1}})
		})
	}
	mustRegister("cache_aware", func(pc framework.PolicyConfig, defaultAbs float64) (framework.Policy, error) {
		if pc.Scorers != nil {
			return nil, fmt.Errorf("cache_aware does not compose scorers; use weighted")
		}
		p := &policy.CacheAware{CacheThreshold: pc.CacheThreshold, BalanceAbs: pc.BalanceAbs, BalanceRel: pc.BalanceRel, LoadPenalty: pc.LoadPenalty}
		if p.CacheThreshold <= 0 {
			p.CacheThreshold = .3
		}
		if p.BalanceAbs <= 0 {
			p.BalanceAbs = defaultAbs
		}
		if p.BalanceRel <= 0 {
			p.BalanceRel = 1.5
		}
		if p.LoadPenalty <= 0 {
			p.LoadPenalty = .1
		}
		return p, nil
	})
	mustRegister("least_load", withoutScorers(func(framework.PolicyConfig, float64) (framework.Policy, error) { return policy.LeastLoad{}, nil }))
	mustRegister("power_of_two", withoutScorers(func(framework.PolicyConfig, float64) (framework.Policy, error) { return policy.PowerOfTwo{}, nil }))
	mustRegister("round_robin", withoutScorers(func(framework.PolicyConfig, float64) (framework.Policy, error) { return &policy.RoundRobin{}, nil }))
	mustRegister("random", withoutScorers(func(framework.PolicyConfig, float64) (framework.Policy, error) { return policy.Random{}, nil }))
	mustRegister("session_affinity", func(pc framework.PolicyConfig, abs float64) (framework.Policy, error) {
		if pc.Scorers != nil && pc.Fallback == "" {
			return nil, fmt.Errorf("session_affinity.scorers requires a weighted fallback")
		}
		p := &policy.SessionAffinity{LoadFactor: pc.LoadFactor}
		if p.LoadFactor <= 0 {
			p.LoadFactor = .25
		}
		if pc.Fallback == "session_affinity" {
			return nil, fmt.Errorf("session_affinity cannot fall back to itself")
		}
		if pc.Fallback != "" {
			pc.Name, pc.Fallback = pc.Fallback, ""
			fallback, err := runtime.BuildPolicy(pc, abs)
			if err != nil {
				return nil, err
			}
			p.Fallback = fallback
		}
		return p, nil
	})
}

func withoutScorers(factory framework.PolicyFactory) framework.PolicyFactory {
	return func(pc framework.PolicyConfig, abs float64) (framework.Policy, error) {
		if pc.Scorers != nil {
			return nil, fmt.Errorf("policy does not compose scorers; use weighted")
		}
		return factory(pc, abs)
	}
}
