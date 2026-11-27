package runtime

import (
	"fmt"
	"maps"
	"math"
	"strings"

	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
	scoring "github.com/leonyang215/ModelWeave/internal/scheduling"
)

type scorerTerm struct {
	legacy framework.Scorer
	plugin scoring.Scorer
	config framework.ScorerConfig
}

func (t scorerTerm) usesCache() bool {
	if t.plugin != nil {
		return t.plugin.UsesCache()
	}
	return t.legacy.UsesCache()
}

// Scored executes a configured composition after filtering. It bridges the
// repository-private data-only contract and the legacy scorer interface.
type Scored struct {
	terms []scorerTerm
	name  string
}

func (p *Scored) Name() string { return p.name }
func (p *Scored) UsesCache() bool {
	for _, term := range p.terms {
		if term.config.Weight > 0 && term.usesCache() {
			return true
		}
	}
	return false
}

func (p *Scored) Scorers() []framework.ScorerConfig {
	out := make([]framework.ScorerConfig, 0, len(p.terms))
	for _, term := range p.terms {
		copy, _ := term.config.Clone() // validated and detached during construction
		out = append(out, copy)
	}
	return out
}

func (p *Scored) Pick(req *framework.Request, candidates []*framework.Candidate) (*framework.Candidate, string) {
	var views []*scoring.Candidate
	for _, term := range p.terms {
		if term.plugin != nil && term.config.Weight > 0 {
			views = make([]*scoring.Candidate, len(candidates))
			values := make([]scoring.Candidate, len(candidates))
			for i, c := range candidates {
				values[i] = scoring.Candidate{
					ID: c.Snap.Spec.ID, Model: c.Snap.Spec.Model, Role: string(c.Snap.Spec.Role), Labels: maps.Clone(c.Snap.Spec.Labels),
					CachedBlocks: c.CachedBlocks, CachedTokens: c.CachedTokens, CacheRatio: c.CacheRatio,
					Load: c.Load, NormLoad: c.NormLoad, Inflight: c.Snap.Inflight, KVUsed: c.Snap.EffectiveKVUsed, KVCapacity: c.Snap.KVCapacity,
				}
				views[i] = &values[i]
			}
			break
		}
	}
	viewReq := &scoring.Request{Model: req.Model, Role: string(req.Role), PromptTokens: req.PromptTokens, SessionKey: req.SessionKey}
	for i, c := range candidates {
		c.Score, c.Scores = 0, make(map[string]float64, len(p.terms))
		for _, term := range p.terms {
			if term.config.Weight == 0 {
				continue
			}
			var value float64
			if term.plugin != nil {
				value = term.plugin.Score(viewReq, views[i], views)
			} else {
				value = term.legacy.Score(req, c, candidates)
			}
			if math.IsNaN(value) || math.IsInf(value, 0) {
				value = 0
			}
			value = max(0, min(1, value))
			c.Scores[term.config.Name] = value
			c.Score += term.config.Weight * value
		}
	}
	best := framework.PickMin(candidates, func(c *framework.Candidate) float64 { return -c.Score })
	if best.Scores["affinity"] > 0 {
		return best, "affinity"
	}
	if best.Scores["prefix-cache"] > 0 {
		return best, "score_cache"
	}
	for name, value := range best.Scores {
		if name != "load" && value > 0 {
			return best, "score_weighted"
		}
	}
	if _, ok := best.Scores["load"]; !ok {
		return best, "score_weighted"
	}
	return best, "score_load"
}

func BuildScored(configs []framework.ScorerConfig) (*Scored, error) {
	return BuildScoredNamed("weighted", configs)
}

func BuildScoredNamed(name string, configs []framework.ScorerConfig) (*Scored, error) {
	if len(configs) == 0 {
		configs = []framework.ScorerConfig{{Name: "prefix-cache", Weight: 1}, {Name: "load", Weight: 1}}
	}
	policy := &Scored{name: name}
	seen, sum := make(map[string]bool), 0.0
	for _, config := range configs {
		if seen[config.Name] {
			return nil, fmt.Errorf("duplicate scorer %q", config.Name)
		}
		seen[config.Name] = true
		if config.Weight < 0 || math.IsNaN(config.Weight) || math.IsInf(config.Weight, 0) {
			return nil, fmt.Errorf("scorer %q: weight must be finite and non-negative", config.Name)
		}
		plugins.RLock()
		factory := plugins.scorers[config.Name]
		plugins.RUnlock()
		_, sdk := scoring.Lookup(config.Name)
		if factory != nil && sdk {
			return nil, fmt.Errorf("scorer %q registered through both SDK and legacy interfaces", config.Name)
		}
		if factory == nil && !sdk {
			return nil, fmt.Errorf("unknown scorer %q (registered: %s)", config.Name, strings.Join(RegisteredScorers(), ", "))
		}
		copy, err := config.Clone()
		if err != nil {
			return nil, fmt.Errorf("scorer %q: invalid params: %w", config.Name, err)
		}
		term := scorerTerm{config: copy}
		if sdk {
			term.plugin, err = scoring.Build(config)
		} else {
			factoryConfig, _ := copy.Clone()
			term.legacy, err = factory(factoryConfig)
			if err == nil && term.legacy == nil {
				err = fmt.Errorf("factory returned nil")
			}
		}
		if err != nil {
			return nil, fmt.Errorf("scorer %q: %w", config.Name, err)
		}
		policy.terms = append(policy.terms, term)
		sum += config.Weight
	}
	if sum <= 0 || math.IsInf(sum, 0) {
		return nil, fmt.Errorf("scorer weights must have a finite, positive sum")
	}
	return policy, nil
}
