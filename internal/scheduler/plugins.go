package scheduler

import (
	"fmt"
	"maps"
	"math"
	"sort"
	"strings"
	"sync"

	scoring "github.com/leonyang215/ModelWeave/pkg/scheduling"
	_ "github.com/leonyang215/ModelWeave/plugins/scheduling"
	"github.com/leonyang215/ModelWeave/plugins/scheduling/load"
	"github.com/leonyang215/ModelWeave/plugins/scheduling/prefixcache"
)

// PolicyFactory creates a policy per model/role. Plugins register at startup
// (usually in init); adding one never requires editing BuildPolicy.
type PolicyFactory func(PolicyConfig, float64) (Policy, error)

// Scorer is the legacy in-module extension interface. New plugins should use
// pkg/scheduling.Scorer, which exposes no registry/reservation handles. Both
// forms can be composed without changing existing in-module integrations.
type Scorer interface {
	Name() string
	UsesCache() bool
	Score(req *Request, candidate *Candidate, candidates []*Candidate) float64
}

type ScorerFactory func(ScorerConfig) (Scorer, error)

// ScorerConfig configures one term of the weighted score. An explicit zero
// weight disables the term; an omitted list uses prefix-cache=1 and load=1.
type ScorerConfig = scoring.Config

var plugins = struct {
	sync.RWMutex
	policies map[string]PolicyFactory
	scorers  map[string]ScorerFactory
}{policies: make(map[string]PolicyFactory), scorers: make(map[string]ScorerFactory)}

// RegisterPolicy registers a named factory, rejecting duplicates and nil
// factories rather than silently replacing a built-in.
func RegisterPolicy(name string, factory PolicyFactory) error {
	plugins.Lock()
	defer plugins.Unlock()
	if strings.TrimSpace(name) == "" || name != strings.TrimSpace(name) || factory == nil {
		return fmt.Errorf("policy registration needs a name and factory")
	}
	if _, ok := plugins.policies[name]; ok {
		return fmt.Errorf("policy %q already registered", name)
	}
	plugins.policies[name] = factory
	return nil
}

func RegisterScorer(name string, factory ScorerFactory) error {
	plugins.Lock()
	defer plugins.Unlock()
	if strings.TrimSpace(name) == "" || name != strings.TrimSpace(name) || factory == nil {
		return fmt.Errorf("scorer registration needs a name and factory")
	}
	if _, ok := plugins.scorers[name]; ok {
		return fmt.Errorf("scorer %q already registered", name)
	}
	if _, ok := scoring.Lookup(name); ok {
		return fmt.Errorf("scorer %q already registered", name)
	}
	plugins.scorers[name] = factory
	return nil
}

func RegisteredPolicies() []string {
	plugins.RLock()
	defer plugins.RUnlock()
	names := make([]string, 0, len(plugins.policies))
	for n := range plugins.policies {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func RegisteredScorers() []string {
	catalog := ScorerCatalog()
	names := make([]string, len(catalog))
	for i, d := range catalog {
		names[i] = d.Name
	}
	return names
}

// ScorerCatalog lists the compiled SDK plugins plus legacy registrations.
// It is metadata only; discovery never instantiates parameterized plugins.
func ScorerCatalog() []scoring.Descriptor {
	out := scoring.Catalog()
	plugins.RLock()
	defer plugins.RUnlock()
	for n := range plugins.scorers {
		out = append(out, scoring.Descriptor{Name: n, Version: "legacy", Description: "In-module scheduling extension.", Parameters: []scoring.Parameter{}})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

type scorerTerm struct {
	scorer Scorer
	plugin scoring.Scorer
	config ScorerConfig
}

func (t scorerTerm) usesCache() bool {
	if t.plugin != nil {
		return t.plugin.UsesCache()
	}
	return t.scorer.UsesCache()
}

// Scored ranks candidates by sum(weight * plugin score). Scores are exposed
// on the decision so routing decisions can be explained without re-scoring.
type Scored struct {
	terms []scorerTerm
	name  string
}

func (p *Scored) Name() string { return p.name }
func (p *Scored) UsesCache() bool {
	for _, t := range p.terms {
		if t.config.Weight > 0 && t.usesCache() {
			return true
		}
	}
	return false
}

func (p *Scored) Scorers() []ScorerConfig {
	out := make([]ScorerConfig, 0, len(p.terms))
	for _, t := range p.terms {
		// Every config was JSON-validated and detached on construction. Never
		// expose its private nested maps to API callers or plugin factories.
		copy, _ := t.config.Clone()
		out = append(out, copy)
	}
	return out
}

func (p *Scored) Pick(req *Request, cands []*Candidate) (*Candidate, string) {
	var views []*scoring.Candidate
	for _, term := range p.terms {
		if term.plugin != nil && term.config.Weight > 0 {
			views = make([]*scoring.Candidate, len(cands))
			values := make([]scoring.Candidate, len(cands))
			for i, c := range cands {
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
	for i, c := range cands {
		c.Score, c.Scores = 0, make(map[string]float64, len(p.terms))
		for _, term := range p.terms {
			if term.config.Weight == 0 {
				continue
			}
			var v float64
			if term.plugin != nil {
				v = term.plugin.Score(viewReq, views[i], views)
			} else {
				v = term.scorer.Score(req, c, cands)
			}
			// A faulty plugin cannot poison the ranking with NaN/Inf.
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = 0
			}
			v = max(0, min(1, v))
			c.Scores[term.config.Name] = v
			c.Score += term.config.Weight * v
		}
	}
	best := pickMin(cands, func(c *Candidate) float64 { return -c.Score })
	if best.Scores["affinity"] > 0 {
		return best, "affinity"
	}
	if best.Scores["prefix-cache"] > 0 {
		return best, "score_cache"
	}
	for name, v := range best.Scores {
		if name != "load" && v > 0 {
			return best, "score_weighted"
		}
	}
	if _, ok := best.Scores["load"]; !ok {
		return best, "score_weighted"
	}
	return best, "score_load"
}

// BuildScored creates a composition from the registered scorer factories.
func BuildScored(configs []ScorerConfig) (*Scored, error) {
	if len(configs) == 0 {
		configs = []ScorerConfig{{Name: "prefix-cache", Weight: 1}, {Name: "load", Weight: 1}}
	}
	p := &Scored{name: "weighted"}
	seen, sum := make(map[string]bool), 0.0
	for _, cfg := range configs {
		if seen[cfg.Name] {
			return nil, fmt.Errorf("duplicate scorer %q", cfg.Name)
		}
		seen[cfg.Name] = true
		if cfg.Weight < 0 || math.IsNaN(cfg.Weight) || math.IsInf(cfg.Weight, 0) {
			return nil, fmt.Errorf("scorer %q: weight must be finite and non-negative", cfg.Name)
		}
		plugins.RLock()
		f := plugins.scorers[cfg.Name]
		plugins.RUnlock()
		_, sdk := scoring.Lookup(cfg.Name)
		if f != nil && sdk {
			return nil, fmt.Errorf("scorer %q registered through both SDK and legacy interfaces", cfg.Name)
		}
		if f == nil && !sdk {
			return nil, fmt.Errorf("unknown scorer %q (registered: %s)", cfg.Name, strings.Join(RegisteredScorers(), ", "))
		}
		copy, err := cfg.Clone()
		if err != nil {
			return nil, fmt.Errorf("scorer %q: invalid params: %w", cfg.Name, err)
		}
		term := scorerTerm{config: copy}
		if sdk {
			term.plugin, err = scoring.Build(cfg)
		} else {
			factoryConfig, _ := copy.Clone()
			term.scorer, err = f(factoryConfig)
			if err == nil && term.scorer == nil {
				err = fmt.Errorf("factory returned nil")
			}
		}
		if err != nil {
			return nil, fmt.Errorf("scorer %q: %w", cfg.Name, err)
		}
		p.terms = append(p.terms, term)
		sum += cfg.Weight
	}
	if sum <= 0 || math.IsInf(sum, 0) {
		return nil, fmt.Errorf("scorer weights must have a finite, positive sum")
	}
	return p, nil
}

// PrefixCacheScorer rewards the fraction of prompt tokens already resident.
type PrefixCacheScorer struct{}

func (PrefixCacheScorer) Name() string    { return "prefix-cache" }
func (PrefixCacheScorer) UsesCache() bool { return true }
func (PrefixCacheScorer) Score(_ *Request, c *Candidate, _ []*Candidate) float64 {
	return (prefixcache.Scorer{}).Score(nil, &scoring.Candidate{CacheRatio: c.CacheRatio}, nil)
}

// LoadScorer rewards spare capacity. NormLoad incorporates prefill backlog,
// running/waiting requests and KV utilization according to the instance role.
type LoadScorer struct{}

func (LoadScorer) Name() string    { return "load" }
func (LoadScorer) UsesCache() bool { return false }
func (LoadScorer) Score(_ *Request, c *Candidate, _ []*Candidate) float64 {
	return (load.Scorer{}).Score(nil, &scoring.Candidate{NormLoad: c.NormLoad}, nil)
}

func init() {
	_ = RegisterPolicy("weighted", func(pc PolicyConfig, _ float64) (Policy, error) {
		if pc.Scorers != nil && (pc.CacheWeight != 0 || pc.LoadWeight != 0 || pc.AffinityWeight != 0) {
			return nil, fmt.Errorf("use scorers or legacy cache/load/affinity weights, not both")
		}
		configs := pc.Scorers
		if configs == nil && (pc.CacheWeight != 0 || pc.LoadWeight != 0 || pc.AffinityWeight != 0) {
			configs = []ScorerConfig{{Name: "prefix-cache", Weight: pc.CacheWeight}, {Name: "load", Weight: pc.LoadWeight}, {Name: "affinity", Weight: pc.AffinityWeight}}
		}
		return BuildScored(configs)
	})
	for _, name := range []string{"prefix-cache", "load"} {
		_ = RegisterPolicy(name, func(pc PolicyConfig, _ float64) (Policy, error) {
			if pc.Scorers != nil {
				return nil, fmt.Errorf("single-scorer policy does not compose scorers; use weighted")
			}
			p, err := BuildScored([]ScorerConfig{{Name: name, Weight: 1}})
			if p != nil {
				p.name = name
			}
			return p, err
		})
	}
	_ = RegisterPolicy("cache_aware", buildCacheAware)
	_ = RegisterPolicy("least_load", withoutScorers(func(PolicyConfig, float64) (Policy, error) { return LeastLoad{}, nil }))
	_ = RegisterPolicy("power_of_two", withoutScorers(func(PolicyConfig, float64) (Policy, error) { return PowerOfTwo{}, nil }))
	_ = RegisterPolicy("round_robin", withoutScorers(func(PolicyConfig, float64) (Policy, error) { return &RoundRobin{}, nil }))
	_ = RegisterPolicy("random", withoutScorers(func(PolicyConfig, float64) (Policy, error) { return Random{}, nil }))
	_ = RegisterPolicy("session_affinity", func(pc PolicyConfig, abs float64) (Policy, error) {
		if pc.Scorers != nil && pc.Fallback == "" {
			return nil, fmt.Errorf("session_affinity.scorers requires a weighted fallback")
		}
		p := &SessionAffinity{LoadFactor: pc.LoadFactor}
		if p.LoadFactor <= 0 {
			p.LoadFactor = .25
		}
		if pc.Fallback == "session_affinity" {
			return nil, fmt.Errorf("session_affinity cannot fall back to itself")
		}
		if pc.Fallback != "" {
			pc.Name, pc.Fallback = pc.Fallback, ""
			f, err := BuildPolicy(pc, abs)
			if err != nil {
				return nil, err
			}
			p.Fallback = f
		}
		return p, nil
	})
}

func withoutScorers(f PolicyFactory) PolicyFactory {
	return func(pc PolicyConfig, abs float64) (Policy, error) {
		if pc.Scorers != nil {
			return nil, fmt.Errorf("policy does not compose scorers; use weighted")
		}
		return f(pc, abs)
	}
}
