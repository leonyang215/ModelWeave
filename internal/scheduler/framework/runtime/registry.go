package runtime

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
	scoring "github.com/leonyang215/ModelWeave/internal/scheduling"
)

var plugins = struct {
	sync.RWMutex
	policies map[string]framework.PolicyFactory
	scorers  map[string]framework.ScorerFactory
}{policies: make(map[string]framework.PolicyFactory), scorers: make(map[string]framework.ScorerFactory)}

func RegisterPolicy(name string, factory framework.PolicyFactory) error {
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

func RegisterScorer(name string, factory framework.ScorerFactory) error {
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
	for name := range plugins.policies {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func RegisteredScorers() []string {
	catalog := ScorerCatalog()
	names := make([]string, len(catalog))
	for i, descriptor := range catalog {
		names[i] = descriptor.Name
	}
	return names
}

// Discovery is metadata-only, including for parameterized SDK extensions.
func ScorerCatalog() []scoring.Descriptor {
	out := scoring.Catalog()
	plugins.RLock()
	defer plugins.RUnlock()
	for name := range plugins.scorers {
		out = append(out, scoring.Descriptor{Name: name, Version: "legacy", Description: "In-module scheduling extension.", Parameters: []scoring.Parameter{}})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func BuildPolicy(config framework.PolicyConfig, defaultAbs float64) (framework.Policy, error) {
	if config.Name == "" {
		config.Name = "weighted"
	}
	plugins.RLock()
	factory := plugins.policies[config.Name]
	plugins.RUnlock()
	if factory == nil {
		return nil, fmt.Errorf("unknown scheduling policy %q (registered: %s)", config.Name, strings.Join(RegisteredPolicies(), ", "))
	}
	policy, err := factory(config, defaultAbs)
	if err != nil {
		return nil, fmt.Errorf("policy %q: %w", config.Name, err)
	}
	if policy == nil {
		return nil, fmt.Errorf("policy %q: factory returned nil", config.Name)
	}
	return policy, nil
}
