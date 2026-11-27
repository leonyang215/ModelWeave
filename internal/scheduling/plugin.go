// Package scheduling defines the repository-private contract and registry for
// composable routing scorers. It is not an externally importable Go SDK.
// Plugins see scheduling data, never registry/store or reservation handles.
// Implementations are trusted, compiled-in Go code, not sandboxed scripts.
package scheduling

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
)

// Request is the immutable request context visible to scoring plugins.
type Request struct {
	Model        string
	Role         string
	PromptTokens int
	SessionKey   string
}

// Candidate contains only a scheduling snapshot. Plugins must not mutate it
// or the candidates slice. Labels are detached from the live registry.
type Candidate struct {
	ID           string
	Model        string
	Role         string
	Labels       map[string]string
	CachedBlocks int
	CachedTokens int
	CacheRatio   float64
	Load         float64
	NormLoad     float64
	Inflight     int64
	KVUsed       int64
	KVCapacity   int64
}

// Scorer returns a preference in [0,1], higher is better. The gateway clamps
// invalid scores and applies configured weights; plugins never admit requests.
// A factory creates a separate instance for every model/role composition.
type Scorer interface {
	Name() string
	UsesCache() bool
	Score(*Request, *Candidate, []*Candidate) float64
}

type Config struct {
	Name   string         `yaml:"name" json:"name"`
	Weight float64        `yaml:"weight" json:"weight"`
	Params map[string]any `yaml:"params,omitempty" json:"params,omitempty"`
}

type Factory func(Config) (Scorer, error)

type Parameter struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Required    bool   `json:"required"`
	Description string `json:"description"`
}

// Descriptor is the discoverable plugin contract shown by the admin catalog.
// Parameters document the schema; each factory must validate its own params.
type Descriptor struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Version     string      `json:"version"`
	Source      string      `json:"source,omitempty"`
	Parameters  []Parameter `json:"parameters"`
}

type registration struct {
	descriptor Descriptor
	factory    Factory
}

// Registry permits independent catalogs in tests/embedders. The gateway and
// init-time plugin registration use the process-wide registry.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]registration
}

func NewRegistry() *Registry { return &Registry{entries: make(map[string]registration)} }

var defaultRegistry = NewRegistry()

func Register(d Descriptor, f Factory) error { return defaultRegistry.Register(d, f) }
func Catalog() []Descriptor                  { return defaultRegistry.Catalog() }
func Names() []string                        { return defaultRegistry.Names() }
func Build(c Config) (Scorer, error)         { return defaultRegistry.Build(c) }
func Lookup(name string) (Descriptor, bool)  { return defaultRegistry.Lookup(name) }

func (r *Registry) Lookup(name string) (Descriptor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	entry, ok := r.entries[name]
	d := entry.descriptor
	d.Parameters = append([]Parameter{}, d.Parameters...)
	return d, ok
}

func (r *Registry) Register(d Descriptor, f Factory) error {
	if strings.TrimSpace(d.Name) == "" || strings.TrimSpace(d.Name) != d.Name || f == nil {
		return fmt.Errorf("scorer registration needs a name and factory")
	}
	if d.Description == "" || d.Version == "" {
		return fmt.Errorf("scorer %q: description and version are required", d.Name)
	}
	seen := make(map[string]bool)
	for _, p := range d.Parameters {
		if p.Name == "" || seen[p.Name] || p.Type == "" {
			return fmt.Errorf("scorer %q: parameters need unique names and types", d.Name)
		}
		seen[p.Name] = true
	}
	d.Parameters = append([]Parameter{}, d.Parameters...)
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.entries == nil {
		r.entries = make(map[string]registration)
	}
	if _, ok := r.entries[d.Name]; ok {
		return fmt.Errorf("scorer %q already registered", d.Name)
	}
	r.entries[d.Name] = registration{descriptor: d, factory: f}
	return nil
}

func (r *Registry) Catalog() []Descriptor {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Descriptor, 0, len(r.entries))
	for _, entry := range r.entries {
		d := entry.descriptor
		d.Parameters = append([]Parameter{}, d.Parameters...)
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (r *Registry) Names() []string {
	catalog := r.Catalog()
	names := make([]string, len(catalog))
	for i, d := range catalog {
		names[i] = d.Name
	}
	return names
}

func (r *Registry) Build(c Config) (Scorer, error) {
	r.mu.RLock()
	entry, ok := r.entries[c.Name]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("unknown scorer %q (registered: %s)", c.Name, strings.Join(r.Names(), ", "))
	}
	copy, err := c.Clone()
	if err != nil {
		return nil, fmt.Errorf("scorer %q: params must be JSON-compatible: %w", c.Name, err)
	}
	scorer, err := entry.factory(copy)
	if err != nil {
		return nil, fmt.Errorf("scorer %q: %w", c.Name, err)
	}
	if scorer == nil || isNil(scorer) {
		return nil, fmt.Errorf("scorer %q: factory returned nil", c.Name)
	}
	return scorer, nil
}

func isNil(v any) bool {
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Func, reflect.Map, reflect.Slice, reflect.Chan:
		return rv.IsNil()
	}
	return false
}

// Clone detaches nested parameter maps/slices. The JSON value representation
// is also the admin wire contract; use DecodeParams rather than type-asserting
// YAML-specific numeric types inside a plugin factory.
func (c Config) Clone() (Config, error) {
	if c.Params == nil {
		return c, nil
	}
	b, err := json.Marshal(c.Params)
	if err != nil {
		return Config{}, err
	}
	var params map[string]any
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	if err := dec.Decode(&params); err != nil {
		return Config{}, err
	}
	c.Params = params
	return c, nil
}

// DecodeParams decodes a typed parameter struct and rejects unknown fields.
// Plugins additionally validate required fields and value ranges themselves.
func DecodeParams(params map[string]any, out any) error {
	if params == nil {
		params = map[string]any{}
	}
	b, err := json.Marshal(params)
	if err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	return dec.Decode(out)
}
