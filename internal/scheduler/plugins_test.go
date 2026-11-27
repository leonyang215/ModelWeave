package scheduler

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/registry"
)

func TestDefaultScoringPluginsAndWeights(t *testing.T) {
	p, err := BuildPolicy(PolicyConfig{}, 32)
	if err != nil {
		t.Fatal(err)
	}
	terms := p.(*Scored).Scorers()
	if p.Name() != "weighted" || !p.UsesCache() || len(terms) != 2 || terms[0].Name != "prefix-cache" || terms[0].Weight != 1 || terms[1].Name != "load" || terms[1].Weight != 1 {
		t.Fatalf("defaults: %s %+v", p.Name(), terms)
	}
	for _, name := range []string{"prefix-cache", "load"} {
		if _, err := BuildPolicy(PolicyConfig{Name: name}, 1); err != nil {
			t.Fatal(err)
		}
	}
}

func TestScoreWeightsChangeRoutingAndExplainDecision(t *testing.T) {
	a := &Candidate{Snap: registry.Snapshot{Spec: registry.Spec{ID: "cached"}}, CacheRatio: 1, NormLoad: .9}
	b := &Candidate{Snap: registry.Snapshot{Spec: registry.Spec{ID: "idle"}}, CacheRatio: 0, NormLoad: 0}
	for _, tc := range []struct {
		cache, load float64
		want        *Candidate
		usesCache   bool
	}{{2, 1, a, true}, {.1, 2, b, true}, {0, 1, b, false}, {1, 0, a, true}} {
		p, err := BuildPolicy(PolicyConfig{Name: "weighted", Scorers: []ScorerConfig{{Name: "prefix-cache", Weight: tc.cache}, {Name: "load", Weight: tc.load}}}, 1)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := p.Pick(&Request{}, []*Candidate{a, b})
		if got != tc.want || p.UsesCache() != tc.usesCache {
			t.Fatalf("weights %g/%g: got %s, uses_cache=%v", tc.cache, tc.load, got.Snap.Spec.ID, p.UsesCache())
		}
		want := tc.cache*got.CacheRatio + tc.load*(1-got.NormLoad)
		if math.Abs(got.Score-want) > 1e-9 {
			t.Fatalf("score %g want %g", got.Score, want)
		}
		if tc.cache > 0 && got.Scores["prefix-cache"] != got.CacheRatio {
			t.Fatal("missing plugin breakdown")
		}
	}
}

type testScorer struct{}

func (testScorer) Name() string    { return "test-preferred" }
func (testScorer) UsesCache() bool { return false }
func (testScorer) Score(_ *Request, c *Candidate, _ []*Candidate) float64 {
	if c.Snap.Spec.ID == "preferred" {
		return 1
	}
	return 0
}

func TestRegisterCustomScorerAndPolicy(t *testing.T) {
	name := fmt.Sprintf("custom-%s-%d", t.Name(), time.Now().UnixNano())
	if err := RegisterScorer(name, func(ScorerConfig) (Scorer, error) { return testScorer{}, nil }); err != nil {
		t.Fatal(err)
	}
	p, err := BuildPolicy(PolicyConfig{Scorers: []ScorerConfig{{Name: name, Weight: 5}, {Name: "load", Weight: 1}}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	a := &Candidate{Snap: registry.Snapshot{Spec: registry.Spec{ID: "preferred"}}, NormLoad: 1}
	b := &Candidate{Snap: registry.Snapshot{Spec: registry.Spec{ID: "other"}}, NormLoad: 0}
	if c, _ := p.Pick(&Request{}, []*Candidate{a, b}); c != a {
		t.Fatal("registered scorer was not used")
	}
	if err := RegisterPolicy(name, func(PolicyConfig, float64) (Policy, error) { return &RoundRobin{}, nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildPolicy(PolicyConfig{Name: name}, 1); err != nil {
		t.Fatal(err)
	}
	if err := RegisterScorer(name, func(ScorerConfig) (Scorer, error) { return testScorer{}, nil }); err == nil {
		t.Fatal("duplicate scorer accepted")
	}
	if err := RegisterPolicy(name, func(PolicyConfig, float64) (Policy, error) { return Random{}, nil }); err == nil {
		t.Fatal("duplicate policy accepted")
	}
	if err := RegisterScorer("", nil); err == nil {
		t.Fatal("empty scorer registration accepted")
	}
	if err := RegisterPolicy("bad", nil); err == nil {
		t.Fatal("nil policy factory accepted")
	}
}

func TestInvalidScoringConfiguration(t *testing.T) {
	for _, cs := range [][]ScorerConfig{
		{{Name: "not-registered", Weight: 1}},
		{{Name: "load", Weight: -1}},
		{{Name: "load", Weight: math.NaN()}},
		{{Name: "load", Weight: math.Inf(1)}},
		{{Name: "load", Weight: 0}},
		{{Name: "load", Weight: 1}, {Name: "load", Weight: 2}},
		{{Name: "load", Weight: 1, Params: map[string]any{"bogus": true}}},
	} {
		if _, err := BuildScored(cs); err == nil {
			t.Errorf("accepted %+v", cs)
		}
	}
	if _, err := BuildPolicy(PolicyConfig{Name: "unknown"}, 1); err == nil || !strings.Contains(err.Error(), "registered:") {
		t.Fatalf("unknown policy: %v", err)
	}
	if _, err := BuildPolicy(PolicyConfig{Scorers: []ScorerConfig{{Name: "load", Weight: 1}}, LoadWeight: 1}, 1); err == nil {
		t.Fatal("mixed legacy/new weights accepted")
	}
	if _, err := BuildPolicy(PolicyConfig{Name: "session_affinity", Fallback: "session_affinity"}, 1); err == nil {
		t.Fatal("recursive fallback accepted")
	}
}
