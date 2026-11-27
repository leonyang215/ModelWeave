package scheduler

import (
	"fmt"
	"math"
	"sync/atomic"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/registry"
	scoring "github.com/leonyang215/ModelWeave/internal/scheduling"
)

var sdkTestID atomic.Uint64

type sdkTestScorer struct {
	name  string
	cache bool
	score func(*scoring.Candidate) float64
}

func (s *sdkTestScorer) Name() string    { return s.name }
func (s *sdkTestScorer) UsesCache() bool { return s.cache }
func (s *sdkTestScorer) Score(_ *scoring.Request, c *scoring.Candidate, _ []*scoring.Candidate) float64 {
	return s.score(c)
}

func registerSDKTestScorer(t *testing.T, s *sdkTestScorer) string {
	t.Helper()
	s.name = fmt.Sprintf("sdk-%s-%d", t.Name(), sdkTestID.Add(1))
	if err := scoring.Register(scoring.Descriptor{Name: s.name, Version: "test", Description: "Repository-private test extension."}, func(scoring.Config) (scoring.Scorer, error) {
		return s, nil
	}); err != nil {
		t.Fatal(err)
	}
	return s.name
}

func TestSDKPluginsClampScoresAndRespectZeroWeights(t *testing.T) {
	for _, tc := range []struct{ input, want float64 }{
		{math.NaN(), 0}, {math.Inf(1), 0}, {math.Inf(-1), 0}, {-1, 0}, {2, 1}, {.7, .7},
	} {
		calls := 0
		name := registerSDKTestScorer(t, &sdkTestScorer{cache: true, score: func(*scoring.Candidate) float64 {
			calls++
			return tc.input
		}})
		p, err := BuildScored([]ScorerConfig{{Name: name, Weight: 2}})
		if err != nil {
			t.Fatal(err)
		}
		pick, _ := p.Pick(&Request{}, []*Candidate{{}})
		if pick.Score != 2*tc.want || pick.Scores[name] != tc.want || calls != 1 || !p.UsesCache() {
			t.Fatalf("invalid score handling: %+v calls=%d", pick, calls)
		}
		p, err = BuildScored([]ScorerConfig{{Name: name, Weight: 0}, {Name: "load", Weight: 1}})
		if err != nil {
			t.Fatal(err)
		}
		p.Pick(&Request{}, []*Candidate{{}})
		if calls != 1 || p.UsesCache() {
			t.Fatal("zero-weight plugin still executed or required a cache lookup")
		}
	}
	if _, err := BuildScored([]ScorerConfig{{Name: "load", Weight: math.MaxFloat64}, {Name: "prefix-cache", Weight: math.MaxFloat64}}); err == nil {
		t.Fatal("overflowing weight sum accepted")
	}
}

func TestParameterizedPluginsAreIsolatedAndIntrospectable(t *testing.T) {
	params := map[string]any{"key": "zone", "value": "west"}
	west, err := BuildScored([]ScorerConfig{{Name: "label", Weight: 3, Params: params}, {Name: "load", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	east, err := BuildScored([]ScorerConfig{{Name: "label", Weight: 3, Params: map[string]any{"key": "zone", "value": "east"}}, {Name: "load", Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	params["value"] = "changed after build"
	view := west.Scorers()
	view[0].Params["value"] = "changed through introspection"
	a := &Candidate{Snap: registry.Snapshot{Spec: registry.Spec{ID: "west", Labels: map[string]string{"zone": "west"}}}, NormLoad: .8}
	b := &Candidate{Snap: registry.Snapshot{Spec: registry.Spec{ID: "east", Labels: map[string]string{"zone": "east"}}}, NormLoad: 0}
	for _, tc := range []struct {
		p    *Scored
		want *Candidate
	}{{west, a}, {east, b}} {
		pick, reason := tc.p.Pick(&Request{}, []*Candidate{a, b})
		if pick != tc.want || reason != "score_weighted" || pick.Scores["label"] != 1 {
			t.Fatalf("plugin configurations crossed models: %+v %s", pick, reason)
		}
	}
	if west.Scorers()[0].Params["value"] != "west" {
		t.Fatal("private parameter config was mutated")
	}
}

func TestSDKCannotMutateLiveRegistryLabels(t *testing.T) {
	name := registerSDKTestScorer(t, &sdkTestScorer{score: func(c *scoring.Candidate) float64 {
		c.Labels["zone"] = "plugin-local"
		return 1
	}})
	p, err := BuildScored([]ScorerConfig{{Name: name, Weight: 1}})
	if err != nil {
		t.Fatal(err)
	}
	labels := map[string]string{"zone": "original"}
	p.Pick(&Request{Model: "m"}, []*Candidate{{Snap: registry.Snapshot{Spec: registry.Spec{ID: "w", Labels: labels}}}})
	if labels["zone"] != "original" {
		t.Fatal("SDK candidate labels alias live registry labels")
	}
	if err := RegisterScorer("label", func(ScorerConfig) (Scorer, error) { return testScorer{}, nil }); err == nil {
		t.Fatal("legacy registration silently replaced a directory plugin")
	}
}
