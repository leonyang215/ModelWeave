package plugins_test

import (
	"math"
	"reflect"
	"testing"

	_ "github.com/leonyang215/ModelWeave/internal/scheduler/plugins"
	"github.com/leonyang215/ModelWeave/internal/scheduling"
)

func TestPluginDirectoryInstallsCatalog(t *testing.T) {
	if names := scheduling.Names(); !reflect.DeepEqual(names, []string{"affinity", "label", "load", "prefix-cache"}) {
		t.Fatalf("directory catalog: %v", names)
	}
	for _, d := range scheduling.Catalog() {
		if d.Source == "" || d.Description == "" || d.Version != "1.0.0" || d.Parameters == nil {
			t.Fatalf("incomplete plugin metadata: %+v", d)
		}
	}
}

func TestBuiltInPluginScores(t *testing.T) {
	cached := &scheduling.Candidate{ID: "cached", CacheRatio: .75, NormLoad: .8, Labels: map[string]string{"zone": "west"}}
	idle := &scheduling.Candidate{ID: "idle", CacheRatio: 0, NormLoad: 0}
	candidates := []*scheduling.Candidate{cached, idle}
	for _, tc := range []struct {
		name   string
		params map[string]any
		want   float64
		cache  bool
	}{
		{"prefix-cache", nil, .75, true}, {"load", nil, .2, false},
		{"label", map[string]any{"key": "zone", "value": "west"}, 1, false},
	} {
		s, err := scheduling.Build(scheduling.Config{Name: tc.name, Params: tc.params})
		if err != nil {
			t.Fatal(err)
		}
		if s.UsesCache() != tc.cache || math.Abs(s.Score(&scheduling.Request{}, cached, candidates)-tc.want) > 1e-10 {
			t.Fatalf("%s: wrong score or cache requirement", tc.name)
		}
	}
	affinity, err := scheduling.Build(scheduling.Config{Name: "affinity"})
	if err != nil {
		t.Fatal(err)
	}
	if affinity.Score(&scheduling.Request{}, cached, candidates) != 0 {
		t.Fatal("affinity without a session")
	}
	req := &scheduling.Request{SessionKey: "session"}
	a, b := affinity.Score(req, cached, candidates), affinity.Score(req, idle, candidates)
	if a+b != 1 || affinity.Score(req, cached, candidates) != a {
		t.Fatal("affinity must select exactly one stable instance")
	}
	label, err := scheduling.Build(scheduling.Config{Name: "label", Params: map[string]any{"key": "zone", "value": ""}})
	if err != nil {
		t.Fatal(err)
	}
	if label.Score(req, idle, candidates) != 0 {
		t.Fatal("missing label matched an empty value")
	}
	idle.Labels = map[string]string{"zone": ""}
	if label.Score(req, idle, candidates) != 1 {
		t.Fatal("explicit empty label did not match")
	}
}

func TestPluginParameterValidation(t *testing.T) {
	for _, name := range []string{"affinity", "prefix-cache", "load"} {
		if _, err := scheduling.Build(scheduling.Config{Name: name, Params: map[string]any{"bogus": true}}); err == nil {
			t.Fatalf("%s accepted unknown parameters", name)
		}
	}
	for _, params := range []map[string]any{
		nil, {}, {"key": "zone"}, {"key": " ", "value": "west"},
		{"key": "zone", "value": nil}, {"key": "zone", "value": true},
		{"key": "zone", "value": "west", "bogus": true},
	} {
		if _, err := scheduling.Build(scheduling.Config{Name: "label", Params: params}); err == nil {
			t.Fatalf("label accepted invalid params: %v", params)
		}
	}
}
