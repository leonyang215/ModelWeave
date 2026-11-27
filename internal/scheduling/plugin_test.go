package scheduling_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/scheduling"
)

type testScorer struct{ name string }

func (s *testScorer) Name() string  { return s.name }
func (*testScorer) UsesCache() bool { return false }
func (*testScorer) Score(_ *scheduling.Request, _ *scheduling.Candidate, _ []*scheduling.Candidate) float64 {
	return 0.5
}

func descriptor(name string) scheduling.Descriptor {
	return scheduling.Descriptor{Name: name, Version: "1.0.0", Description: "Test plugin.", Parameters: []scheduling.Parameter{{Name: "nested", Type: "object"}}}
}

func factory(c scheduling.Config) (scheduling.Scorer, error) { return &testScorer{name: c.Name}, nil }

func TestRegistrationAndCatalogAreDefensiveAndDeterministic(t *testing.T) {
	var r scheduling.Registry
	d := descriptor("z")
	if err := r.Register(d, factory); err != nil {
		t.Fatal(err)
	}
	d.Parameters[0].Name = "mutated"
	if err := r.Register(descriptor("a"), factory); err != nil {
		t.Fatal(err)
	}
	if got := r.Names(); !reflect.DeepEqual(got, []string{"a", "z"}) {
		t.Fatalf("names: %v", got)
	}
	catalog := r.Catalog()
	catalog[1].Parameters[0].Name = "also-mutated"
	info, ok := r.Lookup("z")
	if !ok || info.Parameters[0].Name != "nested" {
		t.Fatalf("catalog leaked mutable state: %+v", info)
	}
	if err := r.Register(descriptor("z"), factory); err == nil {
		t.Fatal("duplicate scorer accepted")
	}
	for _, bad := range []scheduling.Descriptor{
		{}, {Name: " padded ", Version: "1", Description: "bad"},
		{Name: "no-description", Version: "1"}, {Name: "no-version", Description: "bad"},
		{Name: "duplicate-params", Version: "1", Description: "bad", Parameters: []scheduling.Parameter{{Name: "key", Type: "string"}, {Name: "key", Type: "string"}}},
	} {
		if err := r.Register(bad, factory); err == nil {
			t.Fatalf("invalid descriptor accepted: %+v", bad)
		}
	}
	if err := r.Register(descriptor("nil-factory"), nil); err == nil {
		t.Fatal("nil factory accepted")
	}
}

func TestFactoriesReceivePrivateParametersWithoutLosingIntegerPrecision(t *testing.T) {
	r := scheduling.NewRegistry()
	const large int64 = 9007199254740993
	config := scheduling.Config{Name: "private", Weight: 1, Params: map[string]any{"nested": map[string]any{"value": large}}}
	if err := r.Register(descriptor(config.Name), func(c scheduling.Config) (scheduling.Scorer, error) {
		var p struct {
			Nested struct{ Value int64 } `json:"nested"`
		}
		if err := scheduling.DecodeParams(c.Params, &p); err != nil || p.Nested.Value != large {
			return nil, fmt.Errorf("lost parameter precision: %+v %v", p, err)
		}
		c.Params["nested"].(map[string]any)["value"] = "mutated by factory"
		return &testScorer{name: c.Name}, nil
	}); err != nil {
		t.Fatal(err)
	}
	first, err := r.Build(config)
	if err != nil {
		t.Fatal(err)
	}
	second, err := r.Build(config)
	if err != nil || first == second {
		t.Fatalf("factory instance shared across compositions: %v %v", first == second, err)
	}
	if got := config.Params["nested"].(map[string]any)["value"]; got != large {
		t.Fatalf("factory changed caller configuration: %v", got)
	}
	if _, err := (scheduling.Config{Params: map[string]any{"bad": func() {}}}).Clone(); err == nil {
		t.Fatal("non-JSON params accepted")
	}
	copy, err := config.Clone()
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(copy.Params)
	if !strings.Contains(string(b), fmt.Sprint(large)) {
		t.Fatalf("JSON round trip changed integer: %s", b)
	}
}

func TestCatalogDoesNotInstantiateFactoriesAndBuildReportsFailures(t *testing.T) {
	r := scheduling.NewRegistry()
	calls := 0
	want := errors.New("params required")
	if err := r.Register(descriptor("failure"), func(scheduling.Config) (scheduling.Scorer, error) {
		calls++
		return nil, want
	}); err != nil {
		t.Fatal(err)
	}
	if len(r.Catalog()) != 1 || calls != 0 {
		t.Fatal("discovery instantiated a plugin")
	}
	if _, err := r.Build(scheduling.Config{Name: "failure"}); !errors.Is(err, want) || calls != 1 {
		t.Fatalf("factory failure: %v (calls %d)", err, calls)
	}
	for _, typedNil := range []bool{false, true} {
		name := fmt.Sprint("nil-", typedNil)
		if err := r.Register(descriptor(name), func(scheduling.Config) (scheduling.Scorer, error) {
			if typedNil {
				return (*testScorer)(nil), nil
			}
			return nil, nil
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := r.Build(scheduling.Config{Name: name}); err == nil || !strings.Contains(err.Error(), "returned nil") {
			t.Fatalf("nil plugin accepted: %v", err)
		}
	}
	if _, err := r.Build(scheduling.Config{Name: "missing"}); err == nil || !strings.Contains(err.Error(), "registered:") {
		t.Fatalf("unknown scorer: %v", err)
	}
}

func TestConcurrentRegistrationDiscoveryAndConstruction(t *testing.T) {
	r := scheduling.NewRegistry()
	var wg sync.WaitGroup
	for i := range 32 {
		wg.Go(func() {
			name := fmt.Sprintf("plugin-%02d", i)
			if err := r.Register(descriptor(name), factory); err != nil {
				t.Error(err)
			}
			if _, err := r.Build(scheduling.Config{Name: name}); err != nil {
				t.Error(err)
			}
			_ = r.Catalog()
			_ = r.Names()
		})
	}
	wg.Wait()
	if len(r.Catalog()) != 32 {
		t.Fatal("registrations lost")
	}
}
