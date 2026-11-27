package runtime_test

import (
	"reflect"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework/plugins"
	"github.com/leonyang215/ModelWeave/internal/scheduler/framework/runtime"
)

type recordingFilter struct {
	name  string
	code  framework.Code
	order *[]string
}

func (f recordingFilter) Name() string { return f.name }
func (f recordingFilter) Filter(*framework.Request, *framework.Candidate) framework.Status {
	*f.order = append(*f.order, f.name)
	return framework.Status{Code: f.code}
}

type selector struct{ pick *framework.Candidate }

func (selector) Name() string    { return "test" }
func (selector) UsesCache() bool { return false }
func (p selector) Pick(_ *framework.Request, _ []*framework.Candidate) (*framework.Candidate, string) {
	return p.pick, "test"
}

func TestFilterOrderAndShortCircuit(t *testing.T) {
	var order []string
	f := runtime.New(selector{},
		recordingFilter{name: "eligibility", order: &order},
		recordingFilter{name: "capacity", code: framework.Unschedulable, order: &order},
		recordingFilter{name: "must-not-run", order: &order},
	)
	if status := f.RunFilterPlugins(&framework.Request{}, &framework.Candidate{}); status.Code != framework.Unschedulable {
		t.Fatalf("filter result: %+v", status)
	}
	if !reflect.DeepEqual(order, []string{"eligibility", "capacity"}) {
		t.Fatalf("filter order: %v", order)
	}
}

func TestMandatoryFiltersPreserveQueueClassification(t *testing.T) {
	f := runtime.New(selector{}, plugins.Eligibility{}, plugins.Capacity{})
	req := &framework.Request{Model: "m", Role: registry.RoleUnified, KVReserve: 100}
	base := framework.Candidate{Snap: registry.Snapshot{
		Spec:    registry.Spec{ID: "w", Model: "m", Role: registry.RoleUnified},
		Healthy: true, MaxRunning: 1, KVCapacity: 1000,
	}}
	if status := f.RunFilterPlugins(req, &base); status.Code != framework.Success {
		t.Fatal(status)
	}
	for _, tc := range []struct {
		name   string
		change func(*framework.Candidate)
		code   framework.Code
	}{
		{"blocked", func(c *framework.Candidate) { c.Snap.Blocked = true }, framework.UnschedulableAndUnresolvable},
		{"draining", func(c *framework.Candidate) { c.Snap.Draining = true }, framework.UnschedulableAndUnresolvable},
		{"unhealthy", func(c *framework.Candidate) { c.Snap.Healthy = false }, framework.UnschedulableAndUnresolvable},
		{"gone", func(c *framework.Candidate) { c.Snap.Gone = true }, framework.UnschedulableAndUnresolvable},
		{"wrong model", func(c *framework.Candidate) { c.Snap.Spec.Model = "other" }, framework.UnschedulableAndUnresolvable},
		{"running limit", func(c *framework.Candidate) { c.Snap.Inflight = 1 }, framework.Unschedulable},
		{"KV limit", func(c *framework.Candidate) { c.Snap.EffectiveKVUsed = 950 }, framework.Unschedulable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := base
			tc.change(&c)
			if status := f.RunFilterPlugins(req, &c); status.Code != tc.code {
				t.Fatalf("got %+v want %v", status, tc.code)
			}
		})
	}
}

func TestScoreSelectionCannotEscapeFilteredCandidates(t *testing.T) {
	eligible, foreign := &framework.Candidate{}, &framework.Candidate{}
	for _, pick := range []*framework.Candidate{nil, foreign} {
		if _, _, err := runtime.New(selector{pick: pick}).RunScorePlugins(&framework.Request{}, []*framework.Candidate{eligible}); err == nil {
			t.Fatal("invalid policy selection accepted")
		}
	}
	if pick, _, err := runtime.New(selector{pick: eligible}).RunScorePlugins(&framework.Request{}, []*framework.Candidate{eligible}); err != nil || pick != eligible {
		t.Fatalf("valid selection: %v %v", pick, err)
	}
	if _, _, err := runtime.New(selector{}).RunScorePlugins(&framework.Request{}, nil); err == nil {
		t.Fatal("empty scoring pass accepted")
	}
}
