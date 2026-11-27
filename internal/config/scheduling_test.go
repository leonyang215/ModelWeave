package config

import (
	"strings"
	"testing"

	"github.com/leonyang215/ModelWeave/internal/registry"
)

func TestModelWideRoutingDefaultsAndRoleOverrides(t *testing.T) {
	t.Setenv("MW_ADMIN_TOKEN", "test-operator")
	c, err := Load("../../configs/gateway.scheduling.example.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Models) != 3 {
		t.Fatal("multi-model example did not load")
	}
	cache, latency, label := c.Models[0].Routing, c.Models[1].Routing, c.Models[2].Routing
	for _, role := range []registry.Role{registry.RolePrefill, registry.RoleDecode, registry.RoleUnified} {
		p := cache.Policy(role)
		if p.Name != "weighted" || len(p.Scorers) != 2 || p.Scorers[0].Weight != 3 {
			t.Fatalf("model default not inherited for %s: %+v", role, p)
		}
	}
	if p := cache.Policy(registry.RoleEncode); p.Name != "load" || p.Scorers != nil {
		t.Fatalf("role override merged instead of replacing: %+v", p)
	}
	if p := latency.Policy(registry.RoleUnified); p.Scorers[0].Weight != .25 || p.Scorers[1].Weight != 4 {
		t.Fatalf("model-specific weights lost: %+v", p)
	}
	if p := latency.Policy(registry.RolePrefill); len(p.Scorers) != 1 || p.Scorers[0].Name != "load" {
		t.Fatalf("role composition override lost: %+v", p)
	}
	if p := label.Policy(registry.RoleUnified); p.Scorers[0].Params["key"] != "zone" {
		t.Fatalf("parameterized plugin not configured: %+v", p)
	}
	legacy := Routing{}
	if p := legacy.Policy(registry.RoleEncode); p.Name != "load" {
		t.Fatal("legacy encode default changed")
	}
	if p := legacy.Policy(registry.RoleDecode); p.Name != "" || p.Scorers != nil {
		t.Fatal("legacy decode default changed")
	}
}

func TestInvalidModelRoutingDefaultsFailAtStartup(t *testing.T) {
	for _, policy := range []string{
		"{scorers: [{name: load, weight: -1}]}",
		"{scorers: [{name: load, weight: .nan}]}",
		"{scorers: [{name: load, weight: .inf}]}",
		"{scorers: [{name: load, weight: 0}]}",
		"{scorers: [{name: missing, weight: 1}]}",
		"{scorers: [{name: load, weight: 1}, {name: load, weight: 2}]}",
		"{scorers: [{name: label, weight: 1}]}",
		"{scorers: [{name: label, weight: 1, params: {key: zone, value: west, typo: true}}]}",
		"{policy: round_robin, scorers: [{name: load, weight: 1}]}",
	} {
		yaml := "tokenizer: {endpoints: [x]}\nmodels:\n  - name: invalid\n    routing:\n      default: " + policy + "\n      unified: {policy: load}\n"
		if _, err := Parse([]byte(yaml)); err == nil || !strings.Contains(err.Error(), "routing.default") {
			t.Fatalf("invalid default accepted or error lacks model context: %s: %v", policy, err)
		}
	}
	for _, yaml := range []string{
		"tokenizer: {endpoints: [x]}\nmodels: [{name: invalid, routing: {default: {bogus: 1}}}]",
		"tokenizer: {endpoints: [x]}\nmodels: [{name: invalid, routing: {default: {scorers: [{name: load, weight: 1, typo: true}]}}}]",
	} {
		if _, err := Parse([]byte(yaml)); err == nil {
			t.Fatal("unknown routing/config field accepted")
		}
	}
}
