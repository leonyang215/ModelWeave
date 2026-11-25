package config

import (
	"os"
	"strings"
	"testing"
	"time"
)

func TestExampleConfigIsValid(t *testing.T) {
	c, err := Load("../../configs/gateway.example.yaml")
	if err != nil {
		t.Fatal(err)
	}
	m := c.ModelByName("DeepSeek-V4.1-Flash")
	if m == nil || m.Name != "deepseek-v4.1" || m.PD.Handshake != "push" || !m.Normal.Enabled || !m.Normal.FallbackToPD || m.EngineProtocol != "mwip" {
		t.Fatalf("model not resolved: %+v", m)
	}
	if c.Server.FirstTokenTimeout.D() != 120*time.Second || c.KVIndex.Type != "approx" {
		t.Fatalf("server/kv: %+v %+v", c.Server, c.KVIndex)
	}
}

func TestDefaultsAndEnv(t *testing.T) {
	os.Setenv("MW_TEST_TOKENIZER", "http://tok:1")
	defer os.Unsetenv("MW_TEST_TOKENIZER")
	c, err := Parse([]byte("tokenizer: {endpoints: [\"${MW_TEST_TOKENIZER}\"]}\nmodels: [{name: m}]\n"))
	if err != nil {
		t.Fatal(err)
	}
	if c.Tokenizer.Endpoints[0] != "http://tok:1" {
		t.Fatalf("env not expanded: %v", c.Tokenizer.Endpoints)
	}
	m := c.Models[0]
	if m.Mode != "auto" || m.Reserve.OutputTokens != 1024 || m.Retry.StageAttempts != 3 || m.TokenizerModel != "m" {
		t.Fatalf("defaults: %+v", m)
	}
	// The context window comes from the tokenizer unless capped.
	if m.ContextLength != 0 {
		t.Fatalf("context_length default %d, want 0 (the tokenizer's)", m.ContextLength)
	}
	if _, err := Parse([]byte("tokenizer: {endpoints: [x]}\nmodels: [{name: m, context_length: -1}]\n")); err == nil {
		t.Fatal("negative context_length accepted")
	}
}

func TestEngineProtocolPrecedence(t *testing.T) {
	y := []byte(`
tokenizer: {endpoints: [x]}
engine: {protocol: vllm}
models:
  - {name: a}
  - {name: b, engine_protocol: sglang, pd: {handshake: bootstrap}}
`)
	c, err := Parse(y)
	if err != nil {
		t.Fatal(err)
	}
	if c.Models[0].EngineProtocol != "vllm" || c.Models[1].EngineProtocol != "sglang" || c.Models[0].PD.Handshake != "bootstrap" {
		t.Fatalf("config: %s %s %s", c.Models[0].EngineProtocol, c.Models[1].EngineProtocol, c.Models[0].PD.Handshake)
	}
	// The startup option replaces engine.protocol; the model override wins.
	c, err = ParseWith(y, Overrides{EngineProtocol: "mwip"})
	if err != nil {
		t.Fatal(err)
	}
	if c.Engine.Protocol != "mwip" || c.Models[0].EngineProtocol != "mwip" || c.Models[1].EngineProtocol != "sglang" || c.Models[0].PD.Handshake != "push" {
		t.Fatalf("override: %s %s %s", c.Engine.Protocol, c.Models[0].EngineProtocol, c.Models[1].EngineProtocol)
	}
	// Starting a config without a handshake as sglang picks bootstrap.
	s, err := ParseWith([]byte("tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\n"), Overrides{EngineProtocol: "sglang"})
	if err != nil || s.Models[0].PD.Handshake != "bootstrap" {
		t.Fatalf("sglang default handshake: %v", err)
	}
	if _, err := ParseWith(y, Overrides{EngineProtocol: "grpc"}); err == nil {
		t.Fatal("invalid startup protocol accepted")
	}
	d, err := Parse([]byte("tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\n"))
	if err != nil || d.Models[0].EngineProtocol != "mwip" {
		t.Fatalf("default protocol: %v %v", d.Models[0].EngineProtocol, err)
	}
}

func TestValidationErrors(t *testing.T) {
	for name, tc := range map[string]struct{ yaml, want string }{
		"no tokenizer":   {"models: [{name: m}]", "tokenizer.endpoints"},
		"bad mode":       {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, mode: wat}]", "mode"},
		"bad handshake":  {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, pd: {handshake: carrier-pigeon}}]", "handshake"},
		"dup alias":      {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, aliases: [a]}, {name: a}]", "used twice"},
		"unknown field":  {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, bogus: 1}]", "bogus"},
		"bad policy":     {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, routing: {decode: {policy: best}}}]", "policy"},
		"worker model":   {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ndiscovery: {workers: [{id: w, model: q, role: decode, url: u}]}", "unknown model"},
		"tenant prio":    {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ntenants: [{name: t, priority: urgent}]", "priority"},
		"shared api key": {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ntenants: [{name: a, api_keys: [k]}, {name: b, api_keys: [k]}]", "shared"},
		"bad protocol":   {"tokenizer: {endpoints: [x]}\nengine: {protocol: grpc}\nmodels: [{name: m}]", "protocol"},
		"sglang pull":    {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, engine_protocol: sglang, pd: {handshake: pull}}]", "bootstrap"},
		"old fused":      {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, mode: fused}]", "mode"},
	} {
		if _, err := Parse([]byte(tc.yaml)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: want error containing %q, got %v", name, tc.want, err)
		}
	}
}
