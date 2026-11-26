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
	if m == nil || m.Name != "deepseek-v4.1" || m.PD.Handshake != "push" || !m.Normal.Enabled || !m.Normal.FallbackToPD {
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

// mwip is the only engine protocol: there is nothing to choose, and the
// options that chose one are gone.
func TestEnginesSpeakMWIPOnly(t *testing.T) {
	c, err := Parse([]byte("tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ndiscovery: {workers: [{id: w, model: m, role: unified, target: \"w:9001\"}]}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if c.Models[0].PD.Handshake != "push" || c.Discovery.Workers[0].Target != "w:9001" || c.Discovery.HeartbeatTTL.D() != 15*time.Second {
		t.Fatalf("defaults: %+v %+v", c.Models[0].PD, c.Discovery)
	}
	for _, y := range []string{
		"tokenizer: {endpoints: [x]}\nengine: {protocol: vllm}\nmodels: [{name: m}]",
		"tokenizer: {endpoints: [x]}\nmodels: [{name: m, engine_protocol: vllm}]",
		"tokenizer: {endpoints: [x]}\nengine_state: {poll: {enabled: true, format: prometheus}}\nmodels: [{name: m}]",
		"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ndiscovery: {workers: [{id: w, model: m, role: unified, target: t, state_url: u}]}",
	} {
		if _, err := Parse([]byte(y)); err == nil || !strings.Contains(err.Error(), "not found") {
			t.Errorf("accepted %q: %v", y, err)
		}
	}
}

func TestValidationErrors(t *testing.T) {
	for name, tc := range map[string]struct{ yaml, want string }{
		"no tokenizer":     {"models: [{name: m}]", "tokenizer.endpoints"},
		"bad mode":         {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, mode: wat}]", "mode"},
		"bad handshake":    {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, pd: {handshake: carrier-pigeon}}]", "handshake"},
		"dup alias":        {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, aliases: [a]}, {name: a}]", "used twice"},
		"unknown field":    {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, bogus: 1}]", "bogus"},
		"bad policy":       {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, routing: {decode: {policy: best}}}]", "policy"},
		"worker model":     {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ndiscovery: {workers: [{id: w, model: q, role: decode, target: u}]}", "unknown model"},
		"worker target":    {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ndiscovery: {workers: [{id: w, model: m, role: decode}]}", "target"},
		"worker role":      {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ndiscovery: {workers: [{id: w, model: m, role: leader, target: u}]}", "role"},
		"tenant prio":      {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ntenants: [{name: t, priority: urgent}]", "priority"},
		"tenant duplicate": {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ntenants: [{name: t}, {name: t}]", "used twice"},
		"tenant negative":  {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ntenants: [{name: t, rpm: -1}]", "non-negative"},
		"tenant NaN":       {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ntenants: [{name: t, tpm: .nan}]", "finite"},
		"shared api key":   {"tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\ntenants: [{name: a, api_keys: [k]}, {name: b, api_keys: [k]}]", "shared"},
		"old fused":        {"tokenizer: {endpoints: [x]}\nmodels: [{name: m, mode: fused}]", "mode"},
	} {
		if _, err := Parse([]byte(tc.yaml)); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: want error containing %q, got %v", name, tc.want, err)
		}
	}
}

func TestStateAndScorerConfiguration(t *testing.T) {
	t.Setenv("MW_API_LISTEN", "127.0.0.1:8080")
	t.Setenv("MW_REGISTRY_LISTEN", "127.0.0.1:8081")
	t.Setenv("MW_CONTROL_TOKEN", "test-token")
	t.Setenv("MW_REDIS_URL", "redis://127.0.0.1:6379/0")
	if _, err := Load("../../configs/gateway.redis.example.yaml"); err != nil {
		t.Fatal(err)
	}
	c, err := Parse([]byte(`
tokenizer: {endpoints: [x]}
state:
  backend: redis
  redis: {url: "rediss://user:secret@redis:6379/2", namespace: staging}
models:
  - name: m
    routing:
      decode:
        policy: weighted
        scorers:
          - {name: prefix-cache, weight: 3}
          - {name: load, weight: 1}
`))
	if err != nil {
		t.Fatal(err)
	}
	if c.State.Backend != "redis" || c.State.Redis.Namespace != "staging" || c.State.Redis.LeaseTTL.D() != 15*time.Second || c.State.SyncInterval.D() != 250*time.Millisecond {
		t.Fatalf("state defaults: %+v", c.State)
	}
	if cs := c.Models[0].Routing.Decode.Scorers; len(cs) != 2 || cs[0].Weight != 3 || cs[1].Name != "load" {
		t.Fatalf("scorers: %+v", cs)
	}
	base := "tokenizer: {endpoints: [x]}\nmodels: [{name: m}]\n"
	for _, extra := range []string{
		"state: {backend: etcd}",
		"state: {backend: redis}",
		"state: {backend: redis, redis: {url: 'https://secret@redis'}}",
		"state: {backend: redis, redis: {url: 'redis://redis', timeout: -1s}}",
		"state: {backend: redis, redis: {url: 'redis://redis', timeout: 1s, lease_ttl: 2s}}",
		"state: {backend: redis, sync_interval: 10s, redis: {url: 'redis://redis'}}",
		"state: {sync_interval: -1s}",
	} {
		if _, err := Parse([]byte(base + extra)); err == nil {
			t.Errorf("accepted %s", extra)
		} else if strings.Contains(err.Error(), "secret") {
			t.Errorf("error leaks URL credential: %v", err)
		}
	}
	for _, scorers := range []string{"[{name: missing, weight: 1}]", "[{name: load, weight: -1}]", "[{name: load, weight: .nan}]", "[{name: load, weight: 0}]"} {
		if _, err := Parse([]byte("tokenizer: {endpoints: [x]}\nmodels: [{name: m, routing: {decode: {scorers: " + scorers + "}}}]")); err == nil {
			t.Errorf("accepted %s", scorers)
		}
	}
}
