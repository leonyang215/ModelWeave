// Package config loads the gateway configuration.
package config

import (
	"fmt"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/leonyang215/ModelWeave/internal/qos"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

// Duration is a time.Duration that unmarshals from strings like "250ms".
type Duration time.Duration

func (d *Duration) UnmarshalYAML(n *yaml.Node) error {
	var s string
	if err := n.Decode(&s); err != nil {
		return err
	}
	v, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", s, err)
	}
	*d = Duration(v)
	return nil
}

func (d Duration) MarshalYAML() (any, error) { return time.Duration(d).String(), nil }

// D returns the time.Duration.
func (d Duration) D() time.Duration { return time.Duration(d) }

// Config is the root configuration.
type Config struct {
	Server      Server      `yaml:"server"`
	Tokenizer   Tokenizer   `yaml:"tokenizer"`
	Engine      Engine      `yaml:"engine"`
	Auth        Auth        `yaml:"auth"`
	Tenants     []Tenant    `yaml:"tenants"`
	Discovery   Discovery   `yaml:"discovery"`
	EngineState EngineState `yaml:"engine_state"`
	Health      Health      `yaml:"health"`
	KVIndex     KVIndex     `yaml:"kv_index"`
	Models      []Model     `yaml:"models"`
}

// Server holds listener and request-lifecycle settings.
type Server struct {
	Listen              string   `yaml:"listen"`
	MaxBodyBytes        int64    `yaml:"max_body_bytes"`
	LogLevel            string   `yaml:"log_level"`
	LogFormat           string   `yaml:"log_format"`
	RequestTimeout      Duration `yaml:"request_timeout"`
	FirstTokenTimeout   Duration `yaml:"first_token_timeout"`
	IdleTimeout         Duration `yaml:"idle_timeout"`
	ExposeRoutingHeader bool     `yaml:"expose_routing_headers"`
	ShutdownGrace       Duration `yaml:"shutdown_grace"`
	// ControlToken, when set, is required (X-ModelWeave-Token header) on the
	// worker control plane (/mwip/v1/*) and the admin API (/admin/*).
	ControlToken string `yaml:"control_token"`
}

// Tokenizer configures the tokenizer client.
type Tokenizer struct {
	Endpoints []string `yaml:"endpoints"`
	Timeout   Duration `yaml:"timeout"`
}

// Engine selects how the gateway talks to inference workers.
type Engine struct {
	// Protocol is the default engine protocol: "mwip" (send the tokenizer's
	// intermediate state as is, POST /mwip/v1/generate), "vllm" (adapt to
	// POST /inference/v1/generate) or "sglang" (adapt to POST /generate).
	// The gateway's -engine-protocol flag overrides it; models may override
	// both with engine_protocol.
	Protocol string `yaml:"protocol"`
}

// EngineProtocols lists the supported engine protocols.
var EngineProtocols = []string{"mwip", "vllm", "sglang"}

// ValidEngineProtocol reports whether p is supported.
func ValidEngineProtocol(p string) bool {
	for _, x := range EngineProtocols {
		if p == x {
			return true
		}
	}
	return false
}

// Auth configures API-key authentication.
type Auth struct {
	Enabled         bool   `yaml:"enabled"`
	AnonymousTenant string `yaml:"anonymous_tenant"`
}

// Tenant is one API consumer with its QoS policy.
type Tenant struct {
	Name           string   `yaml:"name"`
	APIKeys        []string `yaml:"api_keys"`
	Priority       string   `yaml:"priority"`
	Weight         float64  `yaml:"weight"`
	RPM            float64  `yaml:"rpm"`
	TPM            float64  `yaml:"tpm"`
	MaxConcurrency int      `yaml:"max_concurrency"`
}

// Discovery configures worker discovery.
type Discovery struct {
	HeartbeatTTL Duration        `yaml:"heartbeat_ttl"`
	Workers      []registry.Spec `yaml:"workers"`
}

// EngineState configures engine-state adapters.
type EngineState struct {
	StaleAfter Duration `yaml:"stale_after"`
	Poll       Poll     `yaml:"poll"`
}

// Poll configures pull-based engine state collection.
type Poll struct {
	Enabled  bool     `yaml:"enabled"`
	Interval Duration `yaml:"interval"`
	Timeout  Duration `yaml:"timeout"`
	// Format is "mwip" (GET /mwip/v1/state) or "prometheus" (scrape).
	Format string `yaml:"format"`
	// Preset selects the Prometheus metric names: "vllm" or "sglang".
	Preset string `yaml:"preset"`
	// Metrics overrides individual Prometheus metric names.
	Metrics map[string]string `yaml:"metrics"`
}

// Health configures passive health checking.
type Health struct {
	FailureThreshold int      `yaml:"failure_threshold"`
	Cooldown         Duration `yaml:"cooldown"`
}

// KVIndex configures cache awareness.
type KVIndex struct {
	Type           string   `yaml:"type"` // approx | events | remote | none
	BlockSize      int      `yaml:"block_size"`
	TTL            Duration `yaml:"ttl"`
	CapacityBlocks int      `yaml:"capacity_blocks"`
	OptimisticTTL  Duration `yaml:"optimistic_ttl"`
	RemoteURL      string   `yaml:"remote_url"`
	RemoteTimeout  Duration `yaml:"remote_timeout"`
}

// Model configures one served model.
type Model struct {
	Name           string   `yaml:"name"`
	Aliases        []string `yaml:"aliases"`
	TokenizerModel string   `yaml:"tokenizer_model"`
	// EngineModel is the model name engines serve; sent as the generate
	// request's model when set (vLLM checks it), omitted otherwise.
	EngineModel string `yaml:"engine_model"`
	// EngineProtocol overrides engine.protocol for this model's workers.
	EngineProtocol string `yaml:"engine_protocol"`
	// ContextLength caps the context window (for engines started with a
	// shorter one); 0 uses the window the tokenizer registers for the model.
	ContextLength int `yaml:"context_length"`
	// Mode: auto (normal or pd per request) | normal (no PD split) | pd.
	Mode      string    `yaml:"mode"`
	PD        PD        `yaml:"pd"`
	Normal    Normal    `yaml:"normal"`
	Encode    Encode    `yaml:"encode"`
	Reserve   Reserve   `yaml:"reservation"`
	Routing   Routing   `yaml:"routing"`
	Admission Admission `yaml:"admission"`
	Retry     Retry     `yaml:"retry"`
}

// PD configures prefill/decode disaggregation.
type PD struct {
	// Handshake: push | bootstrap | pull. Default: push for mwip, bootstrap
	// for vllm and sglang.
	Handshake string `yaml:"handshake"`
}

// Normal configures normal (non-PD) execution on decode workers in auto
// mode: a decode worker that accepts normal requests runs prefill itself
// when that is cheaper than a KV transfer.
type Normal struct {
	Enabled bool `yaml:"enabled"`
	// MaxUncachedTokens: run normal when the decode worker would compute at
	// most this many uncached prompt tokens.
	MaxUncachedTokens int `yaml:"max_uncached_tokens"`
	// MinCacheRatio: run normal when the decode worker holds at least this
	// fraction of the prompt.
	MinCacheRatio float64 `yaml:"min_cache_ratio"`
	// SessionAffinity runs follow-up turns normal on the decode worker that
	// served the session.
	SessionAffinity bool     `yaml:"session_affinity"`
	SessionTTL      Duration `yaml:"session_ttl"`
	// FallbackToPD retries a failed normal attempt as pd (once, before
	// any output).
	FallbackToPD bool `yaml:"fallback_to_pd"`
}

// Encode configures the encode (E) stage.
type Encode struct {
	Policy           string   `yaml:"policy"` // auto | always | never
	ArtifactCacheTTL Duration `yaml:"artifact_cache_ttl"`
	ArtifactCacheMax int      `yaml:"artifact_cache_max"`
}

// Reserve configures the gateway's capacity reservations on workers.
type Reserve struct {
	// OutputTokens caps the output length reserved on the decode worker:
	// prompt + min(max_tokens, OutputTokens) is held while a request runs;
	// the engine grows longer outputs itself.
	OutputTokens int `yaml:"output_tokens"`
	// BlockSize rounds reservations up to KV blocks.
	BlockSize int `yaml:"block_size"`
	// UncachedBeta weights prefill work by attention cost:
	// w = uncached * (1 + beta*(cached + uncached/2)).
	UncachedBeta float64 `yaml:"uncached_beta"`
}

// Routing holds the scheduling policy per role.
type Routing struct {
	Prefill scheduler.PolicyConfig `yaml:"prefill"`
	Decode  scheduler.PolicyConfig `yaml:"decode"`
	Encode  scheduler.PolicyConfig `yaml:"encode"`
	Unified scheduler.PolicyConfig `yaml:"unified"`
}

// Admission configures queueing and overload protection.
type Admission struct {
	QueueSize     int      `yaml:"queue_size"`
	MaxWait       Duration `yaml:"max_wait"`
	BackfillDepth int      `yaml:"backfill_depth"`
	Overload      Overload `yaml:"overload"`
}

// Overload configures load shedding by utilization level and priority.
type Overload struct {
	Enabled    bool      `yaml:"enabled"`
	Thresholds []float64 `yaml:"thresholds"` // utilization of levels L1..L4
	Hysteresis float64   `yaml:"hysteresis"`
	// CacheExemptRatio exempts requests whose predicted prefix-cache hit
	// ratio is at least this from probabilistic drops.
	CacheExemptRatio float64 `yaml:"cache_exempt_ratio"`
}

// Retry configures retries before the first output token.
type Retry struct {
	StageAttempts int      `yaml:"stage_attempts"`
	WholeRetries  int      `yaml:"whole_retries"`
	Backoff       Duration `yaml:"backoff"`
}

// Overrides are startup options that take precedence over the file.
type Overrides struct {
	// EngineProtocol replaces engine.protocol (models without their own
	// engine_protocol follow it).
	EngineProtocol string
}

// Load reads and validates a config file.
func Load(path string) (*Config, error) { return LoadWith(path, Overrides{}) }

// LoadWith reads a config file and applies startup overrides.
func LoadWith(path string, ov Overrides) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseWith(b, ov)
}

// Parse parses YAML (with ${ENV} expansion) and applies defaults.
func Parse(b []byte) (*Config, error) { return ParseWith(b, Overrides{}) }

// ParseWith parses YAML, applies startup overrides and defaults.
func ParseWith(b []byte, ov Overrides) (*Config, error) {
	expanded := os.Expand(string(b), func(k string) string {
		if v, ok := os.LookupEnv(k); ok {
			return v
		}
		return "${" + k + "}"
	})
	var c Config
	dec := yaml.NewDecoder(strings.NewReader(expanded))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if ov.EngineProtocol != "" {
		c.Engine.Protocol = ov.EngineProtocol
	}
	c.applyDefaults()
	if err := c.validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

func defDur(d *Duration, v time.Duration) {
	if *d <= 0 {
		*d = Duration(v)
	}
}

func (c *Config) applyDefaults() {
	s := &c.Server
	if s.Listen == "" {
		s.Listen = ":8080"
	}
	if s.MaxBodyBytes <= 0 {
		s.MaxBodyBytes = 32 << 20
	}
	if s.LogLevel == "" {
		s.LogLevel = "info"
	}
	if s.LogFormat == "" {
		s.LogFormat = "json"
	}
	defDur(&s.RequestTimeout, 30*time.Minute)
	defDur(&s.FirstTokenTimeout, 120*time.Second)
	defDur(&s.IdleTimeout, 120*time.Second)
	defDur(&s.ShutdownGrace, 30*time.Second)
	defDur(&c.Tokenizer.Timeout, 30*time.Second)
	if c.Engine.Protocol == "" {
		c.Engine.Protocol = "mwip"
	}
	if c.Auth.AnonymousTenant == "" {
		c.Auth.AnonymousTenant = "anonymous"
	}
	defDur(&c.Discovery.HeartbeatTTL, 15*time.Second)
	defDur(&c.EngineState.StaleAfter, 10*time.Second)
	p := &c.EngineState.Poll
	defDur(&p.Interval, 2*time.Second)
	defDur(&p.Timeout, time.Second)
	if p.Format == "" {
		p.Format = "mwip"
	}
	if c.Health.FailureThreshold <= 0 {
		c.Health.FailureThreshold = 3
	}
	defDur(&c.Health.Cooldown, 10*time.Second)
	k := &c.KVIndex
	if k.Type == "" {
		k.Type = "approx"
	}
	if k.BlockSize <= 0 {
		k.BlockSize = 64
	}
	defDur(&k.TTL, 10*time.Minute)
	defDur(&k.OptimisticTTL, 30*time.Second)
	defDur(&k.RemoteTimeout, 50*time.Millisecond)
	for i := range c.Tenants {
		if c.Tenants[i].Weight <= 0 {
			c.Tenants[i].Weight = 1
		}
	}
	for i := range c.Models {
		m := &c.Models[i]
		if m.TokenizerModel == "" {
			m.TokenizerModel = m.Name
		}
		if m.Mode == "" {
			m.Mode = "auto"
		}
		if m.EngineProtocol == "" {
			m.EngineProtocol = c.Engine.Protocol
		}
		if m.PD.Handshake == "" {
			// The native protocol's own handshake; vLLM (MooncakeConnector)
			// and SGLang pair P and D through a bootstrap server.
			m.PD.Handshake = "push"
			if m.EngineProtocol == "vllm" || m.EngineProtocol == "sglang" {
				m.PD.Handshake = "bootstrap"
			}
		}
		if m.Normal.MaxUncachedTokens <= 0 {
			m.Normal.MaxUncachedTokens = 1024
		}
		if m.Normal.MinCacheRatio <= 0 {
			m.Normal.MinCacheRatio = 0.6
		}
		defDur(&m.Normal.SessionTTL, 30*time.Minute)
		if m.Encode.Policy == "" {
			m.Encode.Policy = "auto"
		}
		defDur(&m.Encode.ArtifactCacheTTL, time.Hour)
		if m.Encode.ArtifactCacheMax <= 0 {
			m.Encode.ArtifactCacheMax = 10000
		}
		if m.Reserve.OutputTokens <= 0 {
			m.Reserve.OutputTokens = 1024
		}
		if m.Reserve.BlockSize <= 0 {
			m.Reserve.BlockSize = k.BlockSize
		}
		if m.Admission.QueueSize <= 0 {
			m.Admission.QueueSize = 512
		}
		defDur(&m.Admission.MaxWait, 30*time.Second)
		if m.Admission.BackfillDepth < 0 {
			m.Admission.BackfillDepth = 0
		}
		if len(m.Admission.Overload.Thresholds) == 0 {
			m.Admission.Overload.Thresholds = []float64{0.75, 0.85, 0.90, 0.95}
		}
		if m.Admission.Overload.Hysteresis <= 0 {
			m.Admission.Overload.Hysteresis = 0.02
		}
		if m.Admission.Overload.CacheExemptRatio <= 0 {
			m.Admission.Overload.CacheExemptRatio = 0.8
		}
		if m.Retry.StageAttempts <= 0 {
			m.Retry.StageAttempts = 3
		}
		if m.Retry.WholeRetries < 0 {
			m.Retry.WholeRetries = 0
		}
		defDur(&m.Retry.Backoff, 20*time.Millisecond)
	}
}

func (c *Config) validate() error {
	if len(c.Tokenizer.Endpoints) == 0 {
		return fmt.Errorf("tokenizer.endpoints must list at least one tokenizer URL")
	}
	if !ValidEngineProtocol(c.Engine.Protocol) {
		return fmt.Errorf("engine.protocol %q: want mwip, vllm or sglang", c.Engine.Protocol)
	}
	switch c.KVIndex.Type {
	case "approx", "events", "none":
	case "remote":
		if c.KVIndex.RemoteURL == "" {
			return fmt.Errorf("kv_index.remote_url is required for kv_index.type=%s", c.KVIndex.Type)
		}
	default:
		return fmt.Errorf("kv_index.type %q: want approx, events, remote or none", c.KVIndex.Type)
	}
	switch c.EngineState.Poll.Format {
	case "mwip", "prometheus":
	default:
		return fmt.Errorf("engine_state.poll.format %q: want mwip or prometheus", c.EngineState.Poll.Format)
	}
	keys := map[string]string{}
	for _, t := range c.Tenants {
		if t.Name == "" {
			return fmt.Errorf("tenant without name")
		}
		if _, err := qos.ParsePriority(t.Priority); err != nil {
			return fmt.Errorf("tenant %s: %w", t.Name, err)
		}
		for _, k := range t.APIKeys {
			if other, dup := keys[k]; dup {
				return fmt.Errorf("api key shared by tenants %s and %s", other, t.Name)
			}
			keys[k] = t.Name
		}
	}
	if len(c.Models) == 0 {
		return fmt.Errorf("models must list at least one model")
	}
	names := map[string]bool{}
	for _, m := range c.Models {
		if m.Name == "" {
			return fmt.Errorf("model without name")
		}
		for _, n := range append([]string{m.Name}, m.Aliases...) {
			if names[n] {
				return fmt.Errorf("model name %q is used twice", n)
			}
			names[n] = true
		}
		if m.ContextLength < 0 {
			return fmt.Errorf("model %s: context_length must not be negative", m.Name)
		}
		switch m.Mode {
		case "auto", "normal", "pd":
		default:
			return fmt.Errorf("model %s: mode %q: want auto, normal or pd", m.Name, m.Mode)
		}
		if !ValidEngineProtocol(m.EngineProtocol) {
			return fmt.Errorf("model %s: engine_protocol %q: want mwip, vllm or sglang", m.Name, m.EngineProtocol)
		}
		if m.EngineProtocol == "sglang" && m.PD.Handshake != "bootstrap" {
			return fmt.Errorf("model %s: sglang workers support pd.handshake bootstrap only", m.Name)
		}
		switch m.PD.Handshake {
		case "push", "bootstrap", "pull":
		default:
			return fmt.Errorf("model %s: pd.handshake %q: want push, bootstrap or pull", m.Name, m.PD.Handshake)
		}
		switch m.Encode.Policy {
		case "auto", "always", "never":
		default:
			return fmt.Errorf("model %s: encode.policy %q: want auto, always or never", m.Name, m.Encode.Policy)
		}
		if len(m.Admission.Overload.Thresholds) != 4 {
			return fmt.Errorf("model %s: admission.overload.thresholds needs 4 levels", m.Name)
		}
		for _, pc := range []scheduler.PolicyConfig{m.Routing.Prefill, m.Routing.Decode, m.Routing.Encode, m.Routing.Unified} {
			if _, err := scheduler.BuildPolicy(pc, 1); err != nil {
				return fmt.Errorf("model %s: %w", m.Name, err)
			}
		}
	}
	for _, w := range c.Discovery.Workers {
		if w.ID == "" || w.URL == "" || w.Model == "" {
			return fmt.Errorf("static worker needs id, model and url: %+v", w)
		}
		if _, err := registry.ParseRole(string(w.Role)); err != nil {
			return fmt.Errorf("worker %s: %w", w.ID, err)
		}
		if !names[w.Model] {
			return fmt.Errorf("worker %s serves unknown model %q", w.ID, w.Model)
		}
	}
	return nil
}

// ModelByName resolves a served model name or alias.
func (c *Config) ModelByName(name string) *Model {
	for i := range c.Models {
		m := &c.Models[i]
		if m.Name == name {
			return m
		}
		for _, a := range m.Aliases {
			if a == name {
				return m
			}
		}
	}
	return nil
}
