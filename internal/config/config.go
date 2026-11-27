// Package config loads the gateway configuration.
package config

import (
	"fmt"
	"math"
	"net/url"
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
	Auth        Auth        `yaml:"auth"`
	Tenants     []Tenant    `yaml:"tenants"`
	Discovery   Discovery   `yaml:"discovery"`
	State       State       `yaml:"state"`
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
	// GRPCListen is where the gateway serves the mwip Registry, the gRPC
	// control plane engine instances register with (default :8081).
	GRPCListen string `yaml:"grpc_listen"`
	// ControlToken, when set, is required on the mwip Registry
	// (x-modelweave-token metadata). It also protects the admin API unless
	// AdminToken overrides it (X-ModelWeave-Token HTTP header).
	ControlToken string `yaml:"control_token"`
	// AdminToken overrides ControlToken for the HTTP admin API only. Engines
	// keep using ControlToken. Empty preserves the existing shared-token setup.
	AdminToken string `yaml:"admin_token"`
	// DisableAdminUI removes the embedded console, but not the admin API.
	DisableAdminUI bool `yaml:"disable_admin_ui"`
}

// Tokenizer configures the tokenizer client.
type Tokenizer struct {
	Endpoints []string `yaml:"endpoints"`
	Timeout   Duration `yaml:"timeout"`
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

// Discovery configures how engine instances are found. Instances register
// with heartbeats on the mwip Registry (server.grpc_listen) and expire when
// no heartbeat arrives for HeartbeatTTL; Workers lists static instances,
// which never expire.
type Discovery struct {
	HeartbeatTTL Duration        `yaml:"heartbeat_ttl"`
	Workers      []registry.Spec `yaml:"workers"`
}

// State selects single-process memory or a Redis namespace shared by gateway
// replicas. Static model/tenant configuration remains local to each replica.
type State struct {
	Backend      string   `yaml:"backend"` // memory | redis
	SyncInterval Duration `yaml:"sync_interval"`
	Redis        Redis    `yaml:"redis"`
}

type Redis struct {
	URL       string   `yaml:"url"`
	Namespace string   `yaml:"namespace"`
	Timeout   Duration `yaml:"timeout"`
	LeaseTTL  Duration `yaml:"lease_ttl"`
}

// EngineState configures how the gateway learns the load of instances:
// heartbeats carry it, and Poll asks instances for it.
type EngineState struct {
	// StaleAfter bounds how long a load report is trusted.
	StaleAfter Duration `yaml:"stale_after"`
	Poll       Poll     `yaml:"poll"`
}

// Poll configures polling every instance with Engine.GetState (for static
// instances, which send no heartbeats).
type Poll struct {
	Enabled  bool     `yaml:"enabled"`
	Interval Duration `yaml:"interval"`
	Timeout  Duration `yaml:"timeout"`
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
	// EngineModel is the model name engines serve; sent as the Generate
	// request's model when set, empty otherwise.
	EngineModel string `yaml:"engine_model"`
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
	// Handshake: push (default) | bootstrap | pull.
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

// Load reads and validates a config file.
func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(b)
}

// Parse parses YAML (with ${ENV} expansion), applies defaults and
// validates.
func Parse(b []byte) (*Config, error) {
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
	if s.GRPCListen == "" {
		s.GRPCListen = ":8081"
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
	if c.Auth.AnonymousTenant == "" {
		c.Auth.AnonymousTenant = "anonymous"
	}
	defDur(&c.Discovery.HeartbeatTTL, 15*time.Second)
	if c.State.Backend == "" {
		c.State.Backend = "memory"
	}
	if c.State.SyncInterval == 0 {
		c.State.SyncInterval = Duration(250 * time.Millisecond)
	}
	if c.State.Redis.Namespace == "" {
		c.State.Redis.Namespace = "modelweave"
	}
	if c.State.Redis.Timeout == 0 {
		c.State.Redis.Timeout = Duration(time.Second)
	}
	if c.State.Redis.LeaseTTL == 0 {
		c.State.Redis.LeaseTTL = Duration(15 * time.Second)
	}
	defDur(&c.EngineState.StaleAfter, 10*time.Second)
	p := &c.EngineState.Poll
	defDur(&p.Interval, 2*time.Second)
	defDur(&p.Timeout, time.Second)
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
		if m.PD.Handshake == "" {
			m.PD.Handshake = "push"
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
	if c.State.Backend != "memory" && c.State.Backend != "redis" {
		return fmt.Errorf("state.backend: want memory or redis")
	}
	if c.State.SyncInterval <= 0 {
		return fmt.Errorf("state.sync_interval must be positive")
	}
	if c.State.Backend == "redis" {
		u, err := url.Parse(c.State.Redis.URL)
		if err != nil || u == nil || (u.Scheme != "redis" && u.Scheme != "rediss") || u.Hostname() == "" {
			return fmt.Errorf("state.redis.url must be a redis:// or rediss:// URL")
		}
		if c.State.Redis.Timeout <= 0 || c.State.Redis.LeaseTTL.D() < 3*c.State.Redis.Timeout.D() || c.State.SyncInterval.D() > c.State.Redis.LeaseTTL.D()/3 {
			return fmt.Errorf("state.redis requires timeout > 0, lease_ttl >= 3*timeout and sync_interval <= lease_ttl/3")
		}
	}
	if len(c.Tokenizer.Endpoints) == 0 {
		return fmt.Errorf("tokenizer.endpoints must list at least one tokenizer URL")
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
	keys := map[string]string{}
	tenantNames := map[string]bool{}
	for _, t := range c.Tenants {
		if t.Name == "" {
			return fmt.Errorf("tenant without name")
		}
		if tenantNames[t.Name] {
			return fmt.Errorf("tenant name %q is used twice", t.Name)
		}
		tenantNames[t.Name] = true
		if t.RPM < 0 || t.TPM < 0 || t.MaxConcurrency < 0 {
			return fmt.Errorf("tenant %s: limits must be non-negative", t.Name)
		}
		for _, v := range []float64{t.RPM, t.TPM, t.Weight} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("tenant %s: limits/weight must be finite", t.Name)
			}
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
		if w.ID == "" || w.Target == "" || w.Model == "" {
			return fmt.Errorf("static worker needs id, model and target: %+v", w)
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
