package qos

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/redis/go-redis/v9"
)

//go:embed redis_limiter.lua
var limiterSource string
var limiterScript = redis.NewScript(limiterSource)

type redisLimiter struct {
	backend  *RedisBackend
	policies map[string]TenantPolicy
	names    []string
}
type limiterPolicy struct {
	RPM            float64 `json:"rpm"`
	TPM            float64 `json:"tpm"`
	MaxConcurrency int     `json:"max_concurrency"`
}
type limiterReply struct {
	Code         string  `json:"code"`
	Reason       string  `json:"reason"`
	RetryAfterMS int64   `json:"retry_after_ms"`
	Inflight     int     `json:"inflight"`
	RPMAvailable float64 `json:"rpm_available"`
	TPMAvailable float64 `json:"tpm_available"`
}

func newRedisLimiter(ctx context.Context, b *RedisBackend, policies []TenantPolicy) (*redisLimiter, error) {
	l := &redisLimiter{backend: b, policies: make(map[string]TenantPolicy)}
	ps := sortedPolicies(policies)
	for _, p := range ps {
		if p.Name == "" || p.MaxConcurrency < 0 {
			return nil, fmt.Errorf("invalid tenant policy")
		}
		for _, v := range []float64{p.RPM, p.TPM, p.Weight} {
			if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("tenant %s: limits/weight must be finite and non-negative", p.Name)
			}
		}
		if _, exists := l.policies[p.Name]; exists {
			return nil, fmt.Errorf("duplicate tenant %q", p.Name)
		}
		l.policies[p.Name] = p
		l.names = append(l.names, p.Name)
	}
	if err := b.ensureConfig(ctx, b.state.Key("qos:settings"), struct {
		Schema  int
		Tenants []TenantPolicy
	}{1, ps}); err != nil {
		return nil, err
	}
	return l, nil
}
func (l *redisLimiter) Policy(tenant string) TenantPolicy {
	if p, ok := l.policies[tenant]; ok {
		return p
	}
	return TenantPolicy{Name: tenant, Priority: PriorityNormal, Weight: 1}
}
func (l *redisLimiter) command(ctx context.Context, op, tenant, id string, estimate, actual int64) (limiterReply, error) {
	p := l.Policy(tenant)
	body, _ := json.Marshal(limiterPolicy{p.RPM, p.TPM, p.MaxConcurrency})
	base := l.backend.state.Key("qos:tenant:" + base64.RawURLEncoding.EncodeToString([]byte(tenant)) + ":")
	raw, err := l.backend.run(ctx, limiterScript, []string{base + "bucket", base + "inflight", l.backend.state.Key("leases")}, op, l.backend.state.OwnerID(), id, string(body), max(estimate, 0), actual)
	var r limiterReply
	if err == nil {
		err = json.Unmarshal([]byte(raw), &r)
	}
	if err != nil {
		return r, err
	}
	if r.Code == "owner_lost" {
		return r, l.backend.ownerLost()
	}
	if r.Code != "ok" && r.Code != "rejected" {
		return r, qosUnavailable("limiter", fmt.Errorf("unexpected result %q", r.Code))
	}
	return r, nil
}
func (l *redisLimiter) AdmitContext(ctx context.Context, tenant string, estimate int64) (*Lease, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if l.backend.closed.Load() {
		return nil, ErrClosed
	}
	id, err := uniqueID()
	if err != nil {
		return nil, err
	}
	finish := func(ctx context.Context, actual int64) error {
		_, err := l.command(ctx, "finish", tenant, id, 0, actual)
		return err
	}
	r, err := l.command(ctx, "admit", tenant, id, estimate, 0)
	if err != nil {
		// An ambiguous admission may already have consumed a slot/RPM/TPM.
		// Retain the request charge conservatively, but refund estimated TPM
		// and release its slot when storage recovers; no engine call is sent.
		l.backend.remember("limit:"+id, func(ctx context.Context) error { return finish(ctx, 0) })
		return nil, err
	}
	if r.Code == "rejected" {
		return nil, &Rejection{Reason: r.Reason, RetryAfter: time.Duration(r.RetryAfterMS) * time.Millisecond}
	}
	return &Lease{finish: func(actual int64) error {
		err := finish(context.Background(), actual)
		if err != nil {
			l.backend.remember("limit:"+id, func(ctx context.Context) error { return finish(ctx, actual) })
		}
		return err
	}}, nil
}
func (l *redisLimiter) StatsContext(ctx context.Context) ([]TenantStats, error) {
	out := make([]TenantStats, 0, len(l.names))
	for _, name := range l.names {
		r, err := l.command(ctx, "stats", name, "", 0, 0)
		if err != nil {
			return nil, err
		}
		p := l.policies[name]
		out = append(out, TenantStats{Name: name, Priority: p.Priority.String(), Inflight: r.Inflight, RPMAvailable: r.RPMAvailable, TPMAvailable: r.TPMAvailable, RPMLimit: p.RPM, TPMLimit: p.TPM, MaxConcurrent: p.MaxConcurrency})
	}
	return out, nil
}
