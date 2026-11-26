package qos

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Priority is a request priority class. Lower values are served first.
type Priority int

const (
	PriorityCritical Priority = iota
	PriorityHigh
	PriorityNormal
	PriorityLow
	numPriorities
)

var priorityNames = [...]string{"critical", "high", "normal", "low"}

func (p Priority) String() string {
	if p >= 0 && p < numPriorities {
		return priorityNames[p]
	}
	return fmt.Sprintf("priority(%d)", int(p))
}

// ParsePriority parses a priority name; the empty string means normal.
func ParsePriority(s string) (Priority, error) {
	if s == "" {
		return PriorityNormal, nil
	}
	for i, n := range priorityNames {
		if strings.EqualFold(s, n) {
			return Priority(i), nil
		}
	}
	return PriorityNormal, fmt.Errorf("unknown priority %q (want critical, high, normal or low)", s)
}

// TenantPolicy holds the QoS settings of one tenant. Zero limits are unlimited.
type TenantPolicy struct {
	Name           string
	Priority       Priority
	Weight         float64 // fair-share weight inside a priority class
	RPM            float64 // requests per minute
	TPM            float64 // tokens (prompt + completion) per minute
	MaxConcurrency int
}

// Rejection is returned when a tenant exceeds a limit.
type Rejection struct {
	Reason     string // "rpm", "tpm", "concurrency"
	RetryAfter time.Duration
}

func (r *Rejection) Error() string {
	return fmt.Sprintf("rate limited (%s), retry after %s", r.Reason, r.RetryAfter.Round(time.Millisecond))
}

// Limiter enforces per-tenant RPM/TPM/concurrency limits.
type Limiter struct {
	mu      sync.Mutex
	tenants map[string]*tenantState
	closed  bool
}

type tenantState struct {
	policy   TenantPolicy
	rpm      *Bucket
	tpm      *Bucket
	inflight int
}

// NewLimiter returns a limiter configured with the given policies. Tenants
// that are not configured are unlimited.
func NewLimiter(policies []TenantPolicy) *Limiter {
	l := &Limiter{tenants: make(map[string]*tenantState)}
	for _, p := range policies {
		l.Set(p)
	}
	return l
}

// Set installs or replaces the policy of a tenant.
func (l *Limiter) Set(p TenantPolicy) {
	st := &tenantState{policy: p}
	if p.RPM > 0 {
		st.rpm = NewPerMinute(p.RPM, 0)
	}
	if p.TPM > 0 {
		st.tpm = NewPerMinute(p.TPM, 0)
	}
	l.mu.Lock()
	if old, ok := l.tenants[p.Name]; ok {
		st.inflight = old.inflight
	}
	l.tenants[p.Name] = st
	l.mu.Unlock()
}

// Policy returns the policy of tenant (an unlimited normal-priority policy if
// the tenant is unknown).
func (l *Limiter) Policy(tenant string) TenantPolicy {
	l.mu.Lock()
	defer l.mu.Unlock()
	if st, ok := l.tenants[tenant]; ok {
		return st.policy
	}
	return TenantPolicy{Name: tenant, Priority: PriorityNormal, Weight: 1}
}

// Lease is held by an admitted request until it finishes.
type Lease struct {
	l         *Limiter
	tenant    string
	estimated float64
	done      bool
	mu        sync.Mutex
	finish    func(int64) error // shared backend; it queues failed reconciliation
}

// Admit charges one request and estimatedTokens against the tenant's limits.
func (l *Limiter) Admit(tenant string, estimatedTokens int64) (*Lease, error) {
	return l.AdmitContext(context.Background(), tenant, estimatedTokens)
}

func (l *Limiter) AdmitContext(ctx context.Context, tenant string, estimatedTokens int64) (*Lease, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil, ErrClosed
	}
	st, ok := l.tenants[tenant]
	if !ok {
		return &Lease{l: l, tenant: tenant}, nil
	}
	if st.policy.MaxConcurrency > 0 && st.inflight >= st.policy.MaxConcurrency {
		return nil, &Rejection{Reason: "concurrency", RetryAfter: time.Second}
	}
	rpm, tpm := st.rpm, st.tpm
	if rpm != nil {
		if ok, wait := rpm.Take(1); !ok {
			return nil, &Rejection{Reason: "rpm", RetryAfter: wait}
		}
	}
	est := float64(max(estimatedTokens, 0))
	if tpm != nil && est > 0 {
		if ok, wait := tpm.Take(est); !ok {
			if rpm != nil {
				rpm.Adjust(1)
			}
			return nil, &Rejection{Reason: "tpm", RetryAfter: wait}
		}
	}
	st.inflight++
	return &Lease{l: l, tenant: tenant, estimated: est}, nil
}

// Finish releases the concurrency slot and reconciles the token estimate with
// the tokens actually used (prompt + completion). It is idempotent.
func (ls *Lease) Finish(actualTokens int64) error {
	if ls == nil {
		return nil
	}
	ls.mu.Lock()
	if ls.done {
		ls.mu.Unlock()
		return nil
	}
	ls.done = true
	ls.mu.Unlock()
	if ls.finish != nil {
		return ls.finish(actualTokens)
	}

	ls.l.mu.Lock()
	st, ok := ls.l.tenants[ls.tenant]
	if ok && st.inflight > 0 {
		st.inflight--
	}
	if ok && st.tpm != nil && actualTokens >= 0 {
		st.tpm.Adjust(ls.estimated - float64(actualTokens))
	}
	ls.l.mu.Unlock()
	return nil
}

// TenantStats is a snapshot used for metrics and the admin API.
type TenantStats struct {
	Name          string  `json:"name"`
	Priority      string  `json:"priority"`
	Inflight      int     `json:"inflight"`
	RPMAvailable  float64 `json:"rpm_available,omitempty"`
	TPMAvailable  float64 `json:"tpm_available,omitempty"`
	RPMLimit      float64 `json:"rpm_limit,omitempty"`
	TPMLimit      float64 `json:"tpm_limit,omitempty"`
	MaxConcurrent int     `json:"max_concurrency,omitempty"`
}

// Stats returns a snapshot of every configured tenant.
func (l *Limiter) Stats() []TenantStats {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]TenantStats, 0, len(l.tenants))
	for name, st := range l.tenants {
		s := TenantStats{
			Name: name, Priority: st.policy.Priority.String(), Inflight: st.inflight,
			RPMLimit: st.policy.RPM, TPMLimit: st.policy.TPM, MaxConcurrent: st.policy.MaxConcurrency,
		}
		if st.rpm != nil {
			s.RPMAvailable = st.rpm.Available()
		}
		if st.tpm != nil {
			s.TPMAvailable = st.tpm.Available()
		}
		out = append(out, s)
	}
	return out
}

func (l *Limiter) StatsContext(ctx context.Context) ([]TenantStats, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return l.Stats(), nil
}

func (l *Limiter) Close() { l.mu.Lock(); l.closed = true; l.mu.Unlock() }
