package qos

import (
	"context"
	"sync"
)

// TenantLimiter has identical semantics with process-local or shared state.
// Storage failures are errors, never an unlimited/local fallback.
type TenantLimiter interface {
	Policy(string) TenantPolicy
	AdmitContext(context.Context, string, int64) (*Lease, error)
	StatsContext(context.Context) ([]TenantStats, error)
}

// Attempt is the resource reservation produced by a scheduling attempt.
// Release must undo a ready attempt if admission cannot be committed (for
// example cancellation, loss of its turn or an ambiguous Redis response).
type Attempt struct {
	Ready   bool
	Release func()
}

type TryAdmit func(context.Context) (Attempt, error)

type QueueStats struct {
	Total int                `json:"total"`
	Depth [numPriorities]int `json:"depth"`
}

// AdmissionQueue gates the fast path too: a new request cannot jump ahead of
// existing waiters just because a worker has become free on another gateway.
type AdmissionQueue interface {
	Admit(context.Context, Ticket, TryAdmit) (queued bool, err error)
	Notify()
	LenContext(context.Context) (int, error)
	StatsContext(context.Context) (QueueStats, error)
	Close()
}

// Backend selects the complete QoS implementation, not just its counters.
// Maintain retries idempotent cleanup after transient storage errors. Close
// stops this gateway's waiters, never a sibling's work. Redis request leases
// share the instance store's owner lifetime, closed by the gateway after QoS.
type Backend interface {
	Kind() string
	Limiter() TenantLimiter
	NewQueue(context.Context, string, QueueConfig, QueueObserver) (AdmissionQueue, error)
	Maintain(context.Context) error
	Close()
}

type memoryBackend struct {
	limiter *Limiter
	mu      sync.Mutex
	queues  []*Queue
	closed  bool
}

func NewMemoryBackend(policies []TenantPolicy) Backend {
	return &memoryBackend{limiter: NewLimiter(policies)}
}
func (b *memoryBackend) Kind() string                   { return "memory" }
func (b *memoryBackend) Limiter() TenantLimiter         { return b.limiter }
func (b *memoryBackend) Maintain(context.Context) error { return nil }
func (b *memoryBackend) NewQueue(ctx context.Context, _ string, cfg QueueConfig, obs QueueObserver) (AdmissionQueue, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil, ErrClosed
	}
	q := NewQueue(cfg, obs)
	b.queues = append(b.queues, q)
	return q, nil
}
func (b *memoryBackend) Close() {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.closed = true
	qs := append([]*Queue(nil), b.queues...)
	b.mu.Unlock()
	b.limiter.Close()
	for _, q := range qs {
		q.Close()
	}
}
