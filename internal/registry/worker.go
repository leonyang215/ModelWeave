// Package registry keeps the set of engine instances (workers), their
// reported engine state, and the gateway's own accounting of in-flight work
// per instance. Every instance serves one model in one role; the scheduler
// asks for the candidates of a model and role.
//
// Instances come from static configuration or register themselves with
// heartbeats on the mwip Registry; those expire when their heartbeats stop.
// Engine state arrives by push (heartbeats) or pull (Engine.GetState polls
// in package enginestate); both end up in Worker.UpdateState.
package registry

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// Role is the job a worker performs in a disaggregated deployment.
type Role string

const (
	RoleEncode  Role = "encode"  // multimodal encoder (E)
	RolePrefill Role = "prefill" // prefill-only (P)
	RoleDecode  Role = "decode"  // decode (D); also runs normal requests if Normal
	RoleUnified Role = "unified" // runs whole requests (normal mode only)
)

// ParseRole validates a role name.
func ParseRole(s string) (Role, error) {
	switch Role(s) {
	case RoleEncode, RolePrefill, RoleDecode, RoleUnified:
		return Role(s), nil
	}
	return "", fmt.Errorf("unknown worker role %q (want encode, prefill, decode or unified)", s)
}

// Spec is the identity and static description of a worker.
type Spec struct {
	ID    string `json:"id" yaml:"id"`
	Model string `json:"model" yaml:"model"`
	Role  Role   `json:"role" yaml:"role"`
	// Target is the instance's mwip target: where it serves the Engine
	// service ("10.0.0.7:9001"; "https://" for TLS).
	Target string `json:"target" yaml:"target"`
	// Normal reports that a decode worker also runs normal (non-PD)
	// requests: it prefills itself. Unified workers always do.
	Normal bool `json:"normal,omitempty" yaml:"normal,omitempty"`
	// BootstrapAddr is where decode instances fetch a prefill instance's KV
	// in the bootstrap PD handshake (engine-defined).
	BootstrapAddr string `json:"bootstrap_addr,omitempty" yaml:"bootstrap_addr,omitempty"`
	// EngineID is the instance's id in the engines' KV connector; defaults
	// to the instance ID.
	EngineID string `json:"engine_id,omitempty" yaml:"engine_id,omitempty"`
	// Static capacity hints; reported engine state takes precedence.
	KVCapacityTokens   int64             `json:"kv_capacity_tokens,omitempty" yaml:"kv_capacity_tokens,omitempty"`
	MaxRunningRequests int               `json:"max_running_requests,omitempty" yaml:"max_running_requests,omitempty"`
	MaxPrefillTokens   int64             `json:"max_prefill_queue_tokens,omitempty" yaml:"max_prefill_queue_tokens,omitempty"`
	Labels             map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
}

// ServesNormal reports whether the worker runs normal (non-PD) requests.
func (s Spec) ServesNormal() bool {
	return s.Role == RoleUnified || (s.Role == RoleDecode && s.Normal)
}

// EngineState is the load reported by an engine. Zero values mean unknown.
type EngineState struct {
	RunningRequests     int     `json:"running_requests"`
	WaitingRequests     int     `json:"waiting_requests"`
	KVUsedTokens        int64   `json:"kv_used_tokens"`
	KVCapacityTokens    int64   `json:"kv_capacity_tokens"`
	KVUsage             float64 `json:"kv_usage,omitempty"` // 0..1, used when tokens are not reported
	PrefillQueueTokens  int64   `json:"prefill_queue_tokens"`
	MaxRunningRequests  int     `json:"max_running_requests,omitempty"`
	DecodeTokensPerSec  float64 `json:"decode_tokens_per_sec,omitempty"`
	PrefillTokensPerSec float64 `json:"prefill_tokens_per_sec,omitempty"`
	CachedBlocks        int64   `json:"cached_blocks,omitempty"`
	Draining            bool    `json:"draining,omitempty"`
}

// Worker is an immutable handle to one registration. Dynamic state and
// reservations live in its Store, shared by all gateway replicas in Redis
// mode. An epoch prevents late polls/releases from modifying a re-registration.
type Worker struct {
	entry     Entry
	store     Store
	completed atomic.Int64 // per-gateway counters; sum across replicas
	errors    atomic.Int64
}

// Spec returns the worker's current spec.
func (w *Worker) Spec() Spec { return w.entry.Spec }

// ID is shorthand for Spec().ID.
func (w *Worker) ID() string { return w.entry.Spec.ID }

func (w *Worker) Source() string { return w.entry.Source }
func (w *Worker) Ref() Ref       { return Ref{ID: w.ID(), Epoch: w.entry.Epoch} }

// CacheKey scopes advisory state to this registration, so a new instance
// reusing an ID cannot inherit its predecessor's prefixes or sessions.
func (w *Worker) CacheKey() string { return fmt.Sprintf("%s@%d", w.ID(), w.entry.Epoch) }

// UpdateState records an engine state report (a heartbeat's or a poll's).
// It does not refresh liveness: registration lives on heartbeats alone.
func (w *Worker) UpdateState(st EngineState, at time.Time) error {
	return w.UpdateStateContext(context.Background(), st, at)
}

func (w *Worker) UpdateStateContext(ctx context.Context, st EngineState, at time.Time) error {
	if err := st.Validate(); err != nil {
		return err
	}
	return w.store.SetState(ctx, w.Ref(), st, at)
}

// UpdateStateIfRevision prevents a slow GetState response from overwriting a
// heartbeat/state mutation that arrived while the poll was in flight.
func (w *Worker) UpdateStateIfRevision(ctx context.Context, st EngineState, at time.Time, revision int64) error {
	if err := st.Validate(); err != nil {
		return err
	}
	return w.store.SetState(ctx, w.Ref(), st, at, revision)
}

func (st EngineState) Validate() error {
	if st.RunningRequests < 0 || st.WaitingRequests < 0 || st.KVUsedTokens < 0 || st.KVCapacityTokens < 0 || st.PrefillQueueTokens < 0 || st.MaxRunningRequests < 0 || st.CachedBlocks < 0 {
		return fmt.Errorf("engine state counters must be non-negative")
	}
	for _, v := range []float64{st.KVUsage, st.DecodeTokensPerSec, st.PrefillTokensPerSec} {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("engine state ratios/throughput must be finite and non-negative")
		}
	}
	return nil
}

func (s Spec) Validate() error {
	if s.ID == "" || s.Model == "" || s.Target == "" {
		return fmt.Errorf("worker spec needs id, model and target")
	}
	if _, err := ParseRole(string(s.Role)); err != nil {
		return err
	}
	if s.KVCapacityTokens < 0 || s.MaxRunningRequests < 0 || s.MaxPrefillTokens < 0 {
		return fmt.Errorf("worker capacity hints must be non-negative")
	}
	return nil
}

// LastSeen is the last heartbeat/registration; state polls do not refresh it.
func (w *Worker) LastSeen() time.Time {
	return w.Snapshot(time.Now(), 0).LastSeen
}

// SetDraining drains the worker for an operator (admin API): it takes no
// new requests until undrained, whatever the engine reports.
func (w *Worker) SetDraining(d bool) error {
	return w.SetDrainingContext(context.Background(), d)
}

func (w *Worker) SetDrainingContext(ctx context.Context, d bool) error {
	return w.store.SetDrain(ctx, w.Ref(), DrainAdmin, d)
}

// SetEngineDraining records whether the engine reports draining (its
// heartbeats).
func (w *Worker) SetEngineDraining(d bool) error {
	return w.store.SetDrain(context.Background(), w.Ref(), DrainEngine, d)
}

// AdminDrained reports whether an operator drained the worker.
func (w *Worker) AdminDrained() bool {
	return w.Snapshot(time.Now(), 0).AdminDrained
}

// Snapshot is an immutable view of a worker used by one scheduling pass.
type Snapshot struct {
	Worker       *Worker
	Spec         Spec
	Gone         bool
	Revision     int64
	AdminDrained bool
	Blocked      bool
	Block        *Block

	HasState      bool
	StateReported bool
	State         EngineState
	StateAge      time.Duration
	Healthy       bool
	Draining      bool
	LastSeen      time.Time
	Failures      int
	Inflight      int64
	PrefillTok    int64 // gateway-tracked uncached prefill tokens in flight
	ReservedKV    int64 // gateway-tracked KV reservation
	// EffectiveKVUsed merges the reported usage with reservations the
	// report does not reflect yet (and with all live gateways' reservations).
	EffectiveKVUsed  int64
	KVCapacity       int64
	MaxRunning       int
	MaxPrefillTokens int64
}

// KVFree returns the estimated free KV tokens (math.MaxInt64 when unknown).
func (s *Snapshot) KVFree() int64 {
	if s.KVCapacity <= 0 {
		return 1<<62 - 1
	}
	return s.KVCapacity - s.EffectiveKVUsed
}

// KVUtil returns KV utilization in [0, 1+] (0 when unknown).
func (s *Snapshot) KVUtil() float64 {
	if s.KVCapacity > 0 {
		return float64(s.EffectiveKVUsed) / float64(s.KVCapacity)
	}
	if s.HasState {
		return s.State.KVUsage
	}
	return 0
}

// QueuedPrefillTokens estimates prefill work waiting on the worker.
func (s *Snapshot) QueuedPrefillTokens() int64 {
	q := s.PrefillTok
	if s.HasState && s.State.PrefillQueueTokens > q {
		q = s.State.PrefillQueueTokens
	}
	return q
}

// Running returns the larger of reported running+waiting and live in-flight.
func (s *Snapshot) Running() int64 {
	r := s.Inflight
	if s.HasState {
		if rep := int64(s.State.RunningRequests + s.State.WaitingRequests); rep > r {
			r = rep
		}
	}
	return r
}

// ReportedKVUsed resolves either engine usage representation. Saturating a
// ratio at capacity also prevents huge but finite ratios from overflowing an
// int64 and accidentally making an overloaded instance appear idle.
func (s *Snapshot) ReportedKVUsed() int64 {
	if !s.HasState {
		return 0
	}
	if s.State.KVUsedTokens > 0 {
		return s.State.KVUsedTokens
	}
	if s.State.KVUsage >= 1 {
		return s.KVCapacity
	}
	if s.State.KVUsage > 0 && s.KVCapacity > 0 {
		return int64(s.State.KVUsage * float64(s.KVCapacity))
	}
	return 0
}

// Snapshot captures the worker state. staleAfter bounds how long a state
// report is trusted.
func (w *Worker) Snapshot(now time.Time, staleAfter time.Duration) Snapshot {
	s, err := w.SnapshotContext(context.Background(), now, staleAfter)
	if err != nil {
		return Snapshot{Worker: w, Spec: w.Spec(), Gone: true}
	}
	return s
}

func (w *Worker) SnapshotContext(ctx context.Context, now time.Time, staleAfter time.Duration) (Snapshot, error) {
	recs, err := w.store.Load(ctx, []Ref{w.Ref()}, now)
	if err != nil {
		return Snapshot{}, err
	}
	return w.snapshot(recs[0], now, staleAfter), nil
}

func (w *Worker) snapshot(rec Record, now time.Time, staleAfter time.Duration) Snapshot {
	s := Snapshot{
		Worker:       w,
		Spec:         rec.Entry.Spec,
		Gone:         rec.Gone,
		Revision:     rec.Revision,
		LastSeen:     rec.LastSeen,
		Failures:     rec.Failures,
		Draining:     rec.AdminDrain || rec.EngineDrain,
		AdminDrained: rec.AdminDrain,
		Blocked:      rec.Block != nil,
		Block:        rec.Block,
	}
	if rec.Gone {
		s.Spec = w.Spec()
		return s
	}
	clock := now
	if !rec.Now.IsZero() {
		clock = rec.Now
		// Expose liveness in the caller's clock while computing the age
		// and cooldown using the store's clock.
		s.LastSeen = now.Add(-clock.Sub(rec.LastSeen))
	}
	if rec.HasState {
		s.StateReported = true
		s.StateAge = max(clock.Sub(rec.StateAt), 0)
	}
	if rec.HasState && (staleAfter <= 0 || clock.Sub(rec.StateAt) <= staleAfter) {
		s.HasState = true
		s.State = rec.State
		s.Draining = s.Draining || rec.State.Draining
	}
	s.Healthy = !clock.Before(rec.UnhealthyTill)
	s.Inflight, s.PrefillTok, s.ReservedKV = rec.Inflight, rec.Prefill, rec.ReservedKV
	s.KVCapacity = s.Spec.KVCapacityTokens
	s.MaxRunning = s.Spec.MaxRunningRequests
	s.MaxPrefillTokens = s.Spec.MaxPrefillTokens
	used := s.ReservedKV
	if s.HasState {
		if s.State.KVCapacityTokens > 0 {
			s.KVCapacity = s.State.KVCapacityTokens
		}
		if s.State.MaxRunningRequests > 0 {
			s.MaxRunning = s.State.MaxRunningRequests
		}
		reported := s.ReportedKVUsed()
		pending := min(rec.PendingKV, math.MaxInt64-reported)
		if r := reported + pending; r > used {
			used = r
		}
	}
	s.EffectiveKVUsed = used
	return s
}

// Reservation is the gateway-side accounting of one request on one worker.
// Release must be called exactly once (it is idempotent).
type Reservation struct {
	w           *Worker
	claim       Claim
	mu          sync.Mutex
	released    bool
	prefillDone bool
	onFree      func()
}

// Reserve accounts a request on the worker: kvTokens of KV memory (decode
// side) and prefillTokens of uncached prefill work (prefill side).
func (w *Worker) Reserve(kvTokens, prefillTokens int64, onFree func()) *Reservation {
	r, _ := w.TryReserve(context.Background(), kvTokens, prefillTokens, Unlimited(), onFree)
	return r
}

// TryReserve conditionally reserves capacity at the store's linearization
// point, not under a process-local scheduler lock.
func (w *Worker) TryReserve(ctx context.Context, kvTokens, prefillTokens int64, lim Limits, onFree func()) (*Reservation, error) {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return nil, err
	}
	c := Claim{ID: hex.EncodeToString(id[:]), Requests: 1, KV: max(kvTokens, 0), Prefill: max(prefillTokens, 0)}
	if err := w.store.Reserve(ctx, w.Ref(), c, lim); err != nil {
		return nil, err
	}
	return &Reservation{w: w, claim: c, onFree: onFree}, nil
}

// Worker returns the reserved worker.
func (r *Reservation) Worker() *Worker { return r.w }

// PrefillDone releases the prefill part early (prefill finished, decode
// continues).
func (r *Reservation) PrefillDone() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.released || r.prefillDone || r.claim.Prefill == 0 {
		return nil
	}
	r.prefillDone = true
	c := r.claim
	c.Requests = 0
	return r.w.store.Release(context.Background(), r.w.Ref(), c)
}

// Release frees the reservation.
func (r *Reservation) Release(success bool) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	if r.released {
		r.mu.Unlock()
		return nil
	}
	r.released = true
	err := r.w.store.Release(context.Background(), r.w.Ref(), r.claim)
	r.mu.Unlock()
	if success {
		r.w.completed.Add(1)
	}
	if r.onFree != nil {
		r.onFree()
	}
	return err
}

// ReportSuccess resets the failure counter (passive health checking).
func (w *Worker) ReportSuccess() {
	_, _ = w.store.Health(context.Background(), w.Ref(), false, 0, 0, time.Now())
}

// ReportFailure records a transport-level failure. After threshold
// consecutive failures the worker is ejected for cooldown.
func (w *Worker) ReportFailure(now time.Time, threshold int, cooldown time.Duration) (ejected bool) {
	w.errors.Add(1)
	ejected, _ = w.store.Health(context.Background(), w.Ref(), true, threshold, cooldown, now)
	return ejected
}

// Counters returns lifetime completed and error counts.
func (w *Worker) Counters() (completed, errors int64) {
	return w.completed.Load(), w.errors.Load()
}
