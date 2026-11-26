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
	"fmt"
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

// Worker is a registered worker with its state and local accounting.
type Worker struct {
	spec   atomic.Pointer[Spec]
	Source string // "static" or "heartbeat"

	mu            sync.Mutex
	state         EngineState
	stateAt       time.Time
	hasState      bool
	lastSeen      time.Time
	pendingSince  int64 // KV tokens reserved since the last state report
	failures      int
	unhealthyTill time.Time
	draining      bool // drained by an operator (admin API)
	engineDrain   bool // the engine reports draining (heartbeat)

	inflight        atomic.Int64 // requests currently assigned (any stage)
	inflightPrefill atomic.Int64 // uncached prompt tokens being prefilled
	reservedKV      atomic.Int64 // KV tokens reserved by in-flight requests
	completed       atomic.Int64
	errors          atomic.Int64
}

// Spec returns the worker's current spec.
func (w *Worker) Spec() Spec { return *w.spec.Load() }

// ID is shorthand for Spec().ID.
func (w *Worker) ID() string { return w.spec.Load().ID }

// UpdateState records an engine state report (a heartbeat's or a poll's).
// It does not refresh liveness: registration lives on heartbeats alone.
func (w *Worker) UpdateState(st EngineState, at time.Time) {
	w.mu.Lock()
	w.state = st
	w.stateAt = at
	w.hasState = true
	w.pendingSince = 0
	w.mu.Unlock()
}

// LastSeen is when the worker last reported (a heartbeat or a state poll).
func (w *Worker) LastSeen() time.Time {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.lastSeen
}

// Touch records liveness without a state report.
func (w *Worker) Touch(at time.Time) {
	w.mu.Lock()
	w.lastSeen = at
	w.mu.Unlock()
}

// SetDraining drains the worker for an operator (admin API): it takes no
// new requests until undrained, whatever the engine reports.
func (w *Worker) SetDraining(d bool) {
	w.mu.Lock()
	w.draining = d
	w.mu.Unlock()
}

// SetEngineDraining records whether the engine reports draining (its
// heartbeats).
func (w *Worker) SetEngineDraining(d bool) {
	w.mu.Lock()
	w.engineDrain = d
	w.mu.Unlock()
}

// AdminDrained reports whether an operator drained the worker.
func (w *Worker) AdminDrained() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.draining
}

// Snapshot is an immutable view of a worker used by one scheduling pass.
type Snapshot struct {
	Worker *Worker
	Spec   Spec

	HasState   bool
	State      EngineState
	StateAge   time.Duration
	Healthy    bool
	Draining   bool
	LastSeen   time.Time
	Failures   int
	Inflight   int64
	PrefillTok int64 // gateway-tracked uncached prefill tokens in flight
	ReservedKV int64 // gateway-tracked KV reservation
	// EffectiveKVUsed merges the reported usage with reservations the
	// report does not reflect yet (and with this gateway's reservations).
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

// Running returns the larger of reported running+waiting and local in-flight.
func (s *Snapshot) Running() int64 {
	r := s.Inflight
	if s.HasState {
		if rep := int64(s.State.RunningRequests + s.State.WaitingRequests); rep > r {
			r = rep
		}
	}
	return r
}

// Snapshot captures the worker state. staleAfter bounds how long a state
// report is trusted.
func (w *Worker) Snapshot(now time.Time, staleAfter time.Duration) Snapshot {
	spec := w.Spec()
	w.mu.Lock()
	s := Snapshot{
		Worker:   w,
		Spec:     spec,
		LastSeen: w.lastSeen,
		Failures: w.failures,
		Draining: w.draining || w.engineDrain,
	}
	if w.hasState && (staleAfter <= 0 || now.Sub(w.stateAt) <= staleAfter) {
		s.HasState = true
		s.State = w.state
		s.StateAge = now.Sub(w.stateAt)
		s.Draining = s.Draining || w.state.Draining
	}
	pending := w.pendingSince
	s.Healthy = !now.Before(w.unhealthyTill)
	w.mu.Unlock()

	s.Inflight = w.inflight.Load()
	s.PrefillTok = w.inflightPrefill.Load()
	s.ReservedKV = w.reservedKV.Load()

	s.KVCapacity = spec.KVCapacityTokens
	s.MaxRunning = spec.MaxRunningRequests
	s.MaxPrefillTokens = spec.MaxPrefillTokens
	used := s.ReservedKV
	if s.HasState {
		if s.State.KVCapacityTokens > 0 {
			s.KVCapacity = s.State.KVCapacityTokens
		}
		if s.State.MaxRunningRequests > 0 {
			s.MaxRunning = s.State.MaxRunningRequests
		}
		reported := s.State.KVUsedTokens
		if reported == 0 && s.State.KVUsage > 0 && s.KVCapacity > 0 {
			reported = int64(s.State.KVUsage * float64(s.KVCapacity))
		}
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
	w        *Worker
	kv       int64
	prefill  int64
	released atomic.Bool
	onFree   func()
}

// Reserve accounts a request on the worker: kvTokens of KV memory (decode
// side) and prefillTokens of uncached prefill work (prefill side).
func (w *Worker) Reserve(kvTokens, prefillTokens int64, onFree func()) *Reservation {
	w.inflight.Add(1)
	if kvTokens > 0 {
		w.reservedKV.Add(kvTokens)
		w.mu.Lock()
		w.pendingSince += kvTokens
		w.mu.Unlock()
	}
	if prefillTokens > 0 {
		w.inflightPrefill.Add(prefillTokens)
	}
	return &Reservation{w: w, kv: kvTokens, prefill: prefillTokens, onFree: onFree}
}

// Worker returns the reserved worker.
func (r *Reservation) Worker() *Worker { return r.w }

// PrefillDone releases the prefill part early (prefill finished, decode
// continues).
func (r *Reservation) PrefillDone() {
	if r == nil || r.prefill == 0 {
		return
	}
	if p := r.prefill; p > 0 {
		r.prefill = 0
		r.w.inflightPrefill.Add(-p)
	}
}

// Release frees the reservation.
func (r *Reservation) Release(success bool) {
	if r == nil || !r.released.CompareAndSwap(false, true) {
		return
	}
	r.w.inflight.Add(-1)
	if r.kv > 0 {
		r.w.reservedKV.Add(-r.kv)
	}
	if r.prefill > 0 {
		r.w.inflightPrefill.Add(-r.prefill)
	}
	if success {
		r.w.completed.Add(1)
	}
	if r.onFree != nil {
		r.onFree()
	}
}

// ReportSuccess resets the failure counter (passive health checking).
func (w *Worker) ReportSuccess() {
	w.mu.Lock()
	w.failures = 0
	w.mu.Unlock()
}

// ReportFailure records a transport-level failure. After threshold
// consecutive failures the worker is ejected for cooldown.
func (w *Worker) ReportFailure(now time.Time, threshold int, cooldown time.Duration) (ejected bool) {
	w.errors.Add(1)
	w.mu.Lock()
	defer w.mu.Unlock()
	w.failures++
	if threshold > 0 && w.failures >= threshold {
		w.unhealthyTill = now.Add(cooldown)
		w.failures = 0
		return true
	}
	return false
}

// Counters returns lifetime completed and error counts.
func (w *Worker) Counters() (completed, errors int64) {
	return w.completed.Load(), w.errors.Load()
}
