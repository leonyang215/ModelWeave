package registry

import (
	"context"
	"errors"
	"math"
	"time"
)

// Instance sources.
const (
	SourceStatic    = "static"    // listed in the gateway configuration
	SourceHeartbeat = "heartbeat" // registered through the mwip Registry
)

// Errors of store operations on one registration.
var (
	// ErrGone means the registration no longer exists: the instance left,
	// expired, or registered again under a newer epoch.
	ErrGone = errors.New("instance registration is gone")
	// ErrConflict means a conditional reservation was refused: the
	// instance's reservations exceed the caller's limits, or its engine
	// reported a new state, since the caller's snapshot.
	ErrConflict = errors.New("instance state changed since the snapshot")
	// ErrStoreUnavailable wraps failures to reach a shared store.
	ErrStoreUnavailable = errors.New("state store unavailable")
)

// NoLimit is the bound of an unbounded Limits field.
const NoLimit int64 = math.MaxInt64

// Store keeps the instance records a Registry is built on: registrations
// (spec, source, epoch), liveness, the reported engine state, drain flags,
// passive health and the gateways' reservations.
//
// The memory store keeps them in the gateway process. A shared store
// (package redisstate) keeps them where several gateway replicas read and
// write the same records; it timestamps with its own clock, so the now and
// at arguments only drive the memory store.
type Store interface {
	// Kind names the backend ("memory", "redis").
	Kind() string
	// Shared reports whether other gateway replicas change the store too:
	// a Registry then follows their changes (Sync, Run).
	Shared() bool

	// Upsert registers an instance or refreshes its liveness; beat, when
	// set, records a heartbeat's report in the same step. A registration
	// keeps its epoch until removal or an identity (model/role/target)
	// change; a static one keeps its spec against heartbeats.
	Upsert(ctx context.Context, spec Spec, source string, beat *Beat, now time.Time) (UpsertResult, error)
	// Remove deletes an instance's registration.
	Remove(ctx context.Context, id string) (Entry, uint64, bool, error)
	// Expire removes the registrations not refreshed for ttl and returns
	// them. Concurrent calls from several replicas remove each one once.
	Expire(ctx context.Context, ttl time.Duration, now time.Time) ([]Entry, uint64, error)
	// Members lists every registration when the membership changed since
	// version known (changed false otherwise).
	Members(ctx context.Context, known uint64) (m Membership, changed bool, err error)

	// Load reads the dynamic state of registrations, in order.
	Load(ctx context.Context, refs []Ref, now time.Time) ([]Record, error)
	// SetState records an engine state report.
	SetState(ctx context.Context, ref Ref, st EngineState, at time.Time, revision ...int64) error
	// SetDrain sets the operator's or the engine's drain flag.
	SetDrain(ctx context.Context, ref Ref, kind DrainKind, on bool) error
	// Reserve accounts a claim on the instance if its reservations, as the
	// store sees them now, are within lim and its admission revision is
	// still lim.Revision; otherwise it fails with ErrConflict (or ErrGone).
	Reserve(ctx context.Context, ref Ref, c Claim, lim Limits) error
	// Release gives back (part of) a claim; releases of a gone
	// registration are ignored.
	Release(ctx context.Context, ref Ref, c Claim) error
	// Health records the outcome of a call: a failure counts towards
	// ejecting the instance for cooldown after threshold consecutive ones
	// (ejected reports that this one did), a success resets the count.
	Health(ctx context.Context, ref Ref, failed bool, threshold int, cooldown time.Duration, now time.Time) (ejected bool, err error)

	// Claim takes a named lease for ttl if nobody holds it, so that one
	// replica does periodic work (an engine state poll) for all of them.
	Claim(ctx context.Context, name string, ttl time.Duration) (bool, error)
	// Maintain runs the store's periodic duties (a shared store: the
	// replica's lease, reconciling its reservations).
	Maintain(ctx context.Context) error
	// Close releases the store's resources.
	Close() error
}

// Entry is one registration of an instance.
type Entry struct {
	Spec   Spec   `json:"spec"`
	Source string `json:"source"` // SourceStatic or SourceHeartbeat
	// Epoch identifies the registration: an instance that registers again
	// after it left or expired gets a newer one.
	Epoch int64 `json:"epoch"`
}

// Ref names one registration in a store.
type Ref struct {
	ID    string
	Epoch int64
}

// Beat is the part of a heartbeat recorded with the registration.
type Beat struct {
	State    *EngineState // nil: the heartbeat carries no report
	Draining bool
}

// UpsertResult is the outcome of Store.Upsert.
type UpsertResult struct {
	Entry   Entry
	Created bool // a new registration
	Changed bool // the spec or the source of the registration changed
	// Version is the membership version after the call.
	Version uint64
}

// Membership lists every registration at one version of the membership
// (bumped by every registration, removal, spec or source change).
type Membership struct {
	Entries []Entry
	Version uint64
}

// DrainKind tells who drains an instance.
type DrainKind int

const (
	DrainAdmin  DrainKind = iota // an operator (admin API)
	DrainEngine                  // the engine (heartbeats)
)

// Record is the dynamic state of a registration.
type Record struct {
	// Gone: the registration no longer exists (the other fields are zero).
	Gone bool
	// Now is the store's clock at the read. A shared store uses its own
	// clock for report ages and health cooldowns, not the gateway's clock.
	Now   time.Time
	Entry Entry

	LastSeen time.Time // last heartbeat/registration
	State    EngineState
	StateAt  time.Time
	HasState bool
	// StateVer counts the state reports of the registration.
	StateVer int64
	// Revision changes on reports, spec, health and drain mutations. A
	// conditional reservation fences decisions made before those changes.
	Revision int64
	// PendingKV is the KV reserved since the last state report.
	PendingKV     int64
	Failures      int
	UnhealthyTill time.Time
	AdminDrain    bool
	EngineDrain   bool

	// Reservations of every live gateway sharing the store.
	Inflight   int64
	ReservedKV int64
	Prefill    int64
}

// Claim is the accounting of a request (or of a part of one) on an
// instance.
type Claim struct {
	// ID makes Reserve and Release idempotent, including after an
	// ambiguous network error. Requests=0 releases only the prefill part.
	ID       string
	Requests int64 // running requests
	KV       int64 // KV tokens (decode, unified and normal runs)
	Prefill  int64 // uncached prompt tokens to prefill (prefill and normal runs)
}

// Limits bound a conditional reservation. Each Max field bounds the
// instance's current value of the matching reservation total (NoLimit:
// unbounded); Revision, unless negative, must still be the instance's
// admission revision.
type Limits struct {
	Revision      int64
	MaxInflight   int64
	MaxReservedKV int64
	MaxPendingKV  int64
	MaxPrefill    int64
}

// Unlimited returns limits that accept any reservation.
func Unlimited() Limits {
	return Limits{Revision: -1, MaxInflight: NoLimit, MaxReservedKV: NoLimit, MaxPendingKV: NoLimit, MaxPrefill: NoLimit}
}

// Allow reports whether reservation totals are within the limits.
func (l Limits) Allow(inflight, reservedKV, pendingKV, prefill int64) bool {
	return inflight <= l.MaxInflight && reservedKV <= l.MaxReservedKV && pendingKV <= l.MaxPendingKV && prefill <= l.MaxPrefill
}
