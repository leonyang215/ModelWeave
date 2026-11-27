// Package redisstate implements the shared instance state store. Mutations
// and conditional capacity reservations are atomic Lua operations on Redis.
package redisstate

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/leonyang215/ModelWeave/internal/admission"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

//go:embed state.lua
var source string
var script = redis.NewScript(source)

type Options struct {
	URL       string
	Namespace string
	Timeout   time.Duration
	LeaseTTL  time.Duration
}

// Store owns one gateway's reservation lease. Prefix/session/artifact caches
// can use Client and Key to share its Redis connection pool and namespace.
type Store struct {
	Client            *redis.Client
	root, owner       string
	timeout, leaseTTL time.Duration
	lost              atomic.Bool
	mu                sync.Mutex
	releases          map[releaseKey]registry.Claim
	once              sync.Once
	closeErr          error
}

type releaseKey struct {
	ref registry.Ref
	id  string
}

var _ registry.Store = (*Store)(nil)

func New(ctx context.Context, opts Options) (*Store, error) {
	if opts.Timeout <= 0 {
		opts.Timeout = time.Second
	}
	if opts.LeaseTTL <= 0 {
		opts.LeaseTTL = 15 * time.Second
	}
	if opts.Namespace == "" {
		opts.Namespace = "modelweave"
	}
	if opts.LeaseTTL < 3*opts.Timeout {
		return nil, fmt.Errorf("state lease TTL must be at least 3 times the operation timeout")
	}
	ro, err := redis.ParseURL(opts.URL)
	if err != nil {
		return nil, fmt.Errorf("invalid state.redis.url")
	} // never echo credentials
	ro.DialTimeout, ro.ReadTimeout, ro.WriteTimeout = opts.Timeout, opts.Timeout, opts.Timeout
	ro.ContextTimeoutEnabled, ro.DisableIdentity, ro.Protocol, ro.MaxRetries = true, true, 2, -1
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return nil, err
	}
	ns := sha256.Sum256([]byte(opts.Namespace))
	s := &Store{Client: redis.NewClient(ro), root: "modelweave:{" + hex.EncodeToString(ns[:16]) + "}:", owner: hex.EncodeToString(id[:]), timeout: opts.Timeout, leaseTTL: opts.LeaseTTL, releases: make(map[releaseKey]registry.Claim)}
	ctx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()
	if err := s.Client.Ping(ctx).Err(); err != nil {
		_ = s.Client.Close()
		return nil, unavailable("connect", err)
	}
	if err := s.integer(ctx, "join", "", opts.LeaseTTL.Milliseconds()); err != nil {
		_ = s.Client.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Kind() string             { return "redis" }
func (s *Store) Shared() bool             { return true }
func (s *Store) Key(suffix string) string { return s.root + suffix }

// OwnerID and OperationContext let the distributed QoS backend use the same
// fenced gateway lease and operation timeout as instance reservations.
func (s *Store) OwnerID() string { return s.owner }
func (s *Store) OperationContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, s.timeout)
}

// CheckSettings prevents replicas with incompatible liveness/hash settings
// (or a different storage schema) from silently sharing a namespace.
func (s *Store) CheckSettings(ctx context.Context, heartbeatTTL time.Duration, blockSize int) error {
	b, _ := json.Marshal(struct {
		Schema       int   `json:"schema"`
		HeartbeatTTL int64 `json:"heartbeat_ttl_ms"`
		BlockSize    int   `json:"block_size"`
	}{3, heartbeatTTL.Milliseconds(), blockSize})
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if err := s.Client.SetNX(ctx, s.Key("settings"), string(b), 0).Err(); err != nil {
		return unavailable("settings", err)
	}
	got, err := s.Client.Get(ctx, s.Key("settings")).Result()
	if err != nil {
		return unavailable("settings", err)
	}
	if got != string(b) {
		return fmt.Errorf("Redis state namespace has incompatible schema, heartbeat_ttl or block_size; use consistent settings or another namespace")
	}
	return nil
}

// Context bounds cache operations that have no request context.
func (s *Store) Context() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), s.timeout)
}

func unavailable(op string, err error) error {
	return fmt.Errorf("%w: redis %s: %v", registry.ErrStoreUnavailable, op, err)
}

func (s *Store) command(ctx context.Context, op, id string, args ...any) *redis.Cmd {
	keys := []string{s.Key("members"), s.Key("version"), s.Key("epochs"), s.Key("leases"), s.Key("liveness"), s.Key("worker:" + id), s.Key("claims:" + id)}
	p := admission.FromContext(ctx)
	control := p.Control
	if control == "" {
		control = s.Key("no-admission-turn")
	}
	keys = append(keys, control)
	all := []any{op, s.root, s.owner, id}
	all = append(all, args...)
	return script.Run(ctx, s.Client, keys, all...)
}

func (s *Store) integer(ctx context.Context, op, id string, args ...any) error {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	n, err := s.command(ctx, op, id, args...).Int()
	if err != nil {
		return unavailable(op, err)
	}
	switch n {
	case -1:
		return registry.ErrGone
	case -2:
		return registry.ErrConflict
	case -3:
		s.lost.Store(true)
		return unavailable(op, errors.New("gateway reservation lease expired; restart this gateway"))
	case -4:
		return admission.ErrTurnLost
	}
	return nil
}

func (s *Store) Upsert(ctx context.Context, spec registry.Spec, source string, beat *registry.Beat, _ time.Time) (registry.UpsertResult, error) {
	b, err := json.Marshal(spec)
	if err != nil {
		return registry.UpsertResult{}, err
	}
	state, drain := "", ""
	if beat != nil {
		drain = strconv.Itoa(boolInt(beat.Draining))
		if beat.State != nil {
			b, err := json.Marshal(beat.State)
			if err != nil {
				return registry.UpsertResult{}, err
			}
			state = string(b)
		}
	}
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	raw, err := s.command(ctx, "upsert", spec.ID, string(b), source, state, drain).Text()
	var res registry.UpsertResult
	if err == nil {
		err = json.Unmarshal([]byte(raw), &res)
	}
	if err != nil {
		return res, unavailable("upsert", err)
	}
	return res, nil
}

func (s *Store) Members(ctx context.Context, known uint64) (registry.Membership, bool, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	a, err := s.command(ctx, "members", "", known).StringSlice()
	if err != nil {
		return registry.Membership{}, false, unavailable("members", err)
	}
	v, err := strconv.ParseUint(a[0], 10, 64)
	if err != nil {
		return registry.Membership{}, false, unavailable("members", err)
	}
	m := registry.Membership{Version: v}
	for _, raw := range a[1:] {
		var e registry.Entry
		if err := json.Unmarshal([]byte(raw), &e); err != nil {
			return m, false, unavailable("members", err)
		}
		m.Entries = append(m.Entries, e)
	}
	return m, v != known, nil
}

func (s *Store) Remove(ctx context.Context, id string) (registry.Entry, uint64, bool, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	a, err := s.command(ctx, "remove", id).StringSlice()
	var e registry.Entry
	if err != nil {
		return e, 0, false, unavailable("remove", err)
	}
	v, _ := strconv.ParseUint(a[0], 10, 64)
	if a[1] == "" {
		return e, v, false, nil
	}
	if err := json.Unmarshal([]byte(a[1]), &e); err != nil {
		return e, v, false, unavailable("remove", err)
	}
	return e, v, true, nil
}

func (s *Store) Expire(ctx context.Context, ttl time.Duration, _ time.Time) ([]registry.Entry, uint64, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	a, err := s.command(ctx, "expire", "", ttl.Milliseconds()).StringSlice()
	if err != nil {
		return nil, 0, unavailable("expire", err)
	}
	v, _ := strconv.ParseUint(a[0], 10, 64)
	out := make([]registry.Entry, 0, len(a)-1)
	for _, raw := range a[1:] {
		var e registry.Entry
		if err := json.Unmarshal([]byte(raw), &e); err != nil {
			return nil, v, unavailable("expire", err)
		}
		out = append(out, e)
	}
	return out, v, nil
}

type wireRecord struct {
	Gone          bool                  `json:"gone"`
	Entry         registry.Entry        `json:"entry"`
	Now           int64                 `json:"now"`
	LastSeen      int64                 `json:"last_seen"`
	StateAt       int64                 `json:"state_at"`
	State         *registry.EngineState `json:"state"`
	StateVer      int64                 `json:"state_ver"`
	Revision      int64                 `json:"revision"`
	PendingKV     int64                 `json:"pending_kv"`
	Inflight      int64                 `json:"inflight"`
	ReservedKV    int64                 `json:"reserved_kv"`
	Prefill       int64                 `json:"prefill"`
	Failures      int                   `json:"failures"`
	UnhealthyTill int64                 `json:"unhealthy_till"`
	AdminDrain    bool                  `json:"admin_drain"`
	EngineDrain   bool                  `json:"engine_drain"`
	Block         *registry.Block       `json:"block"`
}

func (s *Store) Load(ctx context.Context, refs []registry.Ref, _ time.Time) ([]registry.Record, error) {
	if len(refs) == 0 {
		return []registry.Record{}, nil
	}
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	// EVALSHA keeps the request small even for large candidate pools. A
	// NOSCRIPT after SCRIPT FLUSH/reconnect is retried through Script.Run;
	// correctness never relies on a process-local script-cache flag.
	pipe := s.Client.Pipeline()
	cmds := make([]*redis.Cmd, len(refs))
	for i, ref := range refs {
		keys := []string{s.Key("members"), s.Key("version"), s.Key("epochs"), s.Key("leases"), s.Key("liveness"), s.Key("worker:" + ref.ID), s.Key("claims:" + ref.ID)}
		cmds[i] = pipe.EvalSha(ctx, script.Hash(), keys, "load", s.root, s.owner, ref.ID, ref.Epoch)
	}
	if _, err := pipe.Exec(ctx); err != nil && !strings.HasPrefix(err.Error(), "NOSCRIPT") {
		return nil, unavailable("load", err)
	}
	out := make([]registry.Record, len(refs))
	for i, cmd := range cmds {
		if err := cmd.Err(); err != nil && strings.HasPrefix(err.Error(), "NOSCRIPT") {
			cmd = s.command(ctx, "load", refs[i].ID, refs[i].Epoch)
		}
		raw, err := cmd.Text()
		var w wireRecord
		if err == nil {
			err = json.Unmarshal([]byte(raw), &w)
		}
		if err != nil {
			return nil, unavailable("load", err)
		}
		r := registry.Record{Gone: w.Gone, Entry: w.Entry, Now: time.UnixMilli(w.Now), LastSeen: time.UnixMilli(w.LastSeen), StateAt: time.UnixMilli(w.StateAt), StateVer: w.StateVer, Revision: w.Revision, PendingKV: w.PendingKV, Inflight: w.Inflight, ReservedKV: w.ReservedKV, Prefill: w.Prefill, Failures: w.Failures, AdminDrain: w.AdminDrain, EngineDrain: w.EngineDrain, Block: w.Block, UnhealthyTill: time.UnixMilli(w.UnhealthyTill)}
		if w.State != nil {
			r.HasState, r.State = true, *w.State
		}
		out[i] = r
	}
	return out, nil
}

func (s *Store) SetState(ctx context.Context, ref registry.Ref, st registry.EngineState, _ time.Time, revision ...int64) error {
	b, err := json.Marshal(st)
	if err != nil {
		return err
	}
	expected := int64(-1)
	if len(revision) > 0 {
		expected = revision[0]
	}
	return s.integer(ctx, "state", ref.ID, ref.Epoch, string(b), expected)
}
func (s *Store) SetDrain(ctx context.Context, ref registry.Ref, kind registry.DrainKind, on bool) error {
	return s.integer(ctx, "drain", ref.ID, ref.Epoch, int(kind), boolInt(on))
}

func (s *Store) Block(ctx context.Context, id, reason string, _ time.Time) (registry.Block, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	raw, err := s.command(ctx, "block", id, reason).Text()
	var b registry.Block
	if err == nil {
		err = json.Unmarshal([]byte(raw), &b)
	}
	if err != nil {
		return b, unavailable("block", err)
	}
	return b, nil
}

func (s *Store) Unblock(ctx context.Context, id string) error {
	return s.integer(ctx, "unblock", id)
}

func (s *Store) Blocks(ctx context.Context) ([]registry.Block, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	raws, err := s.command(ctx, "blocks", "").StringSlice()
	if err != nil {
		return nil, unavailable("blocks", err)
	}
	out := make([]registry.Block, 0, len(raws))
	for _, raw := range raws {
		var b registry.Block
		if err := json.Unmarshal([]byte(raw), &b); err != nil {
			return nil, unavailable("blocks", err)
		}
		out = append(out, b)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}
func (s *Store) Reserve(ctx context.Context, ref registry.Ref, c registry.Claim, l registry.Limits) error {
	if s.lost.Load() {
		return unavailable("reserve", errors.New("gateway reservation lease expired"))
	}
	p := admission.FromContext(ctx)
	err := s.integer(ctx, "reserve", ref.ID, ref.Epoch, c.ID, c.KV, c.Prefill, l.Revision, l.MaxInflight, l.MaxReservedKV, l.MaxPrefill, l.MaxPendingKV, p.Ticket, p.Token)
	if errors.Is(err, registry.ErrStoreUnavailable) {
		// The script may have committed before the connection failed. Never
		// dispatch an engine call on an ambiguous reservation; reconcile its
		// idempotent release when Redis becomes reachable again.
		s.queueRelease(ref, c)
	}
	return err
}

func (s *Store) queueRelease(ref registry.Ref, c registry.Claim) {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := releaseKey{ref, c.ID}
	if old, ok := s.releases[k]; !ok || c.Requests > old.Requests {
		s.releases[k] = c
	}
}

func (s *Store) release(ctx context.Context, ref registry.Ref, c registry.Claim) error {
	return s.integer(ctx, "release", ref.ID, ref.Epoch, c.ID, boolInt(c.Requests > 0))
}
func (s *Store) Release(ctx context.Context, ref registry.Ref, c registry.Claim) error {
	err := s.release(ctx, ref, c)
	if err != nil {
		s.queueRelease(ref, c)
	}
	return err
}
func (s *Store) Health(ctx context.Context, ref registry.Ref, failed bool, threshold int, cooldown time.Duration, _ time.Time) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	n, err := s.command(ctx, "health", ref.ID, ref.Epoch, boolInt(failed), threshold, cooldown.Milliseconds()).Int()
	if err != nil {
		return false, unavailable("health", err)
	}
	return n == 1, nil
}
func (s *Store) Claim(ctx context.Context, name string, ttl time.Duration) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	n, err := s.command(ctx, "claim", name, max(ttl.Milliseconds(), 1)).Int()
	if err != nil {
		return false, unavailable("claim", err)
	}
	return n == 1, nil
}

func (s *Store) Maintain(ctx context.Context) error {
	if s.lost.Load() {
		return unavailable("maintain", errors.New("gateway reservation lease expired; restart this gateway"))
	}
	if err := s.integer(ctx, "maintain", "", s.leaseTTL.Milliseconds()); err != nil {
		return err
	}
	s.mu.Lock()
	batch := s.releases
	s.releases = make(map[releaseKey]registry.Claim)
	s.mu.Unlock()
	var first error
	for k, c := range batch {
		if err := s.Release(ctx, k.ref, c); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (s *Store) Close() error {
	s.once.Do(func() {
		ctx, cancel := s.Context()
		defer cancel()
		s.closeErr = s.integer(ctx, "leave", "")
		if err := s.Client.Close(); s.closeErr == nil {
			s.closeErr = err
		}
	})
	return s.closeErr
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
