package registry

import (
	"context"
	"fmt"
	"maps"
	"sort"
	"sync"
	"time"
)

// Event is a registry membership change notification. Previous is set on
// updates so consumers can invalidate the old registration's caches.
type Event struct {
	Kind     string // "added", "updated", "removed"
	Worker   *Worker
	Previous *Worker
}

// Registry maintains a local view of the Store's membership. Instance state
// and reservations are always read/written through the store, not this view.
type Registry struct {
	mu        sync.RWMutex
	syncMu    sync.Mutex
	workers   map[string]*Worker
	byKey     map[string][]*Worker
	listeners []func(Event)
	version   uint64
	store     Store
	ttl       time.Duration
	now       func() time.Time
}

// New returns a registry backed by single-process memory.
func New(ttl time.Duration) *Registry { return NewWithStore(ttl, NewMemoryStore()) }

// NewWithStore returns a registry backed by store. Replicas sharing a store
// see the same registrations, load, drains, health and capacity reservations.
func NewWithStore(ttl time.Duration, store Store) *Registry {
	if ttl <= 0 {
		ttl = 15 * time.Second
	}
	return &Registry{
		workers: make(map[string]*Worker), byKey: make(map[string][]*Worker),
		store: store, ttl: ttl, now: time.Now,
	}
}

func (r *Registry) Store() Store       { return r.store }
func (r *Registry) TTL() time.Duration { return r.ttl }

// OnChange registers a listener. Notifications are local observations; a
// shared-store removal is observed by every replica, but committed only once.
func (r *Registry) OnChange(fn func(Event)) {
	r.mu.Lock()
	r.listeners = append(r.listeners, fn)
	r.mu.Unlock()
}

func key(model string, role Role) string { return model + "\x00" + string(role) }

func (r *Registry) emit(events []Event) {
	r.mu.RLock()
	ls := append([]func(Event){}, r.listeners...)
	r.mu.RUnlock()
	for _, ev := range events {
		for _, fn := range ls {
			fn(ev)
		}
	}
}

// Sync refreshes the membership from its durable version. Polling the version
// also recovers changes missed during a disconnect; correctness does not
// depend on Redis Pub/Sub delivery.
func (r *Registry) Sync(ctx context.Context) error {
	r.syncMu.Lock()
	m, changed, err := r.store.Members(ctx, r.version)
	var events []Event
	if err == nil && changed {
		r.mu.Lock()
		next := make(map[string]*Worker, len(m.Entries))
		byKey := make(map[string][]*Worker)
		for _, e := range m.Entries {
			old := r.workers[e.Spec.ID]
			w := old
			if old == nil || old.Ref().Epoch != e.Epoch || old.Source() != e.Source || !specEqual(old.Spec(), e.Spec) {
				// A new immutable handle also fences snapshots of an older
				// spec. Existing requests retain their original handle.
				w = &Worker{entry: cloneEntry(e), store: r.store}
				kind := "updated"
				if old == nil {
					kind = "added"
				}
				events = append(events, Event{Kind: kind, Worker: w, Previous: old})
			}
			next[e.Spec.ID] = w
			k := key(e.Spec.Model, e.Spec.Role)
			byKey[k] = append(byKey[k], w)
		}
		for id, w := range r.workers {
			if next[id] == nil {
				events = append(events, Event{Kind: "removed", Worker: w})
			}
		}
		for _, ws := range byKey {
			sort.Slice(ws, func(i, j int) bool { return ws[i].ID() < ws[j].ID() })
		}
		r.workers, r.byKey, r.version = next, byKey, m.Version
		r.mu.Unlock()
	}
	r.syncMu.Unlock()
	r.emit(events)
	return err
}

func cloneEntry(e Entry) Entry {
	e.Spec.Labels = maps.Clone(e.Spec.Labels)
	return e
}

// Upsert adds or refreshes an instance. Heartbeats cannot replace a static
// instance's configured identity.
func (r *Registry) Upsert(spec Spec, source string) (*Worker, bool, error) {
	return r.UpsertContext(context.Background(), spec, source, nil)
}

// UpsertContext atomically records registration, liveness and the heartbeat's
// state/draining flag (if present).
func (r *Registry) UpsertContext(ctx context.Context, spec Spec, source string, beat *Beat) (*Worker, bool, error) {
	if err := spec.Validate(); err != nil {
		return nil, false, err
	}
	if beat != nil && beat.State != nil {
		if err := beat.State.Validate(); err != nil {
			return nil, false, err
		}
	}
	if source != SourceStatic && source != SourceHeartbeat {
		return nil, false, fmt.Errorf("unknown instance source %q", source)
	}
	res, err := r.store.Upsert(ctx, cloneEntry(Entry{Spec: spec}).Spec, source, beat, r.now())
	if err != nil {
		return nil, false, err
	}
	if err := r.Sync(ctx); err != nil {
		return nil, false, err
	}
	r.mu.RLock()
	w := r.workers[spec.ID]
	r.mu.RUnlock()
	if w == nil || w.Ref().Epoch != res.Entry.Epoch {
		return nil, false, ErrGone
	}
	return w, res.Created, nil
}

func (r *Registry) Remove(id string) (Spec, bool) {
	spec, ok, _ := r.RemoveContext(context.Background(), id)
	return spec, ok
}

func (r *Registry) RemoveContext(ctx context.Context, id string) (Spec, bool, error) {
	e, _, ok, err := r.store.Remove(ctx, id)
	if err == nil {
		err = r.Sync(ctx)
	}
	return e.Spec, ok, err
}

// Get returns a worker by ID. Convenience reads fail closed on a store
// outage; request handlers call Sync explicitly to report the cause.
func (r *Registry) Get(id string) *Worker {
	if r.store.Shared() && r.Sync(context.Background()) != nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.workers[id]
}

// Candidates returns a copy of the workers serving model in role.
func (r *Registry) Candidates(model string, role Role) []*Worker {
	if r.store.Shared() && r.Sync(context.Background()) != nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return append([]*Worker(nil), r.byKey[key(model, role)]...)
}

func (r *Registry) All() []*Worker {
	if r.store.Shared() && r.Sync(context.Background()) != nil {
		return nil
	}
	r.mu.RLock()
	out := make([]*Worker, 0, len(r.workers))
	for _, w := range r.workers {
		out = append(out, w)
	}
	r.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].ID() < out[j].ID() })
	return out
}

func (r *Registry) Sweep() []Spec {
	specs, _ := r.SweepContext(context.Background())
	return specs
}

// SweepContext atomically expires heartbeat registrations. Static instances
// never expire. Only the replica committing a removal returns it for metrics.
func (r *Registry) SweepContext(ctx context.Context) ([]Spec, error) {
	es, _, err := r.store.Expire(ctx, r.ttl, r.now())
	if err != nil {
		return nil, err
	}
	if err := r.Sync(ctx); err != nil {
		return nil, err
	}
	specs := make([]Spec, 0, len(es))
	for _, e := range es {
		specs = append(specs, e.Spec)
	}
	sort.Slice(specs, func(i, j int) bool { return specs[i].ID < specs[j].ID })
	return specs, nil
}

// Snapshots reads the candidates' state in a batch (one Redis pipeline).
func (r *Registry) Snapshots(ctx context.Context, ws []*Worker, now time.Time, staleAfter time.Duration) ([]Snapshot, error) {
	refs := make([]Ref, len(ws))
	for i, w := range ws {
		refs[i] = w.Ref()
	}
	recs, err := r.store.Load(ctx, refs, now)
	if err != nil {
		return nil, err
	}
	out := make([]Snapshot, len(ws))
	for i, w := range ws {
		out[i] = w.snapshot(recs[i], now, staleAfter)
	}
	return out, nil
}
