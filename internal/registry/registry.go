package registry

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Event is a registry change notification.
type Event struct {
	Kind   string // "added", "updated", "removed"
	Worker *Worker
}

// Registry holds all known workers.
type Registry struct {
	mu        sync.RWMutex
	workers   map[string]*Worker
	byKey     map[string][]*Worker // model|role -> workers (copy on write)
	listeners []func(Event)
	ttl       time.Duration // heartbeat-registered workers expire after ttl
	now       func() time.Time
}

// New returns an empty registry. Workers registered by heartbeat are removed
// when no heartbeat arrives for ttl.
func New(ttl time.Duration) *Registry {
	if ttl <= 0 {
		ttl = 15 * time.Second
	}
	return &Registry{
		workers: make(map[string]*Worker),
		byKey:   make(map[string][]*Worker),
		ttl:     ttl,
		now:     time.Now,
	}
}

// OnChange registers a listener for worker additions/updates/removals.
func (r *Registry) OnChange(fn func(Event)) {
	r.mu.Lock()
	r.listeners = append(r.listeners, fn)
	r.mu.Unlock()
}

func key(model string, role Role) string { return model + "|" + string(role) }

func (r *Registry) emit(ev Event) {
	r.mu.RLock()
	ls := r.listeners
	r.mu.RUnlock()
	for _, fn := range ls {
		fn(ev)
	}
}

func (r *Registry) rebuildLocked() {
	byKey := make(map[string][]*Worker)
	for _, w := range r.workers {
		s := w.Spec()
		k := key(s.Model, s.Role)
		byKey[k] = append(byKey[k], w)
	}
	for _, ws := range byKey {
		sort.Slice(ws, func(i, j int) bool { return ws[i].ID() < ws[j].ID() })
	}
	r.byKey = byKey
}

// Upsert adds a worker or updates its spec. It returns the worker and
// whether it was newly added.
func (r *Registry) Upsert(spec Spec, source string) (*Worker, bool, error) {
	if spec.ID == "" || spec.Model == "" || spec.Target == "" {
		return nil, false, fmt.Errorf("worker spec needs id, model and target: %+v", spec)
	}
	if _, err := ParseRole(string(spec.Role)); err != nil {
		return nil, false, err
	}

	now := r.now()
	r.mu.Lock()
	w, ok := r.workers[spec.ID]
	if ok {
		old := w.Spec()
		w.spec.Store(&spec)
		if source == "static" {
			w.Source = source
		}
		if old.Model != spec.Model || old.Role != spec.Role {
			r.rebuildLocked()
		}
		// Refreshed under the registry lock, so Sweep cannot expire it in
		// between.
		w.Touch(now)
		r.mu.Unlock()
		if old.Target != spec.Target || old.Role != spec.Role || old.Model != spec.Model {
			r.emit(Event{Kind: "updated", Worker: w})
		}
		return w, false, nil
	}
	w = &Worker{Source: source}
	w.spec.Store(&spec)
	w.lastSeen = now
	r.workers[spec.ID] = w
	r.rebuildLocked()
	r.mu.Unlock()
	r.emit(Event{Kind: "added", Worker: w})
	return w, true, nil
}

// Remove deletes a worker and returns its spec.
func (r *Registry) Remove(id string) (Spec, bool) {
	r.mu.Lock()
	w, ok := r.workers[id]
	if ok {
		delete(r.workers, id)
		r.rebuildLocked()
	}
	r.mu.Unlock()
	if !ok {
		return Spec{}, false
	}
	r.emit(Event{Kind: "removed", Worker: w})
	return w.Spec(), true
}

// Get returns a worker by ID.
func (r *Registry) Get(id string) *Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.workers[id]
}

// Candidates returns the workers serving model in role. The slice must not
// be modified.
func (r *Registry) Candidates(model string, role Role) []*Worker {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.byKey[key(model, role)]
}

// All returns every worker sorted by ID.
func (r *Registry) All() []*Worker {
	r.mu.RLock()
	out := make([]*Worker, 0, len(r.workers))
	for _, w := range r.workers {
		out = append(out, w)
	}
	r.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].ID() < out[j].ID() })
	return out
}

// Sweep removes heartbeat-registered workers whose last heartbeat is older
// than the TTL, and returns their specs by ID. Static workers never expire.
func (r *Registry) Sweep() []Spec {
	now := r.now()
	var expired []*Worker
	r.mu.Lock()
	for id, w := range r.workers {
		// Checked under the registry lock: a heartbeat that refreshed the
		// worker since cannot be lost.
		if w.Source == "heartbeat" && now.Sub(w.LastSeen()) > r.ttl {
			delete(r.workers, id)
			expired = append(expired, w)
		}
	}
	if len(expired) > 0 {
		r.rebuildLocked()
	}
	r.mu.Unlock()
	specs := make([]Spec, 0, len(expired))
	for _, w := range expired {
		specs = append(specs, w.Spec())
		r.emit(Event{Kind: "removed", Worker: w})
	}
	sort.Slice(specs, func(i, j int) bool { return specs[i].ID < specs[j].ID })
	return specs
}

// TTL returns the heartbeat TTL.
func (r *Registry) TTL() time.Duration { return r.ttl }
