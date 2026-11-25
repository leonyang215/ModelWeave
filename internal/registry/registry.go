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
	if spec.ID == "" || spec.Model == "" || spec.URL == "" {
		return nil, false, fmt.Errorf("worker spec needs id, model and url: %+v", spec)
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
		r.mu.Unlock()
		w.Touch(now)
		if old.URL != spec.URL || old.Role != spec.Role || old.Model != spec.Model {
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

// Remove deletes a worker.
func (r *Registry) Remove(id string) bool {
	r.mu.Lock()
	w, ok := r.workers[id]
	if ok {
		delete(r.workers, id)
		r.rebuildLocked()
	}
	r.mu.Unlock()
	if ok {
		r.emit(Event{Kind: "removed", Worker: w})
	}
	return ok
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
// than the TTL. It returns the removed IDs.
func (r *Registry) Sweep() []string {
	now := r.now()
	var expired []string
	r.mu.RLock()
	for id, w := range r.workers {
		if w.Source != "heartbeat" {
			continue
		}
		w.mu.Lock()
		last := w.lastSeen
		w.mu.Unlock()
		if now.Sub(last) > r.ttl {
			expired = append(expired, id)
		}
	}
	r.mu.RUnlock()
	for _, id := range expired {
		r.Remove(id)
	}
	return expired
}

// TTL returns the heartbeat TTL.
func (r *Registry) TTL() time.Duration { return r.ttl }
