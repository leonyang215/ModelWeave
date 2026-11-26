package registry

import (
	"context"
	"reflect"
	"sort"
	"sync"
	"time"
)

// memoryStore keeps the records in the gateway process: the store of a
// single gateway.
type memoryStore struct {
	mu    sync.RWMutex
	recs  map[string]*memRecord
	ver   uint64 // membership version
	epoch int64
}

type memRecord struct {
	e      Entry
	rec    Record // Gone is never set here
	claims map[string]Claim
}

// NewMemoryStore returns a store that keeps everything in this process.
func NewMemoryStore() Store {
	return &memoryStore{recs: make(map[string]*memRecord)}
}

func (s *memoryStore) Kind() string { return "memory" }
func (s *memoryStore) Shared() bool { return false }
func (s *memoryStore) Maintain(context.Context) error {
	return nil
}
func (s *memoryStore) Close() error { return nil }

func (s *memoryStore) Claim(context.Context, string, time.Duration) (bool, error) { return true, nil }

func specEqual(a, b Spec) bool {
	return reflect.DeepEqual(a, b)
}

func (s *memoryStore) Upsert(_ context.Context, spec Spec, source string, beat *Beat, now time.Time) (UpsertResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var res UpsertResult
	r := s.recs[spec.ID]
	switch {
	case r == nil:
		s.epoch++
		s.ver++
		r = &memRecord{e: Entry{Spec: spec, Source: source, Epoch: s.epoch}, claims: make(map[string]Claim)}
		s.recs[spec.ID] = r
		res.Created = true
	case r.e.Source == SourceStatic && source != SourceStatic:
		// A static instance keeps its configured identity; its heartbeats
		// only refresh it and report its state.
	case r.e.Source != source || !specEqual(r.e.Spec, spec):
		old := r.e.Spec
		if old.Target != spec.Target || old.Model != spec.Model || old.Role != spec.Role {
			s.epoch++
			drain := r.rec.AdminDrain
			r.e.Epoch = s.epoch
			r.rec = Record{AdminDrain: drain}
			r.claims = make(map[string]Claim)
		}
		r.e.Spec, r.e.Source = spec, source
		r.rec.Revision++
		s.ver++
		res.Changed = true
	}
	r.rec.LastSeen = now
	if beat != nil {
		if beat.State != nil {
			r.rec.State, r.rec.StateAt, r.rec.HasState = *beat.State, now, true
			r.rec.StateVer++
			r.rec.Revision++
			r.rec.PendingKV = 0
		}
		if r.rec.EngineDrain != beat.Draining {
			r.rec.EngineDrain = beat.Draining
			r.rec.Revision++
		}
	}
	res.Entry, res.Version = r.e, s.ver
	return res, nil
}

func (s *memoryStore) Remove(_ context.Context, id string) (Entry, uint64, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.recs[id]
	if !ok {
		return Entry{}, s.ver, false, nil
	}
	delete(s.recs, id)
	s.ver++
	return r.e, s.ver, true, nil
}

func (s *memoryStore) Expire(_ context.Context, ttl time.Duration, now time.Time) ([]Entry, uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []Entry
	for id, r := range s.recs {
		seen := r.rec.LastSeen
		// Checked under the store lock: a heartbeat that refreshed the
		// instance since cannot be lost.
		if r.e.Source == SourceHeartbeat && now.Sub(seen) > ttl {
			delete(s.recs, id)
			out = append(out, r.e)
		}
	}
	if len(out) > 0 {
		s.ver++
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Spec.ID < out[j].Spec.ID })
	return out, s.ver, nil
}

func (s *memoryStore) Members(_ context.Context, known uint64) (Membership, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if known == s.ver {
		return Membership{Version: s.ver}, false, nil
	}
	m := Membership{Version: s.ver, Entries: make([]Entry, 0, len(s.recs))}
	for _, r := range s.recs {
		m.Entries = append(m.Entries, r.e)
	}
	return m, true, nil
}

// get returns the record of a live registration.
func (s *memoryStore) get(ref Ref) *memRecord {
	if r := s.recs[ref.ID]; r != nil && r.e.Epoch == ref.Epoch {
		return r
	}
	return nil
}

func (s *memoryStore) Load(_ context.Context, refs []Ref, now time.Time) ([]Record, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Record, len(refs))
	for i, ref := range refs {
		r := s.get(ref)
		if r == nil {
			out[i] = Record{Gone: true}
			continue
		}
		out[i] = r.rec
		out[i].Entry, out[i].Now = r.e, now
	}
	return out, nil
}

func (s *memoryStore) SetState(_ context.Context, ref Ref, st EngineState, at time.Time, revision ...int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.get(ref)
	if r == nil {
		return ErrGone
	}
	if len(revision) > 0 && revision[0] >= 0 && r.rec.Revision != revision[0] {
		return ErrConflict
	}
	if r.rec.HasState && at.Before(r.rec.StateAt) {
		return nil
	}
	r.rec.State, r.rec.StateAt, r.rec.HasState = st, at, true
	r.rec.StateVer++
	r.rec.Revision++
	r.rec.PendingKV = 0
	return nil
}

func (s *memoryStore) SetDrain(_ context.Context, ref Ref, kind DrainKind, on bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.get(ref)
	if r == nil {
		return ErrGone
	}
	if kind == DrainAdmin {
		r.rec.AdminDrain = on
	} else {
		r.rec.EngineDrain = on
	}
	r.rec.Revision++
	return nil
}

func (s *memoryStore) Reserve(_ context.Context, ref Ref, c Claim, lim Limits) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.get(ref)
	if r == nil {
		return ErrGone
	}
	if _, exists := r.claims[c.ID]; exists {
		return nil
	}
	rec := &r.rec
	if lim.Revision >= 0 && rec.Revision != lim.Revision {
		return ErrConflict
	}
	if !lim.Allow(rec.Inflight, rec.ReservedKV, rec.PendingKV, rec.Prefill) {
		return ErrConflict
	}
	rec.Inflight += c.Requests
	rec.ReservedKV += c.KV
	rec.PendingKV += c.KV
	rec.Prefill += c.Prefill
	r.claims[c.ID] = c
	return nil
}

func (s *memoryStore) Release(_ context.Context, ref Ref, c Claim) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.get(ref)
	if r == nil {
		return nil
	}
	held, ok := r.claims[c.ID]
	if !ok {
		return nil
	}
	rec := &r.rec
	if c.Requests > 0 {
		rec.Inflight -= held.Requests
		rec.ReservedKV -= held.KV
		delete(r.claims, c.ID)
	} else {
		r.claims[c.ID] = Claim{ID: held.ID, Requests: held.Requests, KV: held.KV}
	}
	rec.Prefill -= held.Prefill
	return nil
}

func (s *memoryStore) Health(_ context.Context, ref Ref, failed bool, threshold int, cooldown time.Duration, now time.Time) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.get(ref)
	if r == nil {
		return false, nil
	}
	if !failed {
		if r.rec.Failures != 0 {
			r.rec.Revision++
		}
		r.rec.Failures = 0
		return false, nil
	}
	r.rec.Revision++
	r.rec.Failures++
	if threshold > 0 && r.rec.Failures >= threshold {
		r.rec.UnhealthyTill = now.Add(cooldown)
		r.rec.Failures = 0
		return true, nil
	}
	return false, nil
}
