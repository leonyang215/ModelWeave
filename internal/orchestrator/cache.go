package orchestrator

import (
	"container/list"
	"sync"
	"time"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// ArtifactStore and SessionStore select local or shared advisory caches.
// A backend error is a cache miss, never permission to skip capacity checks.
type ArtifactStore interface {
	Get(string) (mwip.EmbeddingLocation, bool)
	Put(string, mwip.EmbeddingLocation)
	Invalidate(...string)
	Len() int
}

type SessionStore interface {
	Get(string) (string, bool)
	Set(string, string)
	Delete(string)
	ForgetWorker(string)
	Len() int
}

// ArtifactCache maps media digests to the embedding artifacts the encode
// stage built for them (one packed tensor per item). Entries are identified
// by the complete content+plan fingerprint, never prompt offsets or hints.
type ArtifactCache struct {
	mu    sync.Mutex
	ttl   time.Duration
	max   int
	lru   *list.List // *artifactEntry, front = most recent
	items map[string]*list.Element
	now   func() time.Time
}

type artifactEntry struct {
	key       string
	embedding mwip.EmbeddingLocation
	expires   time.Time
}

// NewArtifactCache returns a cache bounded by ttl and max entries.
func NewArtifactCache(ttl time.Duration, max int) *ArtifactCache {
	return &ArtifactCache{ttl: ttl, max: max, lru: list.New(), items: map[string]*list.Element{}, now: time.Now}
}

// ArtifactKey identifies the artifacts of an item: its digest and plan (a
// different resize, processor or exact frame selection gives a different key).
func ArtifactKey(it *mwip.MediaItem) string {
	return it.PlanID
}

// Get returns the artifacts of key.
func (c *ArtifactCache) Get(key string) (mwip.EmbeddingLocation, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.items[key]
	if !ok {
		return mwip.EmbeddingLocation{}, false
	}
	ent := e.Value.(*artifactEntry)
	if c.now().After(ent.expires) {
		c.lru.Remove(e)
		delete(c.items, key)
		return mwip.EmbeddingLocation{}, false
	}
	c.lru.MoveToFront(e)
	return ent.embedding, true
}

// Put stores the artifacts of key.
func (c *ArtifactCache) Put(key string, embedding mwip.EmbeddingLocation) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.items[key]; ok {
		ent := e.Value.(*artifactEntry)
		ent.embedding, ent.expires = embedding, c.now().Add(c.ttl)
		c.lru.MoveToFront(e)
		return
	}
	c.items[key] = c.lru.PushFront(&artifactEntry{key: key, embedding: embedding, expires: c.now().Add(c.ttl)})
	for c.max > 0 && c.lru.Len() > c.max {
		last := c.lru.Back()
		c.lru.Remove(last)
		delete(c.items, last.Value.(*artifactEntry).key)
	}
}

// Invalidate drops keys (an engine reported the artifacts missing).
func (c *ArtifactCache) Invalidate(keys ...string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, d := range keys {
		if e, ok := c.items[d]; ok {
			c.lru.Remove(e)
			delete(c.items, d)
		}
	}
}

// Len returns the number of cached artifacts.
func (c *ArtifactCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lru.Len()
}

// SessionMap remembers which decode worker served a session, so follow-up
// turns can run normal on the worker that already holds the conversation's
// KV cache.
type SessionMap struct {
	mu  sync.Mutex
	ttl time.Duration
	m   map[string]sessionEntry
	now func() time.Time
}

type sessionEntry struct {
	worker  string
	expires time.Time
}

// NewSessionMap returns an empty map with the given entry TTL.
func NewSessionMap(ttl time.Duration) *SessionMap {
	return &SessionMap{ttl: ttl, m: map[string]sessionEntry{}, now: time.Now}
}

// Get returns the worker of a session.
func (s *SessionMap) Get(key string) (string, bool) {
	if key == "" {
		return "", false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.m[key]
	if !ok || s.now().After(e.expires) {
		delete(s.m, key)
		return "", false
	}
	return e.worker, true
}

// Set records the worker of a session.
func (s *SessionMap) Set(key, worker string) {
	if key == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.m[key] = sessionEntry{worker: worker, expires: s.now().Add(s.ttl)}
	if len(s.m) > 100000 {
		now := s.now()
		for k, e := range s.m {
			if now.After(e.expires) {
				delete(s.m, k)
			}
		}
	}
}

// Delete forgets a session.
func (s *SessionMap) Delete(key string) {
	if key == "" {
		return
	}
	s.mu.Lock()
	delete(s.m, key)
	s.mu.Unlock()
}

// ForgetWorker drops sessions of a removed worker.
func (s *SessionMap) ForgetWorker(worker string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, e := range s.m {
		if e.worker == worker {
			delete(s.m, k)
		}
	}
}

// Len returns the number of sessions.
func (s *SessionMap) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.m)
}
