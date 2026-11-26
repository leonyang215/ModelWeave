package kvindex

import (
	"context"
	"time"
)

// Query asks which candidate workers hold the leading blocks of a prompt.
type Query struct {
	Model      string
	Role       string // worker role being scheduled
	BlockSize  int
	Blocks     []uint64 // chained block hashes of the prompt (see BlockHashes)
	Candidates []string // worker IDs
	// WorkerKeys optionally scopes each instance ID to a registration epoch.
	// External metadata services still receive the original instance IDs.
	WorkerKeys map[string]string
}

// Index is the cache-awareness adapter used by the scheduler. Implementations
// range from a purely local approximation to an external KV metadata service.
type Index interface {
	// Kind names the implementation ("approx", "events", "remote", "none").
	Kind() string
	// Match returns the number of leading blocks of q.Blocks cached on each
	// candidate. Missing candidates mean 0.
	Match(ctx context.Context, q Query) (map[string]int, error)
	// Observe tells the index that the blocks were routed to worker, i.e. the
	// worker is about to hold them.
	Observe(worker string, blocks []uint64)
	// ForgetWorker drops everything known about worker (it left or restarted).
	ForgetWorker(worker string)
	// SetWorkerCapacity bounds the blocks remembered for worker.
	SetWorkerCapacity(worker string, blocks int)
}

// EventSink receives KV cache events published by engines (MWIP kv_events).
type EventSink interface {
	ApplyEvents(worker string, batch EventBatch) error
}

// EventBatch is a batch of KV cache events from one worker.
type EventBatch struct {
	// Stored lists blocks that became resident, in prefix order.
	Stored []uint64 `json:"stored,omitempty"`
	// Removed lists blocks that were evicted.
	Removed []uint64 `json:"removed,omitempty"`
	// Cleared means the worker dropped its entire cache (restart, reset).
	Cleared bool `json:"cleared,omitempty"`
}

// Approx learns cache contents from routing decisions only. It is the default
// because it needs nothing from the engines; it is an approximation since it
// cannot see evictions, which TTL and per-worker capacity compensate for.
type Approx struct {
	idx *BlockIndex
}

// NewApprox returns an approximate index.
func NewApprox(ttl time.Duration, defaultCapacityBlocks int) *Approx {
	return &Approx{idx: NewBlockIndex(ttl, defaultCapacityBlocks)}
}

func (a *Approx) Kind() string { return "approx" }

func (a *Approx) Match(_ context.Context, q Query) (map[string]int, error) {
	return matchKeys(q, a.idx.Match), nil
}

func (a *Approx) Observe(worker string, blocks []uint64) { a.idx.Insert(worker, blocks) }
func (a *Approx) ForgetWorker(worker string)             { a.idx.Clear(worker) }
func (a *Approx) SetWorkerCapacity(worker string, n int) { a.idx.SetWorkerCapacity(worker, n) }
func (a *Approx) Blocks() *BlockIndex                    { return a.idx }

// Events is an exact index fed by engine KV events. Between a routing decision
// and the engine's "stored" event, it keeps an optimistic entry with a short
// TTL so that a burst of requests sharing a prefix is not scattered.
type Events struct {
	exact      *BlockIndex
	optimistic *BlockIndex
}

// NewEvents returns an event-driven index. optimisticTTL <= 0 disables
// optimistic entries.
func NewEvents(optimisticTTL time.Duration) *Events {
	e := &Events{exact: NewBlockIndex(0, 0)}
	if optimisticTTL > 0 {
		e.optimistic = NewBlockIndex(optimisticTTL, 0)
	}
	return e
}

func (e *Events) Kind() string { return "events" }

func (e *Events) Match(_ context.Context, q Query) (map[string]int, error) {
	out := matchKeys(q, e.exact.Match)
	if e.optimistic != nil {
		for w, n := range matchKeys(q, e.optimistic.Match) {
			if n > out[w] {
				out[w] = n
			}
		}
	}
	return out, nil
}

func (e *Events) Observe(worker string, blocks []uint64) {
	if e.optimistic != nil {
		e.optimistic.Insert(worker, blocks)
	}
}

func (e *Events) ForgetWorker(worker string) {
	e.exact.Clear(worker)
	if e.optimistic != nil {
		e.optimistic.Clear(worker)
	}
}

func (e *Events) SetWorkerCapacity(string, int) {}

// ApplyEvents updates the exact index.
func (e *Events) ApplyEvents(worker string, batch EventBatch) error {
	if batch.Cleared {
		e.exact.Clear(worker)
		if e.optimistic != nil {
			e.optimistic.Clear(worker)
		}
	}
	if len(batch.Removed) > 0 {
		e.exact.Remove(worker, batch.Removed)
		if e.optimistic != nil {
			e.optimistic.Remove(worker, batch.Removed)
		}
	}
	if len(batch.Stored) > 0 {
		e.exact.Insert(worker, batch.Stored)
	}
	return nil
}

// Sweep expires optimistic entries.
func (e *Events) Sweep() int {
	if e.optimistic == nil {
		return 0
	}
	return e.optimistic.Sweep()
}

// Exact exposes the exact block index (for stats).
func (e *Events) Exact() *BlockIndex { return e.exact }

// None disables cache awareness.
type None struct{}

func (None) Kind() string                                             { return "none" }
func (None) Match(_ context.Context, q Query) (map[string]int, error) { return map[string]int{}, nil }
func (None) Observe(string, []uint64)                                 {}
func (None) ForgetWorker(string)                                      {}
func (None) SetWorkerCapacity(string, int)                            {}

func matchKeys(q Query, match func([]uint64, []string) map[string]int) map[string]int {
	keys := make([]string, len(q.Candidates))
	for i, id := range q.Candidates {
		keys[i] = id
		if key := q.WorkerKeys[id]; key != "" {
			keys[i] = key
		}
	}
	m := match(q.Blocks, keys)
	out := make(map[string]int, len(keys))
	for i, id := range q.Candidates {
		out[id] = m[keys[i]]
	}
	return out
}
