package kvindex

import (
	"container/list"
	"sync"
	"time"
)

// BlockIndex is an in-memory map from block hash to the workers holding that
// block. Each worker's blocks are kept in LRU order so the index can mirror the
// bounded capacity of the worker's KV cache. Entries older than the TTL are
// treated as absent and removed by Sweep.
//
// BlockIndex backs both the approximate index (learned from routing decisions)
// and the event-driven index (fed by engine KV events).
type BlockIndex struct {
	mu       sync.RWMutex
	blocks   map[uint64]map[string]int64 // block -> worker -> last seen (unix nanos)
	workers  map[string]*workerBlocks
	ttl      time.Duration
	capacity int // default per-worker capacity in blocks; 0 = unbounded
	now      func() time.Time
}

type workerBlocks struct {
	lru      *list.List // values are uint64 block hashes, front = most recent
	elems    map[uint64]*list.Element
	capacity int
}

// NewBlockIndex returns an empty index. ttl <= 0 disables expiry and
// capacity <= 0 disables per-worker eviction.
func NewBlockIndex(ttl time.Duration, capacity int) *BlockIndex {
	return &BlockIndex{
		blocks:   make(map[uint64]map[string]int64),
		workers:  make(map[string]*workerBlocks),
		ttl:      ttl,
		capacity: capacity,
		now:      time.Now,
	}
}

// SetWorkerCapacity bounds the number of blocks remembered for worker,
// typically KV capacity tokens / block size.
func (b *BlockIndex) SetWorkerCapacity(worker string, blocks int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	wb := b.workerLocked(worker)
	wb.capacity = blocks
	b.evictLocked(worker, wb)
}

func (b *BlockIndex) workerLocked(worker string) *workerBlocks {
	wb, ok := b.workers[worker]
	if !ok {
		wb = &workerBlocks{lru: list.New(), elems: make(map[uint64]*list.Element), capacity: b.capacity}
		b.workers[worker] = wb
	}
	return wb
}

// Insert records that worker holds the given blocks.
func (b *BlockIndex) Insert(worker string, hashes []uint64) {
	if len(hashes) == 0 {
		return
	}
	ts := b.now().UnixNano()
	b.mu.Lock()
	defer b.mu.Unlock()
	wb := b.workerLocked(worker)
	// Insert in reverse so the first (shared) blocks end up most recent: an
	// LRU cache evicts the tail of a prefix before its head.
	for i := len(hashes) - 1; i >= 0; i-- {
		h := hashes[i]
		ws, ok := b.blocks[h]
		if !ok {
			ws = make(map[string]int64, 1)
			b.blocks[h] = ws
		}
		ws[worker] = ts
		if e, ok := wb.elems[h]; ok {
			wb.lru.MoveToFront(e)
		} else {
			wb.elems[h] = wb.lru.PushFront(h)
		}
	}
	b.evictLocked(worker, wb)
}

func (b *BlockIndex) evictLocked(worker string, wb *workerBlocks) {
	if wb.capacity <= 0 {
		return
	}
	for wb.lru.Len() > wb.capacity {
		e := wb.lru.Back()
		h := e.Value.(uint64)
		wb.lru.Remove(e)
		delete(wb.elems, h)
		b.dropLocked(h, worker)
	}
}

func (b *BlockIndex) dropLocked(h uint64, worker string) {
	if ws, ok := b.blocks[h]; ok {
		delete(ws, worker)
		if len(ws) == 0 {
			delete(b.blocks, h)
		}
	}
}

// Remove forgets the given blocks for worker.
func (b *BlockIndex) Remove(worker string, hashes []uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	wb, ok := b.workers[worker]
	for _, h := range hashes {
		b.dropLocked(h, worker)
		if ok {
			if e, found := wb.elems[h]; found {
				wb.lru.Remove(e)
				delete(wb.elems, h)
			}
		}
	}
}

// Clear forgets every block of worker.
func (b *BlockIndex) Clear(worker string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	wb, ok := b.workers[worker]
	if !ok {
		return
	}
	for h := range wb.elems {
		b.dropLocked(h, worker)
	}
	delete(b.workers, worker)
}

// Match returns, for every candidate, the number of leading hashes the
// candidate holds. Candidates without any match are reported as 0.
func (b *BlockIndex) Match(hashes []uint64, candidates []string) map[string]int {
	out := make(map[string]int, len(candidates))
	for _, c := range candidates {
		out[c] = 0
	}
	if len(hashes) == 0 || len(candidates) == 0 {
		return out
	}
	var cutoff int64
	if b.ttl > 0 {
		cutoff = b.now().Add(-b.ttl).UnixNano()
	}
	alive := make([]string, len(candidates))
	copy(alive, candidates)

	b.mu.RLock()
	defer b.mu.RUnlock()
	for i, h := range hashes {
		ws := b.blocks[h]
		if len(ws) == 0 {
			break
		}
		next := alive[:0]
		for _, c := range alive {
			if seen, ok := ws[c]; ok && seen >= cutoff {
				out[c] = i + 1
				next = append(next, c)
			}
		}
		alive = next
		if len(alive) == 0 {
			break
		}
	}
	return out
}

// Sweep removes entries older than the TTL. It returns the number of removed
// (block, worker) entries.
func (b *BlockIndex) Sweep() int {
	if b.ttl <= 0 {
		return 0
	}
	cutoff := b.now().Add(-b.ttl).UnixNano()
	removed := 0
	b.mu.Lock()
	defer b.mu.Unlock()
	for worker, wb := range b.workers {
		// LRU tail holds the least recently inserted blocks.
		for e := wb.lru.Back(); e != nil; {
			h := e.Value.(uint64)
			seen, ok := b.blocks[h][worker]
			if ok && seen >= cutoff {
				break
			}
			prev := e.Prev()
			wb.lru.Remove(e)
			delete(wb.elems, h)
			b.dropLocked(h, worker)
			removed++
			e = prev
		}
		if wb.lru.Len() == 0 {
			delete(b.workers, worker)
		}
	}
	return removed
}

// Stats reports the number of distinct blocks and per-worker block counts.
func (b *BlockIndex) Stats() (blocks int, perWorker map[string]int) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	perWorker = make(map[string]int, len(b.workers))
	for w, wb := range b.workers {
		perWorker[w] = wb.lru.Len()
	}
	return len(b.blocks), perWorker
}
