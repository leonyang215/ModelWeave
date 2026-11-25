package qos

import (
	"container/list"
	"context"
	"errors"
	"sync"
	"time"
)

// Queue errors.
var (
	ErrQueueFull = errors.New("admission queue full")
	ErrShed      = errors.New("request shed by a higher-priority request")
	ErrTimeout   = errors.New("timed out waiting for capacity")
	ErrClosed    = errors.New("admission queue closed")
)

// QueueConfig configures an admission queue.
type QueueConfig struct {
	MaxSize int           // maximum number of waiting requests (<= 0: 1024)
	MaxWait time.Duration // maximum time a request may wait (<= 0: 30s)
	// BackfillDepth is how many further waiters are tried after the next
	// waiter in line failed to get capacity, so that smaller requests can use
	// capacity a large request cannot. 0 disables backfill.
	BackfillDepth int
	// Tick is the period of retries when no explicit capacity notification
	// arrives (<= 0: 25ms). Engine state refreshes can free capacity without
	// a local release.
	Tick time.Duration
}

// QueueObserver receives queue events for metrics.
type QueueObserver interface {
	// Outcome is called once per waiter: granted, timeout, shed, full, canceled.
	Outcome(prio Priority, outcome string, waited time.Duration)
	// Depth is called whenever the depth of a priority class changes.
	Depth(prio Priority, depth int)
}

// Queue holds requests that could not be scheduled for lack of capacity. It
// serves strict priority classes; inside a class, tenants share capacity by
// virtual token counters (VTC fair queuing): the tenant that has received the
// least weighted service goes next.
type Queue struct {
	cfg      QueueConfig
	observer QueueObserver

	mu      sync.Mutex
	classes [numPriorities]map[string]*tenantQueue
	depth   [numPriorities]int
	vtc     map[string]float64
	size    int
	// current is the waiter trying capacity (the dispatcher hands out one
	// turn at a time). It is out of its class but still occupies the queue
	// until it is granted or pushed back.
	current *waiter
	closed  bool

	notify chan struct{}
	stop   chan struct{}
	wg     sync.WaitGroup
}

type tenantQueue struct {
	weight float64
	items  *list.List // *waiter
}

const (
	stateWaiting = iota
	stateInTurn
	stateDone
)

type waiter struct {
	tenant   string
	weight   float64
	prio     Priority
	cost     float64
	enqueued time.Time
	turn     chan struct{}
	result   chan bool
	shed     chan struct{}
	state    int
	elem     *list.Element
	// shedPending: shed while in turn; the waiter leaves unless its try
	// succeeds.
	shedPending bool
}

// NewQueue starts a queue and its dispatcher goroutine.
func NewQueue(cfg QueueConfig, observer QueueObserver) *Queue {
	if cfg.MaxSize <= 0 {
		cfg.MaxSize = 1024
	}
	if cfg.MaxWait <= 0 {
		cfg.MaxWait = 30 * time.Second
	}
	if cfg.Tick <= 0 {
		cfg.Tick = 25 * time.Millisecond
	}
	q := &Queue{
		cfg:      cfg,
		observer: observer,
		vtc:      make(map[string]float64),
		notify:   make(chan struct{}, 1),
		stop:     make(chan struct{}),
	}
	for i := range q.classes {
		q.classes[i] = make(map[string]*tenantQueue)
	}
	q.wg.Add(1)
	go q.run()
	return q
}

// Close stops the dispatcher; waiting requests fail with ErrClosed.
func (q *Queue) Close() {
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return
	}
	q.closed = true
	q.mu.Unlock()
	close(q.stop)
	q.wg.Wait()
}

// Notify signals that capacity may have been freed.
func (q *Queue) Notify() {
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// Len returns the number of waiting requests.
func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.size + q.occupiedByTurnLocked()
}

// occupiedByTurnLocked is 1 when the waiter in turn still holds its place.
func (q *Queue) occupiedByTurnLocked() int {
	if q.current != nil && !q.current.shedPending {
		return 1
	}
	return 0
}

// Ticket describes a request entering the queue.
type Ticket struct {
	Tenant   string
	Weight   float64
	Priority Priority
	Cost     float64 // estimated tokens, used for fair sharing
}

// Wait queues the request and blocks until try succeeds, the wait limit
// expires, the request is shed, or ctx is done. try is invoked by the waiter
// itself when it is its turn and must reserve capacity atomically: it returns
// true when the request got its resources.
func (q *Queue) Wait(ctx context.Context, t Ticket, try func() bool) error {
	if t.Weight <= 0 {
		t.Weight = 1
	}
	w := &waiter{
		tenant: t.Tenant, weight: t.Weight, prio: t.Priority, cost: t.Cost,
		enqueued: time.Now(),
		turn:     make(chan struct{}, 1),
		result:   make(chan bool, 1),
		shed:     make(chan struct{}),
	}
	if t.Priority < 0 || t.Priority >= numPriorities {
		w.prio = PriorityNormal
	}
	if err := q.enqueue(w); err != nil {
		q.outcome(w, "full")
		return err
	}
	q.Notify()

	timer := time.NewTimer(q.cfg.MaxWait)
	defer timer.Stop()
	for {
		select {
		case <-w.turn:
			// The waiter owns its state transition: it updates the queue
			// under the lock before releasing the dispatcher, so that a
			// later timeout never observes a stale in-turn state.
			if ctx.Err() != nil {
				q.mu.Lock()
				w.state = stateDone
				q.current = nil
				q.mu.Unlock()
				w.result <- false
				q.outcome(w, "canceled")
				return ctx.Err()
			}
			ok := try()
			q.mu.Lock()
			q.current = nil
			shed := false
			switch {
			case ok:
				w.state = stateDone
				q.vtc[w.tenant] += w.cost / w.weight
			case w.shedPending:
				w.state, shed = stateDone, true
			default:
				q.pushLocked(w, true)
			}
			q.mu.Unlock()
			w.result <- ok
			if ok {
				q.outcome(w, "granted")
				return nil
			}
			if shed {
				q.outcome(w, "shed")
				return ErrShed
			}
		case <-w.shed:
			q.outcome(w, "shed")
			return ErrShed
		case <-timer.C:
			q.leave(w)
			q.outcome(w, "timeout")
			return ErrTimeout
		case <-ctx.Done():
			q.leave(w)
			q.outcome(w, "canceled")
			return ctx.Err()
		case <-q.stop:
			q.leave(w)
			q.outcome(w, "canceled")
			return ErrClosed
		}
	}
}

func (q *Queue) outcome(w *waiter, outcome string) {
	if q.observer != nil {
		q.observer.Outcome(w.prio, outcome, time.Since(w.enqueued))
	}
}

func (q *Queue) setDepthLocked(p Priority, delta int) {
	q.depth[p] += delta
	q.size += delta
	if q.observer != nil {
		q.observer.Depth(p, q.depth[p])
	}
}

func (q *Queue) enqueue(w *waiter) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return ErrClosed
	}
	if q.size+q.occupiedByTurnLocked() >= q.cfg.MaxSize && !q.shedLowerLocked(w.prio) {
		return ErrQueueFull
	}
	q.pushLocked(w, false)
	return nil
}

func (q *Queue) pushLocked(w *waiter, front bool) {
	cls := q.classes[w.prio]
	tq, ok := cls[w.tenant]
	if !ok {
		tq = &tenantQueue{weight: w.weight, items: list.New()}
		cls[w.tenant] = tq
	}
	if tq.items.Len() == 0 && !front {
		// VTC lift: a tenant becoming active starts at the minimum counter of
		// the active tenants so idle time does not turn into a burst credit.
		minV, found := 0.0, false
		for name, other := range cls {
			if other.items.Len() == 0 || name == w.tenant {
				continue
			}
			if v := q.vtc[name]; !found || v < minV {
				minV, found = v, true
			}
		}
		if found && q.vtc[w.tenant] < minV {
			q.vtc[w.tenant] = minV
		}
	}
	if front {
		w.elem = tq.items.PushFront(w)
	} else {
		w.elem = tq.items.PushBack(w)
	}
	w.state = stateWaiting
	q.setDepthLocked(w.prio, 1)
}

func (q *Queue) removeLocked(w *waiter) {
	if tq, ok := q.classes[w.prio][w.tenant]; ok && w.elem != nil {
		tq.items.Remove(w.elem)
		if tq.items.Len() == 0 {
			delete(q.classes[w.prio], w.tenant)
		}
	}
	w.elem = nil
	q.setDepthLocked(w.prio, -1)
}

// shedLowerLocked evicts the newest waiter of the lowest priority class below
// prio. It reports whether room was made. A class whose only waiter is in
// turn sheds that one: it leaves when its try fails (if the try succeeds it
// is granted, and its place is free all the same).
func (q *Queue) shedLowerLocked(prio Priority) bool {
	for p := numPriorities - 1; p > prio; p-- {
		var victim *waiter
		for _, tq := range q.classes[p] {
			for e := tq.items.Back(); e != nil; e = e.Prev() {
				cand := e.Value.(*waiter)
				if cand.state != stateWaiting {
					continue
				}
				if victim == nil || cand.enqueued.After(victim.enqueued) {
					victim = cand
				}
				break
			}
		}
		if victim != nil {
			q.removeLocked(victim)
			victim.state = stateDone
			close(victim.shed)
			return true
		}
		if c := q.current; c != nil && c.prio == p && !c.shedPending {
			c.shedPending = true
			return true
		}
	}
	return false
}

// leave removes a waiter that gave up. It runs on the waiter's goroutine, so
// an in-turn state means the turn token is still buffered: the waiter takes it
// and answers so the dispatcher does not block.
func (q *Queue) leave(w *waiter) {
	q.mu.Lock()
	switch w.state {
	case stateWaiting:
		q.removeLocked(w)
		w.state = stateDone
		q.mu.Unlock()
	case stateInTurn:
		w.state = stateDone
		q.current = nil
		q.mu.Unlock()
		<-w.turn
		w.result <- false
	default:
		q.mu.Unlock()
	}
}

// nextLocked picks the next waiter: strict priority, then lowest VTC, then
// FIFO. Waiters in skip are not considered.
func (q *Queue) nextLocked(skip map[*waiter]bool) *waiter {
	for p := Priority(0); p < numPriorities; p++ {
		var best *waiter
		bestV := 0.0
		for name, tq := range q.classes[p] {
			var head *waiter
			for e := tq.items.Front(); e != nil; e = e.Next() {
				if c := e.Value.(*waiter); !skip[c] {
					head = c
					break
				}
			}
			if head == nil {
				continue
			}
			v := q.vtc[name]
			if best == nil || v < bestV || (v == bestV && head.enqueued.Before(best.enqueued)) {
				best, bestV = head, v
			}
		}
		if best != nil {
			return best
		}
	}
	return nil
}

func (q *Queue) run() {
	defer q.wg.Done()
	ticker := time.NewTicker(q.cfg.Tick)
	defer ticker.Stop()
	for {
		select {
		case <-q.stop:
			return
		case <-q.notify:
		case <-ticker.C:
		}
		q.dispatch()
	}
}

// dispatch hands turns to waiters until one fails to get capacity (plus the
// configured backfill depth).
func (q *Queue) dispatch() {
	skip := map[*waiter]bool{}
	failures := 0
	for failures <= q.cfg.BackfillDepth {
		q.mu.Lock()
		w := q.nextLocked(skip)
		if w == nil {
			q.mu.Unlock()
			return
		}
		q.removeLocked(w)
		w.state = stateInTurn
		q.current = w
		w.turn <- struct{}{}
		q.mu.Unlock()

		if ok := <-w.result; !ok {
			skip[w] = true
			failures++
		}
	}
}
