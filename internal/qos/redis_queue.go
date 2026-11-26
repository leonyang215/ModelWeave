package qos

import (
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/leonyang215/ModelWeave/internal/admission"
)

//go:embed redis_queue.lua
var queueSource string
var queueScript = redis.NewScript(queueSource)

type queueSettings struct {
	MaxSize       int   `json:"max_size"`
	MaxWaitMS     int64 `json:"max_wait_ms"`
	BackfillDepth int   `json:"backfill_depth"`
	TickMS        int64 `json:"tick_ms"`
}
type queueStatus struct {
	State  string `json:"state"`
	Token  string `json:"token"`
	Queued bool   `json:"queued"`
}
type queueReply struct {
	Code    string                 `json:"code"`
	Changed bool                   `json:"changed"`
	Total   int                    `json:"total"`
	Depth   [numPriorities]int     `json:"depth"`
	Own     map[string]queueStatus `json:"own"`
	queueStatus
}
type queueUpdate struct {
	status queueStatus
	err    error
}
type redisWaiter struct {
	id        string
	updates   chan queueUpdate
	lastToken uint64
}
type redisQueue struct {
	backend              *RedisBackend
	name, root, settings string
	cfg                  QueueConfig
	observer             QueueObserver
	mu                   sync.Mutex
	waiters              map[string]*redisWaiter
	closed               bool
	notify               chan struct{}
	stop                 chan struct{}
	wg                   sync.WaitGroup
}

var _ AdmissionQueue = (*redisQueue)(nil)

func newRedisQueue(ctx context.Context, b *RedisBackend, name string, cfg QueueConfig, obs QueueObserver) (*redisQueue, error) {
	cfg = normalizeQueueConfig(cfg)
	encoded := base64.RawURLEncoding.EncodeToString([]byte(name))
	settings := queueSettings{cfg.MaxSize, max(cfg.MaxWait.Milliseconds(), 1), cfg.BackfillDepth, max(cfg.Tick.Milliseconds(), 1)}
	root := b.state.Key("qos:queue:" + encoded + ":")
	if err := b.ensureConfig(ctx, root+"settings", settings); err != nil {
		return nil, err
	}
	body, _ := json.Marshal(settings)
	return &redisQueue{backend: b, name: encoded, root: root, settings: string(body), cfg: cfg, observer: obs, waiters: make(map[string]*redisWaiter), notify: make(chan struct{}, 1), stop: make(chan struct{})}, nil
}
func (q *redisQueue) keys() []string {
	return []string{q.root + "tickets", q.root + "waiting", q.root + "control", q.root + "vtc", q.root + "skipped", q.backend.state.Key("leases")}
}
func (q *redisQueue) command(ctx context.Context, op, id string, args ...any) (queueReply, error) {
	all := []any{op, q.backend.state.OwnerID(), id, q.settings}
	all = append(all, args...)
	raw, err := q.backend.run(ctx, queueScript, q.keys(), all...)
	var r queueReply
	if err == nil {
		err = json.Unmarshal([]byte(raw), &r)
	}
	if err != nil {
		return r, err
	}
	if r.Code == "owner_lost" {
		return r, q.backend.ownerLost()
	}
	switch r.Code {
	case "ok", "full", "gone", "timeout", "shed", "turn_lost":
	default:
		return r, qosUnavailable("queue", fmt.Errorf("unexpected result %q", r.Code))
	}
	return r, nil
}
func (q *redisQueue) start() { q.wg.Add(1); go q.run() }
func (q *redisQueue) wake() {
	select {
	case q.notify <- struct{}{}:
	default:
	}
}
func (q *redisQueue) signal() { q.wake(); q.backend.notify(q.name) }
func (q *redisQueue) Notify() { _, _ = q.command(context.Background(), "wake", ""); q.signal() }

func (q *redisQueue) run() {
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
		q.mu.Lock()
		ws := make([]*redisWaiter, 0, len(q.waiters))
		for _, w := range q.waiters {
			ws = append(ws, w)
		}
		q.mu.Unlock()
		if len(ws) == 0 {
			continue
		}
		r, err := q.command(context.Background(), "step", "")
		q.mu.Lock()
		for _, w := range ws {
			if q.waiters[w.id] != w {
				continue
			}
			if err != nil {
				select {
				case w.updates <- queueUpdate{err: err}:
				default:
				}
				continue
			}
			s, ok := r.Own[w.id]
			if !ok {
				s = queueStatus{State: "gone"}
			}
			if s.State == "turn" {
				token, _ := strconv.ParseUint(s.Token, 10, 64)
				if token <= w.lastToken {
					continue
				}
				select {
				case w.updates <- queueUpdate{status: s}:
					w.lastToken = token
				default:
					// Retry delivery on the next durable read; never mark a
					// dropped notification as an already delivered turn.
				}
				continue
			}
			if s.State != "waiting" {
				select {
				case w.updates <- queueUpdate{status: s}:
				default:
				}
			}
		}
		q.mu.Unlock()
		if err == nil {
			q.depth(r.Depth)
			if r.Changed {
				q.backend.notify(q.name)
			}
		}
	}
}
func (q *redisQueue) depth(depth [numPriorities]int) {
	if q.observer != nil {
		for p, d := range depth {
			q.observer.Depth(Priority(p), d)
		}
	}
}
func (q *redisQueue) cancel(ctx context.Context, id string) error {
	r, err := q.command(ctx, "cancel", id)
	if err == nil && r.Changed {
		q.wake()
		q.backend.notify(q.name)
	}
	return err
}
func (q *redisQueue) cleanup(id string) {
	q.mu.Lock()
	delete(q.waiters, id)
	closed := q.closed
	q.mu.Unlock()
	if closed {
		return
	} // Close drops this owner's tickets in one operation.
	fn := func(ctx context.Context) error { return q.cancel(ctx, id) }
	if err := fn(context.Background()); err != nil {
		q.backend.remember("queue:"+q.name+":"+id, fn)
	}
}
func (q *redisQueue) Admit(parent context.Context, t Ticket, try TryAdmit) (queued bool, err error) {
	t = normalizeTicket(t)
	if err := parent.Err(); err != nil {
		return false, err
	}
	q.mu.Lock()
	closed := q.closed
	q.mu.Unlock()
	if closed {
		return false, ErrClosed
	}
	id, err := uniqueID()
	if err != nil {
		return false, err
	}
	ctx, cancel := context.WithTimeout(parent, q.cfg.MaxWait)
	defer cancel()
	started := time.Now()
	contextError := func() error {
		if parent.Err() != nil {
			return parent.Err()
		}
		return ErrTimeout
	}
	// Only metadata is stored; input JSON cannot contain user request bodies.
	input, _ := json.Marshal(struct {
		Tenant   string   `json:"tenant"`
		Priority Priority `json:"priority"`
		Weight   float64  `json:"weight"`
		Cost     float64  `json:"cost"`
	}{t.Tenant, t.Priority, t.Weight, t.Cost})
	deadline, _ := ctx.Deadline()
	r, err := q.command(ctx, "enqueue", id, string(input), max(time.Until(deadline).Milliseconds(), 1))
	defer q.cleanup(id)
	outcome := "error"
	defer func() {
		if queued && q.observer != nil {
			q.observer.Outcome(t.Priority, outcome, time.Since(started))
		}
	}()
	if err != nil {
		if ctx.Err() != nil {
			return false, contextError()
		}
		return false, err
	}
	q.depth(r.Depth)
	if r.Code == "full" {
		queued, outcome = true, "full"
		return true, ErrQueueFull
	}
	queued = r.Queued
	w := &redisWaiter{id: id, updates: make(chan queueUpdate, 4)}
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return queued, ErrClosed
	}
	if r.State == "turn" {
		w.lastToken, _ = strconv.ParseUint(r.Token, 10, 64)
		w.updates <- queueUpdate{status: r.queueStatus}
	}
	q.waiters[id] = w
	q.mu.Unlock()
	// Cancellation must revoke the shared turn even while the caller's
	// planning callback is blocked. Worker reservations carry its fencing
	// token, and a late callback is rolled back rather than dispatched.
	watchStop, watchDone := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(watchDone)
		select {
		case <-ctx.Done():
		case <-q.stop:
		case <-watchStop:
			return
		}
		q.mu.Lock()
		closed := q.closed
		q.mu.Unlock()
		if !closed {
			fn := func(ctx context.Context) error { return q.cancel(ctx, id) }
			if err := fn(context.Background()); err != nil {
				q.backend.remember("queue:"+q.name+":"+id, fn)
			}
		}
	}()
	defer func() { close(watchStop); <-watchDone }()
	q.signal()
	for {
		select {
		case <-ctx.Done():
			if parent.Err() != nil {
				outcome = "canceled"
			} else {
				outcome = "timeout"
			}
			return queued, contextError()
		case <-q.stop:
			outcome = "canceled"
			return queued, ErrClosed
		case update := <-w.updates:
			// Cancellation's cleanup can be observed by the dispatcher before
			// this goroutine selects ctx.Done. Preserve the caller's terminal
			// result instead of reporting the intentionally removed turn as lost.
			if ctx.Err() != nil {
				if parent.Err() != nil {
					outcome = "canceled"
				} else {
					outcome = "timeout"
				}
				return queued, contextError()
			}
			q.mu.Lock()
			closed := q.closed
			q.mu.Unlock()
			if closed {
				outcome = "canceled"
				return queued, ErrClosed
			}
			if update.err != nil {
				return queued, update.err
			}
			s := update.status
			switch s.State {
			case "timeout":
				outcome = "timeout"
				return queued, ErrTimeout
			case "shed":
				outcome = "shed"
				return queued, ErrShed
			case "gone", "error":
				return queued, admission.ErrTurnLost
			case "turn":
				if ctx.Err() != nil {
					return queued, contextError()
				}
				turnCtx := admission.WithPermit(ctx, admission.Permit{Control: q.root + "control", Ticket: id, Token: s.Token})
				at, tryErr := try(turnCtx)
				rollback := func() {
					if at.Ready && at.Release != nil {
						at.Release()
					}
				}
				if ctx.Err() != nil {
					rollback()
					if parent.Err() != nil {
						outcome = "canceled"
					} else {
						outcome = "timeout"
					}
					return queued, contextError()
				}
				q.mu.Lock()
				closed := q.closed
				q.mu.Unlock()
				if closed {
					rollback()
					outcome = "canceled"
					return queued, ErrClosed
				}
				result := "retry"
				if tryErr != nil {
					result = "error"
				} else if at.Ready {
					result = "granted"
				}
				ack, ackErr := q.command(ctx, "finish", id, s.Token, result)
				if ackErr != nil {
					rollback()
					if ctx.Err() != nil {
						if parent.Err() != nil {
							outcome = "canceled"
						} else {
							outcome = "timeout"
						}
						return queued, contextError()
					}
					return queued, ackErr
				}
				if ack.Code != "ok" {
					rollback()
					if ctx.Err() != nil {
						if parent.Err() != nil {
							outcome = "canceled"
						} else {
							outcome = "timeout"
						}
						return queued, contextError()
					}
					if ack.Code == "timeout" {
						outcome = "timeout"
						return queued, ErrTimeout
					}
					if ack.Code == "shed" {
						outcome = "shed"
						return queued, ErrShed
					}
					return queued, admission.ErrTurnLost
				}
				q.depth(ack.Depth)
				if tryErr != nil {
					rollback()
					q.signal()
					return queued, tryErr
				}
				if at.Ready {
					outcome = "granted"
					q.signal()
					return queued, nil
				}
				queued = true
				if ack.State == "shed" {
					outcome = "shed"
					q.signal()
					return queued, ErrShed
				}
				q.signal()
			}
		}
	}
}
func (q *redisQueue) LenContext(ctx context.Context) (int, error) {
	s, err := q.StatsContext(ctx)
	return s.Total, err
}
func (q *redisQueue) StatsContext(ctx context.Context) (QueueStats, error) {
	r, err := q.command(ctx, "stats", "")
	return QueueStats{Total: r.Total, Depth: r.Depth}, err
}
func (q *redisQueue) Close() {
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return
	}
	q.closed = true
	q.mu.Unlock()
	close(q.stop)
	q.wg.Wait()
	_, _ = q.command(context.Background(), "drop_owner", "")
	q.backend.notify(q.name)
}

// Wait is a compatibility helper; callers reserving resources should use
// Admit and provide Attempt.Release for rollback on failed queue commit.
func (q *redisQueue) Wait(ctx context.Context, t Ticket, try func() bool) error {
	_, err := q.Admit(ctx, t, func(context.Context) (Attempt, error) { return Attempt{Ready: try()}, nil })
	return err
}
