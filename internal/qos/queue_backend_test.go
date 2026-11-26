package qos

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/admission"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/redis/go-redis/v9"
)

func capacityAttempt(c *capacity) TryAdmit {
	return func(context.Context) (Attempt, error) {
		if !c.try() {
			return Attempt{}, nil
		}
		return Attempt{Ready: true, Release: func() { c.mu.Lock(); c.free++; c.mu.Unlock() }}, nil
	}
}
func addCapacity(c *capacity, q AdmissionQueue) { c.mu.Lock(); c.free++; c.mu.Unlock(); q.Notify() }
func nextGrant(t *testing.T, ch <-chan string) string {
	t.Helper()
	select {
	case s := <-ch:
		return s
	case <-time.After(5 * time.Second):
		t.Fatal("no queue grant")
		return ""
	}
}

func TestAdmissionBackendPriorityAndSharedBound(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, nil)
			a, b := f.queues(t, QueueConfig{MaxSize: 3, MaxWait: 5 * time.Second, Tick: time.Hour})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			c := &capacity{}
			grants := make(chan string, 3)
			var wg sync.WaitGroup
			for i, item := range []struct {
				name string
				p    Priority
			}{{"low", PriorityLow}, {"normal", PriorityNormal}, {"critical", PriorityCritical}} {
				q := a
				if i%2 == 1 {
					q = b
				}
				wg.Add(1)
				go func() {
					defer wg.Done()
					queued, err := q.Admit(ctx, Ticket{Tenant: item.name, Priority: item.p, Cost: 1}, capacityAttempt(c))
					if err != nil {
						t.Errorf("%s: %v", item.name, err)
						return
					}
					if !queued {
						t.Errorf("%s was not queued", item.name)
					}
					grants <- item.name
				}()
				n := i + 1
				awaitQoS(t, "global enqueue", func() bool { return queueLength(t, b) == n })
			}
			if _, err := a.Admit(ctx, Ticket{Tenant: "same-priority", Priority: PriorityLow}, capacityAttempt(c)); !errors.Is(err, ErrQueueFull) {
				t.Fatalf("queue bound is per replica: %v", err)
			}
			for _, want := range []string{"critical", "normal", "low"} {
				addCapacity(c, b)
				if got := nextGrant(t, grants); got != want {
					t.Fatalf("grant %s want %s", got, want)
				}
			}
			wg.Wait()
			if n := queueLength(t, a); n != 0 {
				t.Fatalf("leaked queue entries: %d", n)
			}
		})
	}
}

func TestAdmissionBackendWeightedFairSharing(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, nil)
			a, b := f.queues(t, QueueConfig{MaxSize: 32, MaxWait: 5 * time.Second, Tick: time.Hour})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			c := &capacity{}
			grants := make(chan string, 16)
			var wg sync.WaitGroup
			for i := 0; i < 16; i++ {
				name, weight, q := "heavy", 2.0, a
				if i >= 8 {
					name, weight, q = "light", 1, b
				}
				wg.Add(1)
				go func() {
					defer wg.Done()
					_, err := q.Admit(ctx, Ticket{Tenant: name, Priority: PriorityNormal, Weight: weight, Cost: 100}, capacityAttempt(c))
					if err == nil {
						grants <- name
					} else if !errors.Is(err, context.Canceled) {
						t.Errorf("wait: %v", err)
					}
				}()
			}
			awaitQoS(t, "all tenants queued", func() bool { return queueLength(t, b) == 16 })
			served := map[string]int{}
			for i := 0; i < 6; i++ {
				addCapacity(c, a)
				served[nextGrant(t, grants)]++
			}
			if served["heavy"] != 4 || served["light"] != 2 {
				t.Fatalf("weights 2:1 not shared globally: %+v", served)
			}
			cancel()
			wg.Wait()
			awaitQoS(t, "canceled waiters cleaned", func() bool { return queueLength(t, a) == 0 })
		})
	}
}

func TestAdmissionBackendBackfillAndNoFastPathBypass(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, nil)
			a, b := f.queues(t, QueueConfig{MaxSize: 8, MaxWait: 5 * time.Second, BackfillDepth: 1, Tick: time.Hour})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			largeDone := make(chan error, 1)
			go func() {
				_, err := a.Admit(ctx, Ticket{Tenant: "large", Priority: PriorityNormal, Cost: 100}, func(context.Context) (Attempt, error) { return Attempt{}, nil })
				largeDone <- err
			}()
			awaitQoS(t, "large head queued", func() bool { return queueLength(t, a) == 1 })
			// Small requests may backfill, but not take an uncoordinated fast
			// path ahead of a waiter. Its global turn is still a queued turn.
			queued, err := b.Admit(ctx, Ticket{Tenant: "small", Priority: PriorityNormal, Cost: 1}, func(context.Context) (Attempt, error) { return Attempt{Ready: true, Release: func() {}}, nil })
			if err != nil || !queued {
				t.Fatalf("backfill bypassed global queue or stalled: queued=%v err=%v", queued, err)
			}
			cancel()
			if err := <-largeDone; !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel: %v", err)
			}
		})
	}
}

func TestAdmissionBackendDoesNotLoseWakeDuringFailedTurn(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, nil)
			a, _ := f.queues(t, QueueConfig{MaxSize: 8, MaxWait: 5 * time.Second, Tick: time.Hour})
			entered, release := make(chan struct{}), make(chan struct{})
			var calls atomic.Int64
			done := make(chan error, 1)
			go func() {
				_, err := a.Admit(context.Background(), Ticket{Tenant: "t"}, func(context.Context) (Attempt, error) {
					if calls.Add(1) == 1 {
						close(entered)
						<-release
						return Attempt{}, nil
					}
					return Attempt{Ready: true, Release: func() {}}, nil
				})
				done <- err
			}()
			<-entered
			a.Notify() // arrives before the failed in-turn result is committed
			close(release)
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("lost capacity wakeup")
			}
		})
	}
}

func TestAdmissionBackendShedTimeoutAndCancel(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, nil)
			a, b := f.queues(t, QueueConfig{MaxSize: 1, MaxWait: 150 * time.Millisecond, Tick: 2 * time.Millisecond})
			never := func(context.Context) (Attempt, error) { return Attempt{}, nil }
			low := make(chan error, 1)
			go func() {
				_, err := a.Admit(context.Background(), Ticket{Tenant: "low", Priority: PriorityLow}, never)
				low <- err
			}()
			awaitQoS(t, "low queued", func() bool { return queueLength(t, b) == 1 })
			ctx, cancel := context.WithCancel(context.Background())
			critical := make(chan error, 1)
			go func() {
				_, err := b.Admit(ctx, Ticket{Tenant: "critical", Priority: PriorityCritical}, never)
				critical <- err
			}()
			if err := <-low; !errors.Is(err, ErrShed) {
				t.Fatalf("remote shedding: %v", err)
			}
			awaitQoS(t, "critical queued", func() bool { return queueLength(t, a) == 1 })
			cancel()
			if err := <-critical; !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel: %v", err)
			}
			if _, err := a.Admit(context.Background(), Ticket{Tenant: "timeout"}, never); !errors.Is(err, ErrTimeout) {
				t.Fatalf("timeout: %v", err)
			}
			if n := queueLength(t, b); n != 0 {
				t.Fatalf("timeout leaked ticket: %d", n)
			}
		})
	}
}

func TestRedisCanceledTurnIsFencedAtWorkerReservation(t *testing.T) {
	f := newQoSFixture(t, "redis", nil)
	a, b := f.queues(t, QueueConfig{MaxSize: 8, MaxWait: 5 * time.Second, Tick: 2 * time.Millisecond})
	ra, rb := registry.NewWithStore(time.Minute, f.sa), registry.NewWithStore(time.Minute, f.sb)
	w, _, err := ra.Upsert(registry.Spec{ID: "w", Model: "m", Role: registry.RoleUnified, Target: "w:9001", KVCapacityTokens: 1000}, registry.SourceHeartbeat)
	if err != nil {
		t.Fatal(err)
	}
	entered, resume := make(chan struct{}), make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	late := make(chan error, 1)
	done := make(chan error, 1)
	go func() {
		_, err := a.Admit(ctx, Ticket{Tenant: "first", Priority: PriorityNormal}, func(turn context.Context) (Attempt, error) {
			close(entered)
			<-resume
			// Deliberately keep only the old permit, ignoring cancellation.
			r, err := w.TryReserve(context.WithoutCancel(turn), 100, 0, registry.Unlimited(), nil)
			late <- err
			if r != nil {
				return Attempt{Ready: true, Release: func() { _ = r.Release(false) }}, nil
			}
			return Attempt{}, err
		})
		done <- err
	}()
	<-entered
	cancel()
	awaitQoS(t, "canceled in-turn entry revoked", func() bool { return queueLength(t, b) == 0 })
	_, err = b.Admit(context.Background(), Ticket{Tenant: "second", Priority: PriorityNormal}, func(turn context.Context) (Attempt, error) {
		r, err := rb.Get("w").TryReserve(turn, 100, 0, registry.Unlimited(), nil)
		if err != nil {
			return Attempt{}, err
		}
		return Attempt{Ready: true, Release: func() { _ = r.Release(false) }}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	close(resume)
	if err := <-late; !errors.Is(err, admission.ErrTurnLost) {
		t.Fatalf("stale turn reserved capacity: %v", err)
	}
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled turn: %v", err)
	}
	if s := rb.Get("w").Snapshot(time.Now(), time.Minute); s.Inflight != 1 || s.ReservedKV != 100 {
		t.Fatalf("late attempt changed reservations: %+v", s)
	}
}

func TestRedisQueueExpiredOwnerAndReplicaClose(t *testing.T) {
	f := newQoSFixture(t, "redis", nil)
	a, b := f.queues(t, QueueConfig{MaxSize: 2, MaxWait: 5 * time.Second, Tick: 2 * time.Millisecond})
	never := func(context.Context) (Attempt, error) { return Attempt{}, nil }
	one := make(chan error, 1)
	go func() {
		_, err := a.Admit(context.Background(), Ticket{Tenant: "a", Priority: PriorityNormal}, never)
		one <- err
	}()
	awaitQoS(t, "A queued", func() bool { return queueLength(t, b) == 1 })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	two := make(chan error, 1)
	go func() { _, err := b.Admit(ctx, Ticket{Tenant: "b", Priority: PriorityNormal}, never); two <- err }()
	awaitQoS(t, "both replicas queued", func() bool { return queueLength(t, b) == 2 })
	if err := f.sb.Client.ZAdd(context.Background(), f.sb.Key("leases"), redis.Z{Score: 0, Member: f.sa.OwnerID()}).Err(); err != nil {
		t.Fatal(err)
	}
	awaitQoS(t, "dead gateway removed", func() bool { return queueLength(t, b) == 1 })
	f.a.Close()
	if n := queueLength(t, b); n != 1 {
		t.Fatal("closing A erased B's queue")
	}
	cancel()
	<-one
	if err := <-two; !errors.Is(err, context.Canceled) {
		t.Fatalf("B canceled: %v", err)
	}
}

func TestRedisQueueCommitFailureRollsBackAndRecovers(t *testing.T) {
	f := newQoSFixture(t, "redis", nil)
	if f.mini == nil {
		t.Skip("fault injection uses miniredis")
	}
	a, b := f.queues(t, QueueConfig{MaxSize: 8, MaxWait: 5 * time.Second, Tick: time.Hour})
	var rolled atomic.Int64
	_, err := a.Admit(context.Background(), Ticket{Tenant: "a"}, func(context.Context) (Attempt, error) {
		f.mini.SetError("ERR injected unavailable")
		return Attempt{Ready: true, Release: func() { rolled.Add(1) }}, nil
	})
	if !errors.Is(err, registry.ErrStoreUnavailable) || rolled.Load() != 1 {
		t.Fatalf("failed commit dispatched/leaked: rollback=%d err=%v", rolled.Load(), err)
	}
	if _, err := b.Admit(context.Background(), Ticket{Tenant: "b"}, func(context.Context) (Attempt, error) {
		t.Error("executed while Redis unavailable")
		return Attempt{Ready: true}, nil
	}); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("outage: %v", err)
	}
	f.mini.SetError("")
	if err := f.a.Maintain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := f.b.Maintain(context.Background()); err != nil {
		t.Fatal(err)
	}
	if n := queueLength(t, b); n != 0 {
		t.Fatalf("failed admission ticket not reconciled: %d", n)
	}
}

func TestRedisQueueGlobalBoundUnderConcurrency(t *testing.T) {
	f := newQoSFixture(t, "redis", nil)
	a, b := f.queues(t, QueueConfig{MaxSize: 4, MaxWait: 5 * time.Second, Tick: 2 * time.Millisecond})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var wg sync.WaitGroup
	var full atomic.Int64
	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			q := a
			if i%2 == 1 {
				q = b
			}
			_, err := q.Admit(ctx, Ticket{Tenant: fmt.Sprint(i), Priority: PriorityNormal}, func(context.Context) (Attempt, error) { return Attempt{}, nil })
			if errors.Is(err, ErrQueueFull) {
				full.Add(1)
			} else if !errors.Is(err, context.Canceled) {
				t.Errorf("wait: %v", err)
			}
		}(i)
	}
	awaitQoS(t, "global full rejections", func() bool { return full.Load() == 26 })
	if n := queueLength(t, a); n != 4 {
		t.Fatalf("queue size %d, want 4 globally", n)
	}
	cancel()
	wg.Wait()
	if n := queueLength(t, b); n != 0 {
		t.Fatalf("cancel leaked %d entries", n)
	}
}

func TestRedisQueuePollingRecoversMissedNotifications(t *testing.T) {
	f := newQoSFixture(t, "redis", nil)
	a, b := f.queues(t, QueueConfig{MaxSize: 8, MaxWait: 5 * time.Second, Tick: 2 * time.Millisecond})
	// Disable delivery on these two test clients, not the Redis server or
	// any other namespace. Durable queue state must remain sufficient.
	_ = f.a.(*RedisBackend).pubsub.Close()
	_ = f.b.(*RedisBackend).pubsub.Close()
	c := &capacity{}
	done := make(chan error, 1)
	go func() { _, err := b.Admit(context.Background(), Ticket{Tenant: "b"}, capacityAttempt(c)); done <- err }()
	awaitQoS(t, "ticket without Pub/Sub", func() bool { return queueLength(t, a) == 1 })
	addCapacity(c, a)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("missed notification stranded shared queue")
	}
}

func TestRedisQueueRejectsIncompatibleReplicaSettings(t *testing.T) {
	f := newQoSFixture(t, "redis", nil)
	if _, err := f.a.NewQueue(context.Background(), "m", QueueConfig{MaxSize: 4}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := f.b.NewQueue(context.Background(), "m", QueueConfig{MaxSize: 5}, nil); err == nil {
		t.Fatal("replicas silently use different queue bounds")
	}
}
