package qos

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBucketTakeAndDebt(t *testing.T) {
	b := NewPerMinute(60, 2) // 1 token/s, burst 2
	now := time.Unix(0, 0)
	b.now = func() time.Time { return now }
	b.last = now
	if ok, _ := b.Take(2); !ok {
		t.Fatal("burst not available")
	}
	ok, wait := b.Take(1)
	if ok || wait < 900*time.Millisecond || wait > 1100*time.Millisecond {
		t.Fatalf("want ~1s wait, got ok=%v wait=%v", ok, wait)
	}
	now = now.Add(time.Second)
	if ok, _ := b.Take(1); !ok {
		t.Fatal("refill not applied")
	}
	b.Adjust(-3) // reconciliation pushes the bucket into debt
	if b.Available() >= 0 {
		t.Fatal("debt not recorded")
	}
	// Oversized request is admitted only from a full bucket.
	now = now.Add(time.Hour)
	if ok, _ := b.Take(10); !ok {
		t.Fatal("oversized request should pass on a full bucket")
	}
}

func TestLimiter(t *testing.T) {
	l := NewLimiter([]TenantPolicy{{Name: "a", RPM: 2, TPM: 1000, MaxConcurrency: 2}})
	l1, err := l.Admit("a", 100)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := l.Admit("a", 100)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := l.Admit("a", 100); err == nil || err.(*Rejection).Reason != "concurrency" {
		t.Fatalf("want concurrency rejection, got %v", err)
	}
	l1.Finish(50)
	l1.Finish(50) // idempotent
	if _, err := l.Admit("a", 100); err == nil || err.(*Rejection).Reason != "rpm" {
		t.Fatalf("want rpm rejection, got %v", err)
	}
	l2.Finish(100)
	// Unknown tenants are unlimited.
	for i := 0; i < 100; i++ {
		if _, err := l.Admit("other", 1e9); err != nil {
			t.Fatal(err)
		}
	}
	l3 := NewLimiter([]TenantPolicy{{Name: "t", TPM: 100}})
	if _, err := l3.Admit("t", 80); err != nil {
		t.Fatal(err)
	}
	if _, err := l3.Admit("t", 80); err == nil || err.(*Rejection).Reason != "tpm" {
		t.Fatalf("want tpm rejection, got %v", err)
	}
}

type recObserver struct {
	mu       sync.Mutex
	outcomes map[string]int
	depth    [numPriorities]int
}

func (r *recObserver) Outcome(_ Priority, o string, _ time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.outcomes == nil {
		r.outcomes = map[string]int{}
	}
	r.outcomes[o]++
}
func (r *recObserver) Depth(p Priority, d int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.depth[p] = d
}

func (r *recObserver) waitDepth(t *testing.T, p Priority, d int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		r.mu.Lock()
		got := r.depth[p]
		r.mu.Unlock()
		if got == d {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("depth of class %d is %d, want %d", p, got, d)
		}
		time.Sleep(time.Millisecond)
	}
}

// capacity is a toy resource the queue grants.
type capacity struct {
	mu   sync.Mutex
	free int
}

func (c *capacity) try() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.free > 0 {
		c.free--
		return true
	}
	return false
}

func (c *capacity) release(q *Queue) {
	c.mu.Lock()
	c.free++
	c.mu.Unlock()
	q.Notify()
}

func TestQueuePriorityOrder(t *testing.T) {
	q := NewQueue(QueueConfig{MaxSize: 10, MaxWait: 5 * time.Second, Tick: time.Hour}, nil)
	defer q.Close()
	c := &capacity{}
	var order []string
	var mu sync.Mutex
	var wg sync.WaitGroup
	start := func(name string, p Priority) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := q.Wait(context.Background(), Ticket{Tenant: name, Priority: p, Cost: 1}, c.try); err != nil {
				t.Errorf("%s: %v", name, err)
				return
			}
			mu.Lock()
			order = append(order, name)
			mu.Unlock()
		}()
		// Let the waiter enqueue before the next one.
		for q.Len() == 0 || !queued(q, name) {
			time.Sleep(time.Millisecond)
		}
	}
	start("low", PriorityLow)
	start("normal", PriorityNormal)
	start("critical", PriorityCritical)
	for i := 0; i < 3; i++ {
		c.release(q)
		waitLen(t, q, 2-i)
	}
	wg.Wait()
	want := []string{"critical", "normal", "low"}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order %v, want %v", order, want)
		}
	}
}

func queued(q *Queue, tenant string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, cls := range q.classes {
		if tq, ok := cls[tenant]; ok && tq.items.Len() > 0 {
			return true
		}
	}
	return false
}

func waitLen(t *testing.T, q *Queue, n int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for q.Len() != n {
		if time.Now().After(deadline) {
			t.Fatalf("queue len %d, want %d", q.Len(), n)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestQueueFairShareAcrossTenants(t *testing.T) {
	q := NewQueue(QueueConfig{MaxSize: 100, MaxWait: 5 * time.Second, Tick: time.Hour}, nil)
	defer q.Close()
	c := &capacity{}
	var mu sync.Mutex
	var order []string
	var wg sync.WaitGroup
	// Tenant "heavy" enqueues 6 requests before "light" enqueues 2.
	enqueue := func(name string, n int) {
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := q.Wait(context.Background(), Ticket{Tenant: name, Priority: PriorityNormal, Cost: 100}, c.try); err == nil {
					mu.Lock()
					order = append(order, name)
					mu.Unlock()
				}
			}()
		}
	}
	enqueue("heavy", 6)
	waitLen(t, q, 6)
	enqueue("light", 2)
	waitLen(t, q, 8)
	for i := 0; i < 4; i++ {
		c.release(q)
		waitLen(t, q, 7-i)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		mu.Lock()
		n := len(order)
		mu.Unlock()
		if n >= 4 || time.Now().After(deadline) {
			break
		}
		time.Sleep(time.Millisecond)
	}
	mu.Lock()
	first4 := append([]string(nil), order...)
	mu.Unlock()
	light := 0
	for _, n := range first4 {
		if n == "light" {
			light++
		}
	}
	if light != 2 {
		t.Fatalf("light tenant starved: first grants %v", first4)
	}
	for i := 0; i < 4; i++ {
		c.release(q)
	}
	wg.Wait()
}

func TestQueueTimeoutShedAndFull(t *testing.T) {
	obs := &recObserver{}
	never := func() bool { return false }
	short := NewQueue(QueueConfig{MaxSize: 1, MaxWait: 50 * time.Millisecond, Tick: 5 * time.Millisecond}, obs)
	defer short.Close()
	if err := short.Wait(context.Background(), Ticket{Tenant: "a"}, never); !errors.Is(err, ErrTimeout) {
		t.Fatalf("want timeout, got %v", err)
	}

	q := NewQueue(QueueConfig{MaxSize: 1, MaxWait: 5 * time.Second, Tick: 5 * time.Millisecond}, obs)
	defer q.Close()
	// A low priority waiter is shed by a critical one when the queue is full.
	errc := make(chan error, 1)
	go func() {
		errc <- q.Wait(context.Background(), Ticket{Tenant: "low", Priority: PriorityLow}, never)
	}()
	waitLen(t, q, 1)
	ctx, cancel := context.WithCancel(context.Background())
	critDone := make(chan error, 1)
	go func() { critDone <- q.Wait(ctx, Ticket{Tenant: "crit", Priority: PriorityCritical}, never) }()
	if err := <-errc; !errors.Is(err, ErrShed) {
		t.Fatalf("want shed, got %v", err)
	}
	// Same priority cannot shed: queue full.
	waitLen(t, q, 1)
	if err := q.Wait(context.Background(), Ticket{Tenant: "x", Priority: PriorityCritical}, never); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("want full, got %v", err)
	}
	cancel()
	if err := <-critDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("want canceled, got %v", err)
	}
	pre, precancel := context.WithCancel(context.Background())
	precancel()
	if err := q.Wait(pre, Ticket{Tenant: "y"}, never); !errors.Is(err, context.Canceled) {
		t.Fatalf("want canceled, got %v", err)
	}
	obs.mu.Lock()
	defer obs.mu.Unlock()
	if obs.outcomes["timeout"] != 1 || obs.outcomes["shed"] != 1 || obs.outcomes["full"] != 1 || obs.outcomes["canceled"] != 2 {
		t.Fatalf("outcomes %v", obs.outcomes)
	}
}

// A full queue whose only lower-priority waiter is trying capacity: a
// critical arrival takes its place. The waiter leaves when its try fails,
// or is granted when it succeeds; either way the queue stays within bounds.
func TestQueueShedsWaiterInTurn(t *testing.T) {
	for _, grant := range []bool{false, true} {
		obs := &recObserver{}
		q := NewQueue(QueueConfig{MaxSize: 1, MaxWait: 5 * time.Second, Tick: time.Hour}, obs)
		inTurn, release := make(chan struct{}), make(chan struct{})
		var once sync.Once
		try := func() bool {
			once.Do(func() { close(inTurn) })
			<-release
			return grant
		}
		lowDone := make(chan error, 1)
		go func() { lowDone <- q.Wait(context.Background(), Ticket{Tenant: "low", Priority: PriorityLow}, try) }()
		<-inTurn // low is out of its class, trying capacity
		if q.Len() != 1 {
			t.Fatalf("queue len %d while low is in turn, want 1", q.Len())
		}
		ctx, cancel := context.WithCancel(context.Background())
		critDone := make(chan error, 1)
		go func() {
			critDone <- q.Wait(ctx, Ticket{Tenant: "crit", Priority: PriorityCritical}, func() bool { return false })
		}()
		obs.waitDepth(t, PriorityCritical, 1) // crit took the place low is giving up
		if q.Len() != 1 {
			t.Fatalf("grant=%v: queue len %d, want 1 (crit; low is leaving)", grant, q.Len())
		}
		if err := q.Wait(context.Background(), Ticket{Tenant: "x", Priority: PriorityCritical}, func() bool { return false }); !errors.Is(err, ErrQueueFull) {
			t.Fatalf("grant=%v: want full, got %v", grant, err)
		}
		close(release)
		err := <-lowDone
		if grant && err != nil || !grant && !errors.Is(err, ErrShed) {
			t.Fatalf("grant=%v: low got %v", grant, err)
		}
		if q.Len() != 1 {
			t.Fatalf("grant=%v: queue len %d after low left, want 1 (crit)", grant, q.Len())
		}
		cancel()
		if err := <-critDone; !errors.Is(err, context.Canceled) {
			t.Fatalf("want canceled, got %v", err)
		}
		q.Close()
	}
}

func TestQueueStress(t *testing.T) {
	q := NewQueue(QueueConfig{MaxSize: 10000, MaxWait: 200 * time.Millisecond, BackfillDepth: 2, Tick: time.Millisecond}, nil)
	defer q.Close()
	c := &capacity{free: 4}
	var granted, failed atomic.Int64
	var wg sync.WaitGroup
	for i := 0; i < 400; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx := context.Background()
			if i%7 == 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, time.Duration(i%5)*time.Millisecond)
				defer cancel()
			}
			err := q.Wait(ctx, Ticket{Tenant: []string{"a", "b", "c"}[i%3], Priority: Priority(i % 4), Cost: float64(i%10 + 1)}, c.try)
			if err == nil {
				granted.Add(1)
				time.Sleep(time.Millisecond)
				c.release(q)
				return
			}
			failed.Add(1)
		}(i)
	}
	wg.Wait()
	if granted.Load()+failed.Load() != 400 {
		t.Fatalf("lost waiters: %d + %d", granted.Load(), failed.Load())
	}
	if q.Len() != 0 {
		t.Fatalf("queue not drained: %d", q.Len())
	}
	if c.free != 4 {
		t.Fatalf("capacity leaked: %d", c.free)
	}
}

func TestOverloadLevelsAndMatrix(t *testing.T) {
	o := NewOverload([]float64{0.75, 0.85, 0.90, 0.95}, 0.02, 0.8)
	o.rand = func() float64 { return 0.5 }
	if lv := o.Update(0.5); lv != 0 {
		t.Fatalf("level %d", lv)
	}
	if ok, _ := o.Decide(PriorityLow, 0); !ok {
		t.Fatal("L0 must admit everything")
	}
	if lv := o.Update(0.91); lv != 3 {
		t.Fatalf("want L3, got %d", lv)
	}
	if ok, why := o.Decide(PriorityNormal, 0); ok || why != "overload_reject" {
		t.Fatalf("normal at L3: %v %s", ok, why)
	}
	if ok, _ := o.Decide(PriorityCritical, 0); !ok {
		t.Fatal("critical must pass at L3")
	}
	// High degrades at L3: drop probability (0.91-0.90)/(0.10)=0.1 < 0.5 draw.
	if ok, why := o.Decide(PriorityHigh, 0); !ok || why != "degrade_pass" {
		t.Fatalf("high at L3: %v %s", ok, why)
	}
	o.Update(0.99)
	if ok, why := o.Decide(PriorityHigh, 0); ok || why != "overload_reject" {
		t.Fatalf("high at L4: %v %s", ok, why)
	}
	if ok, why := o.Decide(PriorityCritical, 0.9); !ok || why != "cache_exempt" {
		t.Fatalf("cached critical at L4: %v %s", ok, why)
	}
	// Hysteresis: falling to 0.94 keeps L4 (threshold 0.95 - 0.02 = 0.93).
	if lv := o.Update(0.94); lv != 4 {
		t.Fatalf("hysteresis broken, level %d", lv)
	}
	if lv := o.Update(0.10); lv != 0 {
		t.Fatalf("want L0, got %d", lv)
	}
}
