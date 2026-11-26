package qos

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/leonyang215/ModelWeave/internal/redisstate"
)

type qosFixture struct {
	a, b   Backend
	sa, sb *redisstate.Store
	mini   *miniredis.Miniredis
	mu     sync.Mutex
	now    time.Time
}

func newQoSFixture(t *testing.T, kind string, policies []TenantPolicy) *qosFixture {
	t.Helper()
	f := &qosFixture{now: time.Now().Truncate(time.Millisecond)}
	if kind == "memory" {
		f.a = NewMemoryBackend(policies)
		f.b = f.a
		l := f.a.Limiter().(*Limiter)
		for _, st := range l.tenants {
			for _, bucket := range []*Bucket{st.rpm, st.tpm} {
				if bucket != nil {
					bucket.now = f.clock
					bucket.last = f.clock()
				}
			}
		}
		t.Cleanup(f.a.Close)
		return f
	}
	url := os.Getenv("MW_TEST_REDIS_URL")
	if url == "" {
		f.mini = miniredis.RunT(t)
		f.mini.SetTime(f.clock())
		f.mini.ClockTTL(true)
		url = "redis://" + f.mini.Addr()
	}
	ns := fmt.Sprintf("qos/%s/%d", t.Name(), time.Now().UnixNano())
	newBackend := func() (Backend, *redisstate.Store) {
		s, err := redisstate.New(context.Background(), redisstate.Options{URL: url, Namespace: ns, Timeout: time.Second, LeaseTTL: 2 * time.Minute})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = s.Close() })
		b, err := NewRedisBackend(context.Background(), s, policies)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(b.Close)
		return b, s
	}
	f.a, f.sa = newBackend()
	f.b, f.sb = newBackend()
	return f
}
func (f *qosFixture) clock() time.Time { f.mu.Lock(); defer f.mu.Unlock(); return f.now }
func (f *qosFixture) advance(t *testing.T, d time.Duration) {
	t.Helper()
	if f.sa != nil && f.mini == nil {
		t.Skip("virtual clock uses the memory backend/miniredis")
	}
	f.mu.Lock()
	f.now = f.now.Add(d)
	now := f.now
	f.mu.Unlock()
	if f.mini != nil {
		f.mini.SetTime(now)
	}
}
func (f *qosFixture) queues(t *testing.T, cfg QueueConfig) (AdmissionQueue, AdmissionQueue) {
	t.Helper()
	a, err := f.a.NewQueue(context.Background(), "m", cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if f.a == f.b {
		return a, a
	}
	b, err := f.b.NewQueue(context.Background(), "m", cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	return a, b
}
func awaitQoS(t *testing.T, what string, ready func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !ready() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}
func queueLength(t *testing.T, q AdmissionQueue) int {
	t.Helper()
	n, err := q.LenContext(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func TestMemoryBackendsAreIndependent(t *testing.T) {
	policies := []TenantPolicy{{Name: "t", RPM: 1, TPM: 10, MaxConcurrency: 1}}
	a, b := NewMemoryBackend(policies), NewMemoryBackend(policies)
	defer a.Close()
	defer b.Close()
	for _, backend := range []Backend{a, b} {
		l, err := backend.Limiter().AdmitContext(context.Background(), "t", 10)
		if err != nil {
			t.Fatal(err)
		}
		defer l.Finish(10)
	}
}

func TestBackendCloseAndFastPathDeadline(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, []TenantPolicy{{Name: "t"}})
			q, _ := f.queues(t, QueueConfig{MaxWait: 30 * time.Millisecond})
			var released bool
			_, err := q.Admit(context.Background(), Ticket{Tenant: "t"}, func(ctx context.Context) (Attempt, error) {
				<-ctx.Done()
				return Attempt{Ready: true, Release: func() { released = true }}, nil
			})
			if !errors.Is(err, ErrTimeout) || !released {
				t.Fatalf("fast-path deadline/rollback: %v released=%v", err, released)
			}
			f.a.Close()
			if _, err := f.a.NewQueue(context.Background(), "after-close", QueueConfig{}, nil); !errors.Is(err, ErrClosed) {
				t.Fatalf("queue after close: %v", err)
			}
			if _, err := f.a.Limiter().AdmitContext(context.Background(), "t", 1); !errors.Is(err, ErrClosed) {
				t.Fatalf("limiter after close: %v", err)
			}
		})
	}
}
