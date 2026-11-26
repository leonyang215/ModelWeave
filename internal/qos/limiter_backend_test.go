package qos

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/redis/go-redis/v9"
)

func rejectionReason(err error) string {
	var r *Rejection
	if errors.As(err, &r) {
		return r.Reason
	}
	return ""
}
func tenantStats(t *testing.T, l TenantLimiter) TenantStats {
	t.Helper()
	ss, err := l.StatsContext(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(ss) != 1 {
		t.Fatalf("stats: %+v", ss)
	}
	return ss[0]
}

func TestLimiterBackendContract(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, []TenantPolicy{{Name: "t", RPM: 2, TPM: 100, MaxConcurrency: 2}})
			a, b := f.a.Limiter(), f.b.Limiter()
			ctx := context.Background()
			one, err := a.AdmitContext(ctx, "t", 70)
			if err != nil {
				t.Fatal(err)
			}
			two, err := b.AdmitContext(ctx, "t", 20)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := a.AdmitContext(ctx, "t", 1); rejectionReason(err) != "concurrency" {
				t.Fatalf("concurrency: %v", err)
			}
			if err := one.Finish(40); err != nil {
				t.Fatal(err)
			}
			_ = one.Finish(40)
			if _, err := b.AdmitContext(ctx, "t", 1); rejectionReason(err) != "rpm" {
				t.Fatalf("RPM: %v", err)
			}
			if err := two.Finish(50); err != nil {
				t.Fatal(err)
			}
			s := tenantStats(t, a)
			if s.Inflight != 0 || math.Abs(s.TPMAvailable-10) > .1 {
				t.Fatalf("actual-token reconciliation: %+v", s)
			}
		})
	}
}

func TestLimiterBackendRejectionDoesNotPartiallyCharge(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, []TenantPolicy{{Name: "t", RPM: 2, TPM: 100}})
			a, b := f.a.Limiter(), f.b.Limiter()
			ctx := context.Background()
			one, err := a.AdmitContext(ctx, "t", 80)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := b.AdmitContext(ctx, "t", 80); rejectionReason(err) != "tpm" {
				t.Fatalf("TPM: %v", err)
			}
			_ = one.Finish(0)
			two, err := b.AdmitContext(ctx, "t", 80)
			if err != nil {
				t.Fatalf("TPM rejection consumed RPM or concurrency: %v", err)
			}
			_ = two.Finish(80)
			if _, err := a.AdmitContext(ctx, "t", 0); rejectionReason(err) != "rpm" {
				t.Fatalf("RPM not global: %v", err)
			}
		})
	}
}

func TestLimiterBackendDebtAndRefill(t *testing.T) {
	for _, kind := range []string{"memory", "redis"} {
		t.Run(kind, func(t *testing.T) {
			f := newQoSFixture(t, kind, []TenantPolicy{{Name: "t", TPM: 100}})
			a, b := f.a.Limiter(), f.b.Limiter()
			ctx := context.Background()
			one, err := a.AdmitContext(ctx, "t", 250)
			if err != nil {
				t.Fatalf("oversized full-bucket request: %v", err)
			}
			if _, err := b.AdmitContext(ctx, "t", 1); rejectionReason(err) != "tpm" {
				t.Fatalf("debt ignored: %v", err)
			}
			_ = one.Finish(200)
			if s := tenantStats(t, b); math.Abs(s.TPMAvailable+100) > .1 {
				t.Fatalf("debt reconciliation: %+v", s)
			}
			f.advance(t, 30*time.Second)
			if _, err := b.AdmitContext(ctx, "t", 1); rejectionReason(err) != "tpm" {
				t.Fatalf("debt refilled too early: %v", err)
			}
			f.advance(t, 31*time.Second)
			if lease, err := b.AdmitContext(ctx, "t", 1); err != nil {
				t.Fatalf("refill: %v", err)
			} else {
				_ = lease.Finish(1)
			}
		})
	}
}

func TestLimiterGlobalConcurrencyAtomicAcrossReplicas(t *testing.T) {
	f := newQoSFixture(t, "redis", []TenantPolicy{{Name: "t", MaxConcurrency: 3}})
	var wg sync.WaitGroup
	held := make(chan *Lease, 100)
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			l := f.a.Limiter()
			if i%2 == 1 {
				l = f.b.Limiter()
			}
			lease, err := l.AdmitContext(context.Background(), "t", 1)
			if err == nil {
				held <- lease
			} else if rejectionReason(err) != "concurrency" {
				t.Errorf("admit: %v", err)
			}
		}(i)
	}
	wg.Wait()
	close(held)
	if n := len(held); n != 3 {
		t.Fatalf("admitted %d across replicas, want 3", n)
	}
	if s := tenantStats(t, f.b.Limiter()); s.Inflight != 3 {
		t.Fatalf("global stats: %+v", s)
	}
	for l := range held {
		_ = l.Finish(1)
	}
	if s := tenantStats(t, f.a.Limiter()); s.Inflight != 0 {
		t.Fatalf("leaked slots: %+v", s)
	}
}

func TestLimiterFailedFinishRecoveryAndExpiredOwner(t *testing.T) {
	f := newQoSFixture(t, "redis", []TenantPolicy{{Name: "t", RPM: 4, TPM: 100, MaxConcurrency: 1}})
	if f.mini == nil {
		t.Skip("fault injection uses miniredis")
	}
	ctx := context.Background()
	l, err := f.a.Limiter().AdmitContext(ctx, "t", 80)
	if err != nil {
		t.Fatal(err)
	}
	f.mini.SetError("ERR injected unavailable")
	if err := l.Finish(30); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("finish failure: %v", err)
	}
	if _, err := f.b.Limiter().AdmitContext(ctx, "t", 1); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("outage bypassed limits: %v", err)
	}
	f.mini.SetError("")
	if s := tenantStats(t, f.b.Limiter()); s.Inflight != 1 {
		t.Fatal("ambiguous finish freed quota locally")
	}
	if err := f.a.Maintain(ctx); err != nil {
		t.Fatal(err)
	}
	if err := f.a.Maintain(ctx); err != nil {
		t.Fatal(err)
	}
	if s := tenantStats(t, f.b.Limiter()); s.Inflight != 0 || math.Abs(s.TPMAvailable-70) > .1 {
		t.Fatalf("finish retry double-charged: %+v", s)
	}
	l, err = f.a.Limiter().AdmitContext(ctx, "t", 10)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.sb.Client.ZAdd(ctx, f.sb.Key("leases"), redis.Z{Score: 0, Member: f.sa.OwnerID()}).Err(); err != nil {
		t.Fatal(err)
	}
	if next, err := f.b.Limiter().AdmitContext(ctx, "t", 10); err != nil {
		t.Fatalf("crashed gateway concurrency not reclaimed: %v", err)
	} else {
		_ = next.Finish(10)
	}
	if _, err := f.a.Limiter().AdmitContext(ctx, "t", 1); !errors.Is(err, registry.ErrStoreUnavailable) {
		t.Fatalf("expired owner resurrected: %v", err)
	}
	_ = l.Finish(10)
}

func TestLimiterSettingsAndRestartDoNotResetQuotas(t *testing.T) {
	ps := []TenantPolicy{{Name: "t", RPM: 1}}
	f := newQoSFixture(t, "redis", ps)
	l, err := f.a.Limiter().AdmitContext(context.Background(), "t", 1)
	if err != nil {
		t.Fatal(err)
	}
	_ = l.Finish(1)
	f.a.Close()
	if _, err := f.b.Limiter().AdmitContext(context.Background(), "t", 1); rejectionReason(err) != "rpm" {
		t.Fatalf("replica close reset shared RPM: %v", err)
	}
	if _, err := NewRedisBackend(context.Background(), f.sb, []TenantPolicy{{Name: "t", RPM: 2}}); err == nil {
		t.Fatal("incompatible quota configuration accepted")
	}
}
