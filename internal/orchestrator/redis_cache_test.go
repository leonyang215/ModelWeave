package orchestrator_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/orchestrator"
	"github.com/leonyang215/ModelWeave/internal/redisstate"
)

func sharedCaches(t *testing.T) (*redisstate.Store, *redisstate.Store, *miniredis.Miniredis) {
	t.Helper()
	url := os.Getenv("MW_TEST_REDIS_URL")
	var mini *miniredis.Miniredis
	if url == "" {
		mini = miniredis.RunT(t)
		mini.ClockTTL(true)
		url = "redis://" + mini.Addr()
	}
	ns := fmt.Sprintf("%s/%d", t.Name(), time.Now().UnixNano())
	newStore := func() *redisstate.Store {
		s, err := redisstate.New(context.Background(), redisstate.Options{URL: url, Namespace: ns})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = s.Close() })
		return s
	}
	return newStore(), newStore(), mini
}

func TestRedisSharedSessionsAndModelIsolation(t *testing.T) {
	a, b, mini := sharedCaches(t)
	onError := func(err error) { t.Error(err) }
	sa, sb := orchestrator.NewRedisSessionMap(a, "m", time.Minute, onError), orchestrator.NewRedisSessionMap(b, "m", time.Minute, onError)
	sa.Set("session-a", "w0@1")
	if w, ok := sb.Get("session-a"); !ok || w != "w0@1" {
		t.Fatalf("session: %s %v", w, ok)
	}
	other := orchestrator.NewRedisSessionMap(b, "other-model", time.Minute, onError)
	if _, ok := other.Get("session-a"); ok {
		t.Fatal("session leaked across models")
	}
	sb.Set("session-a", "w1@2")
	sa.ForgetWorker("w0@1")
	if w, ok := sa.Get("session-a"); !ok || w != "w1@2" {
		t.Fatal("late invalidation erased a rebound session")
	}
	sa.Set("session-b", "w1@2")
	if sb.Len() != 2 {
		t.Fatal("shared session count")
	}
	sb.Delete("session-b")
	if sa.Len() != 1 {
		t.Fatal("shared session deletion")
	}
	sb.ForgetWorker("w1@2")
	if _, ok := sa.Get("session-a"); ok || sa.Len() != 0 {
		t.Fatal("removed instance kept sessions")
	}
	if mini != nil {
		sa.Set("expires", "w0@3")
		mini.SetTime(time.Now().Add(2 * time.Minute))
		if _, ok := sb.Get("expires"); ok || sb.Len() != 0 {
			t.Fatal("session TTL")
		}
	}
}

func TestRedisSharedArtifactsBoundedLRUAndInvalidation(t *testing.T) {
	a, b, mini := sharedCaches(t)
	onError := func(err error) { t.Error(err) }
	ca, cb := orchestrator.NewRedisArtifactCache(a, "m", time.Minute, 2, onError), orchestrator.NewRedisArtifactCache(b, "m", time.Minute, 2, onError)
	arts := mwip.EmbeddingLocation{PlanID: "plan-id", URI: "s3://test/artifact", Tokens: 16, Encoder: "e0"}
	ca.Put("a", arts)
	ca.Put("b", arts)
	if got, ok := cb.Get("a"); !ok || got != arts {
		t.Fatalf("shared artifact: %+v %v", got, ok)
	}
	cb.Put("c", arts)
	if _, ok := ca.Get("b"); ok {
		t.Fatal("global LRU capacity was exceeded")
	}
	if ca.Len() != 2 {
		t.Fatal("global artifact count")
	}
	ca.Invalidate("a")
	if _, ok := cb.Get("a"); ok {
		t.Fatal("artifact invalidation not shared")
	}
	other := orchestrator.NewRedisArtifactCache(b, "n", time.Minute, 2, onError)
	if _, ok := other.Get("c"); ok {
		t.Fatal("artifacts leaked across models")
	}
	if mini != nil {
		mini.SetTime(time.Now().Add(2 * time.Minute))
		if _, ok := ca.Get("c"); ok || ca.Len() != 0 {
			t.Fatal("artifact TTL")
		}
	}
}
