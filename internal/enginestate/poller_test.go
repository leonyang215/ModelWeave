package enginestate

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/registry"
)

// A poll records the reported load of every instance; a failed poll leaves
// the instance's last report in place.
func TestPollerUpdatesRegistry(t *testing.T) {
	reg := registry.New(time.Minute)
	w1, _, _ := reg.Upsert(registry.Spec{ID: "a", Model: "m", Role: registry.RoleDecode, Target: "a:9001"}, "static")
	w2, _, _ := reg.Upsert(registry.Spec{ID: "b", Model: "m", Role: registry.RoleDecode, Target: "b:9001"}, "static")
	w2.UpdateState(registry.EngineState{RunningRequests: 2}, time.Now())
	state := func(_ context.Context, spec registry.Spec) (registry.EngineState, error) {
		if spec.ID == "b" {
			return registry.EngineState{}, errors.New("unavailable")
		}
		return registry.EngineState{RunningRequests: 7, WaitingRequests: 1, KVUsedTokens: 100, KVCapacityTokens: 1000}, nil
	}
	outcomes := map[string]int{}
	results := make(chan string, 2)
	NewPoller(reg, Options{State: state, OnResult: func(o string) { results <- o }}).PollAll(context.Background())
	close(results)
	for o := range results {
		outcomes[o]++
	}
	if s := w1.Snapshot(time.Now(), time.Minute); s.State.RunningRequests != 7 || s.KVCapacity != 1000 || s.EffectiveKVUsed != 100 {
		t.Fatalf("polled state: %+v", s)
	}
	if s := w2.Snapshot(time.Now(), time.Minute); s.State.RunningRequests != 2 {
		t.Fatalf("a failed poll replaced the last report: %+v", s.State)
	}
	if outcomes["ok"] != 1 || outcomes["error"] != 1 {
		t.Fatalf("outcomes %v", outcomes)
	}
}
