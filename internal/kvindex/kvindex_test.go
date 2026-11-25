package kvindex

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func seq(n int, start uint32) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = start + uint32(i)
	}
	return out
}

func TestBlockHashesChained(t *testing.T) {
	a := BlockHashes(seq(130, 0), 64, 7)
	if len(a) != 2 {
		t.Fatalf("want 2 full blocks, got %d", len(a))
	}
	// Same first block, different second block: first hash equal, second differs.
	toks := seq(128, 0)
	toks[100] = 99999
	b := BlockHashes(toks, 64, 7)
	if a[0] != b[0] || a[1] == b[1] {
		t.Fatalf("prefix property violated: %v vs %v", a, b)
	}
	// Different first block changes every later hash (chaining).
	toks2 := seq(128, 0)
	toks2[0] = 42
	c := BlockHashes(toks2, 64, 7)
	if c[0] == a[0] || c[1] == a[1] {
		t.Fatalf("chaining violated")
	}
	// Seed separates models.
	if BlockHashes(seq(64, 0), 64, 1)[0] == BlockHashes(seq(64, 0), 64, 2)[0] {
		t.Fatalf("seed ignored")
	}
	if BlockHashes(seq(10, 0), 64, 0) != nil {
		t.Fatalf("partial block must not hash")
	}
}

func TestBlockIndexMatchPrefix(t *testing.T) {
	idx := NewBlockIndex(0, 0)
	h := []uint64{1, 2, 3, 4}
	idx.Insert("w1", h[:3])
	idx.Insert("w2", h[:1])
	got := idx.Match(h, []string{"w1", "w2", "w3"})
	if got["w1"] != 3 || got["w2"] != 1 || got["w3"] != 0 {
		t.Fatalf("unexpected match %v", got)
	}
	// A hole breaks the prefix.
	idx.Insert("w3", []uint64{2, 3})
	if got := idx.Match(h, []string{"w3"}); got["w3"] != 0 {
		t.Fatalf("non-prefix match counted: %v", got)
	}
	idx.Remove("w1", []uint64{2})
	if got := idx.Match(h, []string{"w1"}); got["w1"] != 1 {
		t.Fatalf("remove ignored: %v", got)
	}
	idx.Clear("w1")
	if got := idx.Match(h, []string{"w1"}); got["w1"] != 0 {
		t.Fatalf("clear ignored: %v", got)
	}
}

func TestBlockIndexCapacityEvictsTailFirst(t *testing.T) {
	idx := NewBlockIndex(0, 3)
	idx.Insert("w", []uint64{1, 2, 3, 4, 5})
	// Capacity 3: blocks 1..3 (the head of the prefix) must survive.
	if got := idx.Match([]uint64{1, 2, 3, 4, 5}, []string{"w"}); got["w"] != 3 {
		t.Fatalf("want head of prefix kept, got %v", got)
	}
	blocks, per := idx.Stats()
	if blocks != 3 || per["w"] != 3 {
		t.Fatalf("stats %d %v", blocks, per)
	}
}

func TestBlockIndexTTL(t *testing.T) {
	idx := NewBlockIndex(time.Minute, 0)
	now := time.Unix(1000, 0)
	idx.now = func() time.Time { return now }
	idx.Insert("w", []uint64{1, 2})
	now = now.Add(30 * time.Second)
	idx.Insert("w", []uint64{1})
	now = now.Add(45 * time.Second) // block 2 is 75s old, block 1 is 45s old
	if got := idx.Match([]uint64{1, 2}, []string{"w"}); got["w"] != 1 {
		t.Fatalf("expired block counted: %v", got)
	}
	if n := idx.Sweep(); n != 1 {
		t.Fatalf("sweep removed %d", n)
	}
	blocks, _ := idx.Stats()
	if blocks != 1 {
		t.Fatalf("blocks after sweep %d", blocks)
	}
}

func TestEventsIndex(t *testing.T) {
	e := NewEvents(time.Second)
	e.ApplyEvents("w1", EventBatch{Stored: []uint64{1, 2, 3}})
	e.Observe("w2", []uint64{1, 2})
	got, _ := e.Match(context.Background(), Query{Blocks: []uint64{1, 2, 3}, Candidates: []string{"w1", "w2"}})
	if got["w1"] != 3 || got["w2"] != 2 {
		t.Fatalf("unexpected %v", got)
	}
	e.ApplyEvents("w1", EventBatch{Removed: []uint64{2}})
	got, _ = e.Match(context.Background(), Query{Blocks: []uint64{1, 2, 3}, Candidates: []string{"w1"}})
	if got["w1"] != 1 {
		t.Fatalf("removal ignored %v", got)
	}
	e.ApplyEvents("w1", EventBatch{Cleared: true})
	got, _ = e.Match(context.Background(), Query{Blocks: []uint64{1}, Candidates: []string{"w1"}})
	if got["w1"] != 0 {
		t.Fatalf("clear ignored %v", got)
	}
}

func TestRemoteWithFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req remoteMatchRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "m" || len(req.BlockHashes) != 2 {
			http.Error(w, "bad", 400)
			return
		}
		_ = json.NewEncoder(w).Encode(remoteMatchResponse{MatchedBlocks: map[string]int{"a": 2, "b": 9}})
	}))
	defer srv.Close()
	fb := NewApprox(0, 0)
	fb.Observe("c", []uint64{10})
	r := &Remote{URL: srv.URL, Timeout: time.Second, Fallback: fb}
	got, err := r.Match(context.Background(), Query{Model: "m", Blocks: []uint64{10, 11}, Candidates: []string{"a", "b"}})
	if err != nil || got["a"] != 2 || got["b"] != 2 {
		t.Fatalf("remote match %v %v", got, err)
	}
	r.URL = "http://127.0.0.1:1" // unreachable: fallback answers
	got, err = r.Match(context.Background(), Query{Model: "m", Blocks: []uint64{10, 11}, Candidates: []string{"c"}})
	if err != nil || got["c"] != 1 {
		t.Fatalf("fallback match %v %v", got, err)
	}
}
