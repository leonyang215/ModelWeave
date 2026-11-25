package kvindex

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Remote queries an external KV metadata service, which learns cache contents
// directly from the engines. When the service fails or is slow, the fallback
// index answers instead so routing keeps working.
//
// Wire contract (POST {URL}/v1/prefix/match):
//
//	request:  {"model": "m", "block_size": 64, "block_hashes": [u64...], "candidates": ["w1", ...]}
//	response: {"matched_blocks": {"w1": 12, "w2": 0}}
type Remote struct {
	URL      string
	Client   *http.Client
	Timeout  time.Duration
	Fallback Index
	// OnResult, if set, observes the outcome of every remote lookup
	// ("ok", "error", "fallback") and its latency.
	OnResult func(outcome string, d time.Duration)
}

type remoteMatchRequest struct {
	Model       string   `json:"model"`
	BlockSize   int      `json:"block_size"`
	BlockHashes []uint64 `json:"block_hashes"`
	Candidates  []string `json:"candidates"`
}

type remoteMatchResponse struct {
	MatchedBlocks map[string]int `json:"matched_blocks"`
}

func (r *Remote) Kind() string { return "remote" }

func (r *Remote) Match(ctx context.Context, q Query) (map[string]int, error) {
	start := time.Now()
	out, err := r.match(ctx, q)
	if err == nil {
		r.report("ok", start)
		return out, nil
	}
	if r.Fallback != nil {
		r.report("fallback", start)
		return r.Fallback.Match(ctx, q)
	}
	r.report("error", start)
	return nil, err
}

func (r *Remote) report(outcome string, start time.Time) {
	if r.OnResult != nil {
		r.OnResult(outcome, time.Since(start))
	}
}

func (r *Remote) match(ctx context.Context, q Query) (map[string]int, error) {
	if len(q.Blocks) == 0 {
		out := make(map[string]int, len(q.Candidates))
		for _, c := range q.Candidates {
			out[c] = 0
		}
		return out, nil
	}
	timeout := r.Timeout
	if timeout <= 0 {
		timeout = 50 * time.Millisecond
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	body, err := json.Marshal(remoteMatchRequest{
		Model: q.Model, BlockSize: q.BlockSize, BlockHashes: q.Blocks, Candidates: q.Candidates,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.URL+"/v1/prefix/match", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := r.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("kv index %s: status %d: %s", r.URL, resp.StatusCode, bytes.TrimSpace(b))
	}
	var mr remoteMatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&mr); err != nil {
		return nil, fmt.Errorf("kv index %s: decode: %w", r.URL, err)
	}
	out := make(map[string]int, len(q.Candidates))
	for _, c := range q.Candidates {
		n := mr.MatchedBlocks[c]
		if n > len(q.Blocks) {
			n = len(q.Blocks)
		}
		out[c] = n
	}
	return out, nil
}

func (r *Remote) Observe(worker string, blocks []uint64) {
	if r.Fallback != nil {
		r.Fallback.Observe(worker, blocks)
	}
}

func (r *Remote) ForgetWorker(worker string) {
	if r.Fallback != nil {
		r.Fallback.ForgetWorker(worker)
	}
}

func (r *Remote) SetWorkerCapacity(worker string, n int) {
	if r.Fallback != nil {
		r.Fallback.SetWorkerCapacity(worker, n)
	}
}
