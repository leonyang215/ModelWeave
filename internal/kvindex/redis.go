package kvindex

import (
	"context"
	_ "embed"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
)

//go:embed redis.lua
var redisSource string
var redisScript = redis.NewScript(redisSource)

type RedisOptions struct {
	Type                        string // approx or events
	TTL, OptimisticTTL, Timeout time.Duration
	CapacityBlocks              int
	OnError                     func(error)
}

// Redis shares prefix observations and KV events across gateways. Keys are
// scoped to registration epochs by the gateway, never just instance IDs.
type Redis struct {
	client *redis.Client
	prefix string
	opts   RedisOptions
}

func NewRedis(client *redis.Client, prefix string, opts RedisOptions) *Redis {
	if opts.Timeout <= 0 {
		opts.Timeout = time.Second
	}
	return &Redis{client: client, prefix: prefix, opts: opts}
}
func (r *Redis) Kind() string { return r.opts.Type }
func (r *Redis) keys(worker, part string) []string {
	k := r.prefix + part + ":" + worker
	return []string{k + ":seen", k + ":order", k + ":meta"}
}
func hashArgs(hashes []uint64) []any {
	out := make([]any, len(hashes))
	for i, h := range hashes {
		out[i] = strconv.FormatUint(h, 10)
	}
	return out
}
func (r *Redis) Match(ctx context.Context, q Query) (map[string]int, error) {
	out := make(map[string]int, len(q.Candidates))
	if len(q.Candidates) == 0 || len(q.Blocks) == 0 {
		return out, nil
	}
	ctx, cancel := context.WithTimeout(ctx, r.opts.Timeout)
	defer cancel()
	pipe := r.client.Pipeline()
	cmds := make(map[string][]*redis.Cmd, len(q.Candidates))
	type part struct {
		name string
		ttl  time.Duration
	}
	for _, id := range q.Candidates {
		key := id
		if k := q.WorkerKeys[id]; k != "" {
			key = k
		}
		parts := []part{{"approx", r.opts.TTL}}
		if r.opts.Type == "events" {
			parts = []part{{"exact", 0}}
			if r.opts.OptimisticTTL > 0 {
				parts = append(parts, part{"optimistic", r.opts.OptimisticTTL})
			}
		}
		for _, p := range parts {
			args := append([]any{"match", p.ttl.Milliseconds()}, hashArgs(q.Blocks)...)
			cmds[id] = append(cmds[id], pipe.Eval(ctx, redisSource, r.keys(key, p.name), args...))
		}
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("redis prefix lookup: %w", err)
	}
	for id, cs := range cmds {
		for _, c := range cs {
			n, err := c.Int()
			if err != nil {
				return nil, err
			}
			out[id] = max(out[id], n)
		}
	}
	return out, nil
}
func (r *Redis) mutate(keys []string, args ...any) error {
	ctx, cancel := context.WithTimeout(context.Background(), r.opts.Timeout)
	defer cancel()
	err := redisScript.Run(ctx, r.client, keys, args...).Err()
	if err != nil && r.opts.OnError != nil {
		r.opts.OnError(err)
	}
	return err
}
func (r *Redis) Observe(worker string, hashes []uint64) {
	part, ttl := "approx", r.opts.TTL
	if r.opts.Type == "events" {
		part, ttl = "optimistic", r.opts.OptimisticTTL
		if ttl <= 0 {
			return
		}
	}
	args := append([]any{"observe", ttl.Milliseconds(), r.opts.CapacityBlocks}, hashArgs(hashes)...)
	_ = r.mutate(r.keys(worker, part), args...)
}
func (r *Redis) ForgetWorker(worker string) {
	keys := append(r.keys(worker, "approx"), r.keys(worker, "exact")...)
	keys = append(keys, r.keys(worker, "optimistic")...)
	_ = r.mutate(keys, "forget")
}
func (r *Redis) SetWorkerCapacity(worker string, n int) {
	if r.opts.Type == "approx" {
		_ = r.mutate(r.keys(worker, "approx"), "capacity", n)
	}
}
func (r *Redis) ApplyEvents(worker string, b EventBatch) error {
	if r.opts.Type != "events" {
		return nil
	}
	keys := append(r.keys(worker, "exact"), r.keys(worker, "optimistic")...)
	cleared := 0
	if b.Cleared {
		cleared = 1
	}
	args := []any{"events", cleared, r.opts.OptimisticTTL.Milliseconds(), len(b.Stored), len(b.Removed)}
	args = append(args, hashArgs(b.Stored)...)
	args = append(args, hashArgs(b.Removed)...)
	return r.mutate(keys, args...)
}

// Stats counts resident block-instance entries, excluding expired ones.
func (r *Redis) Stats(ctx context.Context, workers []string) (int, map[string]int, error) {
	ctx, cancel := context.WithTimeout(ctx, r.opts.Timeout)
	defer cancel()
	part, ttl := "approx", r.opts.TTL
	if r.opts.Type == "events" {
		part, ttl = "exact", 0
	}
	pipe := r.client.Pipeline()
	cmds := make(map[string]*redis.Cmd, len(workers))
	for _, w := range workers {
		cmds[w] = pipe.Eval(ctx, redisSource, r.keys(w, part), "stats", ttl.Milliseconds())
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return 0, nil, err
	}
	counts, total := make(map[string]int, len(workers)), 0
	for w, c := range cmds {
		n, err := c.Int()
		if err != nil {
			return 0, nil, err
		}
		counts[w] = n
		total += n
	}
	return total, counts, nil
}
