package qos

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/redis/go-redis/v9"

	"github.com/leonyang215/ModelWeave/internal/redisstate"
	"github.com/leonyang215/ModelWeave/internal/registry"
)

// RedisBackend shares one gateway-owner lease with the instance store. Only
// request bodies/callbacks are local; quotas, queue order and turns are not.
type RedisBackend struct {
	state   *redisstate.Store
	limiter *redisLimiter
	mu      sync.Mutex
	queues  map[string]*redisQueue
	pending map[string]func(context.Context) error
	closed  atomic.Bool
	lost    atomic.Bool
	pubsub  *redis.PubSub
	stop    chan struct{}
	wg      sync.WaitGroup
}

var _ Backend = (*RedisBackend)(nil)

func NewRedisBackend(ctx context.Context, state *redisstate.Store, policies []TenantPolicy) (Backend, error) {
	b := &RedisBackend{state: state, queues: make(map[string]*redisQueue), pending: make(map[string]func(context.Context) error), stop: make(chan struct{})}
	l, err := newRedisLimiter(ctx, b, policies)
	if err != nil {
		return nil, err
	}
	b.limiter = l
	// Pub/Sub only reduces wakeup latency. Periodic durable state reads keep
	// the queue correct if a notification is missed or a connection restarts.
	b.pubsub = state.Client.Subscribe(context.Background(), state.Key("qos:wake"))
	subctx, cancel := state.OperationContext(ctx)
	_, err = b.pubsub.Receive(subctx)
	cancel()
	if err != nil {
		_ = b.pubsub.Close()
		return nil, qosUnavailable("subscribe", err)
	}
	b.wg.Add(1)
	go b.listen()
	return b, nil
}
func (b *RedisBackend) Kind() string           { return "redis" }
func (b *RedisBackend) Limiter() TenantLimiter { return b.limiter }

func qosUnavailable(op string, err error) error {
	return fmt.Errorf("%w: redis QoS %s: %v", registry.ErrStoreUnavailable, op, err)
}
func (b *RedisBackend) run(ctx context.Context, script *redis.Script, keys []string, args ...any) (string, error) {
	if b.lost.Load() {
		return "", qosUnavailable("lease", errors.New("gateway lease expired; restart this gateway"))
	}
	ctx, cancel := b.state.OperationContext(ctx)
	defer cancel()
	raw, err := script.Run(ctx, b.state.Client, keys, args...).Text()
	if err != nil {
		return "", qosUnavailable("operation", err)
	}
	return raw, nil
}
func (b *RedisBackend) ownerLost() error {
	b.lost.Store(true)
	return qosUnavailable("lease", errors.New("gateway lease expired; restart this gateway"))
}
func (b *RedisBackend) ensureConfig(ctx context.Context, key string, value any) error {
	body, err := json.Marshal(value)
	if err != nil {
		return err
	}
	ctx, cancel := b.state.OperationContext(ctx)
	defer cancel()
	if err := b.state.Client.SetNX(ctx, key, string(body), 0).Err(); err != nil {
		return qosUnavailable("settings", err)
	}
	got, err := b.state.Client.Get(ctx, key).Result()
	if err != nil {
		return qosUnavailable("settings", err)
	}
	if got != string(body) {
		return fmt.Errorf("Redis QoS namespace has incompatible tenant or admission settings; use consistent configuration or a fresh namespace")
	}
	return nil
}
func (b *RedisBackend) remember(key string, fn func(context.Context) error) {
	b.mu.Lock()
	b.pending[key] = fn
	b.mu.Unlock()
}
func (b *RedisBackend) Maintain(ctx context.Context) error {
	if b.lost.Load() {
		return b.ownerLost()
	}
	b.mu.Lock()
	batch := b.pending
	b.pending = make(map[string]func(context.Context) error)
	b.mu.Unlock()
	var first error
	for k, fn := range batch {
		if err := fn(ctx); err != nil {
			b.remember(k, fn)
			if first == nil {
				first = err
			}
		}
	}
	return first
}
func (b *RedisBackend) NewQueue(ctx context.Context, name string, cfg QueueConfig, obs QueueObserver) (AdmissionQueue, error) {
	if b.closed.Load() {
		return nil, ErrClosed
	}
	q, err := newRedisQueue(ctx, b, name, cfg, obs)
	if err != nil {
		return nil, err
	}
	b.mu.Lock()
	if b.closed.Load() {
		b.mu.Unlock()
		return nil, ErrClosed
	}
	if _, exists := b.queues[q.name]; exists {
		b.mu.Unlock()
		return nil, fmt.Errorf("admission queue %q already configured", name)
	}
	b.queues[q.name] = q
	b.mu.Unlock()
	q.start()
	return q, nil
}
func (b *RedisBackend) listen() {
	defer b.wg.Done()
	ch := b.pubsub.Channel()
	for {
		select {
		case <-b.stop:
			return
		case msg, ok := <-ch:
			if !ok {
				return
			}
			b.mu.Lock()
			q := b.queues[msg.Payload]
			b.mu.Unlock()
			if q != nil {
				q.wake()
			}
		}
	}
}
func (b *RedisBackend) notify(name string) {
	if b.closed.Load() {
		return
	}
	ctx, cancel := b.state.Context()
	defer cancel()
	// A failed notification never changes admission results; queue polling
	// and state transitions do not depend on Pub/Sub delivery.
	_ = b.state.Client.Publish(ctx, b.state.Key("qos:wake"), name).Err()
}
func (b *RedisBackend) Close() {
	if !b.closed.CompareAndSwap(false, true) {
		return
	}
	b.mu.Lock()
	qs := make([]*redisQueue, 0, len(b.queues))
	for _, q := range b.queues {
		qs = append(qs, q)
	}
	b.mu.Unlock()
	for _, q := range qs {
		q.Close()
	}
	close(b.stop)
	_ = b.pubsub.Close()
	b.wg.Wait()
}

func uniqueID() (string, error) {
	var id [16]byte
	_, err := rand.Read(id[:])
	return hex.EncodeToString(id[:]), err
}

func sortedPolicies(policies []TenantPolicy) []TenantPolicy {
	out := append([]TenantPolicy(nil), policies...)
	for i := range out {
		if out[i].Weight <= 0 {
			out[i].Weight = 1
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
