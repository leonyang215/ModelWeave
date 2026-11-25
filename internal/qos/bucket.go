// Package qos implements per-tenant rate limiting, concurrency limits and the
// admission queue that holds requests while no worker has capacity.
package qos

import (
	"math"
	"sync"
	"time"
)

// Bucket is a token bucket refilled continuously at rate tokens per second.
// The balance may go negative ("debt") when a request turns out to cost more
// than its up-front estimate; new requests wait until the debt is repaid.
type Bucket struct {
	mu       sync.Mutex
	capacity float64
	rate     float64 // tokens per second
	tokens   float64
	last     time.Time
	now      func() time.Time
}

// NewPerMinute returns a bucket that allows perMinute tokens per minute with a
// burst of burst tokens (burst <= 0 means one minute worth).
func NewPerMinute(perMinute float64, burst float64) *Bucket {
	if burst <= 0 {
		burst = perMinute
	}
	b := &Bucket{capacity: burst, rate: perMinute / 60, tokens: burst, now: time.Now}
	b.last = b.now()
	return b
}

func (b *Bucket) refill() {
	now := b.now()
	if el := now.Sub(b.last).Seconds(); el > 0 {
		b.tokens = math.Min(b.capacity, b.tokens+el*b.rate)
		b.last = now
	}
}

// Take consumes n tokens if available. Otherwise it consumes nothing and
// reports how long until n tokens would be available.
func (b *Bucket) Take(n float64) (bool, time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refill()
	need := n
	if need > b.capacity {
		// A single request larger than the burst is admitted when the bucket
		// is full; it then leaves the bucket in debt.
		need = b.capacity
	}
	if b.tokens >= need {
		b.tokens -= n
		return true, 0
	}
	if b.rate <= 0 {
		return false, time.Hour
	}
	wait := time.Duration((need - b.tokens) / b.rate * float64(time.Second))
	return false, wait
}

// Adjust adds delta tokens (positive refunds, negative charges). The balance
// is capped at capacity but may go negative.
func (b *Bucket) Adjust(delta float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refill()
	b.tokens = math.Min(b.capacity, b.tokens+delta)
}

// Available returns the current balance.
func (b *Bucket) Available() float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refill()
	return b.tokens
}
