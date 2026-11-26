// Package enginestate collects engine load by polling instances with
// Engine.GetState. It is the pull side of engine state, for instances that
// send no heartbeats (static ones); the push side is the mwip Registry
// heartbeat handled by the gateway. Both end up in registry.Worker.
package enginestate

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/leonyang215/ModelWeave/internal/registry"
)

// StateFunc asks an instance for its load (Engine.GetState).
type StateFunc func(ctx context.Context, spec registry.Spec) (registry.EngineState, error)

// Options configure a poller.
type Options struct {
	Timeout  time.Duration
	Interval time.Duration // shared poll lease; default 2s
	State    StateFunc
	// OnResult reports every poll: outcome is "ok" or "error".
	OnResult func(outcome string)
}

// Poller fetches the engine state of every registered instance.
type Poller struct {
	reg  *registry.Registry
	opts Options
}

// NewPoller returns a poller.
func NewPoller(reg *registry.Registry, opts Options) *Poller {
	if opts.Timeout <= 0 {
		opts.Timeout = time.Second
	}
	if opts.Interval <= 0 {
		opts.Interval = 2 * time.Second
	}
	return &Poller{reg: reg, opts: opts}
}

// PollAll polls every instance concurrently (at most 32 at a time).
func (p *Poller) PollAll(ctx context.Context) {
	if err := p.reg.Sync(ctx); err != nil {
		if p.opts.OnResult != nil {
			p.opts.OnResult("error")
		}
		return
	}
	sem := make(chan struct{}, 32)
	var wg sync.WaitGroup
	for _, w := range p.reg.All() {
		wg.Add(1)
		sem <- struct{}{}
		go func(w *registry.Worker) {
			defer wg.Done()
			defer func() { <-sem }()
			if p.reg.Store().Shared() {
				claimed, err := p.reg.Store().Claim(ctx, "state:"+w.CacheKey(), max(p.opts.Interval, p.opts.Timeout))
				if err != nil {
					if p.opts.OnResult != nil {
						p.opts.OnResult("error")
					}
					return
				}
				if !claimed {
					return
				}
			}
			snap, err := w.SnapshotContext(ctx, time.Now(), 0)
			var st registry.EngineState
			if err == nil && !snap.Gone {
				st, err = p.Fetch(ctx, snap.Spec)
			}
			if err == nil && !snap.Gone {
				err = w.UpdateStateIfRevision(ctx, st, time.Now(), snap.Revision)
			}
			outcome := "ok"
			if err != nil {
				outcome = "error"
			}
			if p.opts.OnResult != nil {
				p.opts.OnResult(outcome)
			}
		}(w)
	}
	wg.Wait()
}

// Fetch retrieves the state of one instance.
func (p *Poller) Fetch(ctx context.Context, spec registry.Spec) (registry.EngineState, error) {
	if p.opts.State == nil {
		return registry.EngineState{}, errors.New("no engine client")
	}
	ctx, cancel := context.WithTimeout(ctx, p.opts.Timeout)
	defer cancel()
	return p.opts.State(ctx, spec)
}
