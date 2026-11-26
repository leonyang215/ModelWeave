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
	Timeout time.Duration
	State   StateFunc
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
	return &Poller{reg: reg, opts: opts}
}

// PollAll polls every instance concurrently (at most 32 at a time).
func (p *Poller) PollAll(ctx context.Context) {
	sem := make(chan struct{}, 32)
	var wg sync.WaitGroup
	for _, w := range p.reg.All() {
		wg.Add(1)
		sem <- struct{}{}
		go func(w *registry.Worker) {
			defer wg.Done()
			defer func() { <-sem }()
			st, err := p.Fetch(ctx, w.Spec())
			outcome := "ok"
			if err != nil {
				outcome = "error"
			} else {
				w.UpdateState(st, time.Now())
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
