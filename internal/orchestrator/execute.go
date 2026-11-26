package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"time"

	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

// Emit forwards one engine output event (ready, delta, finish) towards the
// client. An error stops the execution: the caller is gone.
type Emit func(ev *mwip.GenerateEvent) error

// ErrFirstTokenTimeout means no output arrived before the deadline.
var ErrFirstTokenTimeout = errors.New("first token deadline exceeded")

// ErrIdleTimeout means the output stream stalled.
var ErrIdleTimeout = errors.New("output stream stalled")

// ErrClientGone is returned by an Emit that could not write to the client.
var ErrClientGone = errors.New("client gone")

// StoppedError wraps the error returned by Emit.
type StoppedError struct{ Err error }

func (e *StoppedError) Error() string { return "stopped: " + e.Err.Error() }
func (e *StoppedError) Unwrap() error { return e.Err }

// Outcome describes a finished execution.
type Outcome struct {
	Mode         string
	Reason       string
	Fallback     string
	Attempts     int
	DWorker      string
	PWorker      string
	CachedTokens int // predicted by the scheduler for the worker that prefilled
	Usage        *mwip.Usage
	FinishReason string
	Timing       *mwip.Timing
	DispatchedAt time.Time
	FirstTokenAt time.Time
	LastTokenAt  time.Time
	OutputEvents int
	RegisterDur  time.Duration
	PrefillDur   time.Duration
	Err          error
}

type stage string

const (
	stageD stage = "decode"
	stageP stage = "prefill"
)

type attemptResult struct {
	err     error
	failed  stage
	emitted bool
}

// Execute runs a planned job to completion. Before the first token reaches
// the client, failed attempts are retried on other workers and a failed
// normal attempt on a decode worker falls back to pd; afterwards errors end
// the request. The plan's reservations are released before Execute returns.
func (o *Orchestrator) Execute(ctx context.Context, job *Job, plan *Plan, emit Emit) *Outcome {
	m := job.Model
	out := &Outcome{Mode: plan.Mode, Reason: plan.Reason}
	excludeD, excludeP := map[string]bool{}, map[string]bool{}
	dAttempts, wholeRetries := 0, 0
	fellBack := false
	cur := plan
	for {
		out.Attempts++
		attemptID := fmt.Sprintf("%s-%d", job.ID, out.Attempts)
		out.Mode, out.Reason, out.DWorker = cur.Mode, cur.Reason, cur.D.Spec.ID
		if cur.P != nil {
			out.PWorker = cur.P.Spec.ID
			out.CachedTokens = cur.P.CachedTokens
		} else {
			out.PWorker = ""
			out.CachedTokens = cur.D.CachedTokens
		}
		if out.DispatchedAt.IsZero() {
			out.DispatchedAt = time.Now()
		}
		if job.OnAttempt != nil {
			job.OnAttempt(AttemptInfo{Attempt: out.Attempts, Mode: cur.Mode, Reason: cur.Reason, Decode: out.DWorker, Prefill: out.PWorker, CachedTokens: out.CachedTokens})
		}
		res := o.runAttempt(ctx, job, cur, attemptID, emit, out)
		if res.err == nil {
			cur.Release(true)
			cur.D.Worker.ReportSuccess()
			m.Sessions.Set(job.SessionKey, cur.D.Spec.ID)
			return out
		}
		cur.Release(false)
		o.noteFailure(job, cur, res)
		if res.emitted || ctx.Err() != nil || isStopped(res.err) || !retriable(res.err) {
			if res.failed == stageD || cur.Mode != ModePD {
				m.Sessions.Delete(job.SessionKey)
			}
			out.Err = res.err
			return out
		}
		var next *Plan
		var err error
		switch {
		case cur.Mode == ModeNormal && cur.D.Spec.Role == registry.RoleDecode && m.Cfg.Normal.FallbackToPD &&
			!fellBack && o.has(m.Name, registry.RolePrefill, false):
			fellBack = true
			out.Fallback = errCode(res.err)
			o.met.Fallbacks.WithLabelValues(m.Name, ModeNormal, ModePD, out.Fallback).Inc()
			m.Sessions.Delete(job.SessionKey)
			if isTransport(res.err) {
				excludeD[cur.D.Spec.ID] = true
			}
			next, err = o.Plan(ctx, job, PlanOpts{ForceMode: ModePD, ExcludeD: excludeD, ExcludeP: excludeP})
			if next != nil {
				next.Reason = "fallback"
			}
		case res.failed == stageP:
			wholeRetries++
			if wholeRetries > m.Cfg.Retry.WholeRetries {
				out.Err = res.err
				return out
			}
			excludeP[cur.P.Spec.ID] = true
			o.met.Retries.WithLabelValues(m.Name, string(stageP), errCode(res.err)).Inc()
			next, err = o.Plan(ctx, job, PlanOpts{ForceMode: cur.Mode, ExcludeD: excludeD, ExcludeP: excludeP})
		default:
			dAttempts++
			if dAttempts >= m.Cfg.Retry.StageAttempts {
				out.Err = res.err
				return out
			}
			excludeD[cur.D.Spec.ID] = true
			o.met.Retries.WithLabelValues(m.Name, string(stageD), errCode(res.err)).Inc()
			mode := cur.Mode
			if fellBack {
				mode = ModePD
			}
			next, err = o.Plan(ctx, job, PlanOpts{ForceMode: mode, ExcludeD: excludeD, ExcludeP: excludeP})
		}
		if err != nil {
			out.Err = fmt.Errorf("%w (retry after: %v)", err, res.err)
			return out
		}
		o.log.Info("retrying request", "request_id", job.ID, "attempt", out.Attempts+1, "mode", next.Mode,
			"decode", next.D.Spec.ID, "prefill", workerOf(next.P), "after_error", res.err.Error())
		cur = next
		select {
		case <-time.After(m.Cfg.Retry.Backoff.D()):
		case <-ctx.Done():
			cur.Release(false)
			out.Err = ctx.Err()
			return out
		}
	}
}

func workerOf(d *scheduler.Decision) string {
	if d == nil {
		return ""
	}
	return d.Spec.ID
}

func (o *Orchestrator) noteFailure(job *Job, plan *Plan, res attemptResult) {
	if isStopped(res.err) {
		return
	}
	var e *engine.Error
	if errors.As(res.err, &e) {
		o.met.UpstreamErrors.WithLabelValues(job.Model.Name, e.Role, e.Code).Inc()
		if countsAgainstHealth(e) {
			w := plan.D
			if res.failed == stageP && plan.P != nil {
				w = plan.P
			}
			if w.Worker.ReportFailure(time.Now(), o.health.FailureThreshold, o.health.Cooldown.D()) {
				o.log.Warn("worker ejected after consecutive failures", "worker", w.Spec.ID, "cooldown", o.health.Cooldown.D().String())
			}
		}
	}
}

// countsAgainstHealth reports whether a worker error indicates a broken
// worker (passive health checking, like consecutive-5xx outlier detection).
// Capacity rejections and declined normal requests are regular answers.
func countsAgainstHealth(e *engine.Error) bool {
	switch e.Code {
	case mwip.ErrCapacity, mwip.ErrNormalRejected, mwip.ErrBadRequest, mwip.ErrArtifact:
		return false
	}
	return e.Transport || e.Code == mwip.ErrInternal
}

func isStopped(err error) bool {
	var s *StoppedError
	return errors.As(err, &s)
}

func isTransport(err error) bool {
	var e *engine.Error
	return errors.As(err, &e) && e.Transport
}

func retriable(err error) bool {
	if errors.Is(err, ErrFirstTokenTimeout) || errors.Is(err, ErrIdleTimeout) {
		return false
	}
	var e *engine.Error
	if errors.As(err, &e) {
		if e.Code == mwip.ErrArtifact || e.Code == mwip.ErrBadRequest {
			// Missing artifacts are rebuilt by the caller; bad requests
			// fail everywhere.
			return false
		}
		return e.Retriable || e.Code == mwip.ErrNormalRejected || e.Code == mwip.ErrCapacity || e.Code == mwip.ErrKVTransfer
	}
	return false
}

func errCode(err error) string {
	var e *engine.Error
	if errors.As(err, &e) {
		return e.Code
	}
	switch {
	case errors.Is(err, ErrFirstTokenTimeout):
		return "first_token_timeout"
	case errors.Is(err, ErrIdleTimeout):
		return "idle_timeout"
	case errors.Is(err, context.Canceled):
		return "canceled"
	}
	return "error"
}

// stage builds one Generate call of an attempt: the intermediate state, the
// request's media (with the encode stage's artifacts) and the call's role.
func (o *Orchestrator) stage(job *Job, attemptID string, d mwip.Disagg) *engine.Stage {
	timeout := time.Until(job.FirstTokenDeadline).Milliseconds()
	if timeout < 1 {
		timeout = 1
	}
	if d.Room == "" {
		d.Room = attemptID
	}
	d.BootstrapRoom = bootstrapRoom(attemptID)
	return &engine.Stage{
		Render: job.Render, AttemptID: attemptID, EngineModel: job.Model.Cfg.EngineModel, Disagg: d,
		Media: job.Media, TimeoutMS: timeout,
		SessionKey: job.SessionKey, Priority: job.Priority.String(),
	}
}

// bootstrapRoom derives a numeric bootstrap room from the attempt, for KV
// connectors that pair by number.
func bootstrapRoom(attemptID string) int64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(attemptID))
	return int64(h.Sum64() & (1<<63 - 1))
}

func peer(d *scheduler.Decision) *mwip.Peer {
	if d == nil {
		return nil
	}
	p := &mwip.Peer{WorkerID: d.Spec.ID, Target: d.Spec.Target, BootstrapAddr: d.Spec.BootstrapAddr, EngineID: d.Spec.EngineID}
	if p.EngineID == "" {
		p.EngineID = d.Spec.ID
	}
	return p
}

// generate opens one engine stream (always streaming) and counts it.
func (o *Orchestrator) generate(ctx context.Context, job *Job, d *scheduler.Decision, st *engine.Stage) (*engine.Stream, error) {
	role := string(d.Spec.Role)
	if st.Disagg.Role != "" {
		role = st.Disagg.Role
	}
	o.met.EngineRequests.WithLabelValues(job.Model.Name, role).Inc()
	return o.eng.Generate(ctx, d.Spec, st)
}

func (o *Orchestrator) runAttempt(ctx context.Context, job *Job, plan *Plan, attemptID string, emit Emit, out *Outcome) attemptResult {
	actx, cancel := context.WithCancel(ctx)
	defer cancel()
	var res attemptResult
	switch plan.Mode {
	case ModeNormal:
		res = o.runSingle(actx, job, plan, attemptID, emit, out)
	case ModePD:
		switch job.Model.Cfg.PD.Handshake {
		case mwip.HandshakeBootstrap:
			res = o.runBootstrap(actx, job, plan, attemptID, emit, out)
		case mwip.HandshakePull:
			res = o.runPull(actx, job, plan, attemptID, emit, out)
		default:
			res = o.runPush(actx, job, plan, attemptID, emit, out)
		}
	default:
		res = attemptResult{err: fmt.Errorf("unknown mode %q", plan.Mode), failed: stageD}
	}
	if res.err != nil {
		reason := "error"
		switch {
		case ctx.Err() != nil, errors.Is(res.err, ErrClientGone):
			reason = "client_gone"
		case isStopped(res.err):
			reason = "stopped"
		case errors.Is(res.err, ErrFirstTokenTimeout), errors.Is(res.err, ErrIdleTimeout):
			reason = "timeout"
		}
		o.met.Aborts.WithLabelValues(job.Model.Name, reason).Inc()
		o.abortAll(job, plan, attemptID)
	}
	return res
}

// abortAll tells involved workers to drop the attempt (in addition to the
// connection close); best effort and asynchronous.
func (o *Orchestrator) abortAll(job *Job, plan *Plan, attemptID string) {
	targets := []*scheduler.Decision{plan.D}
	if plan.P != nil {
		targets = append(targets, plan.P)
	}
	for _, d := range targets {
		go func(d *scheduler.Decision) {
			_ = o.eng.Abort(context.Background(), d.Spec, attemptID)
		}(d)
	}
}

type evResult struct {
	ev  *mwip.GenerateEvent
	err error
}

// readLoop pumps a stream into a channel until it ends.
func readLoop(ctx context.Context, s *engine.Stream) <-chan evResult {
	ch := make(chan evResult, 16)
	go func() {
		defer close(ch)
		for {
			ev, err := s.Next()
			select {
			case ch <- evResult{ev, err}:
			case <-ctx.Done():
				return
			}
			if err != nil || ev.Type == mwip.EventFinish {
				return
			}
		}
	}()
	return ch
}

// pump forwards decode output to the client. prefillDone, when not nil,
// delivers the prefill stage's result: a prefill failure before the first
// token fails the attempt, while a first token makes the prefill result
// irrelevant (the decode side has its KV).
func (o *Orchestrator) pump(ctx context.Context, job *Job, dCh <-chan evResult, prefillDone <-chan prefillResult, emit Emit, out *Outcome, plan *Plan) attemptResult {
	idle := o.timeout.IdleTimeout.D()
	var pending *mwip.GenerateEvent // ready is held back until data exists
	emitted := false
	firstDeadline := time.NewTimer(time.Until(job.FirstTokenDeadline))
	defer firstDeadline.Stop()
	var idleTimer *time.Timer
	var idleC <-chan time.Time
	resetIdle := func() {
		if idleTimer == nil {
			idleTimer = time.NewTimer(idle)
			idleC = idleTimer.C
			return
		}
		idleTimer.Stop()
		idleTimer.Reset(idle)
	}
	defer func() {
		if idleTimer != nil {
			idleTimer.Stop()
		}
	}()
	send := func(ev *mwip.GenerateEvent) error {
		if err := emit(ev); err != nil {
			return &StoppedError{Err: err}
		}
		return nil
	}
	for {
		select {
		case <-ctx.Done():
			return attemptResult{err: ctx.Err(), failed: stageD, emitted: emitted}
		case <-firstDeadline.C:
			if !emitted {
				return attemptResult{err: ErrFirstTokenTimeout, failed: stageD}
			}
		case <-idleC:
			return attemptResult{err: ErrIdleTimeout, failed: stageD, emitted: emitted}
		case pr, ok := <-prefillDone:
			if !ok {
				prefillDone = nil
				continue
			}
			prefillDone = nil
			if pr.err != nil && !emitted {
				return attemptResult{err: pr.err, failed: stageP}
			}
			if pr.err == nil {
				out.PrefillDur = pr.dur
			}
			if plan.P != nil {
				plan.P.Reservation.PrefillDone()
			}
		case r, ok := <-dCh:
			if !ok {
				return attemptResult{err: &engine.Error{Role: "decode", WorkerID: plan.D.Spec.ID, Code: mwip.ErrInternal, Message: "stream closed", Retriable: true, Transport: true}, failed: stageD, emitted: emitted}
			}
			if r.err != nil {
				if errors.Is(r.err, io.EOF) {
					r.err = &engine.Error{Role: "decode", WorkerID: plan.D.Spec.ID, Code: mwip.ErrInternal, Message: "stream ended without finish", Retriable: true, Transport: true}
				}
				failed := stageD
				if engine.IsCode(r.err, mwip.ErrKVTransfer) {
					failed = stageP
				}
				return attemptResult{err: r.err, failed: failed, emitted: emitted}
			}
			ev := r.ev
			switch ev.Type {
			case mwip.EventReady:
				pending = ev
			case mwip.EventDelta, mwip.EventFinish:
				if !emitted {
					emitted = true
					out.FirstTokenAt = time.Now()
					firstDeadline.Stop()
					if plan.D != nil && plan.D.Reservation != nil {
						plan.D.Reservation.PrefillDone()
					}
					if pending == nil {
						pending = &mwip.GenerateEvent{Type: mwip.EventReady, PromptTokens: job.Render.PromptTokens}
					}
					if err := send(pending); err != nil {
						return attemptResult{err: err, emitted: true}
					}
				}
				out.LastTokenAt = time.Now()
				if ev.Type == mwip.EventDelta {
					out.OutputEvents++
				}
				if err := send(ev); err != nil {
					return attemptResult{err: err, emitted: true}
				}
				if ev.Type == mwip.EventFinish {
					out.Usage, out.FinishReason, out.Timing = ev.Usage, ev.Reason, ev.Timing
					return attemptResult{emitted: true}
				}
				resetIdle()
			default:
				// accepted / prefill_done on the decode stream: informational.
			}
		}
	}
}

func (o *Orchestrator) runSingle(ctx context.Context, job *Job, plan *Plan, attemptID string, emit Emit, out *Outcome) attemptResult {
	d := plan.D
	s, err := o.generate(ctx, job, d, o.stage(job, attemptID, mwip.Disagg{Mode: mwip.ModeNormal}))
	if err != nil {
		return attemptResult{err: err, failed: stageD}
	}
	defer s.Close()
	return o.pump(ctx, job, readLoop(ctx, s), nil, emit, out, plan)
}

// runPrefill runs the prefill stage (streaming, like every engine call) and
// waits for its end.
func (o *Orchestrator) runPrefill(ctx context.Context, job *Job, p *scheduler.Decision, st *engine.Stage) (*mwip.GenerateEvent, error) {
	s, err := o.generate(ctx, job, p, st)
	if err != nil {
		return nil, err
	}
	defer s.Close()
	for {
		ev, err := s.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				err = &engine.Error{Role: "prefill", WorkerID: p.Spec.ID, Code: mwip.ErrInternal, Message: "prefill stream ended without finish", Retriable: true, Transport: true}
			}
			return nil, err
		}
		if ev.Type == mwip.EventPrefillDone {
			return ev, nil
		}
	}
}

// runPush: D registers and acknowledges with its KV receiver, then P is
// sent and pushes KV to D (Mooncake-style order). P's request goes out only
// after the acknowledgement; D's first token decides success.
func (o *Orchestrator) runPush(ctx context.Context, job *Job, plan *Plan, attemptID string, emit Emit, out *Outcome) attemptResult {
	d, p := plan.D, plan.P
	t0 := time.Now()
	dReq := o.stage(job, attemptID, mwip.Disagg{Mode: mwip.ModeSplit, Role: "decode", Handshake: mwip.HandshakePush, Prefill: peer(p), Decode: peer(d)})
	ds, err := o.generate(ctx, job, d, dReq)
	if err != nil {
		return attemptResult{err: err, failed: stageD}
	}
	defer ds.Close()
	dCh := readLoop(ctx, ds)
	var ack *mwip.GenerateEvent
	deadline := time.NewTimer(time.Until(job.FirstTokenDeadline))
	defer deadline.Stop()
	select {
	case r, ok := <-dCh:
		if !ok || r.err != nil {
			if r.err == nil || errors.Is(r.err, io.EOF) {
				r.err = &engine.Error{Role: "decode", WorkerID: d.Spec.ID, Code: mwip.ErrInternal, Message: "register stream closed", Retriable: true, Transport: true}
			}
			return attemptResult{err: r.err, failed: stageD}
		}
		if r.ev.Type != mwip.EventAccepted || r.ev.Receiver == nil {
			return attemptResult{err: &engine.Error{Role: "decode", WorkerID: d.Spec.ID, Code: mwip.ErrInternal, Message: "expected accepted with kv_receiver, got " + r.ev.Type, Retriable: true}, failed: stageD}
		}
		ack = r.ev
	case <-deadline.C:
		return attemptResult{err: ErrFirstTokenTimeout, failed: stageD}
	case <-ctx.Done():
		return attemptResult{err: ctx.Err(), failed: stageD}
	}
	out.RegisterDur = time.Since(t0)
	o.met.StageDuration.WithLabelValues(job.Model.Name, "decode_register").Observe(out.RegisterDur.Seconds())

	pReq := o.stage(job, attemptID, mwip.Disagg{Mode: mwip.ModeSplit, Role: "prefill", Handshake: mwip.HandshakePush, Prefill: peer(p), Decode: peer(d), Receiver: ack.Receiver})
	pDone := o.startPrefill(ctx, job, p, pReq)
	return o.pump(ctx, job, dCh, pDone, emit, out, plan)
}

// runBootstrap: P and D are sent concurrently with a shared room; D fetches
// KV from P's bootstrap server (disagg.prefill.bootstrap_addr).
func (o *Orchestrator) runBootstrap(ctx context.Context, job *Job, plan *Plan, attemptID string, emit Emit, out *Outcome) attemptResult {
	d, p := plan.D, plan.P
	pReq := o.stage(job, attemptID, mwip.Disagg{Mode: mwip.ModeSplit, Role: "prefill", Handshake: mwip.HandshakeBootstrap, Prefill: peer(p), Decode: peer(d)})
	pDone := o.startPrefill(ctx, job, p, pReq)
	dReq := o.stage(job, attemptID, mwip.Disagg{Mode: mwip.ModeSplit, Role: "decode", Handshake: mwip.HandshakeBootstrap, Prefill: peer(p), Decode: peer(d)})
	ds, err := o.generate(ctx, job, d, dReq)
	if err != nil {
		return attemptResult{err: err, failed: stageD}
	}
	defer ds.Close()
	return o.pump(ctx, job, readLoop(ctx, ds), pDone, emit, out, plan)
}

// runPull: P runs first and returns a KV handle on its stream (kv_handle),
// then D pulls the KV with it.
func (o *Orchestrator) runPull(ctx context.Context, job *Job, plan *Plan, attemptID string, emit Emit, out *Outcome) attemptResult {
	d, p := plan.D, plan.P
	t0 := time.Now()
	pReq := o.stage(job, attemptID, mwip.Disagg{Mode: mwip.ModeSplit, Role: "prefill", Handshake: mwip.HandshakePull, Prefill: peer(p), Decode: peer(d)})
	pctx, cancel := context.WithDeadline(ctx, job.FirstTokenDeadline)
	done, err := o.runPrefill(pctx, job, p, pReq)
	cancel()
	if err != nil {
		if pctx.Err() != nil && ctx.Err() == nil {
			err = ErrFirstTokenTimeout
		}
		return attemptResult{err: err, failed: stageP}
	}
	out.PrefillDur = time.Since(t0)
	o.met.StageDuration.WithLabelValues(job.Model.Name, "prefill").Observe(out.PrefillDur.Seconds())
	p.Reservation.PrefillDone()
	if done.Handle == nil {
		return attemptResult{err: &engine.Error{Role: "prefill", WorkerID: p.Spec.ID, Code: mwip.ErrInternal, Message: "prefill stream ended without a KV handle", Retriable: true}, failed: stageP}
	}
	dReq := o.stage(job, attemptID, mwip.Disagg{Mode: mwip.ModeSplit, Role: "decode", Handshake: mwip.HandshakePull, Prefill: peer(p), Decode: peer(d),
		Handle: done.Handle})
	ds, err := o.generate(ctx, job, d, dReq)
	if err != nil {
		return attemptResult{err: err, failed: stageD}
	}
	defer ds.Close()
	return o.pump(ctx, job, readLoop(ctx, ds), nil, emit, out, plan)
}

type prefillResult struct {
	err error
	dur time.Duration
}

// startPrefill runs the prefill stage in the background; the returned
// channel delivers its result once.
func (o *Orchestrator) startPrefill(ctx context.Context, job *Job, p *scheduler.Decision, req *engine.Stage) <-chan prefillResult {
	ch := make(chan prefillResult, 1)
	t0 := time.Now()
	go func() {
		pctx, cancel := context.WithDeadline(ctx, job.FirstTokenDeadline)
		defer cancel()
		_, err := o.runPrefill(pctx, job, p, req)
		d := time.Since(t0)
		if err == nil {
			o.met.StageDuration.WithLabelValues(job.Model.Name, "prefill").Observe(d.Seconds())
		} else if pctx.Err() != nil && ctx.Err() == nil {
			err = ErrFirstTokenTimeout
		}
		ch <- prefillResult{err: err, dur: d}
	}()
	return ch
}
