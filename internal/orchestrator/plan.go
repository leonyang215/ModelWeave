// Package orchestrator plans and executes requests across encode, prefill
// and decode workers.
//
// Execution modes:
//
//	normal  one worker runs prefill and decode (no PD split): a unified
//	        worker, or a decode worker that also takes normal requests,
//	        chosen when it already caches most of the prompt or the uncached
//	        part is short; falls back to pd on failure
//	pd      prefill on P, KV handed to D, D decodes (handshake push,
//	        bootstrap or pull)
//	epd_*   either of the above after an encode stage for media
//
// Workers are called in the model's engine protocol (mwip, vllm or sglang).
package orchestrator

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/obs"
	"github.com/leonyang215/ModelWeave/internal/qos"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

// Mode names.
const (
	ModeNormal = "normal"
	ModePD     = "pd"
)

// Model is the runtime state of one served model.
type Model struct {
	Cfg       *config.Model
	Name      string
	Protocol  string // engine protocol of the model's workers
	BlockSize int
	Seed      uint64
	Prefill   *scheduler.Scheduler
	Decode    *scheduler.Scheduler
	Encode    *scheduler.Scheduler
	Unified   *scheduler.Scheduler
	Artifacts *ArtifactCache
	Sessions  *SessionMap
}

// Job is one rendered client request.
type Job struct {
	ID       string
	Model    *Model
	Tenant   string
	Priority qos.Priority
	Render   *mwip.RenderResult
	// Media is the request's copy of the multimodal items; the encode stage
	// attaches embedding artifacts (mwip) to it.
	Media *mwip.Multimodal
	// ECParams are the vLLM encoders' ec_transfer_params (EPD).
	ECParams   map[string]any
	Blocks     []uint64 // prefix block hashes of the prompt
	SessionKey string
	Arrived    time.Time
	// FirstTokenDeadline is the single absolute deadline shared by every
	// stage and retry until the first output token.
	FirstTokenDeadline time.Time
	// OnFree is called when a reservation is released (wakes the queue).
	OnFree func()
	// OnAttempt is called (on the executing goroutine, before any Emit of
	// that attempt) when an attempt starts.
	OnAttempt func(AttemptInfo)
}

// AttemptInfo describes the workers of one attempt.
type AttemptInfo struct {
	Attempt      int
	Mode         string
	Reason       string
	Decode       string
	Prefill      string
	CachedTokens int
}

// Plan is a scheduled execution with reserved workers.
type Plan struct {
	Mode   string
	Reason string
	D      *scheduler.Decision // decode / unified worker
	P      *scheduler.Decision // prefill worker (pd only)
}

// Release frees the plan's reservations.
func (p *Plan) Release(success bool) {
	if p == nil {
		return
	}
	if p.D != nil {
		p.D.Reservation.Release(success)
	}
	if p.P != nil {
		p.P.Reservation.Release(success)
	}
}

// PlanOpts constrain planning (used by retries and fallbacks).
type PlanOpts struct {
	ForceMode string
	ExcludeD  map[string]bool
	ExcludeP  map[string]bool
}

// Orchestrator plans and runs jobs.
type Orchestrator struct {
	reg     *registry.Registry
	eng     *engine.Client
	met     *obs.Metrics
	log     *slog.Logger
	health  config.Health
	timeout config.Server
}

// New returns an orchestrator.
func New(reg *registry.Registry, eng *engine.Client, met *obs.Metrics, log *slog.Logger, health config.Health, srv config.Server) *Orchestrator {
	return &Orchestrator{reg: reg, eng: eng, met: met, log: log, health: health, timeout: srv}
}

// ErrNoRoute means the model has no workers able to run the request.
var ErrNoRoute = errors.New("no worker pool can serve this request")

func roundUp(n, block int) int64 {
	if block <= 1 {
		return int64(n)
	}
	return int64((n + block - 1) / block * block)
}

// MaxTokens is the request's output token limit.
func (j *Job) MaxTokens() int { return j.Render.InferenceOptions.MaxTokens }

// TokenIDs are the prompt as the engine lays it out (media expanded).
func (j *Job) TokenIDs() []uint32 { return j.Render.TokenIDs }

// kvReserve is the decode-side KV reservation: prompt + capped output,
// rounded to blocks. Reserving the whole max_tokens would strand capacity
// most requests never use; engines grow the KV of a running request and
// report their real usage.
func (j *Job) kvReserve() int64 {
	out := min(j.MaxTokens(), j.Model.Cfg.Reserve.OutputTokens)
	return roundUp(j.Render.PromptTokens+out, j.Model.Cfg.Reserve.BlockSize)
}

func (j *Job) prefillCost() func(int) int64 {
	prompt, beta := j.Render.PromptTokens, j.Model.Cfg.Reserve.UncachedBeta
	return func(cached int) int64 { return scheduler.UncachedCost(prompt, cached, beta) }
}

func (j *Job) request(role registry.Role) *scheduler.Request {
	return &scheduler.Request{
		Model:        j.Model.Name,
		Role:         role,
		PromptTokens: j.Render.PromptTokens,
		Blocks:       j.Blocks,
		BlockSize:    j.Model.BlockSize,
		SessionKey:   j.SessionKey,
		OnFree:       j.OnFree,
	}
}

func (o *Orchestrator) has(model string, role registry.Role, normal bool) bool {
	for _, w := range o.reg.Candidates(model, role) {
		if !normal || w.Spec().ServesNormal() {
			return true
		}
	}
	return false
}

// Plan schedules a job and reserves its workers. It returns
// scheduler.ErrNoCapacity when the job should wait in the admission queue.
func (o *Orchestrator) Plan(ctx context.Context, job *Job, opts PlanOpts) (*Plan, error) {
	m := job.Model
	mode := m.Cfg.Mode
	if opts.ForceMode != "" {
		mode = opts.ForceMode
	}
	hasDecode := o.has(m.Name, registry.RoleDecode, false)
	hasPrefill := o.has(m.Name, registry.RolePrefill, false)
	hasUnified := o.has(m.Name, registry.RoleUnified, false)
	switch mode {
	case "auto":
		switch {
		case hasDecode:
			return o.planAuto(ctx, job, opts, hasPrefill)
		case hasUnified:
			return o.planUnified(ctx, job, opts)
		}
		return nil, ErrNoRoute
	case ModeNormal:
		if hasUnified {
			return o.planUnified(ctx, job, opts)
		}
		d, err := o.scheduleNormalD(ctx, job, opts, "")
		if err != nil {
			return nil, err
		}
		return &Plan{Mode: ModeNormal, Reason: "forced", D: d}, nil
	case ModePD:
		return o.planPD(ctx, job, opts, nil, "forced")
	}
	return nil, ErrNoRoute
}

func (o *Orchestrator) planUnified(ctx context.Context, job *Job, opts PlanOpts) (*Plan, error) {
	req := job.request(registry.RoleUnified)
	req.KVReserve = job.kvReserve()
	req.PrefillCost = job.prefillCost()
	req.Exclude = opts.ExcludeD
	d, err := job.Model.Unified.Schedule(ctx, req)
	if err != nil {
		return nil, err
	}
	return &Plan{Mode: ModeNormal, Reason: "unified", D: d}, nil
}

func (o *Orchestrator) scheduleNormalD(ctx context.Context, job *Job, opts PlanOpts, pin string) (*scheduler.Decision, error) {
	req := job.request(registry.RoleDecode)
	req.RequireNormal = true
	req.KVReserve = job.kvReserve()
	req.PrefillCost = job.prefillCost()
	req.Exclude = opts.ExcludeD
	req.Pin = pin
	return job.Model.Decode.Schedule(ctx, req)
}

// planAuto decides between normal and pd per request:
//  1. a session pinned to a normal-capable decode worker runs normal there;
//  2. otherwise the decode worker is picked cache-aware among normal-capable
//     workers; if it would compute at most max_uncached_tokens, or caches at
//     least min_cache_ratio of the prompt, the request runs normal;
//  3. otherwise it runs pd with that decode worker and a cache-aware
//     prefill worker.
func (o *Orchestrator) planAuto(ctx context.Context, job *Job, opts PlanOpts, hasPrefill bool) (*Plan, error) {
	m := job.Model
	nc := m.Cfg.Normal
	normalOK := nc.Enabled && o.has(m.Name, registry.RoleDecode, true)
	if !hasPrefill && !normalOK {
		return nil, ErrNoRoute
	}
	if normalOK && nc.SessionAffinity {
		if wid, ok := m.Sessions.Get(job.SessionKey); ok && !opts.ExcludeD[wid] {
			if d, err := o.scheduleNormalD(ctx, job, opts, wid); err == nil {
				return &Plan{Mode: ModeNormal, Reason: "session", D: d}, nil
			}
		}
	}
	if !normalOK {
		return o.planPD(ctx, job, opts, nil, "no_normal_worker")
	}
	d, err := o.scheduleNormalD(ctx, job, opts, "")
	if err != nil {
		if errors.Is(err, scheduler.ErrNoWorkers) && hasPrefill {
			// Normal-capable workers are all down; plain decode workers
			// may still serve pd.
			return o.planPD(ctx, job, opts, nil, "normal_unavailable")
		}
		return nil, err
	}
	uncached := job.Render.PromptTokens - d.CachedTokens
	switch {
	case uncached <= nc.MaxUncachedTokens:
		if d.CachedTokens > 0 {
			return &Plan{Mode: ModeNormal, Reason: "prefix_hit", D: d}, nil
		}
		return &Plan{Mode: ModeNormal, Reason: "short_prompt", D: d}, nil
	case d.CacheRatio >= nc.MinCacheRatio:
		return &Plan{Mode: ModeNormal, Reason: "prefix_hit", D: d}, nil
	case !hasPrefill:
		return &Plan{Mode: ModeNormal, Reason: "no_prefill_worker", D: d}, nil
	}
	// Long uncached prompt: keep this decode worker, prefill elsewhere.
	d.Reservation.PrefillDone()
	reason := "long_prompt"
	if d.CachedTokens > 0 {
		reason = "partial_hit"
	}
	return o.planPD(ctx, job, opts, d, reason)
}

// planPD reserves a decode worker (unless given) and a prefill worker.
func (o *Orchestrator) planPD(ctx context.Context, job *Job, opts PlanOpts, d *scheduler.Decision, reason string) (*Plan, error) {
	m := job.Model
	if d == nil {
		req := job.request(registry.RoleDecode)
		req.KVReserve = job.kvReserve()
		req.Exclude = opts.ExcludeD
		var err error
		if d, err = m.Decode.Schedule(ctx, req); err != nil {
			return nil, err
		}
	}
	preq := job.request(registry.RolePrefill)
	preq.PrefillCost = job.prefillCost()
	preq.Exclude = opts.ExcludeP
	p, err := m.Prefill.Schedule(ctx, preq)
	if err != nil {
		d.Reservation.Release(false)
		return nil, err
	}
	return &Plan{Mode: ModePD, Reason: reason, D: d, P: p}, nil
}
