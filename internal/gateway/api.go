package gateway

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/format"
	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/orchestrator"
	"github.com/leonyang215/ModelWeave/internal/qos"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
	"github.com/leonyang215/ModelWeave/internal/tokenizer"
)

// peekBody holds the request fields the gateway itself needs.
type peekBody struct {
	Model          string `json:"model"`
	Stream         bool   `json:"stream"`
	User           string `json:"user"`
	PromptCacheKey string `json:"prompt_cache_key"`
	Metadata       struct {
		UserID string `json:"user_id"`
	} `json:"metadata"`
}

// record accumulates everything the access log and metrics report.
type record struct {
	ID, API, Model, Tenant, Priority string
	Stream                           bool
	Arrived                          time.Time
	Status                           int
	ErrKind, ErrMsg                  string
	Mode, Reason, Fallback           string
	Decode, Prefill                  string
	Attempts                         int
	PromptTokens, CompletionTokens   int
	ReasoningTokens                  int
	PredictedCached, EngineCached    int
	FinishReason                     string
	Admission                        string
	Queued                           bool
	TokenizeDur, QueueDur, EncodeDur time.Duration
	RegisterDur, PrefillDur          time.Duration
	DispatchedAt, FirstTokenAt       time.Time
	LastTokenAt, FirstWriteAt        time.Time
	Encoded                          int
	ArtifactHits                     int
	Images, Videos                   int
}

func newRequestID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "mw-" + hex.EncodeToString(b[:])
}

func (s *Server) handleAPI(protocol string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) { s.serveAPI(w, r, protocol) }
}

func (s *Server) authenticate(r *http.Request) (string, *apiError) {
	key := r.Header.Get("x-api-key")
	if key == "" {
		if a := r.Header.Get("Authorization"); strings.HasPrefix(a, "Bearer ") {
			key = strings.TrimSpace(strings.TrimPrefix(a, "Bearer "))
		}
	}
	if key != "" {
		if t, ok := s.keys[key]; ok {
			return t, nil
		}
		if s.cfg.Auth.Enabled {
			return "", &apiError{Status: http.StatusUnauthorized, Kind: "invalid_api_key", Message: "invalid API key"}
		}
	} else if s.cfg.Auth.Enabled {
		return "", &apiError{Status: http.StatusUnauthorized, Kind: "invalid_api_key", Message: "missing API key"}
	}
	return s.cfg.Auth.AnonymousTenant, nil
}

func sessionKey(r *http.Request, p *peekBody) string {
	for _, v := range []string{r.Header.Get("X-Session-Id"), p.PromptCacheKey, p.User, p.Metadata.UserID} {
		if v != "" {
			return v
		}
	}
	return ""
}

func (s *Server) serveAPI(w http.ResponseWriter, r *http.Request, protocol string) {
	rec := &record{ID: r.Header.Get("X-Request-Id"), API: protocol, Arrived: time.Now(), Status: http.StatusOK}
	if rec.ID == "" || len(rec.ID) > 128 {
		rec.ID = newRequestID()
	}
	w.Header().Set("X-Request-Id", rec.ID)
	defer s.finish(rec)
	fail := func(e *apiError) {
		rec.Status, rec.ErrKind, rec.ErrMsg = e.Status, e.Kind, e.Message
		writeError(w, protocol, e)
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, s.cfg.Server.MaxBodyBytes))
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			fail(&apiError{Status: http.StatusRequestEntityTooLarge, Kind: "request_too_large", Message: err.Error()})
			return
		}
		fail(badRequest("read body: " + err.Error()))
		return
	}
	var peek peekBody
	if err := json.Unmarshal(body, &peek); err != nil {
		fail(badRequest("invalid JSON body: " + err.Error()))
		return
	}
	rec.Stream = peek.Stream
	tenant, aerr := s.authenticate(r)
	if aerr != nil {
		fail(aerr)
		return
	}
	rec.Tenant = tenant
	policy := s.limiter.Policy(tenant)
	prio := policy.Priority
	if hp, err := qos.ParsePriority(r.Header.Get("X-Priority")); err == nil && r.Header.Get("X-Priority") != "" && hp > prio {
		prio = hp // callers may lower, never raise, their priority
	}
	rec.Priority = prio.String()
	m := s.byName[peek.Model]
	if m == nil {
		fail(&apiError{Status: http.StatusNotFound, Kind: "model_not_found", Message: fmt.Sprintf("model %q is not served", peek.Model)})
		return
	}
	rec.Model = m.cfg.Name
	s.met.Inflight.WithLabelValues(m.cfg.Name).Inc()
	defer s.met.Inflight.WithLabelValues(m.cfg.Name).Dec()

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Server.RequestTimeout.D())
	defer cancel()

	// 1. Render: the tokenizer converts the protocol request into the
	// intermediate state (token list, media placements, options).
	t0 := time.Now()
	rr, err := s.tok.Render(ctx, &mwip.RenderRequest{
		RequestID:     rec.ID,
		Protocol:      protocol,
		Model:         m.cfg.TokenizerModel,
		Body:          body,
		ContextLength: m.cfg.ContextLength,
	})
	rec.TokenizeDur = time.Since(t0)
	s.met.StageDuration.WithLabelValues(m.cfg.Name, "tokenize").Observe(rec.TokenizeDur.Seconds())
	if err != nil {
		var te *tokenizer.Error
		outcome := "error"
		if errors.As(err, &te) && te.Status > 0 && te.Status < 500 {
			outcome = "rejected"
		}
		s.met.TokenizerDuration.WithLabelValues(outcome).Observe(rec.TokenizeDur.Seconds())
		fail(classify(err))
		return
	}
	s.met.TokenizerDuration.WithLabelValues("ok").Observe(rec.TokenizeDur.Seconds())
	rec.PromptTokens, rec.Stream = rr.PromptTokens, rr.Stream
	if mm := rr.Multimodal; mm != nil {
		for _, it := range mm.Items {
			if it.Modality == "video" {
				rec.Videos++
			} else {
				rec.Images++
			}
			s.met.MediaItems.WithLabelValues(m.cfg.Name, it.Modality).Inc()
			s.met.MediaTokens.WithLabelValues(m.cfg.Name, it.Modality).Add(float64(it.Span.Length))
		}
	}
	// 2. Build the job and its prefix-cache block hashes. Media positions
	// hold their pad values, so the token list hashes as is.
	job := &orchestrator.Job{
		ID: rec.ID, Model: m.rt, Tenant: tenant, Priority: prio, Render: rr, Media: rr.Multimodal.Clone(),
		SessionKey: sessionKey(r, &peek), Arrived: rec.Arrived,
		FirstTokenDeadline: rec.Arrived.Add(s.cfg.Server.FirstTokenTimeout.D()),
		OnFree:             m.queue.Notify,
	}
	job.Blocks = kvindex.BlockHashes(rr.TokenIDs, m.rt.BlockSize, m.rt.Seed)

	// 3. Tenant rate limits (TPM charged up front, reconciled at the end).
	estimate := int64(rr.PromptTokens + min(job.MaxTokens(), m.cfg.Reserve.OutputTokens))
	lease, err := s.limiter.Admit(tenant, estimate)
	if err != nil {
		var rj *qos.Rejection
		if errors.As(err, &rj) {
			s.met.RateLimited.WithLabelValues(tenant, rj.Reason).Inc()
		}
		rec.Admission = "rejected_rate_limit"
		fail(classify(err))
		return
	}
	defer func() { lease.Finish(int64(rec.PromptTokens + rec.CompletionTokens)) }()

	// 4. Overload protection by load level and priority.
	if level, _ := m.overload.Level(); level > 0 && m.cfg.Admission.Overload.Enabled {
		ok, why := m.overload.Decide(prio, s.predictCacheRatio(ctx, job))
		if !ok {
			rec.Admission = "rejected_overload"
			fail(&apiError{Status: http.StatusServiceUnavailable, Kind: "overloaded", Message: fmt.Sprintf("model %s is overloaded (level L%d, %s)", m.cfg.Name, level, why), RetryAfter: 2 * time.Second})
			return
		}
	}

	// 5-7. Encode (EPD), plan with queueing, execute. A missing embedding
	// artifact reported by P/D before any output triggers one rebuild.
	for round := 0; ; round++ {
		if need, err := s.orch.NeedsEncode(job); err != nil {
			fail(classify(err))
			return
		} else if need {
			st, err := s.orch.Encode(ctx, job)
			rec.EncodeDur += st.Duration
			rec.Encoded += st.Built
			rec.ArtifactHits += st.Hits
			if err != nil {
				fail(classify(err))
				return
			}
		}
		plan, err := s.planWithQueue(ctx, job, m, rec)
		if err != nil {
			if ctx.Err() != nil {
				rec.Status, rec.ErrKind = 499, "client_closed"
				return
			}
			fail(classify(err))
			return
		}
		if rec.Admission == "" {
			rec.Admission = "admitted"
		}
		res := s.execute(ctx, w, protocol, job, plan, body, peek.Model, rec)
		if res.err == nil || res.emitted {
			return
		}
		if engine.IsCode(res.err, mwip.ErrArtifact) && round == 0 && rec.Encoded+rec.ArtifactHits > 0 {
			s.log.Warn("embedding artifact unavailable, rebuilding", "request_id", rec.ID)
			s.orch.InvalidateArtifacts(job)
			continue
		}
		if ctx.Err() != nil {
			rec.Status, rec.ErrKind = 499, "client_closed"
			return
		}
		fail(classify(res.err))
		return
	}
}

// predictCacheRatio returns the best predicted prefix-cache ratio among the
// model's decode/unified workers (used to exempt cheap requests from drops).
func (s *Server) predictCacheRatio(ctx context.Context, job *orchestrator.Job) float64 {
	if len(job.Blocks) == 0 || job.Render.PromptTokens == 0 {
		return 0
	}
	var ids []string
	for _, role := range []registry.Role{registry.RoleDecode, registry.RoleUnified, registry.RolePrefill} {
		for _, w := range s.reg.Candidates(job.Model.Name, role) {
			ids = append(ids, w.ID())
		}
	}
	matches, err := s.index.Match(ctx, kvindex.Query{
		Model: job.Model.Name, Role: string(registry.RoleDecode), BlockSize: job.Model.BlockSize, Blocks: job.Blocks,
		Candidates: ids,
	})
	if err != nil {
		return 0
	}
	best := 0
	for _, n := range matches {
		best = max(best, n)
	}
	return min(1, float64(best*job.Model.BlockSize)/float64(job.Render.PromptTokens))
}

// planWithQueue plans the job, waiting in the model's admission queue while
// no worker has capacity.
func (s *Server) planWithQueue(ctx context.Context, job *orchestrator.Job, m *modelRT, rec *record) (*orchestrator.Plan, error) {
	t0 := time.Now()
	plan, err := s.orch.Plan(ctx, job, orchestrator.PlanOpts{})
	if err == nil || !errors.Is(err, scheduler.ErrNoCapacity) {
		return plan, err
	}
	rec.Queued = true
	var planErr error
	weight := 1.0
	if t, ok := s.tenants[job.Tenant]; ok {
		weight = t.Weight
	}
	werr := m.queue.Wait(ctx, qos.Ticket{
		Tenant: job.Tenant, Weight: weight, Priority: job.Priority,
		Cost: float64(job.Render.PromptTokens + min(job.MaxTokens(), m.cfg.Reserve.OutputTokens)),
	}, func() bool {
		p, e := s.orch.Plan(ctx, job, orchestrator.PlanOpts{})
		if e == nil {
			plan = p
			return true
		}
		if errors.Is(e, scheduler.ErrNoCapacity) {
			return false
		}
		planErr = e
		return true
	})
	rec.QueueDur = time.Since(t0)
	s.met.StageDuration.WithLabelValues(m.cfg.Name, "admission_queue").Observe(rec.QueueDur.Seconds())
	switch {
	case errors.Is(werr, qos.ErrQueueFull):
		rec.Admission = "rejected_queue_full"
	case errors.Is(werr, qos.ErrShed):
		rec.Admission = "shed"
	case errors.Is(werr, qos.ErrTimeout):
		rec.Admission = "rejected_timeout"
	case werr == nil && planErr == nil:
		rec.Admission = "queued"
	}
	if werr != nil {
		return nil, werr
	}
	return plan, planErr
}

type execResult struct {
	err     error
	emitted bool // bytes reached the client
}

// execute runs the plan and writes the client response. Engine output
// arrives already parsed (content, reasoning, tool-call deltas); the
// formatter renders it as the client's protocol, streamed or aggregated.
// Engines are always called streaming, whatever the client asked for.
func (s *Server) execute(ctx context.Context, w http.ResponseWriter, protocol string, job *orchestrator.Job, plan *orchestrator.Plan, body []byte, model string, rec *record) execResult {
	rr := job.Render
	fm := format.New(protocol, format.Meta{
		ID: format.ResponseID(protocol, rec.ID), Model: model, Created: rec.Arrived.Unix(),
		Stream: rr.Stream, IncludeUsage: rr.ResponseOptions.IncludeUsage, Logprobs: rr.InferenceOptions.Logprobs,
		Request: body, PromptTokens: rr.PromptTokens,
	})
	var route *orchestrator.AttemptInfo
	job.OnAttempt = func(a orchestrator.AttemptInfo) { route = &a }
	flusher, _ := w.(http.Flusher)
	wrote := false
	writeHeader := func() {
		wrote = true
		h := w.Header()
		h.Set("Content-Type", fm.ContentType())
		if rr.Stream {
			h.Set("Cache-Control", "no-cache")
			h.Set("X-Accel-Buffering", "no")
		}
		if s.cfg.Server.ExposeRoutingHeader && route != nil {
			h.Set("X-ModelWeave-Mode", route.Mode)
			h.Set("X-ModelWeave-Reason", route.Reason)
			h.Set("X-ModelWeave-Decode-Worker", route.Decode)
			if route.Prefill != "" {
				h.Set("X-ModelWeave-Prefill-Worker", route.Prefill)
			}
			h.Set("X-ModelWeave-Cached-Tokens", strconv.Itoa(route.CachedTokens))
			h.Set("X-ModelWeave-Attempt", strconv.Itoa(route.Attempt))
		}
		w.WriteHeader(http.StatusOK)
	}
	write := func(b []byte) error {
		if len(b) == 0 {
			return nil
		}
		if !wrote {
			writeHeader()
		}
		if rec.FirstWriteAt.IsZero() {
			rec.FirstWriteAt = time.Now()
		}
		if _, err := w.Write(b); err != nil {
			return fmt.Errorf("%w: %v", orchestrator.ErrClientGone, err)
		}
		if flusher != nil && rr.Stream {
			flusher.Flush()
		}
		return nil
	}
	finished := false
	out := s.orch.Execute(ctx, job, plan, func(ev *mwip.GenerateEvent) error {
		switch ev.Type {
		case mwip.EventReady:
			return write(fm.Start())
		case mwip.EventDelta:
			return write(fm.Delta(ev))
		case mwip.EventFinish:
			finished = true
			return write(fm.Finish(ev))
		}
		return nil
	})
	rec.Mode, rec.Reason, rec.Fallback, rec.Attempts = out.Mode, out.Reason, out.Fallback, out.Attempts
	rec.Decode, rec.Prefill, rec.PredictedCached = out.DWorker, out.PWorker, out.CachedTokens
	rec.DispatchedAt, rec.FirstTokenAt, rec.LastTokenAt = out.DispatchedAt, out.FirstTokenAt, out.LastTokenAt
	rec.RegisterDur, rec.PrefillDur = out.RegisterDur, out.PrefillDur
	if finished {
		sum := fm.Summary()
		rec.FinishReason, rec.PromptTokens, rec.CompletionTokens = sum.FinishReason, sum.PromptTokens, sum.CompletionTokens
		rec.ReasoningTokens, rec.EngineCached = sum.ReasoningTokens, sum.CachedTokens
	}
	switch {
	case out.Err == nil:
		if !wrote {
			writeHeader()
		}
		return execResult{emitted: true}
	case ctx.Err() != nil, errors.Is(out.Err, orchestrator.ErrClientGone):
		rec.Status, rec.ErrKind = 499, "client_closed"
		return execResult{err: out.Err, emitted: wrote}
	case !wrote:
		// Nothing reached the client (non-streaming responses are written
		// at the end): answer with a regular error.
		return execResult{err: out.Err}
	}
	e := classify(out.Err)
	rec.Status, rec.ErrKind, rec.ErrMsg = http.StatusOK, e.Kind, e.Message
	_ = write(fm.Fail(e.Kind, e.Message))
	return execResult{err: e, emitted: true}
}

// finish records metrics and the access log of a request.
func (s *Server) finish(rec *record) {
	now := time.Now()
	model := rec.Model
	if model == "" {
		model = "unknown"
	}
	mode := rec.Mode
	if mode == "" {
		mode = "none"
	} else if rec.Encoded+rec.ArtifactHits > 0 {
		mode = "epd_" + mode
	}
	code := strconv.Itoa(rec.Status)
	s.met.Requests.WithLabelValues(model, rec.API, strconv.FormatBool(rec.Stream), mode, code).Inc()
	if rec.FinishReason != "" && rec.Status == http.StatusOK && rec.ErrKind == "" {
		s.met.Finishes.WithLabelValues(model, rec.API, rec.FinishReason).Inc()
	}
	e2e := now.Sub(rec.Arrived)
	s.met.RequestDuration.WithLabelValues(model, rec.API, mode).Observe(e2e.Seconds())
	if rec.Admission != "" {
		s.met.Admission.WithLabelValues(model, rec.Tenant, rec.Priority, rec.Admission).Inc()
	}
	if rec.Mode != "" {
		s.met.ExecutionMode.WithLabelValues(model, mode, rec.Reason).Inc()
	}
	var ttft time.Duration
	if !rec.FirstTokenAt.IsZero() {
		first := rec.FirstTokenAt
		if !rec.FirstWriteAt.IsZero() && rec.Stream {
			first = rec.FirstWriteAt
		}
		ttft = first.Sub(rec.Arrived)
		s.met.TTFT.WithLabelValues(model, mode).Observe(ttft.Seconds())
		gw := rec.DispatchedAt.Sub(rec.Arrived) - rec.EncodeDur
		s.met.TTFTSegment.WithLabelValues(model, mode, "gateway").Observe(max(gw, 0).Seconds())
		if rec.EncodeDur > 0 {
			s.met.TTFTSegment.WithLabelValues(model, mode, "encode").Observe(rec.EncodeDur.Seconds())
		}
		s.met.TTFTSegment.WithLabelValues(model, mode, "engine").Observe(rec.FirstTokenAt.Sub(rec.DispatchedAt).Seconds())
		s.met.StageDuration.WithLabelValues(model, "first_token").Observe(rec.FirstTokenAt.Sub(rec.DispatchedAt).Seconds())
		if !rec.LastTokenAt.IsZero() && rec.CompletionTokens > 1 {
			decode := rec.LastTokenAt.Sub(rec.FirstTokenAt)
			s.met.StageDuration.WithLabelValues(model, "decode").Observe(decode.Seconds())
			s.met.TPOT.WithLabelValues(model, mode).Observe(decode.Seconds() / float64(rec.CompletionTokens-1))
		}
	}
	if rec.Status == http.StatusOK && rec.Mode != "" {
		s.met.Tokens.WithLabelValues(model, rec.Tenant, "prompt").Add(float64(rec.PromptTokens))
		s.met.Tokens.WithLabelValues(model, rec.Tenant, "completion").Add(float64(rec.CompletionTokens))
		if rec.EngineCached > 0 {
			s.met.Tokens.WithLabelValues(model, rec.Tenant, "cached").Add(float64(rec.EngineCached))
		}
		if rec.ReasoningTokens > 0 {
			s.met.Tokens.WithLabelValues(model, rec.Tenant, "reasoning").Add(float64(rec.ReasoningTokens))
		}
	}
	level := slog.LevelInfo
	if rec.Status >= 500 || (rec.Status == http.StatusOK && rec.ErrKind != "") {
		level = slog.LevelWarn
	}
	attrs := []slog.Attr{
		slog.String("request_id", rec.ID), slog.String("api", rec.API), slog.String("model", model),
		slog.String("tenant", rec.Tenant), slog.String("priority", rec.Priority), slog.Bool("stream", rec.Stream),
		slog.Int("status", rec.Status), slog.String("mode", mode), slog.String("reason", rec.Reason),
		slog.String("decode_worker", rec.Decode), slog.String("prefill_worker", rec.Prefill),
		slog.Int("attempts", rec.Attempts), slog.String("fallback", rec.Fallback),
		slog.Int("prompt_tokens", rec.PromptTokens), slog.Int("completion_tokens", rec.CompletionTokens),
		slog.Int("reasoning_tokens", rec.ReasoningTokens), slog.Int("predicted_cached_tokens", rec.PredictedCached),
		slog.Int("engine_cached_tokens", rec.EngineCached), slog.String("finish_reason", rec.FinishReason),
		slog.String("admission", rec.Admission), slog.Bool("queued", rec.Queued),
		slog.Float64("tokenize_ms", ms(rec.TokenizeDur)), slog.Float64("queue_ms", ms(rec.QueueDur)),
		slog.Int("images", rec.Images), slog.Int("videos", rec.Videos),
		slog.Float64("encode_ms", ms(rec.EncodeDur)), slog.Int("encoded_items", rec.Encoded), slog.Int("artifact_hits", rec.ArtifactHits),
		slog.Float64("register_ms", ms(rec.RegisterDur)), slog.Float64("prefill_ms", ms(rec.PrefillDur)),
		slog.Float64("ttft_ms", ms(ttft)), slog.Float64("e2e_ms", ms(e2e)),
	}
	if rec.ErrKind != "" {
		attrs = append(attrs, slog.String("error_kind", rec.ErrKind), slog.String("error", truncate(rec.ErrMsg, 512)))
	}
	s.log.LogAttrs(context.Background(), level, "request", attrs...)
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
