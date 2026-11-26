package orchestrator_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leonyang215/ModelWeave/internal/config"
	"github.com/leonyang215/ModelWeave/internal/engine"
	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/mockengine"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/obs"
	"github.com/leonyang215/ModelWeave/internal/orchestrator"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/scheduler"
)

type env struct {
	t     *testing.T
	reg   *registry.Registry
	orch  *orchestrator.Orchestrator
	model *orchestrator.Model
	met   *obs.Metrics
	mocks map[string]*mockengine.Engine
	urls  map[string]string
}

func newEnv(t *testing.T, mut func(*config.Model)) *env {
	t.Helper()
	cfg, err := config.Parse([]byte(`
tokenizer: {endpoints: ["http://unused"]}
models:
  - name: m
    normal: {enabled: true, max_uncached_tokens: 100, fallback_to_pd: true}
`))
	if err != nil {
		t.Fatal(err)
	}
	mc := &cfg.Models[0]
	if mut != nil {
		mut(mc)
	}
	reg := registry.New(time.Minute)
	met := obs.NewMetrics()
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	idx := kvindex.NewApprox(0, 0)
	build := func(pc scheduler.PolicyConfig) *scheduler.Scheduler {
		p, err := scheduler.BuildPolicy(pc, 1e9)
		if err != nil {
			t.Fatal(err)
		}
		return scheduler.New(reg, scheduler.Config{Policy: p, Index: idx})
	}
	m := &orchestrator.Model{
		Cfg: mc, Name: "m", BlockSize: 16, Seed: kvindex.ModelSeed("m"),
		Prefill: build(mc.Routing.Prefill), Decode: build(mc.Routing.Decode),
		Encode: build(scheduler.PolicyConfig{Name: "least_load"}), Unified: build(mc.Routing.Unified),
		Artifacts: orchestrator.NewArtifactCache(time.Hour, 100), Sessions: orchestrator.NewSessionMap(time.Hour),
	}
	return &env{
		t: t, reg: reg, model: m, met: met, mocks: map[string]*mockengine.Engine{}, urls: map[string]string{},
		orch: orchestrator.New(reg, engine.New(), met, log, cfg.Health, cfg.Server),
	}
}

// worker starts an in-process mock instance (its HTTP endpoints and its
// mwip Engine service) and registers it.
func (e *env) worker(id, role string, mut func(*mockengine.Config)) {
	e.t.Helper()
	var h atomic.Pointer[http.Handler]
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { (*h.Load()).ServeHTTP(w, r) }))
	e.t.Cleanup(srv.Close)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		e.t.Fatal(err)
	}
	cfg := mockengine.Config{
		ID: id, Model: "m", Role: role, AdvertiseURL: srv.URL, BlockSize: 16, DecodeTPS: 2000,
		PrefillTPS: 1e6, MaxOutputTokens: 8, KVCapacityTokens: 1 << 20, MaxRunning: 64,
		Normal: role == "decode", VerifyArtifacts: true, EncodeLatency: time.Millisecond,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	if mut != nil {
		mut(&cfg)
	}
	eng := mockengine.New(cfg)
	handler := eng.Handler()
	h.Store(&handler)
	gs := mockengine.NewGRPCServer(eng)
	go func() { _ = gs.Serve(lis) }()
	e.t.Cleanup(gs.Stop)
	e.mocks[id], e.urls[id] = eng, srv.URL
	if _, _, err := e.reg.Upsert(registry.Spec{ID: id, Model: "m", Role: registry.Role(role), Target: lis.Addr().String(), Normal: cfg.Normal, BootstrapAddr: srv.URL}, "static"); err != nil {
		e.t.Fatal(err)
	}
}

func f64(v float64) *float64 { return &v }

// job builds a rendered request of prompt text tokens, optionally with one
// image of 50 positions after the first 4 tokens.
func (e *env) job(prompt int, image bool) *orchestrator.Job {
	ids := make([]uint32, prompt)
	for i := range ids {
		ids[i] = uint32(1000 + i)
	}
	rr := &mwip.RenderResult{
		Protocol: mwip.ProtocolChatCompletions, Model: "m",
		Conversation:     mwip.Conversation{ToolChoice: "auto", ResponseFormat: "text"},
		InferenceOptions: mwip.InferenceOptions{MaxTokens: 16, StopTokenIDs: []uint32{1}, Temperature: f64(1)},
		Stream:           true,
	}
	if image {
		const n = 50
		digest := fmt.Sprintf("%064x", time.Now().UnixNano())
		plan := mwip.MediaPlan{Processor: "qwen_vl.v1", Width: 320, Height: 160, Tokens: n, NFrames: 1, PatchSize: 16, SpatialMergeSize: 2, TemporalPatchSize: 2}
		planID := mwip.PlanID("image", digest, plan)
		pad := mwip.PadValue(planID)
		img := make([]uint32, n)
		for i := range img {
			img[i] = pad
		}
		ids = append(append(append([]uint32(nil), ids[:4]...), img...), ids[4:]...)
		rr.Multimodal = &mwip.Multimodal{PadShift: 1 << 30, Items: []mwip.MediaItem{{
			Modality: "image", Digest: digest, Source: mwip.MediaSource{Kind: "url", URL: "https://img/" + digest},
			Plan: plan, PlanID: planID,
			Placements: []mwip.Placement{{Offset: 4, Length: n, PadValue: pad, Reference: planID, Modality: "image"}},
		}}}
	}
	rr.TokenIDs, rr.PromptTokens = ids, len(ids)
	return &orchestrator.Job{
		ID: fmt.Sprintf("job-%d", time.Now().UnixNano()), Model: e.model, Render: rr, Media: rr.Multimodal.Clone(),
		Blocks: kvindex.BlockHashes(ids, 16, e.model.Seed), Arrived: time.Now(),
		FirstTokenDeadline: time.Now().Add(10 * time.Second),
	}
}

type sink struct {
	mu     sync.Mutex
	events []*mwip.GenerateEvent
	stopAt int // return an error after this many delta events (0: never)
	deltas int
}

func (s *sink) emit(ev *mwip.GenerateEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, ev)
	if ev.Type == mwip.EventDelta {
		s.deltas++
		if s.stopAt > 0 && s.deltas >= s.stopAt {
			return errors.New("enough")
		}
	}
	return nil
}

func (s *sink) types() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	for _, ev := range s.events {
		if len(out) == 0 || out[len(out)-1] != ev.Type {
			out = append(out, ev.Type)
		}
	}
	return out
}

func (s *sink) text() string {
	var b strings.Builder
	for _, ev := range s.events {
		b.WriteString(ev.Content)
	}
	return b.String()
}

func (e *env) run(job *orchestrator.Job, s *sink) *orchestrator.Outcome {
	e.t.Helper()
	plan, err := e.orch.Plan(context.Background(), job, orchestrator.PlanOpts{})
	if err != nil {
		e.t.Fatalf("plan: %v", err)
	}
	return e.orch.Execute(context.Background(), job, plan, s.emit)
}

func expectOK(t *testing.T, out *orchestrator.Outcome, s *sink, mode string) {
	t.Helper()
	if out.Err != nil {
		t.Fatalf("execution failed: %v", out.Err)
	}
	if out.Mode != mode {
		t.Fatalf("mode %s, want %s (reason %s)", out.Mode, mode, out.Reason)
	}
	got := s.types()
	if len(got) != 3 || got[0] != mwip.EventReady || got[1] != mwip.EventDelta || got[2] != mwip.EventFinish {
		t.Fatalf("event sequence %v", got)
	}
	if out.Usage == nil || out.Usage.CompletionTokens == 0 {
		t.Fatalf("missing usage: %+v", out.Usage)
	}
}

// stats returns a mock worker's counters, including the last request body.
func (e *env) stats(id string) map[string]any {
	e.t.Helper()
	resp, err := http.Get(e.urls[id] + "/mock/stats")
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	var m map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		e.t.Fatal(err)
	}
	return m
}

func lastRequest(st map[string]any) map[string]any {
	m, _ := st["last_request"].(map[string]any)
	return m
}

func num(st map[string]any, k string) float64 { v, _ := st[k].(float64); return v }

func TestNormalShortPrompt(t *testing.T) {
	e := newEnv(t, nil)
	e.worker("p0", "prefill", nil)
	e.worker("d0", "decode", nil)
	s := &sink{}
	out := e.run(e.job(40, false), s)
	expectOK(t, out, s, orchestrator.ModeNormal)
	if out.Reason != "short_prompt" || out.DWorker != "d0" || !strings.HasPrefix(s.text(), "Hello from ModelWeave mock worker d0") {
		t.Fatalf("reason %s worker %s text %q", out.Reason, out.DWorker, s.text())
	}
	if st := e.stats("d0"); num(st, "generate_requests") != 1 || lastRequest(st)["disagg"].(map[string]any)["mode"] != "DISAGG_MODE_NORMAL" {
		t.Fatalf("d0 got %v", st)
	}
	if num(e.stats("p0"), "generate_requests") != 0 {
		t.Fatal("a normal request reached the prefill instance")
	}
}

// Every PD handshake, checked in the request fields the engines received.
func TestPDHandshakes(t *testing.T) {
	type check func(t *testing.T, p, d map[string]any)
	has := func(m map[string]any, path ...string) bool {
		var cur any = m
		for _, k := range path {
			o, ok := cur.(map[string]any)
			if !ok {
				return false
			}
			cur, ok = o[k]
			if !ok {
				return false
			}
		}
		return cur != nil
	}
	cases := []struct {
		hs    string
		check check
	}{
		{mwip.HandshakePush, func(t *testing.T, p, d map[string]any) {
			if !has(p, "disagg", "kv_receiver") || d["disagg"].(map[string]any)["role"] != "ROLE_DECODE" {
				t.Fatalf("push fields: P %v D %v", p["disagg"], d["disagg"])
			}
		}},
		{mwip.HandshakeBootstrap, func(t *testing.T, p, d map[string]any) {
			if !has(d, "disagg", "prefill", "bootstrap_addr") {
				t.Fatalf("bootstrap fields: D %v", d["disagg"])
			}
		}},
		{mwip.HandshakePull, func(t *testing.T, p, d map[string]any) {
			if !has(d, "disagg", "kv_handle", "room") {
				t.Fatalf("pull fields: D %v", d["disagg"])
			}
		}},
	}
	for _, c := range cases {
		t.Run(c.hs, func(t *testing.T) {
			e := newEnv(t, func(m *config.Model) { m.PD.Handshake = c.hs })
			e.worker("p0", "prefill", nil)
			e.worker("d0", "decode", nil)
			s := &sink{}
			out := e.run(e.job(400, false), s)
			expectOK(t, out, s, orchestrator.ModePD)
			if out.PWorker != "p0" || out.DWorker != "d0" || out.Reason != "long_prompt" {
				t.Fatalf("plan %+v", out)
			}
			if !strings.HasPrefix(s.text(), "Hello from ModelWeave mock worker d0") {
				t.Fatalf("answer %q", s.text())
			}
			pReq, dReq := lastRequest(e.stats("p0")), lastRequest(e.stats("d0"))
			c.check(t, pReq, dReq)
			// The repeated prompt is cached on d0, which received the KV:
			// it now runs normal there.
			s2 := &sink{}
			out2 := e.run(e.job(400, false), s2)
			expectOK(t, out2, s2, orchestrator.ModeNormal)
			if out2.Reason != "prefix_hit" || out2.Usage.Cached() == 0 {
				t.Fatalf("second request: reason %s cached %d", out2.Reason, out2.Usage.Cached())
			}
		})
	}
}

// EPD: an encode instance builds the image's artifacts, the prefill
// instance loads them, and the same image again is a cache hit.
func TestEPDWithArtifactCache(t *testing.T) {
	e := newEnv(t, func(m *config.Model) { m.Mode = "pd" })
	e.worker("e0", "encode", nil)
	e.worker("p0", "prefill", nil)
	e.worker("d0", "decode", nil)
	job := e.job(200, true)
	need, err := e.orch.NeedsEncode(job)
	if err != nil || !need {
		t.Fatalf("needs encode %v %v", need, err)
	}
	st, err := e.orch.Encode(context.Background(), job)
	if err != nil || st.Built != 1 {
		t.Fatalf("encode %+v %v", st, err)
	}
	s := &sink{}
	out := e.run(job, s)
	expectOK(t, out, s, orchestrator.ModePD)
	if out.Usage.PromptTokens != 250 {
		t.Fatalf("prompt tokens %d, want 250 (image expanded)", out.Usage.PromptTokens)
	}
	if p := e.stats("p0"); num(p, "artifacts_verified") != 1 || num(e.stats("e0"), "encoded") != 1 {
		t.Fatalf("epd: P %v", p)
	}
	again := e.job(200, false)
	again.Render, again.Media = job.Render, job.Render.Multimodal.Clone()
	if st, err := e.orch.Encode(context.Background(), again); err != nil || st.Hits != 1 || st.Built != 0 {
		t.Fatalf("cache: %+v %v", st, err)
	}
}

func TestNormalFallbackToPD(t *testing.T) {
	e := newEnv(t, nil)
	e.worker("p0", "prefill", nil)
	e.worker("d0", "decode", func(c *mockengine.Config) { c.RejectNormal = true })
	s := &sink{}
	out := e.run(e.job(40, false), s)
	expectOK(t, out, s, orchestrator.ModePD)
	if out.Fallback != mwip.ErrNormalRejected || out.Attempts != 2 || out.Reason != "fallback" {
		t.Fatalf("fallback not recorded: %+v", out)
	}
}

func TestDecodeRetryOnAnotherWorker(t *testing.T) {
	e := newEnv(t, func(m *config.Model) { m.Mode = "normal"; m.Routing.Decode.Name = "round_robin" })
	e.worker("d-bad", "decode", func(c *mockengine.Config) { c.FailRate = 1 })
	e.worker("d-good", "decode", nil)
	for i := 0; i < 8; i++ {
		s := &sink{}
		out := e.run(e.job(40, false), s)
		expectOK(t, out, s, orchestrator.ModeNormal)
		if out.DWorker != "d-good" {
			t.Fatalf("finished on %s", out.DWorker)
		}
	}
	// Internal errors count against health: the bad worker is ejected.
	if snap := e.reg.Get("d-bad").Snapshot(time.Now(), time.Minute); snap.Healthy {
		t.Fatal("d-bad should be ejected after consecutive failures")
	}
	if snap := e.reg.Get("d-good").Snapshot(time.Now(), time.Minute); !snap.Healthy {
		t.Fatal("d-good must stay healthy")
	}
}

func TestPrefillFailureTriggersWholeRetry(t *testing.T) {
	e := newEnv(t, func(m *config.Model) {
		m.Mode = "pd"
		m.Retry.WholeRetries = 2
		m.Routing.Prefill.Name = "round_robin"
	})
	// A prefill worker whose port is closed: transport failure.
	l, _ := net.Listen("tcp", "127.0.0.1:0")
	dead := "http://" + l.Addr().String()
	l.Close()
	if _, _, err := e.reg.Upsert(registry.Spec{ID: "p-dead", Model: "m", Role: registry.RolePrefill, Target: dead}, "static"); err != nil {
		t.Fatal(err)
	}
	e.worker("p-ok", "prefill", nil)
	e.worker("d0", "decode", nil)
	for i := 0; i < 3; i++ {
		s := &sink{}
		out := e.run(e.job(400, false), s)
		expectOK(t, out, s, orchestrator.ModePD)
		if out.PWorker != "p-ok" {
			t.Fatalf("prefilled on %s", out.PWorker)
		}
	}
}

func TestUnifiedWorkersRunNormal(t *testing.T) {
	e := newEnv(t, nil)
	e.worker("u0", "unified", nil)
	s := &sink{}
	out := e.run(e.job(300, false), s)
	expectOK(t, out, s, orchestrator.ModeNormal)
	if out.Reason != "unified" {
		t.Fatalf("reason %s", out.Reason)
	}
}

func TestEmitErrorStopsEngine(t *testing.T) {
	e := newEnv(t, func(m *config.Model) { m.Mode = "normal" })
	e.worker("d0", "decode", func(c *mockengine.Config) { c.DecodeTPS = 50; c.MaxOutputTokens = 200 })
	s := &sink{stopAt: 2}
	start := time.Now()
	out := e.run(e.job(40, false), s)
	var stopped *orchestrator.StoppedError
	if !errors.As(out.Err, &stopped) {
		t.Fatalf("want StoppedError, got %v", out.Err)
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("engine kept generating after the stop")
	}
	deadline := time.Now().Add(2 * time.Second)
	for e.mocks["d0"].State().RunningRequests != 0 {
		if time.Now().After(deadline) {
			t.Fatal("mock still running the aborted request")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if snap := e.reg.Get("d0").Snapshot(time.Now(), time.Minute); snap.Inflight != 0 || snap.ReservedKV != 0 {
		t.Fatalf("reservation leaked: %+v", snap)
	}
}

func TestNoCapacityThenQueueable(t *testing.T) {
	e := newEnv(t, func(m *config.Model) { m.Mode = "normal" })
	e.worker("d0", "decode", nil)
	e.reg.Get("d0").UpdateState(registry.EngineState{KVCapacityTokens: 100, KVUsedTokens: 90}, time.Now())
	_, err := e.orch.Plan(context.Background(), e.job(40, false), orchestrator.PlanOpts{})
	if !errors.Is(err, scheduler.ErrNoCapacity) {
		t.Fatalf("want ErrNoCapacity, got %v", err)
	}
}

func TestMidStreamFailureIsNotRetried(t *testing.T) {
	e := newEnv(t, func(m *config.Model) { m.Mode = "normal" })
	e.worker("d0", "decode", func(c *mockengine.Config) { c.FailAfterTokens = 3; c.MaxOutputTokens = 20 })
	e.worker("d1", "decode", nil)
	s := &sink{}
	job := e.job(40, false)
	plan, err := e.orch.Plan(context.Background(), job, orchestrator.PlanOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if plan.D.Spec.ID != "d0" {
		// Force the failing worker for this test.
		plan.Release(false)
		e.reg.Get("d1").SetDraining(true)
		if plan, err = e.orch.Plan(context.Background(), job, orchestrator.PlanOpts{}); err != nil {
			t.Fatal(err)
		}
	}
	out := e.orch.Execute(context.Background(), job, plan, s.emit)
	if out.Err == nil || out.Attempts != 1 {
		t.Fatalf("mid-stream failure must end the request without retry: err=%v attempts=%d", out.Err, out.Attempts)
	}
	if s.deltas == 0 {
		t.Fatal("deltas before the failure should have been emitted")
	}
}
