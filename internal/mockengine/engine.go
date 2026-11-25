// Package mockengine is a simulated inference worker for local testing. It
// serves the three engine protocols the gateway can be started with:
//
//	POST /mwip/v1/generate       the intermediate state (NDJSON events)
//	POST /inference/v1/generate  vLLM token-in generate (output_mode derender)
//	POST /generate               SGLang native generate (raw text)
//
// It models what the gateway depends on: a prefix KV cache with the
// gateway's block hashing and KV events, prefill/decode timing, hard
// admission, the PD handshakes in each protocol's fields with a real KV
// hand-off between mock workers over HTTP, normal (non-PD) execution, the
// encode stage (native artifacts, a vLLM EC producer, SGLang's
// engine-internal encoding), engine-side output parsing, aborts, heartbeats
// and fault injection. Output tokens come from the real tokenizer of the
// worker's model, in that model's markup (DeepSeek V4.1 DSML, Qwen3
// <think> and hermes tool calls), so reasoning, tool calls and stop strings
// behave like the model.
package mockengine

import (
	"container/list"
	"context"
	crand "crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/leonyang215/ModelWeave/internal/kvindex"
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/wire"
)

// Config configures a mock worker.
type Config struct {
	ID                string
	Model             string
	Role              string
	Listen            string
	AdvertiseURL      string
	GatewayURL        string
	ControlToken      string // X-ModelWeave-Token sent with heartbeats
	TokenizerURL      string
	TokenizerModel    string // model name known to the tokenizer (default: its only model)
	HeartbeatInterval time.Duration
	Normal            bool // a decode worker that also runs normal (non-PD) requests
	KVCapacityTokens  int64
	MaxRunning        int
	MaxPrefillTokens  int64
	PrefillTPS        float64 // prompt tokens per second
	DecodeTPS         float64 // output tokens per second per request
	MaxOutputTokens   int     // cap of generated tokens (also bounded by max_tokens)
	AnswerRepeat      int     // extra filler sentences appended to the answer
	BlockSize         int
	KVEvents          bool
	EncodeLatency     time.Duration
	// PlaceholderTokenID is the token of one collapsed image in vLLM and
	// SGLang requests whose media hints name no placeholder (DeepSeek V4.1
	// <｜image｜>).
	PlaceholderTokenID uint32
	// Fault injection.
	RejectNormal    bool    // decline normal requests (tests normal -> pd fallback)
	FailRate        float64 // probability of failing a request before output
	FailAfterTokens int     // break the stream after N tokens (0: never)
	VerifyArtifacts bool    // check embedding artifacts / EC handles with their encoder
	Log             *slog.Logger
}

type kvStore struct {
	mu       sync.Mutex
	cap      int
	lru      *list.List
	elems    map[uint64]*list.Element
	stored   []uint64
	removed  []uint64
	cleared  bool
	recordEv bool
}

func newKVStore(capBlocks int, events bool) *kvStore {
	return &kvStore{cap: capBlocks, lru: list.New(), elems: map[uint64]*list.Element{}, recordEv: events}
}

// match returns the number of leading blocks present.
func (k *kvStore) match(blocks []uint64) int {
	k.mu.Lock()
	defer k.mu.Unlock()
	n := 0
	for _, b := range blocks {
		e, ok := k.elems[b]
		if !ok {
			break
		}
		k.lru.MoveToFront(e)
		n++
	}
	return n
}

func (k *kvStore) insert(blocks []uint64) {
	k.mu.Lock()
	defer k.mu.Unlock()
	for i := len(blocks) - 1; i >= 0; i-- {
		b := blocks[i]
		if e, ok := k.elems[b]; ok {
			k.lru.MoveToFront(e)
			continue
		}
		k.elems[b] = k.lru.PushFront(b)
		if k.recordEv {
			k.stored = append(k.stored, b)
		}
	}
	for k.cap > 0 && k.lru.Len() > k.cap {
		e := k.lru.Back()
		b := e.Value.(uint64)
		k.lru.Remove(e)
		delete(k.elems, b)
		if k.recordEv {
			k.removed = append(k.removed, b)
		}
	}
}

func (k *kvStore) drainEvents() *mwip.KVEvents {
	k.mu.Lock()
	defer k.mu.Unlock()
	if !k.recordEv || (len(k.stored) == 0 && len(k.removed) == 0 && !k.cleared) {
		return nil
	}
	ev := &mwip.KVEvents{Stored: k.stored, Removed: k.removed, Cleared: k.cleared}
	k.stored, k.removed, k.cleared = nil, nil, false
	return ev
}

func (k *kvStore) len() int {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.lru.Len()
}

// kvPayload is the simulated KV hand-off between prefill and decode.
type kvPayload struct {
	Room          string `json:"room"`
	FromWorker    string `json:"from_worker"`
	PromptTokens  int    `json:"prompt_tokens"`
	Transferred   int    `json:"transferred_tokens"`
	CachedOnP     int    `json:"cached_on_prefill"`
	PrefillMillis int64  `json:"prefill_ms"`
}

type running struct {
	requestID string
	cancel    context.CancelFunc
	kvTokens  int64
}

// Engine is a mock worker.
type Engine struct {
	cfg  Config
	log  *slog.Logger
	kv   *kvStore
	seed uint64
	http *http.Client

	mu        sync.Mutex
	running   map[string]*running       // request_id -> running
	rooms     map[string]chan kvPayload // decode: room -> KV arrival (push)
	outbox    map[string]kvPayload      // prefill: room -> KV ready for pickup
	outboxW   map[string]chan struct{}  // prefill: room -> ready signal
	artifacts map[string]string         // encode: reference -> producing request

	kvUsed        atomic.Int64
	prefillQueued atomic.Int64
	waiting       atomic.Int64

	stats struct {
		sync.Mutex
		ByMode            map[string]int
		ByProtocol        map[string]int // generate calls
		EncodeByProtocol  map[string]int // encode calls
		Streaming         int            // generate calls with stream: true
		NonStreaming      int            // generate calls without streaming
		Aborts            int
		Canceled          int
		Rejected          int
		Injected          int
		CachedTokens      int
		PromptTokens      int
		OutputTokens      int
		KVReceived        int
		KVSent            int
		Encoded           int // items encoded by this encoder
		ArtifactsVerified int // native embedding artifacts checked with their encoder
		ECLoaded          int // vLLM embeddings loaded through ec_transfer_params
		LocalEncoded      int // media this worker encoded itself (no encode stage)
		MediaItems        int
		StopMatches       int
		LastRequestID     string
		LastCached        int
		LastMode          string
		LastFinish        string
		LastProtocol      string
		LastRequest       json.RawMessage // body of the last generate call
	}

	vocab vocab
}

// New creates an engine.
func New(cfg Config) *Engine {
	if cfg.BlockSize <= 0 {
		cfg.BlockSize = kvindex.DefaultBlockSize
	}
	if cfg.MaxOutputTokens <= 0 {
		cfg.MaxOutputTokens = 64
	}
	if cfg.PlaceholderTokenID == 0 {
		cfg.PlaceholderTokenID = 129264
	}
	if cfg.Log == nil {
		cfg.Log = slog.Default()
	}
	capBlocks := 0
	if cfg.KVCapacityTokens > 0 {
		capBlocks = int(cfg.KVCapacityTokens) / cfg.BlockSize
	}
	e := &Engine{
		cfg:       cfg,
		log:       cfg.Log.With("worker", cfg.ID, "role", cfg.Role),
		kv:        newKVStore(capBlocks, cfg.KVEvents),
		seed:      kvindex.ModelSeed(cfg.Model),
		http:      &http.Client{Timeout: 0},
		running:   map[string]*running{},
		rooms:     map[string]chan kvPayload{},
		outbox:    map[string]kvPayload{},
		outboxW:   map[string]chan struct{}{},
		artifacts: map[string]string{},
	}
	e.stats.ByMode = map[string]int{}
	e.stats.ByProtocol = map[string]int{}
	e.stats.EncodeByProtocol = map[string]int{}
	e.vocab.engine = e
	return e
}

// Handler returns the worker's HTTP API (every engine protocol).
func (e *Engine) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /mwip/v1/generate", e.handleMWIP)
	mux.HandleFunc("POST /mwip/v1/abort", e.handleAbortMWIP)
	mux.HandleFunc("POST /mwip/v1/encode", e.handleEncode)
	mux.HandleFunc("POST /inference/v1/generate", e.handleVLLM)
	mux.HandleFunc("POST /inference/v1/abort_requests", e.handleAbortVLLM)
	mux.HandleFunc("POST /abort_requests", e.handleAbortVLLM)
	mux.HandleFunc("POST /generate", e.handleSGLang)
	mux.HandleFunc("POST /abort_request", e.handleAbortSGLang)
	mux.HandleFunc("GET /mwip/v1/artifacts/{ref}", e.handleArtifact)
	mux.HandleFunc("POST /mwip/v1/kv/push", e.handleKVPush)
	mux.HandleFunc("GET /mwip/v1/kv/fetch", e.handleKVFetch)
	mux.HandleFunc("GET /mwip/v1/state", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, e.State()) })
	health := func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok")) }
	mux.HandleFunc("GET /mwip/v1/health", health)
	mux.HandleFunc("GET /health", health)
	mux.HandleFunc("GET /metrics", e.handleMetrics)
	mux.HandleFunc("GET /mock/stats", e.handleStats)
	return mux
}

// State returns the current engine state.
func (e *Engine) State() registry.EngineState {
	e.mu.Lock()
	n := len(e.running)
	e.mu.Unlock()
	return registry.EngineState{
		RunningRequests:    n,
		WaitingRequests:    int(e.waiting.Load()),
		KVUsedTokens:       e.kvUsed.Load(),
		KVCapacityTokens:   e.cfg.KVCapacityTokens,
		PrefillQueueTokens: e.prefillQueued.Load(),
		MaxRunningRequests: e.cfg.MaxRunning,
		CachedBlocks:       int64(e.kv.len()),
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, code, msg string, retriable bool) {
	writeJSON(w, status, mwip.ErrorBody{Type: "error", Code: code, Message: msg, Retriable: retriable})
}

func (e *Engine) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	st := e.State()
	e.stats.Lock()
	byProto := make(map[string]int, len(e.stats.ByProtocol))
	for k, v := range e.stats.ByProtocol {
		byProto[k] = v
	}
	e.stats.Unlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# TYPE mwip_engine_running_requests gauge\nmwip_engine_running_requests %d\n", st.RunningRequests)
	fmt.Fprintf(w, "# TYPE mwip_engine_waiting_requests gauge\nmwip_engine_waiting_requests %d\n", st.WaitingRequests)
	fmt.Fprintf(w, "# TYPE mwip_engine_kv_used_tokens gauge\nmwip_engine_kv_used_tokens %d\n", st.KVUsedTokens)
	fmt.Fprintf(w, "# TYPE mwip_engine_kv_capacity_tokens gauge\nmwip_engine_kv_capacity_tokens %d\n", st.KVCapacityTokens)
	fmt.Fprintf(w, "# TYPE mwip_engine_prefill_queue_tokens gauge\nmwip_engine_prefill_queue_tokens %d\n", st.PrefillQueueTokens)
	fmt.Fprintf(w, "# TYPE mwip_engine_cached_blocks gauge\nmwip_engine_cached_blocks %d\n", st.CachedBlocks)
	fmt.Fprintf(w, "# TYPE mock_engine_generate_requests_total counter\n")
	for _, p := range []string{mwip.EngineMWIP, mwip.EngineVLLM, mwip.EngineSGLang} {
		fmt.Fprintf(w, "mock_engine_generate_requests_total{protocol=%q} %d\n", p, byProto[p])
	}
}

func (e *Engine) handleStats(w http.ResponseWriter, _ *http.Request) {
	e.stats.Lock()
	defer e.stats.Unlock()
	writeJSON(w, 200, map[string]any{
		"worker": e.cfg.ID, "role": e.cfg.Role, "by_mode": e.stats.ByMode,
		"by_protocol": e.stats.ByProtocol, "encode_by_protocol": e.stats.EncodeByProtocol,
		"streaming_requests": e.stats.Streaming, "non_streaming_requests": e.stats.NonStreaming,
		"aborts": e.stats.Aborts, "canceled": e.stats.Canceled, "rejected": e.stats.Rejected,
		"injected_failures": e.stats.Injected, "cached_tokens_total": e.stats.CachedTokens,
		"prompt_tokens_total": e.stats.PromptTokens, "output_tokens_total": e.stats.OutputTokens,
		"kv_received": e.stats.KVReceived, "kv_sent": e.stats.KVSent, "encoded": e.stats.Encoded,
		"artifacts_verified": e.stats.ArtifactsVerified, "ec_loaded": e.stats.ECLoaded,
		"local_encoded": e.stats.LocalEncoded, "media_items": e.stats.MediaItems,
		"stop_string_matches": e.stats.StopMatches, "last_request_id": e.stats.LastRequestID,
		"last_cached_tokens": e.stats.LastCached, "last_mode": e.stats.LastMode, "last_finish": e.stats.LastFinish,
		"last_protocol": e.stats.LastProtocol, "last_request": e.stats.LastRequest, "cached_blocks": e.kv.len(),
	})
}

func (e *Engine) count(f func()) {
	e.stats.Lock()
	f()
	e.stats.Unlock()
}

func decodeBody(w http.ResponseWriter, r *http.Request, v any) (json.RawMessage, bool) {
	var raw json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&raw); err == nil {
		if err = json.Unmarshal(raw, v); err == nil {
			return raw, true
		} else {
			writeErr(w, 400, mwip.ErrBadRequest, "invalid generate request: "+err.Error(), false)
		}
	} else {
		writeErr(w, 400, mwip.ErrBadRequest, "invalid generate request: "+err.Error(), false)
	}
	return nil, false
}

// modelInfo returns the markup of the worker's model, answering 503 when
// the tokenizer cannot tell it.
func (e *Engine) modelInfo(w http.ResponseWriter, r *http.Request) (modelInfo, bool) {
	info, err := e.vocab.model(r.Context())
	if err != nil {
		writeErr(w, 503, mwip.ErrInternal, "model info: "+err.Error(), true)
		return info, false
	}
	return info, true
}

func (e *Engine) handleMWIP(w http.ResponseWriter, r *http.Request) {
	var req mwip.GenerateRequest
	raw, ok := decodeBody(w, r, &req)
	if !ok {
		return
	}
	info, ok := e.modelInfo(w, r)
	if !ok {
		return
	}
	e.serve(w, r, fromMWIP(&req, info), raw)
}

func (e *Engine) handleVLLM(w http.ResponseWriter, r *http.Request) {
	var req wire.VLLMGenerateRequest
	raw, ok := decodeBody(w, r, &req)
	if !ok {
		return
	}
	if e.cfg.Role == "encode" {
		e.vllmEncode(w, r, &req, raw)
		return
	}
	info, ok := e.modelInfo(w, r)
	if !ok {
		return
	}
	e.serve(w, r, fromVLLM(&req, e.cfg.PlaceholderTokenID, info), raw)
}

func (e *Engine) handleSGLang(w http.ResponseWriter, r *http.Request) {
	var req wire.SGLangRequest
	raw, ok := decodeBody(w, r, &req)
	if !ok {
		return
	}
	info, ok := e.modelInfo(w, r)
	if !ok {
		return
	}
	e.serve(w, r, fromSGLang(&req, e.cfg.PlaceholderTokenID, e.cfg.Role, info), raw)
}

// blocksOf hashes a prompt exactly like the gateway: the expanded token
// list as is (media positions hold their pad values).
func (e *Engine) blocksOf(g *genReq) []uint64 {
	return kvindex.BlockHashes(g.tokens, e.cfg.BlockSize, e.seed)
}

func (e *Engine) admit(id string, r *running) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.cfg.MaxRunning > 0 && len(e.running) >= e.cfg.MaxRunning {
		return false
	}
	if e.cfg.KVCapacityTokens > 0 && e.kvUsed.Load()+r.kvTokens > e.cfg.KVCapacityTokens {
		return false
	}
	if _, dup := e.running[id]; dup {
		return false
	}
	e.running[id] = r
	e.kvUsed.Add(r.kvTokens)
	return true
}

func (e *Engine) release(id string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if r, ok := e.running[id]; ok {
		e.kvUsed.Add(-r.kvTokens)
		delete(e.running, id)
	}
}

func (e *Engine) abort(ids []string) int {
	n := 0
	e.mu.Lock()
	for _, id := range ids {
		if run, ok := e.running[id]; ok {
			run.cancel()
			n++
		}
	}
	e.mu.Unlock()
	if n > 0 {
		e.count(func() { e.stats.Aborts += n })
	}
	return n
}

func (e *Engine) handleAbortMWIP(w http.ResponseWriter, r *http.Request) {
	var req mwip.AbortRequest
	_ = json.NewDecoder(r.Body).Decode(&req)
	writeJSON(w, 200, map[string]int{"aborted": e.abort([]string{req.RequestID})})
}

// handleAbortVLLM serves vLLM's POST /inference/v1/abort_requests.
func (e *Engine) handleAbortVLLM(w http.ResponseWriter, r *http.Request) {
	var req struct {
		RequestIDs json.RawMessage `json:"request_ids"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	var ids []string
	if json.Unmarshal(req.RequestIDs, &ids) != nil {
		var one string
		if json.Unmarshal(req.RequestIDs, &one) == nil {
			ids = []string{one}
		}
	}
	e.abort(ids)
	w.WriteHeader(http.StatusOK)
}

// handleAbortSGLang serves SGLang's POST /abort_request.
func (e *Engine) handleAbortSGLang(w http.ResponseWriter, r *http.Request) {
	var req struct {
		RID string `json:"rid"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	e.abort([]string{req.RID})
	w.WriteHeader(http.StatusOK)
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (e *Engine) prefillDelay(tokens int) time.Duration {
	if e.cfg.PrefillTPS <= 0 || tokens <= 0 {
		return 0
	}
	return time.Duration(float64(tokens) / e.cfg.PrefillTPS * float64(time.Second))
}

func modeLabel(g *genReq, role string) string {
	switch {
	case g.kv.kind != "local":
		return "split_" + g.kv.kind
	case role == "unified":
		return "normal_unified"
	}
	return "normal"
}

// serve validates and runs a generate call.
func (e *Engine) serve(w http.ResponseWriter, r *http.Request, g *genReq, raw json.RawMessage) {
	e.count(func() {
		if g.stream {
			e.stats.Streaming++
		} else {
			e.stats.NonStreaming++
		}
		e.stats.ByProtocol[g.protocol]++
		e.stats.LastProtocol, e.stats.LastRequest = g.protocol, raw
		e.stats.MediaItems += len(g.media)
	})
	if len(g.tokens) == 0 {
		writeErr(w, 400, mwip.ErrBadRequest, "token_ids required", false)
		return
	}
	if g.requestID == "" {
		g.requestID = "mock-" + randomHex(8)
	}
	label := modeLabel(g, e.cfg.Role)
	switch {
	case e.cfg.Role == "encode":
		writeErr(w, 400, mwip.ErrBadRequest, "an encode worker does not generate", false)
		return
	case g.kv.kind == "local" && e.cfg.Role == "prefill":
		writeErr(w, 400, mwip.ErrBadRequest, "a prefill worker cannot run whole requests", false)
		return
	case g.kv.kind == "local" && e.cfg.Role == "decode" && (e.cfg.RejectNormal || !e.cfg.Normal):
		e.count(func() { e.stats.Rejected++ })
		writeErr(w, 503, mwip.ErrNormalRejected, "this decode worker does not run normal requests", true)
		return
	case g.kv.kind != "local" && g.kv.kind != e.cfg.Role:
		writeErr(w, 400, mwip.ErrBadRequest, fmt.Sprintf("worker role %s cannot serve the %s leg", e.cfg.Role, g.kv.kind), false)
		return
	}
	if e.cfg.FailRate > 0 && rand.Float64() < e.cfg.FailRate {
		e.count(func() { e.stats.Injected++ })
		writeErr(w, 500, mwip.ErrInternal, "injected failure", true)
		return
	}
	// Media are needed where prefill runs: embeddings from the encode stage
	// (native artifacts, vLLM EC handles), else encoded here.
	var mediaDelay time.Duration
	if g.kv.kind != "decode" {
		var err error
		if mediaDelay, err = e.prepareMedia(r.Context(), g); err != nil {
			writeErr(w, 422, mwip.ErrArtifact, err.Error(), false)
			return
		}
	}
	blocks := e.blocksOf(g)
	prompt := len(g.tokens)
	kvTokens := int64(prompt + g.maxTokens)
	if g.kv.kind == "prefill" {
		kvTokens = int64(prompt)
	}
	if e.cfg.KVCapacityTokens > 0 {
		kvTokens = min(kvTokens, int64(prompt)+int64(e.cfg.MaxOutputTokens))
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	if !e.admit(g.requestID, &running{requestID: g.requestID, cancel: cancel, kvTokens: kvTokens}) {
		e.count(func() { e.stats.Rejected++ })
		writeErr(w, 503, mwip.ErrCapacity, "worker is at capacity", true)
		return
	}
	defer e.release(g.requestID)
	e.count(func() {
		e.stats.ByMode[label]++
		e.stats.LastRequestID, e.stats.LastMode = g.requestID, label
		e.stats.PromptTokens += prompt
	})
	out := newSink(w, g, e.cfg.ID)
	var err error
	switch g.kv.kind {
	case "prefill":
		err = e.runPrefill(ctx, out, g, blocks, prompt, mediaDelay)
	case "decode":
		err = e.runSplitDecode(ctx, out, g, blocks, prompt)
	default:
		err = e.runLocal(ctx, out, g, blocks, prompt, mediaDelay)
	}
	if err != nil {
		if ctx.Err() != nil {
			e.count(func() { e.stats.Canceled++ })
			return
		}
		e.log.Warn("request failed", "request_id", g.requestID, "protocol", g.protocol, "error", err)
	}
}

// prepareMedia obtains the embeddings of the request's media: native
// artifacts and vLLM EC handles are checked with their encoder; media
// without either are encoded here (the returned delay).
func (e *Engine) prepareMedia(ctx context.Context, g *genReq) (time.Duration, error) {
	var delay time.Duration
	for _, m := range g.media {
		switch {
		case len(m.artifacts) > 0:
			if e.cfg.VerifyArtifacts {
				for _, a := range m.artifacts {
					if err := e.fetchArtifact(ctx, a); err != nil {
						return 0, err
					}
				}
			}
			e.count(func() { e.stats.ArtifactsVerified += len(m.artifacts) })
		case m.ecURI != "":
			if err := e.fetchArtifact(ctx, m.ecURI); err != nil {
				return 0, fmt.Errorf("ec transfer: %w", err)
			}
			e.count(func() { e.stats.ECLoaded++ })
		default:
			delay += e.cfg.EncodeLatency
			e.count(func() { e.stats.LocalEncoded++ })
		}
	}
	return delay, nil
}

func (e *Engine) fetchArtifact(ctx context.Context, uri string) error {
	u := strings.Replace(uri, "mock://", "http://", 1)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	resp, err := e.http.Do(req)
	if err != nil {
		return fmt.Errorf("embedding %s unreachable: %v", uri, err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("embedding %s missing (%d)", uri, resp.StatusCode)
	}
	return nil
}

// runLocal serves normal requests: local prefill, then decode.
func (e *Engine) runLocal(ctx context.Context, out sink, g *genReq, blocks []uint64, prompt int, mediaDelay time.Duration) error {
	start := time.Now()
	cached := min(e.kv.match(blocks)*e.cfg.BlockSize, prompt)
	uncached := prompt - cached
	e.prefillQueued.Add(int64(uncached))
	err := sleepCtx(ctx, e.prefillDelay(uncached)+mediaDelay)
	e.prefillQueued.Add(-int64(uncached))
	if err != nil {
		return err
	}
	e.kv.insert(blocks)
	return e.decode(ctx, out, g, prompt, cached, &mwip.Timing{PrefillMS: msSince(start), EncodeMS: ms(mediaDelay)}, nil, start)
}

func msSince(t time.Time) float64 { return ms(time.Since(t)) }

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// runPrefill serves the prefill leg: compute (simulated), hand the KV over
// per handshake, then end the leg: a native prefill_done event, or one
// token like vLLM's max_tokens=1 leg.
func (e *Engine) runPrefill(ctx context.Context, out sink, g *genReq, blocks []uint64, prompt int, mediaDelay time.Duration) error {
	start := time.Now()
	cached := min(e.kv.match(blocks)*e.cfg.BlockSize, prompt)
	uncached := prompt - cached
	e.prefillQueued.Add(int64(uncached))
	err := sleepCtx(ctx, e.prefillDelay(uncached)+mediaDelay)
	e.prefillQueued.Add(-int64(uncached))
	if err != nil {
		return err
	}
	e.kv.insert(blocks)
	payload := kvPayload{Room: g.kv.room, FromWorker: e.cfg.ID, PromptTokens: prompt, CachedOnP: cached, PrefillMillis: time.Since(start).Milliseconds()}
	var params map[string]any
	var handle *mwip.KVHandle
	switch g.kv.handshake {
	case mwip.HandshakePush:
		if g.kv.receiver == nil {
			return out.fail(400, mwip.ErrBadRequest, "push handshake without kv_receiver", false)
		}
		payload.Transferred = prompt - g.kv.receiver.CachedTokens
		if err := e.pushKV(ctx, g.kv.receiver.URL, payload); err != nil {
			return out.fail(502, mwip.ErrKVTransfer, err.Error(), true)
		}
	default: // bootstrap and pull: park the KV for the decode leg to fetch
		payload.Transferred = prompt
		e.park(g.kv.room, payload)
		if g.kv.returnParams {
			handle = &mwip.KVHandle{WorkerID: e.cfg.ID, URL: e.cfg.AdvertiseURL, Room: g.kv.room, Tokens: prompt}
			host, port := e.hostPort()
			params = map[string]any{
				"do_remote_prefill": true, "do_remote_decode": false, "remote_engine_id": e.cfg.ID,
				"remote_block_ids": []int{0}, "remote_host": host, "remote_port": port,
				"remote_request_id": g.kv.room, "tp_size": 1,
			}
		}
	}
	e.count(func() { e.stats.KVSent++; e.stats.CachedTokens += cached; e.stats.LastCached = cached })
	timing := &mwip.Timing{PrefillMS: msSince(start), EncodeMS: ms(mediaDelay)}
	if g.protocol == mwip.EngineMWIP {
		return out.prefillDone(&mwip.GenerateEvent{Type: mwip.EventPrefillDone, PromptTokens: prompt, CachedTokens: cached,
			Handle: handle, Timing: timing, WorkerID: e.cfg.ID,
			Usage: &mwip.Usage{PromptTokens: prompt, PromptTokensDetails: &mwip.PromptTokensDetails{CachedTokens: cached}}})
	}
	g.maxTokens = 1
	return e.decode(ctx, out, g, prompt, cached, timing, params, start)
}

func (e *Engine) hostPort() (string, int) {
	u, err := url.Parse(e.cfg.AdvertiseURL)
	if err != nil {
		return "", 0
	}
	port, _ := strconv.Atoi(u.Port())
	return u.Hostname(), port
}

func (e *Engine) park(room string, p kvPayload) {
	e.mu.Lock()
	e.outbox[room] = p
	if ch, ok := e.outboxW[room]; ok {
		close(ch)
		delete(e.outboxW, room)
	}
	e.mu.Unlock()
	go func() {
		time.Sleep(2 * time.Minute)
		e.mu.Lock()
		delete(e.outbox, room)
		e.mu.Unlock()
	}()
}

func (e *Engine) pushKV(ctx context.Context, base string, p kvPayload) error {
	body, _ := json.Marshal(p)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/mwip/v1/kv/push", strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("kv push to %s: status %d", base, resp.StatusCode)
	}
	return nil
}

// handleKVPush receives KV for a registered room (push handshake).
func (e *Engine) handleKVPush(w http.ResponseWriter, r *http.Request) {
	var p kvPayload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		writeErr(w, 400, mwip.ErrBadRequest, err.Error(), false)
		return
	}
	e.mu.Lock()
	ch, ok := e.rooms[p.Room]
	e.mu.Unlock()
	if !ok {
		writeErr(w, 404, mwip.ErrKVTransfer, "unknown room "+p.Room, false)
		return
	}
	select {
	case ch <- p:
		writeJSON(w, 200, map[string]bool{"ok": true})
	default:
		writeErr(w, 409, mwip.ErrKVTransfer, "room already filled", false)
	}
}

// handleKVFetch hands parked KV to a decode worker (bootstrap / pull),
// long-polling until the prefill leg has produced it.
func (e *Engine) handleKVFetch(w http.ResponseWriter, r *http.Request) {
	room := r.URL.Query().Get("room")
	deadline := time.After(60 * time.Second)
	for {
		e.mu.Lock()
		p, ok := e.outbox[room]
		var ch chan struct{}
		if ok {
			delete(e.outbox, room)
		} else {
			ch = e.outboxW[room]
			if ch == nil {
				ch = make(chan struct{})
				e.outboxW[room] = ch
			}
		}
		e.mu.Unlock()
		if ok {
			writeJSON(w, 200, p)
			return
		}
		select {
		case <-ch:
		case <-deadline:
			writeErr(w, 504, mwip.ErrKVTransfer, "kv not produced in time", true)
			return
		case <-r.Context().Done():
			return
		}
	}
}

func (e *Engine) fetchKV(ctx context.Context, base, room string) (kvPayload, error) {
	var p kvPayload
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/mwip/v1/kv/fetch?room="+url.QueryEscape(room), nil)
	if err != nil {
		return p, err
	}
	resp, err := e.http.Do(req)
	if err != nil {
		return p, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return p, fmt.Errorf("kv fetch from %s: status %d", base, resp.StatusCode)
	}
	return p, json.NewDecoder(resp.Body).Decode(&p)
}

// runSplitDecode serves the decode leg: obtain the KV, then decode.
func (e *Engine) runSplitDecode(ctx context.Context, out sink, g *genReq, blocks []uint64, prompt int) error {
	start := time.Now()
	cached := min(e.kv.match(blocks)*e.cfg.BlockSize, prompt)
	timeout := time.Minute
	if g.timeoutMS > 0 {
		timeout = time.Duration(g.timeoutMS) * time.Millisecond
	}
	wait, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	var p kvPayload
	switch g.kv.handshake {
	case mwip.HandshakePush:
		ch := make(chan kvPayload, 1)
		e.mu.Lock()
		e.rooms[g.kv.room] = ch
		e.mu.Unlock()
		defer func() {
			e.mu.Lock()
			delete(e.rooms, g.kv.room)
			e.mu.Unlock()
		}()
		if err := out.accepted(&mwip.KVReceiver{WorkerID: e.cfg.ID, URL: e.cfg.AdvertiseURL, Room: g.kv.room, CachedTokens: cached}); err != nil {
			return err
		}
		select {
		case p = <-ch:
		case <-wait.Done():
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return out.fail(504, mwip.ErrKVTransfer, "kv never arrived", true)
		}
	default:
		if g.kv.source == "" || g.kv.room == "" {
			return out.fail(400, mwip.ErrBadRequest, "no kv source", false)
		}
		var err error
		if p, err = e.fetchKV(wait, g.kv.source, g.kv.room); err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return out.fail(502, mwip.ErrKVTransfer, err.Error(), true)
		}
	}
	e.kv.insert(blocks)
	e.count(func() { e.stats.KVReceived++ })
	timing := &mwip.Timing{PrefillMS: float64(p.PrefillMillis), TransferMS: msSince(start)}
	return e.decode(ctx, out, g, prompt, max(cached, p.CachedOnP), timing, nil, start)
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = crand.Read(b)
	return hex.EncodeToString(b)
}

// decode generates the output: canned text through the real tokenizer,
// stop strings applied to the detokenized text (like vLLM's detokenizer),
// then, when the engine parses (native protocol, vLLM derender), the
// engine-side parser of the model's markup.
func (e *Engine) decode(ctx context.Context, out sink, g *genReq, prompt, cached int, timing *mwip.Timing, kvParams map[string]any, start time.Time) error {
	if g.kv.kind != "prefill" {
		e.count(func() { e.stats.CachedTokens += cached; e.stats.LastCached = cached })
	}
	tk, err := e.vocab.tokenize(ctx, e.vocab.answer(g))
	if err != nil {
		return out.fail(500, mwip.ErrInternal, "mock output: "+err.Error(), true)
	}
	limit := min(g.maxTokens, e.cfg.MaxOutputTokens)
	if limit <= 0 {
		limit = e.cfg.MaxOutputTokens
	}
	eos := uint32(1)
	if len(g.stopTokenIDs) > 0 {
		eos = g.stopTokenIDs[0]
	}
	ids, pieces := tk.ids, tk.pieces
	natural := len(ids)+1 <= limit
	if natural {
		ids = append(append([]uint32(nil), ids...), eos)
		pieces = append(append([]string(nil), pieces...), "")
	} else {
		ids, pieces = ids[:limit], pieces[:limit]
	}
	parser := newParser(parserOpts{reasoning: g.reasoning, tools: g.parseTools, callsOpened: g.callsOpened,
		toolParser: g.toolParser, newID: func() string { return "call_" + randomHex(8) }})
	stop := newStopper(g.stop)
	var interval time.Duration
	if e.cfg.DecodeTPS > 0 {
		interval = time.Duration(float64(time.Second) / e.cfg.DecodeTPS)
	}
	firstAt := time.Now()
	reasoningTokens := 0
	for i, id := range ids {
		if e.cfg.FailAfterTokens > 0 && i >= e.cfg.FailAfterTokens {
			e.count(func() { e.stats.Injected++ })
			return fmt.Errorf("injected stream break after %d tokens", i)
		}
		released, stopped := stop.push(pieces[i])
		last := stopped || i == len(ids)-1
		if last && !stopped {
			released += stop.flush()
		}
		t := tokenOut{id: id, text: released}
		if g.parse {
			if parser.state == stReasoning && !strings.Contains(pieces[i], thinkStart) && !strings.Contains(pieces[i], thinkEnd) {
				reasoningTokens++
			}
			t.delta = parser.feed(released, last)
			if !g.includeReasoning {
				t.delta.reasoning = ""
			}
		}
		if g.topLogprobs >= 0 {
			tok := pieces[i]
			if natural && i == len(ids)-1 {
				tok = e.vocab.tokenText(ctx, eos)
			}
			lp := -0.05 * float64(i%7)
			c := &mwip.LogprobContent{Token: tok, Logprob: lp, Bytes: bytesOf(tok), TopLogprobs: []mwip.TopLogprob{}}
			for k := 0; k < g.topLogprobs; k++ {
				alt := tok
				if k > 0 {
					alt = fmt.Sprintf("%s~%d", tok, k)
				}
				c.TopLogprobs = append(c.TopLogprobs, mwip.TopLogprob{Token: alt, Logprob: lp - float64(k), Bytes: bytesOf(alt)})
			}
			t.logprob = c
		}
		if last {
			f := &finishOut{reason: "length", timing: timing, kvParams: kvParams}
			switch {
			case stopped:
				f.reason, f.stopReason = "stop", stop.matched
				e.count(func() { e.stats.StopMatches++ })
			case natural:
				f.reason = "stop"
			}
			if g.parse && f.reason == "stop" && parser.toolSeen {
				f.reason = "tool_calls"
			}
			n := i + 1
			f.usage = mwip.Usage{PromptTokens: prompt, CompletionTokens: n, TotalTokens: prompt + n,
				PromptTokensDetails: &mwip.PromptTokensDetails{CachedTokens: cached}}
			if g.parse {
				f.usage.CompletionTokensDetails = &mwip.CompletionTokensDetails{ReasoningTokens: reasoningTokens}
			}
			timing.TTFTMS = float64(firstAt.Sub(start).Microseconds()) / 1000
			timing.DecodeMS = msSince(firstAt)
			t.finish = f
			e.count(func() { e.stats.OutputTokens += n; e.stats.LastFinish = f.reason })
		}
		if err := out.token(t); err != nil {
			return err
		}
		if last {
			return nil
		}
		if err := sleepCtx(ctx, interval); err != nil {
			return err
		}
	}
	return nil
}

func bytesOf(s string) []int {
	out := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		out[i] = int(s[i])
	}
	return out
}

func (e *Engine) artifactURI(ref string) string {
	return strings.Replace(e.cfg.AdvertiseURL, "http://", "mock://", 1) + "/mwip/v1/artifacts/" + url.PathEscape(ref)
}

// handleEncode builds native embedding artifacts: one per placement (an
// image, or each kept frame of a video, as the tokenizer planned it).
func (e *Engine) handleEncode(w http.ResponseWriter, r *http.Request) {
	if e.cfg.Role != "encode" {
		writeErr(w, 400, mwip.ErrBadRequest, "not an encode worker", false)
		return
	}
	var req mwip.EncodeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, 400, mwip.ErrBadRequest, err.Error(), false)
		return
	}
	e.count(func() { e.stats.EncodeByProtocol[mwip.EngineMWIP]++ })
	if e.cfg.FailRate > 0 && rand.Float64() < e.cfg.FailRate {
		e.count(func() { e.stats.Injected++ })
		writeErr(w, 500, mwip.ErrInternal, "injected encoder failure", true)
		return
	}
	if !e.admit(req.AttemptID, &running{requestID: req.RequestID, cancel: func() {}}) {
		writeErr(w, 503, mwip.ErrCapacity, "encoder at capacity", true)
		return
	}
	defer e.release(req.AttemptID)
	t0 := time.Now()
	frames := 0
	for _, it := range req.Items {
		frames += len(it.Placements)
	}
	if err := sleepCtx(r.Context(), time.Duration(len(req.Items))*e.cfg.EncodeLatency+time.Duration(frames)*time.Millisecond); err != nil {
		return
	}
	resp := mwip.EncodeResponse{WorkerID: e.cfg.ID}
	for _, it := range req.Items {
		for _, p := range it.Placements {
			e.mu.Lock()
			e.artifacts[p.Reference] = req.RequestID
			e.mu.Unlock()
			resp.Artifacts = append(resp.Artifacts, mwip.EmbeddingArtifact{
				Reference: p.Reference, URI: e.artifactURI(p.Reference), Tokens: p.Length, Encoder: e.cfg.ID,
			})
		}
	}
	resp.EncodeMS = msSince(t0)
	e.count(func() { e.stats.Encoded += len(req.Items); e.stats.ByMode["encode"]++ })
	writeJSON(w, 200, resp)
}

// vllmEncode is a vLLM EC producer: it encodes the content_parts of a
// (non-streaming) generate and returns their ec_transfer_params, keyed by
// mm_hash (the uuid the gateway set).
func (e *Engine) vllmEncode(w http.ResponseWriter, r *http.Request, req *wire.VLLMGenerateRequest, raw json.RawMessage) {
	e.count(func() {
		e.stats.EncodeByProtocol[mwip.EngineVLLM]++
		e.stats.LastProtocol, e.stats.LastRequest = mwip.EngineVLLM, raw
	})
	if len(req.ContentParts) == 0 {
		writeErr(w, 400, mwip.ErrBadRequest, "encode request without content_parts", false)
		return
	}
	if err := sleepCtx(r.Context(), time.Duration(len(req.ContentParts))*e.cfg.EncodeLatency); err != nil {
		return
	}
	host, port := e.hostPort()
	params := map[string]any{}
	for _, p := range req.ContentParts {
		key := p.UUID
		if key == "" {
			key = randomHex(16)
		}
		e.mu.Lock()
		e.artifacts[key] = req.RequestID
		e.mu.Unlock()
		params[key] = map[string]any{
			"modality": strings.TrimSuffix(p.Type, "_url"), "transfer_id": randomHex(16),
			"peer_host": host, "peer_port": port, "uri": e.artifactURI(key),
			"metadata": map[string]any{"source": p.URL},
		}
	}
	e.count(func() { e.stats.Encoded += len(req.ContentParts); e.stats.ByMode["encode"]++ })
	writeJSON(w, 200, wire.VLLMResponse{
		RequestID: req.RequestID, ECTransferParams: params,
		Choices: []wire.VLLMChoice{{Index: 0, FinishReason: "length", TokenIDs: []uint32{0}}},
	})
}

func (e *Engine) handleArtifact(w http.ResponseWriter, r *http.Request) {
	ref := r.PathValue("ref")
	e.mu.Lock()
	_, ok := e.artifacts[ref]
	e.mu.Unlock()
	if !ok {
		writeErr(w, 404, mwip.ErrArtifact, "unknown artifact", false)
		return
	}
	writeJSON(w, 200, map[string]string{"reference": ref, "encoder": e.cfg.ID})
}

// Run serves the worker and sends heartbeats until ctx ends.
func (e *Engine) Run(ctx context.Context) error {
	srv := &http.Server{Addr: e.cfg.Listen, Handler: e.Handler(), ReadHeaderTimeout: 10 * time.Second}
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	e.log.Info("mock engine listening", "addr", e.cfg.Listen, "model", e.cfg.Model, "advertise", e.cfg.AdvertiseURL,
		"normal", e.cfg.Normal, "kv_capacity_tokens", e.cfg.KVCapacityTokens, "prefill_tps", e.cfg.PrefillTPS, "decode_tps", e.cfg.DecodeTPS)
	if e.cfg.GatewayURL != "" {
		go e.heartbeatLoop(ctx)
	}
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	for _, gw := range e.gateways() {
		body, _ := json.Marshal(map[string]string{"worker_id": e.cfg.ID})
		req, _ := http.NewRequest(http.MethodPost, gw+"/mwip/v1/deregister", strings.NewReader(string(body)))
		req.Header.Set("Content-Type", "application/json")
		if e.cfg.ControlToken != "" {
			req.Header.Set("X-ModelWeave-Token", e.cfg.ControlToken)
		}
		if resp, err := (&http.Client{Timeout: 2 * time.Second}).Do(req); err == nil {
			resp.Body.Close()
		}
	}
	sctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return srv.Shutdown(sctx)
}

// gateways returns the heartbeat targets (GatewayURL is comma separated).
func (e *Engine) gateways() []string {
	var out []string
	for _, g := range strings.Split(e.cfg.GatewayURL, ",") {
		if g = strings.TrimSpace(g); g != "" {
			out = append(out, strings.TrimRight(g, "/"))
		}
	}
	return out
}

func (e *Engine) heartbeatLoop(ctx context.Context) {
	interval := e.cfg.HeartbeatInterval
	if interval <= 0 {
		interval = time.Second
	}
	client := &http.Client{Timeout: 2 * time.Second}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		e.sendHeartbeat(ctx, client)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (e *Engine) sendHeartbeat(ctx context.Context, client *http.Client) {
	st, _ := json.Marshal(e.State())
	hb := mwip.Heartbeat{
		Worker: mwip.WorkerInfo{
			ID: e.cfg.ID, Model: e.cfg.Model, Role: e.cfg.Role, URL: e.cfg.AdvertiseURL, Normal: e.cfg.Normal,
			KVCapacityTokens: e.cfg.KVCapacityTokens, MaxRunningRequests: e.cfg.MaxRunning,
			MaxPrefillTokens: e.cfg.MaxPrefillTokens, BlockSize: e.cfg.BlockSize,
		},
		State:    st,
		KVEvents: e.kv.drainEvents(),
	}
	if e.cfg.Role == "prefill" {
		hb.Worker.BootstrapAddr, hb.Worker.EngineID = e.cfg.AdvertiseURL, e.cfg.ID
	}
	body, _ := json.Marshal(hb)
	for _, gw := range e.gateways() {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, gw+"/mwip/v1/heartbeat", strings.NewReader(string(body)))
		if err != nil {
			return
		}
		req.Header.Set("Content-Type", "application/json")
		if e.cfg.ControlToken != "" {
			req.Header.Set("X-ModelWeave-Token", e.cfg.ControlToken)
		}
		resp, err := client.Do(req)
		if err != nil {
			e.log.Debug("heartbeat failed", "gateway", gw, "error", err)
			continue
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			e.log.Warn("heartbeat rejected", "gateway", gw, "status", resp.StatusCode)
		}
	}
}
