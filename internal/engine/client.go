// Package engine adapts the intermediate state to the engine protocol of a
// model's workers, chosen at startup:
//
//	mwip    POST /mwip/v1/generate   the intermediate state as is (NDJSON events)
//	vllm    POST /inference/v1/generate  token-in, output_mode "derender" (SSE)
//	sglang  POST /generate               input_ids (SSE)
//
// Every generate call streams, whatever the client asked for. The encode
// stage is unary (mwip: /mwip/v1/encode, vllm: a non-streaming generate on an
// EC-producer instance); SGLang runs it inside the engine. PD fields go into
// each engine's own request fields.
package engine

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"time"

	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/registry"
	"github.com/leonyang215/ModelWeave/internal/wire"
)

// Error is a failed worker call.
type Error struct {
	Role      string
	WorkerID  string
	Status    int    // HTTP status, 0 for transport errors and stream events
	Code      string // MWIP error code
	Message   string
	Retriable bool
	// Transport is true when the worker could not be reached or the
	// connection broke; it counts against the worker's health.
	Transport bool
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s worker %s: %s: %s", e.Role, e.WorkerID, e.Code, e.Message)
}

// IsCode reports whether err is an *Error with the given code.
func IsCode(err error, code string) bool {
	var e *Error
	return errors.As(err, &e) && e.Code == code
}

// Stage is one engine call of an attempt: the intermediate state plus how
// this call takes part in the execution.
type Stage struct {
	Render      *mwip.RenderResult
	AttemptID   string // request_id of the call; P and D share it
	EngineModel string
	Disagg      mwip.Disagg
	// Media is the request's media with encode-stage results attached
	// (embedding artifacts for mwip); defaults to Render.Multimodal.
	Media *mwip.Multimodal
	// ECParams are the vLLM encoders' ec_transfer_params (EPD).
	ECParams   map[string]any
	TimeoutMS  int64
	SessionKey string
	Priority   string
}

func (s *Stage) media() *mwip.Multimodal {
	if s.Media != nil {
		return s.Media
	}
	return s.Render.Multimodal
}

func (s *Stage) role(spec registry.Spec) string {
	if s.Disagg.Role != "" {
		return s.Disagg.Role
	}
	return string(spec.Role)
}

// EncodeStage asks an encoder for the embeddings of media items.
type EncodeStage struct {
	RequestID   string
	AttemptID   string
	Model       string
	EngineModel string
	Items       []mwip.MediaItem
	// PlaceholderTokenID is the model token of one collapsed image, for
	// items without their own placeholder (vllm).
	PlaceholderTokenID uint32
	TimeoutMS          int64
}

// EncodeResult is what an encoder returned.
type EncodeResult struct {
	Artifacts []mwip.EmbeddingArtifact // mwip: one per placement
	ECParams  map[string]any           // vllm: ec_transfer_params by mm_hash
	EncodeMS  float64
}

// Client is the engine client.
type Client struct {
	http *http.Client
}

// New returns a client with pooled keep-alive connections.
func New() *Client {
	dialer := &net.Dialer{Timeout: 3 * time.Second, KeepAlive: 30 * time.Second}
	tr := &http.Transport{
		DialContext:         dialer.DialContext,
		MaxIdleConns:        1024,
		MaxIdleConnsPerHost: 256,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  true,
	}
	return &Client{http: &http.Client{Transport: tr}}
}

// HTTP exposes the underlying client (used by state pollers).
func (c *Client) HTTP() *http.Client { return c.http }

// EncodeInGateway reports whether the gateway runs the encode stage for a
// protocol. SGLang encodes inside the engine: a language-only prefill
// server dispatches media to its own --encoder-urls.
func EncodeInGateway(protocol string) bool { return protocol != mwip.EngineSGLang }

// Generate starts a streaming generate call on a worker.
func (c *Client) Generate(ctx context.Context, protocol string, spec registry.Spec, st *Stage) (*Stream, error) {
	role := st.role(spec)
	var (
		path, accept string
		body         any
		err          error
	)
	switch protocol {
	case mwip.EngineVLLM:
		path, accept = "/inference/v1/generate", "text/event-stream"
		body, err = toVLLM(st)
	case mwip.EngineSGLang:
		path, accept = "/generate", "text/event-stream"
		body, err = toSGLang(st)
	default:
		path, accept = "/mwip/v1/generate", "application/x-ndjson"
		body = toMWIP(st)
	}
	if err != nil {
		return nil, &Error{Role: role, WorkerID: spec.ID, Code: mwip.ErrBadRequest, Message: err.Error()}
	}
	resp, err := c.post(ctx, spec.URL+path, accept, st.AttemptID, body)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, transportErr(role, spec.ID, err)
	}
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		resp.Body.Close()
		return nil, classifyStatus(role, spec.ID, resp.StatusCode, b)
	}
	return &Stream{
		role: role, worker: spec.ID, protocol: protocol, prefill: role == "prefill",
		body: resp.Body, r: bufio.NewReaderSize(resp.Body, 64<<10),
	}, nil
}

// Encode runs the encode stage of media items on an encoder worker.
func (c *Client) Encode(ctx context.Context, protocol string, spec registry.Spec, st *EncodeStage) (*EncodeResult, error) {
	var (
		path string
		body any
	)
	switch protocol {
	case mwip.EngineVLLM:
		req, err := vllmEncodeRequest(st)
		if err != nil {
			return nil, &Error{Role: "encode", WorkerID: spec.ID, Code: mwip.ErrBadRequest, Message: err.Error()}
		}
		path, body = "/inference/v1/generate", req
	case mwip.EngineSGLang:
		return nil, &Error{Role: "encode", WorkerID: spec.ID, Code: mwip.ErrBadRequest,
			Message: "sglang encodes inside the engine (--language-only with --encoder-urls)"}
	default:
		path, body = "/mwip/v1/encode", &mwip.EncodeRequest{
			RequestID: st.RequestID, AttemptID: st.AttemptID, Model: st.Model, Items: st.Items, TimeoutMS: st.TimeoutMS,
		}
	}
	resp, err := c.post(ctx, spec.URL+path, "application/json", st.AttemptID, body)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, transportErr("encode", spec.ID, err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, transportErr("encode", spec.ID, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, classifyStatus("encode", spec.ID, resp.StatusCode, b)
	}
	bad := func(msg string) error {
		return &Error{Role: "encode", WorkerID: spec.ID, Code: mwip.ErrInternal, Message: msg, Retriable: true}
	}
	if protocol == mwip.EngineVLLM {
		var vr wire.VLLMResponse
		if err := json.Unmarshal(b, &vr); err != nil {
			return nil, bad("bad encode response: " + err.Error())
		}
		if len(vr.ECTransferParams) == 0 {
			return nil, bad("encoder returned no ec_transfer_params")
		}
		return &EncodeResult{ECParams: vr.ECTransferParams}, nil
	}
	var er mwip.EncodeResponse
	if err := json.Unmarshal(b, &er); err != nil {
		return nil, bad("bad encode response: " + err.Error())
	}
	return &EncodeResult{Artifacts: er.Artifacts, EncodeMS: er.EncodeMS}, nil
}

// Abort asks a worker to stop a request; it is best effort (the gateway
// also closes the connection).
func (c *Client) Abort(ctx context.Context, protocol string, spec registry.Spec, requestID string) error {
	var (
		path string
		body any
	)
	switch protocol {
	case mwip.EngineVLLM:
		path, body = "/inference/v1/abort_requests", map[string]any{"request_ids": []string{requestID}}
	case mwip.EngineSGLang:
		path, body = "/abort_request", map[string]any{"rid": requestID}
	default:
		path, body = "/mwip/v1/abort", mwip.AbortRequest{RequestID: requestID}
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	resp, err := c.post(ctx, spec.URL+path, "application/json", requestID, body)
	if err != nil {
		return err
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return nil
}

func (c *Client) post(ctx context.Context, url, accept, requestID string, body any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", accept)
	req.Header.Set("X-Request-Id", requestID)
	req.Header.Set("MWIP-Version", mwip.Version)
	return c.http.Do(req)
}

func classifyStatus(role, worker string, status int, body []byte) *Error {
	e := &Error{Role: role, WorkerID: worker, Status: status}
	var mw mwip.ErrorBody
	var oa struct {
		Error   *wire.ChunkError `json:"error"`
		Message string           `json:"message"`
	}
	switch {
	case json.Unmarshal(body, &mw) == nil && mw.Code != "":
		e.Code, e.Message, e.Retriable = mw.Code, mw.Message, mw.Retriable
	case json.Unmarshal(body, &oa) == nil && oa.Error != nil:
		e.Message = oa.Error.Message
	case oa.Message != "":
		e.Message = oa.Message
	default:
		e.Message = string(bytes.TrimSpace(body))
	}
	switch {
	case status == http.StatusTooManyRequests || status == http.StatusServiceUnavailable:
		if e.Code == "" {
			e.Code = mwip.ErrCapacity
		}
		e.Retriable = true
	case status >= 400 && status < 500:
		if e.Code == "" {
			e.Code = mwip.ErrBadRequest
		}
	case status >= 500:
		if e.Code == "" {
			e.Code = mwip.ErrInternal
		}
		e.Retriable = true
	}
	return e
}

func transportErr(role, worker string, err error) *Error {
	return &Error{Role: role, WorkerID: worker, Code: "unavailable", Message: err.Error(), Retriable: true, Transport: true}
}

// placeholder returns the tokens an item collapses to for engines that
// expand media themselves: the model's placeholder from the tokenizer, or,
// from tokenizers that predate per-item placeholders, the image placeholder
// token.
func placeholder(m *mwip.Multimodal, it *mwip.MediaItem) []uint32 {
	if len(it.Placeholder) > 0 {
		return it.Placeholder
	}
	if it.Modality == "image" && m.PlaceholderTokenID != 0 {
		return []uint32{m.PlaceholderTokenID}
	}
	return nil
}

// Collapsible reports whether engines that expand media themselves (vLLM,
// SGLang) can take the request's media: every item needs a placeholder of
// the model. DeepSeek V4.1 has none for videos, for example.
func Collapsible(m *mwip.Multimodal) error {
	if m == nil {
		return nil
	}
	for i := range m.Items {
		if it := &m.Items[i]; len(placeholder(m, it)) == 0 {
			return fmt.Errorf("the model has no placeholder for %s input: it needs the mwip engine protocol", it.Modality)
		}
	}
	return nil
}

// collapse undoes the media expansion for engines that expand media
// themselves: each item's span becomes the model's placeholder. It returns
// the token list and the items in prompt order.
func collapse(r *mwip.RenderResult, m *mwip.Multimodal) ([]uint32, []mwip.MediaItem, error) {
	if m == nil || len(m.Items) == 0 {
		return r.TokenIDs, nil, nil
	}
	if err := Collapsible(m); err != nil {
		return nil, nil, err
	}
	items := append([]mwip.MediaItem(nil), m.Items...)
	sort.Slice(items, func(i, j int) bool { return items[i].Span.Offset < items[j].Span.Offset })
	out := make([]uint32, 0, len(r.TokenIDs))
	pos := 0
	for i := range items {
		it := &items[i]
		s := it.Span
		if s.Offset < pos || s.Offset+s.Length > len(r.TokenIDs) {
			return nil, nil, fmt.Errorf("media item %d spans [%d,%d) outside the prompt", it.Index, s.Offset, s.Offset+s.Length)
		}
		out = append(out, r.TokenIDs[pos:s.Offset]...)
		it.Placeholder = placeholder(m, it)
		out = append(out, it.Placeholder...)
		pos = s.Offset + s.Length
	}
	out = append(out, r.TokenIDs[pos:]...)
	return out, items, nil
}

// mediaHints describe the collapsed items: their placeholders and expanded
// spans (diagnostics for engines; test engines rebuild the expanded prompt
// from them).
func mediaHints(tokens []uint32, items []mwip.MediaItem) []wire.MediaHint {
	var out []wire.MediaHint
	for _, it := range items {
		h := wire.MediaHint{Digest: it.Digest, Modality: it.Modality, Tokens: it.Span.Length, Placeholder: it.Placeholder}
		if len(it.Placements) > 0 {
			h.PadValue = it.Placements[0].PadValue
		}
		s := it.Span
		if s.Offset >= 0 && s.Offset+s.Length <= len(tokens) {
			if runs := runLength(tokens[s.Offset : s.Offset+s.Length]); len(runs) > 1 {
				h.Layout = runs
			}
		}
		out = append(out, h)
	}
	return out
}

// runLength encodes tokens as [token, count] pairs.
func runLength(tokens []uint32) [][2]uint32 {
	var out [][2]uint32
	for _, t := range tokens {
		if n := len(out); n > 0 && out[n-1][0] == t {
			out[n-1][1]++
			continue
		}
		out = append(out, [2]uint32{t, 1})
	}
	return out
}

func extensions(st *Stage, items []mwip.MediaItem) *wire.Extensions {
	d := st.Disagg
	return &wire.Extensions{
		AttemptID: st.AttemptID, Disagg: &d, TimeoutMS: st.TimeoutMS,
		SessionKey: st.SessionKey, Priority: st.Priority, Media: mediaHints(st.Render.TokenIDs, items),
	}
}
