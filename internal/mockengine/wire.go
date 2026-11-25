package mockengine

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/wire"
)

// tokenOut is one generated token as the wire writers see it.
type tokenOut struct {
	id      uint32
	text    string // detokenized text released by the stop-string check
	delta   parsed // engine-side parse of text (vLLM derender)
	logprob *mwip.LogprobContent
	finish  *finishOut // set on the last token
}

type finishOut struct {
	reason     string // stop | length | tool_calls
	stopReason any    // matched stop string, or nil
	usage      mwip.Usage
	kvParams   map[string]any // vLLM prefill leg of the pull handshake
	timing     *mwip.Timing
}

// sink writes a generate response in one engine protocol, streamed (SSE,
// NDJSON) or as one JSON body.
type sink interface {
	accepted(rcv *mwip.KVReceiver) error
	// prefillDone ends a native prefill leg (vLLM and SGLang prefill legs
	// generate one token instead).
	prefillDone(ev *mwip.GenerateEvent) error
	token(t tokenOut) error
	// fail reports an error: an HTTP error before any output, an error
	// chunk inside a started stream.
	fail(status int, code, msg string, retriable bool) error
}

func newSink(w http.ResponseWriter, g *genReq, worker string) sink {
	base := &out{w: w, rid: g.requestID, worker: worker}
	base.f, _ = w.(http.Flusher)
	switch {
	case g.protocol == mwip.EngineMWIP:
		return &mwipStream{out: base}
	case g.protocol == mwip.EngineSGLang && g.stream:
		return &sglangStream{out: base}
	case g.protocol == mwip.EngineSGLang:
		return &sglangJSON{out: base}
	case g.stream:
		return &vllmStream{out: base, mode: g.outputMode, includeUsage: g.includeUsage}
	}
	return &vllmJSON{out: base, mode: g.outputMode}
}

// out holds a response being written.
type out struct {
	w       http.ResponseWriter
	f       http.Flusher
	rid     string
	worker  string
	started bool
}

func (s *out) data(v any) error {
	if !s.started {
		s.started = true
		s.w.Header().Set("Content-Type", "text/event-stream")
		s.w.WriteHeader(http.StatusOK)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return err
	}
	if s.f != nil {
		s.f.Flush()
	}
	return nil
}

func (s *out) done() error {
	if _, err := s.w.Write([]byte("data: [DONE]\n\n")); err != nil {
		return err
	}
	if s.f != nil {
		s.f.Flush()
	}
	return nil
}

func (s *out) fail(status int, code, msg string, retriable bool) error {
	err := fmt.Errorf("%s: %s", code, msg)
	if !s.started {
		writeErr(s.w, status, code, msg, retriable)
		return err
	}
	_ = s.data(map[string]any{"error": map[string]any{"message": msg, "type": code, "code": status}})
	_ = s.done()
	return err
}

func (s *out) prefillDone(*mwip.GenerateEvent) error {
	return fmt.Errorf("prefill_done is a native protocol event")
}

// --- native: POST /mwip/v1/generate (NDJSON events, engine-parsed) ---------

type mwipStream struct {
	*out
}

func (s *mwipStream) event(ev *mwip.GenerateEvent) error {
	if !s.started {
		s.started = true
		s.w.Header().Set("Content-Type", "application/x-ndjson")
		s.w.WriteHeader(http.StatusOK)
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	if _, err := s.w.Write(append(b, '\n')); err != nil {
		return err
	}
	if s.f != nil {
		s.f.Flush()
	}
	return nil
}

func (s *mwipStream) accepted(rcv *mwip.KVReceiver) error {
	return s.event(&mwip.GenerateEvent{Type: mwip.EventAccepted, Receiver: rcv, CachedTokens: rcv.CachedTokens, WorkerID: s.worker})
}

func (s *mwipStream) prefillDone(ev *mwip.GenerateEvent) error { return s.event(ev) }

func (s *mwipStream) token(t tokenOut) error {
	ev := &mwip.GenerateEvent{Type: mwip.EventDelta, TokenIDs: []uint32{t.id},
		Content: t.delta.content, Reasoning: t.delta.reasoning, ToolCalls: t.delta.calls}
	if t.logprob != nil {
		ev.Logprobs = []mwip.LogprobContent{*t.logprob}
	}
	if err := s.event(ev); err != nil {
		return err
	}
	f := t.finish
	if f == nil {
		return nil
	}
	reason, _ := f.stopReason.(string)
	u := f.usage
	return s.event(&mwip.GenerateEvent{Type: mwip.EventFinish, Reason: f.reason, StopReason: reason, Usage: &u, Timing: f.timing, WorkerID: s.worker})
}

func (s *mwipStream) fail(status int, code, msg string, retriable bool) error {
	err := fmt.Errorf("%s: %s", code, msg)
	if !s.started {
		writeErr(s.w, status, code, msg, retriable)
		return err
	}
	_ = s.event(&mwip.GenerateEvent{Type: mwip.EventError, Code: code, Message: msg, Retriable: retriable, WorkerID: s.worker})
	return err
}

// --- vLLM: POST /inference/v1/generate -------------------------------------

type vllmChoice struct {
	Index        int                    `json:"index"`
	Delta        *wire.VLLMDeltaMessage `json:"delta,omitempty"`
	Text         *string                `json:"text,omitempty"`
	Logprobs     *mwip.ChatLogprobs     `json:"logprobs"`
	FinishReason *string                `json:"finish_reason"`
	StopReason   any                    `json:"stop_reason"`
	TokenIDs     []uint32               `json:"token_ids"`
}

type vllmChunk struct {
	RequestID        string                `json:"request_id"`
	OutputMode       string                `json:"output_mode"`
	Choices          []vllmChoice          `json:"choices"`
	Usage            *mwip.Usage           `json:"usage,omitempty"`
	KVTransferParams map[string]any        `json:"kv_transfer_params,omitempty"`
	ModelWeave       *wire.ChunkExtensions `json:"modelweave,omitempty"`
}

// vllmStream writes GenerateDerenderStreamResponse (or the text/tokens
// variants) chunks, the usage chunk and [DONE].
type vllmStream struct {
	*out
	mode         string
	includeUsage bool
	roleSent     bool
}

func (s *vllmStream) accepted(rcv *mwip.KVReceiver) error {
	return s.data(vllmChunk{RequestID: s.rid, OutputMode: s.mode, Choices: []vllmChoice{},
		ModelWeave: &wire.ChunkExtensions{Event: mwip.EventAccepted, Receiver: rcv, CachedTokens: rcv.CachedTokens, WorkerID: s.worker}})
}

func (s *vllmStream) token(t tokenOut) error {
	ch := vllmChoice{TokenIDs: []uint32{t.id}}
	switch s.mode {
	case "derender":
		d := &wire.VLLMDeltaMessage{ToolCalls: t.delta.calls}
		if !s.roleSent {
			d.Role, s.roleSent = "assistant", true
		}
		if t.delta.reasoning != "" {
			d.Reasoning = &t.delta.reasoning
		}
		if t.delta.content != "" {
			d.Content = &t.delta.content
		}
		ch.Delta = d
	case "text":
		ch.Text = &t.text
	}
	if t.logprob != nil {
		ch.Logprobs = &mwip.ChatLogprobs{Content: []mwip.LogprobContent{*t.logprob}}
	}
	chunk := vllmChunk{RequestID: s.rid, OutputMode: s.mode}
	f := t.finish
	if f != nil {
		ch.FinishReason, ch.StopReason = &f.reason, f.stopReason
		chunk.ModelWeave = &wire.ChunkExtensions{WorkerID: s.worker, Timing: f.timing}
	}
	chunk.Choices = []vllmChoice{ch}
	if err := s.data(chunk); err != nil || f == nil {
		return err
	}
	if s.includeUsage || f.kvParams != nil {
		final := vllmChunk{RequestID: s.rid, OutputMode: s.mode, Choices: []vllmChoice{}, KVTransferParams: f.kvParams}
		if s.includeUsage {
			final.Usage = &f.usage
		}
		if err := s.data(final); err != nil {
			return err
		}
	}
	return s.done()
}

// vllmJSON writes the non-streaming GenerateDerenderResponse.
type vllmJSON struct {
	*out
	mode      string
	content   strings.Builder
	reasoning strings.Builder
	text      strings.Builder
	calls     []*mwip.DeltaToolCall
	ids       []uint32
	logprobs  []mwip.LogprobContent
}

func (s *vllmJSON) accepted(*mwip.KVReceiver) error { return nil }

func (s *vllmJSON) token(t tokenOut) error {
	s.ids = append(s.ids, t.id)
	s.text.WriteString(t.text)
	s.content.WriteString(t.delta.content)
	s.reasoning.WriteString(t.delta.reasoning)
	for _, c := range t.delta.calls {
		if c.Index >= len(s.calls) {
			s.calls = append(s.calls, &mwip.DeltaToolCall{Index: c.Index, ID: c.ID, Type: c.Type, Function: &mwip.DeltaFunction{}})
		}
		if c.Function != nil {
			s.calls[c.Index].Function.Name += c.Function.Name
			s.calls[c.Index].Function.Arguments += c.Function.Arguments
		}
	}
	if t.logprob != nil {
		s.logprobs = append(s.logprobs, *t.logprob)
	}
	f := t.finish
	if f == nil {
		return nil
	}
	choice := map[string]any{"index": 0, "finish_reason": f.reason, "stop_reason": f.stopReason, "token_ids": s.ids}
	switch s.mode {
	case "derender":
		msg := map[string]any{"role": "assistant", "content": s.content.String()}
		if r := s.reasoning.String(); r != "" {
			msg["reasoning"] = r
		}
		if len(s.calls) > 0 {
			msg["tool_calls"] = s.calls
		}
		choice["message"] = msg
	case "text":
		choice["text"] = s.text.String()
	}
	if s.logprobs != nil {
		choice["logprobs"] = map[string]any{"content": s.logprobs}
	}
	writeJSON(s.w, 200, map[string]any{"request_id": s.rid, "output_mode": s.mode, "choices": []any{choice},
		"usage": f.usage, "kv_transfer_params": f.kvParams})
	return nil
}

// --- SGLang: POST /generate -------------------------------------------------

func sglangFinish(f *finishOut, n int) map[string]any {
	switch {
	case f.reason == "length":
		return map[string]any{"type": "length", "length": n}
	case f.stopReason != nil:
		return map[string]any{"type": "stop", "matched": f.stopReason}
	}
	return map[string]any{"type": "stop", "matched": 1}
}

func sglangLogprob(lp *mwip.LogprobContent, id uint32) ([]any, [][]any) {
	tok := []any{lp.Logprob, id, lp.Token}
	var top [][]any
	for _, c := range lp.TopLogprobs {
		top = append(top, []any{c.Logprob, id, c.Token})
	}
	return tok, top
}

// sglangStream writes SGLang's incremental stream (as with
// --incremental-streaming-output): text and output_ids are deltas.
type sglangStream struct {
	*out
	n int
}

func (s *sglangStream) accepted(rcv *mwip.KVReceiver) error {
	return s.data(map[string]any{"text": "", "output_ids": []uint32{}, "meta_info": map[string]any{"id": s.rid},
		"modelweave": wire.ChunkExtensions{Event: mwip.EventAccepted, Receiver: rcv, CachedTokens: rcv.CachedTokens, WorkerID: s.worker}})
}

func (s *sglangStream) token(t tokenOut) error {
	s.n++
	meta := map[string]any{"id": s.rid, "finish_reason": nil, "completion_tokens": s.n}
	if t.logprob != nil {
		tok, top := sglangLogprob(t.logprob, t.id)
		meta["output_token_logprobs"], meta["output_top_logprobs"] = [][]any{tok}, [][][]any{top}
	}
	chunk := map[string]any{"text": t.text, "output_ids": []uint32{t.id}, "meta_info": meta, "index": 0}
	if f := t.finish; f != nil {
		meta["finish_reason"] = sglangFinish(f, s.n)
		meta["prompt_tokens"], meta["cached_tokens"] = f.usage.PromptTokens, f.usage.Cached()
		chunk["modelweave"] = wire.ChunkExtensions{WorkerID: s.worker, Timing: f.timing}
	}
	if err := s.data(chunk); err != nil || t.finish == nil {
		return err
	}
	return s.done()
}

// sglangJSON writes SGLang's non-streaming response.
type sglangJSON struct {
	*out
	text strings.Builder
	ids  []uint32
}

func (s *sglangJSON) accepted(*mwip.KVReceiver) error { return nil }

func (s *sglangJSON) token(t tokenOut) error {
	s.text.WriteString(t.text)
	s.ids = append(s.ids, t.id)
	f := t.finish
	if f == nil {
		return nil
	}
	writeJSON(s.w, 200, map[string]any{"text": s.text.String(), "output_ids": s.ids, "meta_info": map[string]any{
		"id": s.rid, "finish_reason": sglangFinish(f, len(s.ids)), "prompt_tokens": f.usage.PromptTokens,
		"completion_tokens": len(s.ids), "cached_tokens": f.usage.Cached(),
	}})
	return nil
}
