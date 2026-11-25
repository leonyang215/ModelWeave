package engine

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/wire"
)

// Stream is an open generate response, translated into gateway events:
// accepted (push D), prefill_done (end of a prefill leg), delta, finish and
// error.
type Stream struct {
	role, worker, protocol string
	prefill                bool
	body                   io.ReadCloser
	r                      *bufio.Reader
	pending                []*mwip.GenerateEvent
	finish                 *mwip.GenerateEvent
	usage                  *mwip.Usage
	kvParams               map[string]any
	sg                     struct {
		text   string // output so far (cumulative SGLang streams)
		tokens int
		lps    int // logprob entries seen
	}
	done bool
}

// Next returns the next event. A worker-reported error is returned as an
// *Error (with the event), a broken stream as a transport *Error, and io.EOF
// after the final event.
func (s *Stream) Next() (*mwip.GenerateEvent, error) {
	for {
		if len(s.pending) > 0 {
			ev := s.pending[0]
			s.pending = s.pending[1:]
			if ev.Type == mwip.EventError {
				return ev, &Error{Role: s.role, WorkerID: s.worker, Code: ev.Code, Message: ev.Message, Retriable: ev.Retriable}
			}
			return ev, nil
		}
		if s.done {
			return nil, io.EOF
		}
		data, err := s.read()
		if err != nil {
			if errors.Is(err, io.EOF) && s.finish != nil {
				s.emitFinal()
				continue
			}
			if errors.Is(err, io.EOF) {
				return nil, &Error{Role: s.role, WorkerID: s.worker, Code: mwip.ErrInternal, Message: "stream ended without finish", Retriable: true, Transport: true}
			}
			return nil, transportErr(s.role, s.worker, err)
		}
		if data == "[DONE]" {
			if s.finish == nil {
				return nil, &Error{Role: s.role, WorkerID: s.worker, Code: mwip.ErrInternal, Message: "stream ended without finish_reason", Retriable: true}
			}
			s.emitFinal()
			continue
		}
		if err := s.handle([]byte(data)); err != nil {
			return nil, err
		}
	}
}

// read returns the next NDJSON line (mwip) or SSE data payload.
func (s *Stream) read() (string, error) {
	for {
		line, err := s.r.ReadString('\n')
		trimmed := strings.TrimSpace(line)
		if s.protocol == mwip.EngineMWIP {
			if trimmed != "" {
				return trimmed, nil
			}
		} else if strings.HasPrefix(trimmed, "data:") {
			return strings.TrimSpace(strings.TrimPrefix(trimmed, "data:")), nil
		}
		if err != nil {
			return "", err
		}
	}
}

// emitFinal queues the finish event with the stream's usage; the end of a
// prefill leg is prefill_done.
func (s *Stream) emitFinal() {
	f := s.finish
	f.Usage = s.usage
	if s.usage != nil {
		f.PromptTokens, f.CachedTokens = s.usage.PromptTokens, s.usage.Cached()
	}
	if f.KVTransferParams == nil {
		f.KVTransferParams = s.kvParams
	}
	if s.prefill {
		f.Type = mwip.EventPrefillDone
	}
	s.pending = append(s.pending, f)
	s.done = true
}

func (s *Stream) bad(err error) error {
	return &Error{Role: s.role, WorkerID: s.worker, Code: mwip.ErrInternal, Message: "bad stream chunk: " + err.Error(), Retriable: true}
}

func (s *Stream) handle(data []byte) error {
	switch s.protocol {
	case mwip.EngineVLLM:
		return s.handleVLLM(data)
	case mwip.EngineSGLang:
		return s.handleSGLang(data)
	}
	return s.handleMWIP(data)
}

// handleMWIP takes one native event.
func (s *Stream) handleMWIP(data []byte) error {
	var ev mwip.GenerateEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return s.bad(err)
	}
	switch ev.Type {
	case mwip.EventError:
		s.pending = append(s.pending, &ev)
		s.done = true
	case mwip.EventPrefillDone, mwip.EventFinish:
		s.pending = append(s.pending, &ev)
		s.done = true
	case mwip.EventDelta:
		if !s.prefill {
			s.pending = append(s.pending, &ev)
		}
	default: // accepted, ready
		s.pending = append(s.pending, &ev)
	}
	return nil
}

func (s *Stream) accepted(ext *wire.ChunkExtensions) {
	if ext != nil && ext.Event == mwip.EventAccepted {
		s.pending = append(s.pending, &mwip.GenerateEvent{Type: mwip.EventAccepted, Receiver: ext.Receiver, CachedTokens: ext.CachedTokens, WorkerID: ext.WorkerID})
	}
}

func (s *Stream) handleVLLM(data []byte) error {
	var c wire.VLLMStreamChunk
	if err := json.Unmarshal(data, &c); err != nil {
		return s.bad(err)
	}
	if c.Error != nil {
		s.pending = append(s.pending, chunkError(c.Error, c.ModelWeave))
		s.done = true
		return nil
	}
	s.accepted(c.ModelWeave)
	if c.Usage != nil {
		s.usage = c.Usage
	}
	if c.KVTransferParams != nil {
		s.kvParams = c.KVTransferParams
	}
	for i := range c.Choices {
		ch := &c.Choices[i]
		if ch.Index != 0 {
			continue
		}
		ev := &mwip.GenerateEvent{Type: mwip.EventDelta, TokenIDs: ch.TokenIDs}
		if d := ch.Delta; d != nil {
			if d.Content != nil {
				ev.Content = *d.Content
			}
			switch {
			case d.Reasoning != nil:
				ev.Reasoning = *d.Reasoning
			case d.ReasoningContent != nil:
				ev.Reasoning = *d.ReasoningContent
			}
			ev.ToolCalls = d.ToolCalls
		} else if ch.Text != nil {
			ev.Content = *ch.Text
		}
		if ch.Logprobs != nil {
			ev.Logprobs = ch.Logprobs.Content
		}
		if !s.prefill && (ev.Content != "" || ev.Reasoning != "" || len(ev.ToolCalls) > 0 || len(ev.TokenIDs) > 0) {
			s.pending = append(s.pending, ev)
		}
		if ch.FinishReason != nil {
			if *ch.FinishReason == "abort" || *ch.FinishReason == "error" {
				// The engine dropped the request (we did not ask it to).
				s.pending = append(s.pending, &mwip.GenerateEvent{Type: mwip.EventError, Code: mwip.ErrAborted, Message: "engine finished the request with " + *ch.FinishReason, Retriable: true})
				s.done = true
				return nil
			}
			fin := &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: *ch.FinishReason, StopReason: stopReason(ch.StopReason)}
			if c.ModelWeave != nil {
				fin.Timing = c.ModelWeave.Timing
			}
			s.finish = fin
		}
	}
	return nil
}

func chunkError(ce *wire.ChunkError, ext *wire.ChunkExtensions) *mwip.GenerateEvent {
	ev := &mwip.GenerateEvent{Type: mwip.EventError, Message: ce.Message, Code: mwip.ErrInternal, Retriable: true}
	code := fmt.Sprint(ce.Code)
	switch {
	case ce.Type != "" && strings.Contains(ce.Type, "_") && !strings.HasSuffix(ce.Type, "Error"):
		// A ModelWeave error code in the type field (capacity_exhausted, ...).
		ev.Code = ce.Type
		ev.Retriable = ce.Type != mwip.ErrBadRequest && ce.Type != mwip.ErrArtifact
	case code == "400" || ce.Type == "BadRequestError" || ce.Type == "invalid_request_error":
		ev.Code, ev.Retriable = mwip.ErrBadRequest, false
	}
	if ext != nil {
		ev.WorkerID = ext.WorkerID
	}
	return ev
}

// stopReason renders a matched stop string (vLLM stop_reason, SGLang
// matched); token ids are not stop sequences.
func stopReason(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return ""
}

func (s *Stream) handleSGLang(data []byte) error {
	var c wire.SGLangChunk
	if err := json.Unmarshal(data, &c); err != nil {
		return s.bad(err)
	}
	if c.Error != nil {
		s.pending = append(s.pending, chunkError(c.Error, c.ModelWeave))
		s.done = true
		return nil
	}
	s.accepted(c.ModelWeave)
	if c.Index != 0 {
		return nil
	}
	m := &c.MetaInfo
	// SGLang streams cumulative text and output_ids unless the server runs
	// with --incremental-streaming-output. Both look the same in the first
	// chunk; afterwards cumulative chunks carry every output id so far.
	text, ids := c.Text, c.OutputIDs
	if s.sg.tokens > 0 && len(ids) > 0 && len(ids) == m.CompletionTokens && len(ids) >= s.sg.tokens {
		ids = ids[s.sg.tokens:]
		text = strings.TrimPrefix(text, s.sg.text)
	}
	s.sg.tokens += len(ids)
	s.sg.text += text
	ev := &mwip.GenerateEvent{Type: mwip.EventDelta, Content: text, TokenIDs: ids}
	ev.Logprobs = s.sglangLogprobs(m.OutputTokenLogprobs, m.OutputTopLogprobs, len(ids))
	if !s.prefill && (ev.Content != "" || len(ev.TokenIDs) > 0) {
		s.pending = append(s.pending, ev)
	}
	if m.PromptTokens > 0 || m.CompletionTokens > 0 {
		u := &mwip.Usage{PromptTokens: m.PromptTokens, CompletionTokens: m.CompletionTokens, TotalTokens: m.PromptTokens + m.CompletionTokens}
		if m.CachedTokens > 0 {
			u.PromptTokensDetails = &mwip.PromptTokensDetails{CachedTokens: m.CachedTokens}
		}
		s.usage = u
	}
	if fr := m.FinishReason; fr != nil {
		reason := fr.Type
		switch reason {
		case "":
			reason = "stop"
		case "abort":
			s.pending = append(s.pending, &mwip.GenerateEvent{Type: mwip.EventError, Code: mwip.ErrAborted, Message: "engine aborted the request", Retriable: true})
			s.done = true
			return nil
		}
		fin := &mwip.GenerateEvent{Type: mwip.EventFinish, Reason: reason, StopReason: stopReason(fr.Matched)}
		if c.ModelWeave != nil {
			fin.Timing = c.ModelWeave.Timing
		}
		s.finish = fin
	}
	return nil
}

// sglangLogprobs converts SGLang's [logprob, token_id, token_text] entries
// of the newest n tokens (the lists are cumulative in cumulative mode).
func (s *Stream) sglangLogprobs(tok [][]any, top [][][]any, n int) []mwip.LogprobContent {
	if len(tok) == 0 {
		return nil
	}
	from := 0
	if s.sg.lps > 0 && len(tok) == s.sg.lps+n {
		from = s.sg.lps
	} else if len(tok) > n {
		from = len(tok) - n
	}
	s.sg.lps += len(tok) - from
	var out []mwip.LogprobContent
	for i := from; i < len(tok); i++ {
		entry := tok[i]
		lc := mwip.LogprobContent{TopLogprobs: []mwip.TopLogprob{}}
		if len(entry) > 0 {
			lc.Logprob, _ = entry[0].(float64)
		}
		if len(entry) > 2 {
			lc.Token, _ = entry[2].(string)
		} else if len(entry) > 1 {
			lc.Token = fmt.Sprintf("token_id:%v", entry[1])
		}
		if i < len(top) {
			for _, cand := range top[i] {
				tl := mwip.TopLogprob{}
				if len(cand) > 0 {
					tl.Logprob, _ = cand[0].(float64)
				}
				if len(cand) > 2 {
					tl.Token, _ = cand[2].(string)
				}
				lc.TopLogprobs = append(lc.TopLogprobs, tl)
			}
		}
		out = append(out, lc)
	}
	return out
}

// Close releases the stream (canceling the request is the caller's ctx).
func (s *Stream) Close() { _ = s.body.Close() }
