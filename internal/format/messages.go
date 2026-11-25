package format

import (
	"encoding/json"
	"strings"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// messages renders Anthropic Messages.
type messages struct {
	*state
	blocks []*block
	cur    *block
}

type block struct {
	index int
	kind  string // thinking | text | tool_use
	text  strings.Builder
	call  *toolCall
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	OutputTokens             int `json:"output_tokens"`
}

type anthropicMessage struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Model        string          `json:"model"`
	Content      []any           `json:"content"`
	StopReason   *string         `json:"stop_reason"`
	StopSequence *string         `json:"stop_sequence"`
	Usage        *anthropicUsage `json:"usage"`
}

func (m *messages) message() anthropicMessage {
	return anthropicMessage{ID: m.meta.ID, Type: "message", Role: "assistant", Model: m.meta.Model, Content: []any{}}
}

func (m *messages) Start() []byte {
	if !m.stream {
		return nil
	}
	msg := m.message()
	msg.Usage = &anthropicUsage{InputTokens: m.meta.PromptTokens}
	return sseEvent("message_start", o("type", "message_start", "message", msg))
}

func blockStart(b *block) any {
	switch b.kind {
	case "thinking":
		return o("type", "thinking", "thinking", "", "signature", "")
	case "tool_use":
		return o("type", "tool_use", "id", b.call.id, "name", b.call.name, "input", o())
	}
	return o("type", "text", "text", "")
}

// open closes the current block and starts a new one.
func (m *messages) open(kind string, call *toolCall) []byte {
	out := m.closeBlock()
	b := &block{index: len(m.blocks), kind: kind, call: call}
	m.blocks = append(m.blocks, b)
	m.cur = b
	if m.stream {
		out = append(out, sseEvent("content_block_start", o("type", "content_block_start", "index", b.index, "content_block", blockStart(b)))...)
	}
	return out
}

func (m *messages) closeBlock() []byte {
	if m.cur == nil {
		return nil
	}
	b := m.cur
	m.cur = nil
	if !m.stream {
		return nil
	}
	return sseEvent("content_block_stop", o("type", "content_block_stop", "index", b.index))
}

func (m *messages) delta(b *block, delta obj) []byte {
	if !m.stream {
		return nil
	}
	return sseEvent("content_block_delta", o("type", "content_block_delta", "index", b.index, "delta", delta))
}

func (m *messages) Delta(ev *mwip.GenerateEvent) []byte {
	tools := m.absorb(ev, "toolu_")
	var out []byte
	if ev.Reasoning != "" {
		if m.cur == nil || m.cur.kind != "thinking" {
			out = append(out, m.open("thinking", nil)...)
		}
		m.cur.text.WriteString(ev.Reasoning)
		out = append(out, m.delta(m.cur, o("type", "thinking_delta", "thinking", ev.Reasoning))...)
	}
	if ev.Content != "" {
		if m.cur == nil || m.cur.kind != "text" {
			out = append(out, m.open("text", nil)...)
		}
		m.cur.text.WriteString(ev.Content)
		out = append(out, m.delta(m.cur, o("type", "text_delta", "text", ev.Content))...)
	}
	for _, t := range tools {
		if t.new {
			out = append(out, m.open("tool_use", t.call)...)
		}
		if t.args == "" {
			continue
		}
		b := m.cur
		if b == nil || b.call != t.call {
			for _, x := range m.blocks {
				if x.call == t.call {
					b = x
				}
			}
		}
		if b != nil {
			out = append(out, m.delta(b, o("type", "input_json_delta", "partial_json", t.args))...)
		}
	}
	return out
}

func (m *messages) stop() (reason string, sequence *string) {
	switch m.sum.FinishReason {
	case "tool_calls":
		return "tool_use", nil
	case "length":
		return "max_tokens", nil
	}
	if s := m.finish.StopReason; s != "" {
		return "stop_sequence", &s
	}
	return "end_turn", nil
}

func (m *messages) usage() *anthropicUsage {
	s := m.sum
	return &anthropicUsage{InputTokens: s.PromptTokens - s.CachedTokens, CacheReadInputTokens: s.CachedTokens, OutputTokens: s.CompletionTokens}
}

func (m *messages) Finish(ev *mwip.GenerateEvent) []byte {
	m.close(ev)
	reason, seq := m.stop()
	if m.stream {
		out := m.closeBlock()
		out = append(out, sseEvent("message_delta", o("type", "message_delta", "delta", o("stop_reason", reason, "stop_sequence", seq), "usage", m.usage()))...)
		return append(out, sseEvent("message_stop", o("type", "message_stop"))...)
	}
	msg := m.message()
	for _, b := range m.blocks {
		switch b.kind {
		case "thinking":
			msg.Content = append(msg.Content, o("type", "thinking", "thinking", b.text.String(), "signature", ""))
		case "text":
			msg.Content = append(msg.Content, o("type", "text", "text", b.text.String()))
		case "tool_use":
			msg.Content = append(msg.Content, o("type", "tool_use", "id", b.call.id, "name", b.call.name, "input", json.RawMessage(jsonObject(b.call.args.String()))))
		}
	}
	msg.StopReason, msg.StopSequence, msg.Usage = &reason, seq, m.usage()
	return marshal(msg)
}

func (m *messages) Fail(kind, message string) []byte {
	typ := "api_error"
	switch kind {
	case "overloaded", "no_worker_available":
		typ = "overloaded_error"
	case "invalid_request_error":
		typ = kind
	}
	return sseEvent("error", o("type", "error", "error", o("type", typ, "message", message)))
}
