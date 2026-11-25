package format

import (
	"encoding/json"
	"strings"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// responses renders the OpenAI Responses API.
type responses struct {
	*state
	seq   int
	items []*item
	cur   *item
	echo  map[string]json.RawMessage
}

type item struct {
	index    int
	kind     string // reasoning | message | function_call
	id       string
	text     strings.Builder
	call     *toolCall
	logprobs []mwip.LogprobContent
	done     bool
}

type responsesUsage struct {
	InputTokens         int              `json:"input_tokens"`
	InputTokensDetails  cachedDetails    `json:"input_tokens_details"`
	OutputTokens        int              `json:"output_tokens"`
	OutputTokensDetails reasoningDetails `json:"output_tokens_details"`
	TotalTokens         int              `json:"total_tokens"`
}

type responseObject struct {
	ID                 string          `json:"id"`
	Object             string          `json:"object"`
	CreatedAt          int64           `json:"created_at"`
	Status             string          `json:"status"`
	Background         bool            `json:"background"`
	Error              any             `json:"error"`
	IncompleteDetails  any             `json:"incomplete_details"`
	Instructions       json.RawMessage `json:"instructions"`
	MaxOutputTokens    json.RawMessage `json:"max_output_tokens"`
	Model              string          `json:"model"`
	Output             []any           `json:"output"`
	ParallelToolCalls  json.RawMessage `json:"parallel_tool_calls"`
	PreviousResponseID json.RawMessage `json:"previous_response_id"`
	Reasoning          json.RawMessage `json:"reasoning"`
	Store              bool            `json:"store"`
	Temperature        json.RawMessage `json:"temperature"`
	Text               json.RawMessage `json:"text"`
	ToolChoice         json.RawMessage `json:"tool_choice"`
	Tools              json.RawMessage `json:"tools"`
	TopP               json.RawMessage `json:"top_p"`
	Truncation         json.RawMessage `json:"truncation"`
	Usage              *responsesUsage `json:"usage"`
	User               json.RawMessage `json:"user"`
	Metadata           json.RawMessage `json:"metadata"`
}

// echoDefaults are the values the Responses API reports for request
// fields the client did not set.
var echoDefaults = map[string]string{
	"instructions": "null", "max_output_tokens": "null", "parallel_tool_calls": "true",
	"previous_response_id": "null", "reasoning": `{"effort":null,"summary":null}`, "temperature": "1",
	"text": `{"format":{"type":"text"}}`, "tool_choice": `"auto"`, "tools": "[]", "top_p": "1",
	"truncation": `"disabled"`, "user": "null", "metadata": "{}",
}

func newResponses(st *state) *responses {
	r := &responses{state: st, echo: map[string]json.RawMessage{}}
	var req map[string]json.RawMessage
	_ = json.Unmarshal(st.meta.Request, &req)
	for k, def := range echoDefaults {
		if v, ok := req[k]; ok && string(v) != "null" {
			r.echo[k] = v
		} else {
			r.echo[k] = json.RawMessage(def)
		}
	}
	return r
}

func (r *responses) object(status string, output []any, usage *responsesUsage) responseObject {
	e := r.echo
	ro := responseObject{
		ID: r.meta.ID, Object: "response", CreatedAt: r.meta.Created, Status: status, Model: r.meta.Model,
		Output: output, Usage: usage, Instructions: e["instructions"], MaxOutputTokens: e["max_output_tokens"],
		ParallelToolCalls: e["parallel_tool_calls"], PreviousResponseID: e["previous_response_id"],
		Reasoning: e["reasoning"], Temperature: e["temperature"], Text: e["text"], ToolChoice: e["tool_choice"],
		Tools: e["tools"], TopP: e["top_p"], Truncation: e["truncation"], User: e["user"], Metadata: e["metadata"],
	}
	if ro.Output == nil {
		ro.Output = []any{}
	}
	if status == "incomplete" {
		ro.IncompleteDetails = o("reason", "max_output_tokens")
	}
	return ro
}

func (r *responses) event(name string, fields obj) []byte {
	if !r.stream {
		return nil
	}
	ev := append(o("type", name, "sequence_number", r.seq), fields...)
	r.seq++
	return sseEvent(name, ev)
}

func (r *responses) Start() []byte {
	if !r.stream {
		return nil
	}
	obj := r.object("in_progress", nil, nil)
	out := r.event("response.created", o("response", obj))
	return append(out, r.event("response.in_progress", o("response", obj))...)
}

func (it *item) part() obj {
	if it.kind == "reasoning" {
		return o("type", "reasoning_text", "text", it.text.String())
	}
	lp := it.logprobs
	if lp == nil {
		lp = []mwip.LogprobContent{}
	}
	return o("type", "output_text", "text", it.text.String(), "annotations", []any{}, "logprobs", lp)
}

// json renders the item; in progress items have no content yet.
func (it *item) json() obj {
	status := "completed"
	if !it.done {
		status = "in_progress"
	}
	switch it.kind {
	case "reasoning":
		content := []any{}
		if it.done {
			content = append(content, it.part())
		}
		return o("id", it.id, "type", "reasoning", "summary", []any{}, "content", content, "status", status)
	case "function_call":
		args := ""
		if it.done {
			args = it.call.args.String()
		}
		return o("id", it.id, "type", "function_call", "status", status, "arguments", args, "call_id", it.call.id, "name", it.call.name)
	}
	content := []any{}
	if it.done {
		content = append(content, it.part())
	}
	return o("id", it.id, "type", "message", "status", status, "role", "assistant", "content", content)
}

func (r *responses) open(kind string, call *toolCall) []byte {
	out := r.closeItem()
	prefix := map[string]string{"reasoning": "rs_", "message": "msg_", "function_call": "fc_"}[kind]
	it := &item{index: len(r.items), kind: kind, id: prefix + randomID(24), call: call}
	r.items = append(r.items, it)
	r.cur = it
	out = append(out, r.event("response.output_item.added", o("output_index", it.index, "item", it.json()))...)
	if kind != "function_call" {
		empty := &item{kind: kind}
		out = append(out, r.event("response.content_part.added", o("item_id", it.id, "output_index", it.index, "content_index", 0, "part", empty.part()))...)
	}
	return out
}

func (r *responses) closeItem() []byte {
	it := r.cur
	if it == nil {
		return nil
	}
	r.cur = nil
	it.done = true
	var out []byte
	switch it.kind {
	case "reasoning":
		out = append(out, r.event("response.reasoning_text.done", o("item_id", it.id, "output_index", it.index, "content_index", 0, "text", it.text.String()))...)
		out = append(out, r.event("response.content_part.done", o("item_id", it.id, "output_index", it.index, "content_index", 0, "part", it.part()))...)
	case "message":
		lp := it.logprobs
		if lp == nil {
			lp = []mwip.LogprobContent{}
		}
		out = append(out, r.event("response.output_text.done", o("item_id", it.id, "output_index", it.index, "content_index", 0, "text", it.text.String(), "logprobs", lp))...)
		out = append(out, r.event("response.content_part.done", o("item_id", it.id, "output_index", it.index, "content_index", 0, "part", it.part()))...)
	case "function_call":
		out = append(out, r.event("response.function_call_arguments.done", o("item_id", it.id, "output_index", it.index, "arguments", it.call.args.String()))...)
	}
	return append(out, r.event("response.output_item.done", o("output_index", it.index, "item", it.json()))...)
}

func (r *responses) Delta(ev *mwip.GenerateEvent) []byte {
	tools := r.absorb(ev, "call_")
	var out []byte
	if ev.Reasoning != "" {
		if r.cur == nil || r.cur.kind != "reasoning" {
			out = append(out, r.open("reasoning", nil)...)
		}
		r.cur.text.WriteString(ev.Reasoning)
		out = append(out, r.event("response.reasoning_text.delta", o("item_id", r.cur.id, "output_index", r.cur.index, "content_index", 0, "delta", ev.Reasoning))...)
	}
	if ev.Content != "" {
		if r.cur == nil || r.cur.kind != "message" {
			out = append(out, r.open("message", nil)...)
		}
		r.cur.text.WriteString(ev.Content)
		var lp []mwip.LogprobContent
		if r.meta.Logprobs {
			lp = ev.Logprobs
			r.cur.logprobs = append(r.cur.logprobs, lp...)
		}
		if lp == nil {
			lp = []mwip.LogprobContent{}
		}
		out = append(out, r.event("response.output_text.delta", o("item_id", r.cur.id, "output_index", r.cur.index, "content_index", 0, "delta", ev.Content, "logprobs", lp))...)
	}
	for _, t := range tools {
		if t.new {
			out = append(out, r.open("function_call", t.call)...)
		}
		if t.args == "" {
			continue
		}
		for _, it := range r.items {
			if it.call == t.call {
				out = append(out, r.event("response.function_call_arguments.delta", o("item_id", it.id, "output_index", it.index, "delta", t.args))...)
			}
		}
	}
	return out
}

func (r *responses) usage() *responsesUsage {
	s := r.sum
	return &responsesUsage{
		InputTokens: s.PromptTokens, InputTokensDetails: cachedDetails{CachedTokens: s.CachedTokens},
		OutputTokens: s.CompletionTokens, OutputTokensDetails: reasoningDetails{ReasoningTokens: s.ReasoningTokens},
		TotalTokens: s.PromptTokens + s.CompletionTokens,
	}
}

func (r *responses) Finish(ev *mwip.GenerateEvent) []byte {
	r.close(ev)
	out := r.closeItem()
	status := "completed"
	if r.sum.FinishReason == "length" {
		status = "incomplete"
	}
	output := make([]any, 0, len(r.items))
	for _, it := range r.items {
		it.done = true
		output = append(output, it.json())
	}
	obj := r.object(status, output, r.usage())
	if !r.stream {
		return marshal(obj)
	}
	name := "response.completed"
	if status == "incomplete" {
		name = "response.incomplete"
	}
	return append(out, r.event(name, o("response", obj))...)
}

func (r *responses) Fail(kind, message string) []byte {
	ev := o("type", "error", "code", kind, "message", message, "param", nil, "sequence_number", r.seq)
	r.seq++
	return sseEvent("error", ev)
}
