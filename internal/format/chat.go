package format

import "github.com/leonyang215/ModelWeave/internal/mwip"

// chat renders OpenAI Chat Completions (DeepSeek flavor: reasoning in
// reasoning_content).
type chat struct{ *state }

type chatFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

type chatToolCall struct {
	Index    *int         `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function chatFunction `json:"function"`
}

type chatDelta struct {
	Role             string         `json:"role,omitempty"`
	Content          *string        `json:"content,omitempty"`
	ReasoningContent *string        `json:"reasoning_content,omitempty"`
	ToolCalls        []chatToolCall `json:"tool_calls,omitempty"`
}

type chatStreamChoice struct {
	Index        int                `json:"index"`
	Delta        chatDelta          `json:"delta"`
	Logprobs     *mwip.ChatLogprobs `json:"logprobs"`
	FinishReason *string            `json:"finish_reason"`
}

type chatChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []chatStreamChoice `json:"choices"`
	Usage   *chatUsage         `json:"usage,omitempty"`
}

type cachedDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type reasoningDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type chatUsage struct {
	PromptTokens            int              `json:"prompt_tokens"`
	CompletionTokens        int              `json:"completion_tokens"`
	TotalTokens             int              `json:"total_tokens"`
	PromptTokensDetails     cachedDetails    `json:"prompt_tokens_details"`
	CompletionTokensDetails reasoningDetails `json:"completion_tokens_details"`
	// DeepSeek API cache accounting.
	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
}

type chatMessage struct {
	Role             string         `json:"role"`
	Content          *string        `json:"content"`
	ReasoningContent *string        `json:"reasoning_content,omitempty"`
	ToolCalls        []chatToolCall `json:"tool_calls,omitempty"`
}

type chatChoice struct {
	Index        int                `json:"index"`
	Message      chatMessage        `json:"message"`
	Logprobs     *mwip.ChatLogprobs `json:"logprobs"`
	FinishReason string             `json:"finish_reason"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
}

func (c *chat) chunk(choice chatStreamChoice, usage *chatUsage) []byte {
	ch := chatChunk{ID: c.meta.ID, Object: "chat.completion.chunk", Created: c.meta.Created, Model: c.meta.Model, Usage: usage}
	if usage == nil {
		ch.Choices = []chatStreamChoice{choice}
	} else {
		ch.Choices = []chatStreamChoice{}
	}
	return sseData(ch)
}

func (c *chat) Start() []byte {
	if !c.stream {
		return nil
	}
	empty := ""
	return c.chunk(chatStreamChoice{Delta: chatDelta{Role: "assistant", Content: &empty}}, nil)
}

func (c *chat) Delta(ev *mwip.GenerateEvent) []byte {
	tools := c.absorb(ev, "call_")
	if !c.stream {
		return nil
	}
	var d chatDelta
	if ev.Reasoning != "" {
		d.ReasoningContent = &ev.Reasoning
	}
	if ev.Content != "" {
		d.Content = &ev.Content
	}
	for _, t := range tools {
		idx := t.call.index
		tc := chatToolCall{Index: &idx, Function: chatFunction{Arguments: t.args}}
		if t.new {
			tc.ID, tc.Type, tc.Function.Name = t.call.id, "function", t.call.name
		}
		d.ToolCalls = append(d.ToolCalls, tc)
	}
	var lp *mwip.ChatLogprobs
	if c.meta.Logprobs && len(ev.Logprobs) > 0 {
		lp = &mwip.ChatLogprobs{Content: ev.Logprobs}
	}
	if d.Content == nil && d.ReasoningContent == nil && len(d.ToolCalls) == 0 && lp == nil {
		return nil
	}
	return c.chunk(chatStreamChoice{Delta: d, Logprobs: lp}, nil)
}

func (c *chat) usage() chatUsage {
	s := c.sum
	return chatUsage{
		PromptTokens: s.PromptTokens, CompletionTokens: s.CompletionTokens, TotalTokens: s.PromptTokens + s.CompletionTokens,
		PromptTokensDetails:     cachedDetails{CachedTokens: s.CachedTokens},
		CompletionTokensDetails: reasoningDetails{ReasoningTokens: s.ReasoningTokens},
		PromptCacheHitTokens:    s.CachedTokens, PromptCacheMissTokens: s.PromptTokens - s.CachedTokens,
	}
}

func (c *chat) Finish(ev *mwip.GenerateEvent) []byte {
	c.close(ev)
	reason := c.sum.FinishReason
	if c.stream {
		out := c.chunk(chatStreamChoice{Delta: chatDelta{}, FinishReason: &reason}, nil)
		if c.meta.IncludeUsage {
			u := c.usage()
			out = append(out, c.chunk(chatStreamChoice{}, &u)...)
		}
		return append(out, "data: [DONE]\n\n"...)
	}
	msg := chatMessage{Role: "assistant"}
	if content := c.content.String(); content != "" || len(c.calls) == 0 {
		msg.Content = &content
	}
	if r := c.reasoning.String(); r != "" {
		msg.ReasoningContent = &r
	}
	for _, call := range c.calls {
		msg.ToolCalls = append(msg.ToolCalls, chatToolCall{ID: call.id, Type: "function", Function: chatFunction{Name: call.name, Arguments: call.args.String()}})
	}
	var lp *mwip.ChatLogprobs
	if c.meta.Logprobs {
		lp = &mwip.ChatLogprobs{Content: c.logprobs}
		if lp.Content == nil {
			lp.Content = []mwip.LogprobContent{}
		}
	}
	return marshal(chatResponse{
		ID: c.meta.ID, Object: "chat.completion", Created: c.meta.Created, Model: c.meta.Model,
		Choices: []chatChoice{{Message: msg, Logprobs: lp, FinishReason: reason}},
		Usage:   c.usage(),
	})
}

func (c *chat) Fail(kind, message string) []byte {
	return sseData(o("error", o("message", message, "type", kind, "param", nil, "code", kind)))
}
