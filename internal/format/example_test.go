package format_test

import (
	"fmt"

	"github.com/leonyang215/ModelWeave/internal/format"
	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// Engine deltas (already parsed by the engine) become an OpenAI chat stream.
func ExampleNew_chatStream() {
	f := format.New(mwip.ProtocolChatCompletions, format.Meta{
		ID: "chatcmpl-1", Model: "deepseek-v4.1", Created: 1700000000, PromptTokens: 12,
		Stream: true, IncludeUsage: true,
	})
	fmt.Print(string(f.Start()))
	fmt.Print(string(f.Delta(&mwip.GenerateEvent{Type: mwip.EventDelta, Reasoning: "Greet.", TokenIDs: []uint32{7}})))
	fmt.Print(string(f.Delta(&mwip.GenerateEvent{Type: mwip.EventDelta, Content: "Hi!", TokenIDs: []uint32{8}})))
	fmt.Print(string(f.Finish(&mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "stop", Usage: &mwip.Usage{PromptTokens: 12, CompletionTokens: 2}})))
	// Output:
	// data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4.1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}
	//
	// data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4.1","choices":[{"index":0,"delta":{"reasoning_content":"Greet."},"logprobs":null,"finish_reason":null}]}
	//
	// data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4.1","choices":[{"index":0,"delta":{"content":"Hi!"},"logprobs":null,"finish_reason":null}]}
	//
	// data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4.1","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}
	//
	// data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4.1","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":1},"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":12}}
	//
	// data: [DONE]
}

// The same deltas aggregated for a non-streaming Anthropic client.
func ExampleNew_messagesJSON() {
	f := format.New(mwip.ProtocolMessages, format.Meta{ID: "msg_1", Model: "deepseek-v4.1", PromptTokens: 12})
	f.Start()
	f.Delta(&mwip.GenerateEvent{Type: mwip.EventDelta, Reasoning: "Greet.", TokenIDs: []uint32{7}})
	f.Delta(&mwip.GenerateEvent{Type: mwip.EventDelta, Content: "Hi!", TokenIDs: []uint32{8}})
	fmt.Println(string(f.Finish(&mwip.GenerateEvent{Type: mwip.EventFinish, Reason: "stop",
		Usage: &mwip.Usage{PromptTokens: 12, CompletionTokens: 2, PromptTokensDetails: &mwip.PromptTokensDetails{CachedTokens: 8}}})))
	// Output:
	// {"id":"msg_1","type":"message","role":"assistant","model":"deepseek-v4.1","content":[{"type":"thinking","thinking":"Greet.","signature":""},{"type":"text","text":"Hi!"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":4,"cache_creation_input_tokens":0,"cache_read_input_tokens":8,"output_tokens":2}}
}
