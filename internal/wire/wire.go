// Package wire holds the request and response types of the third-party
// engine protocols ModelWeave adapts to: vLLM's token-in generate API and
// SGLang's native /generate. Fields under "modelweave" are ModelWeave
// extensions that stock engines ignore.
package wire

import (
	"encoding/json"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// Extensions are ModelWeave fields of a vLLM or SGLang request.
type Extensions struct {
	AttemptID  string       `json:"attempt_id"`
	Disagg     *mwip.Disagg `json:"disagg,omitempty"`
	TimeoutMS  int64        `json:"timeout_ms,omitempty"`
	SessionKey string       `json:"session_key,omitempty"`
	Priority   string       `json:"priority,omitempty"`
	// Media describes the collapsed media placeholders in prompt order:
	// the positions each item takes once the engine expands it, as the
	// tokenizer planned them (diagnostics; engines compute their own).
	Media []MediaHint `json:"media,omitempty"`
}

// MediaHint is the tokenizer's plan of one collapsed media item.
type MediaHint struct {
	Digest   string `json:"digest"`
	Modality string `json:"modality"`
	Tokens   int    `json:"tokens"` // positions after expansion (all placements)
	// PadValue is the token the expanded positions hold in the
	// intermediate state.
	PadValue uint32 `json:"pad_value"`
	// Placeholder is what the item was collapsed to in the prompt.
	Placeholder []uint32 `json:"placeholder,omitempty"`
	// Layout is the expanded span run-length encoded as [token, count]
	// pairs, when it is more than one run of PadValue (a video's kept
	// frames with their timestamps).
	Layout [][2]uint32 `json:"layout,omitempty"`
}

// ChunkExtensions are ModelWeave fields of a stream chunk.
type ChunkExtensions struct {
	Event        string           `json:"event,omitempty"` // "accepted": D registered (push)
	Receiver     *mwip.KVReceiver `json:"kv_receiver,omitempty"`
	CachedTokens int              `json:"cached_tokens,omitempty"`
	WorkerID     string           `json:"worker_id,omitempty"`
	Timing       *mwip.Timing     `json:"timing,omitempty"`
}

// ChunkError is an error sent inside a stream (data: {"error": ...}).
type ChunkError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    any    `json:"code,omitempty"`
}

// ---------------------------------------------------------------------------
// vLLM: POST /inference/v1/generate
// (vllm/entrypoints/scale_out/token_in_token_out/protocol.py)
// ---------------------------------------------------------------------------

// VLLMSamplingParams is vLLM's SamplingParams (JSON names).
type VLLMSamplingParams struct {
	MaxTokens                  int             `json:"max_tokens"`
	Temperature                *float64        `json:"temperature,omitempty"`
	TopP                       *float64        `json:"top_p,omitempty"`
	TopK                       *int            `json:"top_k,omitempty"`
	MinP                       *float64        `json:"min_p,omitempty"`
	PresencePenalty            *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty           *float64        `json:"frequency_penalty,omitempty"`
	RepetitionPenalty          *float64        `json:"repetition_penalty,omitempty"`
	Seed                       *uint64         `json:"seed,omitempty"`
	Stop                       []string        `json:"stop,omitempty"`
	StopTokenIDs               []uint32        `json:"stop_token_ids,omitempty"`
	SkipSpecialTokens          *bool           `json:"skip_special_tokens,omitempty"`
	SpacesBetweenSpecialTokens *bool           `json:"spaces_between_special_tokens,omitempty"`
	Logprobs                   *int            `json:"logprobs,omitempty"`
	Detokenize                 *bool           `json:"detokenize,omitempty"`
	StructuredOutputs          json.RawMessage `json:"structured_outputs,omitempty"`
}

// VLLMParseContext is vLLM's ParseContext: what its parsers need.
type VLLMParseContext struct {
	Tools              []json.RawMessage `json:"tools,omitempty"`
	ToolChoice         json.RawMessage   `json:"tool_choice,omitempty"`
	ParallelToolCalls  *bool             `json:"parallel_tool_calls,omitempty"`
	IncludeReasoning   bool              `json:"include_reasoning"`
	ChatTemplateKwargs map[string]any    `json:"chat_template_kwargs,omitempty"`
}

// VLLMStreamOptions is vLLM's StreamOptions.
type VLLMStreamOptions struct {
	IncludeUsage         bool `json:"include_usage"`
	ContinuousUsageStats bool `json:"continuous_usage_stats"`
}

// VLLMContentPart is one raw multimodal input (vLLM content_parts): the
// prompt keeps one unexpanded placeholder for it; uuid keys vLLM's
// multimodal and encoder caches.
type VLLMContentPart struct {
	Type string `json:"type"` // image_url | video_url
	URL  string `json:"url"`
	UUID string `json:"uuid,omitempty"`
}

// VLLMGenerateRequest is vLLM's token-in GenerateRequest.
type VLLMGenerateRequest struct {
	RequestID             string             `json:"request_id"`
	TokenIDs              []uint32           `json:"token_ids"`
	SamplingParams        VLLMSamplingParams `json:"sampling_params"`
	Model                 string             `json:"model,omitempty"`
	ReasoningEnded        *bool              `json:"reasoning_ended,omitempty"`
	ReasoningParserKwargs json.RawMessage    `json:"reasoning_parser_kwargs,omitempty"`
	OutputMode            string             `json:"output_mode"` // derender | text | tokens
	ParseContext          *VLLMParseContext  `json:"parse_context,omitempty"`
	Stream                bool               `json:"stream"`
	StreamOptions         *VLLMStreamOptions `json:"stream_options,omitempty"`
	ContentParts          []VLLMContentPart  `json:"content_parts,omitempty"`
	KVTransferParams      map[string]any     `json:"kv_transfer_params,omitempty"`
	ECTransferParams      map[string]any     `json:"ec_transfer_params,omitempty"`
	ModelWeave            *Extensions        `json:"modelweave,omitempty"`
}

// VLLMDeltaMessage is vLLM's parsed chat delta.
type VLLMDeltaMessage struct {
	Role             string               `json:"role,omitempty"`
	Content          *string              `json:"content,omitempty"`
	Reasoning        *string              `json:"reasoning,omitempty"`
	ReasoningContent *string              `json:"reasoning_content,omitempty"` // older engines
	ToolCalls        []mwip.DeltaToolCall `json:"tool_calls,omitempty"`
}

// VLLMStreamChoice is one choice of a streamed generate chunk.
type VLLMStreamChoice struct {
	Index        int                `json:"index"`
	FinishReason *string            `json:"finish_reason"`
	StopReason   json.RawMessage    `json:"stop_reason,omitempty"` // matched stop string or token id
	TokenIDs     []uint32           `json:"token_ids,omitempty"`
	Delta        *VLLMDeltaMessage  `json:"delta,omitempty"` // output_mode derender
	Text         *string            `json:"text,omitempty"`  // output_mode text
	Logprobs     *mwip.ChatLogprobs `json:"logprobs,omitempty"`
}

// VLLMStreamChunk is one SSE chunk of vLLM's streamed generate response.
type VLLMStreamChunk struct {
	RequestID        string             `json:"request_id"`
	OutputMode       string             `json:"output_mode,omitempty"`
	Choices          []VLLMStreamChoice `json:"choices"`
	Usage            *mwip.Usage        `json:"usage,omitempty"`
	KVTransferParams map[string]any     `json:"kv_transfer_params,omitempty"`
	ModelWeave       *ChunkExtensions   `json:"modelweave,omitempty"`
	Error            *ChunkError        `json:"error,omitempty"`
}

// VLLMResponse is vLLM's non-streaming generate response (the encode leg of
// EPD reads ec_transfer_params from it).
type VLLMResponse struct {
	RequestID        string           `json:"request_id"`
	Choices          []VLLMChoice     `json:"choices"`
	Usage            *mwip.Usage      `json:"usage,omitempty"`
	KVTransferParams map[string]any   `json:"kv_transfer_params,omitempty"`
	ECTransferParams map[string]any   `json:"ec_transfer_params,omitempty"`
	ModelWeave       *ChunkExtensions `json:"modelweave,omitempty"`
}

// VLLMChoice is one choice of a non-streaming response.
type VLLMChoice struct {
	Index        int      `json:"index"`
	FinishReason string   `json:"finish_reason"`
	TokenIDs     []uint32 `json:"token_ids,omitempty"`
}

// ---------------------------------------------------------------------------
// SGLang: POST /generate (sglang/srt/managers/io_struct.py GenerateReqInput)
// ---------------------------------------------------------------------------

// SGLangRequest is SGLang's GenerateReqInput (the fields ModelWeave sets).
type SGLangRequest struct {
	RID            string         `json:"rid"`
	InputIDs       []uint32       `json:"input_ids"`
	SamplingParams map[string]any `json:"sampling_params"`
	Stream         bool           `json:"stream"`
	ReturnLogprob  bool           `json:"return_logprob,omitempty"`
	TopLogprobsNum int            `json:"top_logprobs_num,omitempty"`
	ReturnText     bool           `json:"return_text_in_logprobs,omitempty"`
	ImageData      []string       `json:"image_data,omitempty"`
	VideoData      []string       `json:"video_data,omitempty"`
	BootstrapHost  string         `json:"bootstrap_host,omitempty"`
	BootstrapPort  *int           `json:"bootstrap_port,omitempty"`
	BootstrapRoom  *int64         `json:"bootstrap_room,omitempty"`
	ModelWeave     *Extensions    `json:"modelweave,omitempty"`
}

// SGLangFinishReason is meta_info.finish_reason.
type SGLangFinishReason struct {
	Type    string          `json:"type"` // stop | length | abort
	Matched json.RawMessage `json:"matched,omitempty"`
	Length  int             `json:"length,omitempty"`
}

// SGLangMetaInfo is a chunk's meta_info.
type SGLangMetaInfo struct {
	ID                  string              `json:"id"`
	FinishReason        *SGLangFinishReason `json:"finish_reason"`
	PromptTokens        int                 `json:"prompt_tokens,omitempty"`
	CompletionTokens    int                 `json:"completion_tokens,omitempty"`
	CachedTokens        int                 `json:"cached_tokens,omitempty"`
	OutputTokenLogprobs [][]any             `json:"output_token_logprobs,omitempty"`
	OutputTopLogprobs   [][][]any           `json:"output_top_logprobs,omitempty"`
}

// SGLangChunk is one SSE chunk (or the non-streaming body) of /generate.
type SGLangChunk struct {
	Text       string           `json:"text"`
	OutputIDs  []uint32         `json:"output_ids"`
	Index      int              `json:"index,omitempty"`
	MetaInfo   SGLangMetaInfo   `json:"meta_info"`
	Error      *ChunkError      `json:"error,omitempty"`
	ModelWeave *ChunkExtensions `json:"modelweave,omitempty"`
}
