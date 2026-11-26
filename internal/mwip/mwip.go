// Package mwip holds the in-memory form of the ModelWeave contracts
// (docs/protocol.md):
//
//   - the intermediate state the tokenizer returns (JSON): the conversation
//     rendered into one token list by the model's prompt format (DeepSeek,
//     Qwen, any chat template), images and videos expanded to pad-value
//     positions, with the options engines need to generate and parse;
//   - generate requests and the events of their output streams, which the
//     gateway sends to and reads from engines;
//   - encode requests and KV events.
//
// On the wire, engines speak mwip, the gRPC protocol defined in
// proto/modelweave/mwip/v1 (Go code in api/mwip/v1); proto.go converts
// between the two forms.
package mwip

import "encoding/json"

// API protocols understood by the tokenizer.
const (
	ProtocolChatCompletions = "chat_completions" // OpenAI /v1/chat/completions
	ProtocolResponses       = "responses"        // OpenAI /v1/responses
	ProtocolMessages        = "messages"         // Anthropic /v1/messages
)

// ---------------------------------------------------------------------------
// Tokenizer contract: the intermediate state
// ---------------------------------------------------------------------------

// RenderRequest asks the tokenizer to convert an API request.
type RenderRequest struct {
	RequestID     string          `json:"request_id"`
	Protocol      string          `json:"protocol"`
	Model         string          `json:"model,omitempty"` // tokenizer-side model name; defaults to body.model
	Body          json.RawMessage `json:"body"`
	ContextLength int             `json:"context_length,omitempty"`
}

// RenderResult is the intermediate state.
type RenderResult struct {
	RequestID        string           `json:"request_id"`
	Protocol         string           `json:"protocol"` // client API protocol
	Model            string           `json:"model"`
	ModelVersion     string           `json:"model_version"`
	PromptTokens     int              `json:"prompt_tokens"` // == len(TokenIDs)
	TokenIDs         []uint32         `json:"token_ids"`
	Multimodal       *Multimodal      `json:"multimodal,omitempty"`
	Conversation     Conversation     `json:"conversation"`
	InferenceOptions InferenceOptions `json:"inference_options"`
	ParsingOptions   ParsingOptions   `json:"parsing_options"`
	Stream           bool             `json:"stream"` // the client asked for a stream
	ResponseOptions  ResponseOptions  `json:"response_options"`
	RenderArtifact   RenderArtifact   `json:"render_artifact"`
	TimingUS         map[string]int   `json:"timing_us,omitempty"`
}

// Conversation is the conversation without its messages: what engine-side
// parsers need.
type Conversation struct {
	ThinkingMode    bool              `json:"thinking_mode"`
	ReasoningEffort string            `json:"reasoning_effort,omitempty"`
	Tools           []json.RawMessage `json:"tools,omitempty"` // OpenAI function tools
	ToolChoice      string            `json:"tool_choice"`     // auto | none | required
	ResponseFormat  string            `json:"response_format"` // text | json_object | json_schema | regex
	// ChatTemplateKwargs are the template variables the prompt was rendered
	// with (thinking switches), for engines whose parsers read them.
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

// OutputConstraint is a constraint engines may enforce by constrained
// decoding, as kind and schema string.
type OutputConstraint struct {
	Kind   string `json:"kind"` // json | regex | tool_call | tool_call_required
	Schema string `json:"schema"`
}

// InferenceOptions are the sampling options, max_tokens resolved.
type InferenceOptions struct {
	MaxTokens         int               `json:"max_tokens"`
	Temperature       *float64          `json:"temperature,omitempty"`
	TopP              *float64          `json:"top_p,omitempty"`
	TopK              *int              `json:"top_k,omitempty"`
	MinP              *float64          `json:"min_p,omitempty"`
	PresencePenalty   *float64          `json:"presence_penalty,omitempty"`
	FrequencyPenalty  *float64          `json:"frequency_penalty,omitempty"`
	RepetitionPenalty *float64          `json:"repetition_penalty,omitempty"`
	Seed              *uint64           `json:"seed,omitempty"`
	Logprobs          bool              `json:"logprobs"`
	TopLogprobs       *int              `json:"top_logprobs,omitempty"`
	StopTokenIDs      []uint32          `json:"stop_token_ids"`
	OutputConstraint  *OutputConstraint `json:"output_constraint,omitempty"`
}

// Output parsers named in ParsingOptions.
const (
	ParserDeepseekV4 = "deepseek_v4" // </think> reasoning, DSML tool calls
	ParserQwen3      = "qwen3"       // <think>...</think> reasoning
	ParserHermes     = "hermes"      // <tool_call>{"name": ..., "arguments": ...}</tool_call>
)

// ParsingOptions tell engines how to parse their output.
type ParsingOptions struct {
	ReasoningParser      string   `json:"reasoning_parser,omitempty"` // deepseek_v4 | qwen3; empty: no reasoning
	ToolCallParser       string   `json:"tool_call_parser,omitempty"` // deepseek_v4 | hermes
	ParseToolCalls       bool     `json:"parse_tool_calls"`
	StartsInToolCalls    bool     `json:"starts_in_tool_calls"`
	ParseJSONOutput      bool     `json:"parse_json_output"`
	ThinkingInitialStage string   `json:"thinking_initial_stage,omitempty"` // reasoning_start | reasoning_continuation | content_continuation
	StopSequences        []string `json:"stop_sequences,omitempty"`
}

// StartsInReasoning reports whether output begins inside the reasoning.
func (p ParsingOptions) StartsInReasoning() bool {
	return p.ThinkingInitialStage == "reasoning_start" || p.ThinkingInitialStage == "reasoning_continuation"
}

// ResponseOptions are response settings of the client protocol.
type ResponseOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// Span is a token range.
type Span struct {
	Offset int `json:"offset"`
	Length int `json:"length"`
}

// TokenLayout marks structural ranges of the prompt.
type TokenLayout struct {
	CacheablePrefix  Span `json:"cacheable_prefix"`
	GenerationPrimer Span `json:"generation_primer"`
}

// Segment is the token range of one rendered message.
type Segment struct {
	Role   string `json:"role"`
	Offset int    `json:"offset"`
	Length int    `json:"length"`
}

// RenderArtifact is the by-product of rendering.
type RenderArtifact struct {
	Version            int         `json:"version"`
	Encoding           string      `json:"encoding"`
	TokenLayout        TokenLayout `json:"token_layout"`
	Segments           []Segment   `json:"segments,omitempty"`
	PrefixContinuation bool        `json:"prefix_continuation,omitempty"`
}

// Multimodal lists the images and videos of a prompt.
type Multimodal struct {
	PadShift uint32      `json:"pad_shift"` // every pad value is at least this
	Items    []MediaItem `json:"items"`
}

// HasVideo reports whether any item is a video.
func (m *Multimodal) HasVideo() bool {
	if m == nil {
		return false
	}
	for _, it := range m.Items {
		if it.Modality == "video" {
			return true
		}
	}
	return false
}

// Clone returns a deep copy (the gateway attaches embeddings per request).
func (m *Multimodal) Clone() *Multimodal {
	if m == nil {
		return nil
	}
	c := *m
	c.Items = make([]MediaItem, len(m.Items))
	for i, it := range m.Items {
		c.Items[i] = it
		c.Items[i].Placements = append([]Placement(nil), it.Placements...)
		c.Items[i].Plan.FrameIndices = append([]uint64(nil), it.Plan.FrameIndices...)
		c.Items[i].Plan.Timestamps = append([]float64(nil), it.Plan.Timestamps...)
		if it.Embedding != nil {
			e := *it.Embedding
			c.Items[i].Embedding = &e
		}
	}
	return &c
}

// MediaItem is one image or video.
type MediaItem struct {
	Index      int                `json:"index"`
	Modality   string             `json:"modality"` // image | video
	Digest     string             `json:"digest"`   // sha256 hex of the media bytes
	Source     MediaSource        `json:"source"`
	Plan       MediaPlan          `json:"plan"`
	PlanID     string             `json:"plan_id"`
	Placements []Placement        `json:"placements"`
	Embedding  *EmbeddingLocation `json:"embedding,omitempty"`
}

// MediaSource locates a media item.
type MediaSource struct {
	Kind string `json:"kind"` // url | base64
	URL  string `json:"url"`
	MIME string `json:"mime,omitempty"`
}

// MediaPlan is an authoritative execution plan, not a processor hint.
type MediaPlan struct {
	Processor         string    `json:"processor"`
	Width             int       `json:"width"`
	Height            int       `json:"height"`
	Tokens            int       `json:"tokens"`  // per image, or per kept frame
	NFrames           int       `json:"nframes"` // image: 1; video: exact padded index count
	PatchSize         int       `json:"patch_size"`
	SpatialMergeSize  int       `json:"spatial_merge_size"`
	TemporalPatchSize int       `json:"temporal_patch_size"`
	FrameIndices      []uint64  `json:"frame_indices,omitempty"`
	Timestamps        []float64 `json:"timestamps,omitempty"`
}

// Placement is the token range of one image or one kept video frame.
type Placement struct {
	Offset          int      `json:"offset"`
	Length          int      `json:"length"`
	PadValue        uint32   `json:"pad_value"`
	Reference       string   `json:"reference"` // image digest, or "{video digest}_kf_{k:04}"
	Modality        string   `json:"modality"`  // image | frame
	Timestamp       *float64 `json:"timestamp,omitempty"`
	EmbeddingOffset int      `json:"embedding_offset"` // row in the item's packed tensor
}

// ---------------------------------------------------------------------------
// Output types shared by every engine protocol
// ---------------------------------------------------------------------------

// Usage is the token accounting of a request (OpenAI usage).
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens,omitempty"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails details prompt usage.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// CompletionTokensDetails details completion usage.
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// Cached returns the cached prompt tokens.
func (u *Usage) Cached() int {
	if u == nil || u.PromptTokensDetails == nil {
		return 0
	}
	return u.PromptTokensDetails.CachedTokens
}

// Reasoning returns the reasoning tokens.
func (u *Usage) Reasoning() int {
	if u == nil || u.CompletionTokensDetails == nil {
		return 0
	}
	return u.CompletionTokensDetails.ReasoningTokens
}

// DeltaFunction is a tool call fragment.
type DeltaFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// DeltaToolCall is one tool call fragment (OpenAI streaming semantics: the
// first fragment of a call has id, type and name).
type DeltaToolCall struct {
	Index    int            `json:"index"`
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type,omitempty"`
	Function *DeltaFunction `json:"function,omitempty"`
}

// TopLogprob is one candidate of a position.
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// LogprobContent is the logprob of one output token (OpenAI shape).
type LogprobContent struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes,omitempty"`
	TopLogprobs []TopLogprob `json:"top_logprobs"`
}

// ChatLogprobs is ChatCompletionLogProbs.
type ChatLogprobs struct {
	Content []LogprobContent `json:"content"`
}

// Timing reports engine-side latencies in milliseconds.
type Timing struct {
	QueueMS    float64 `json:"queue_ms,omitempty"`
	EncodeMS   float64 `json:"encode_ms,omitempty"`
	PrefillMS  float64 `json:"prefill_ms,omitempty"`
	TransferMS float64 `json:"transfer_ms,omitempty"`
	TTFTMS     float64 `json:"ttft_ms,omitempty"`
	DecodeMS   float64 `json:"decode_ms,omitempty"`
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

// Execution modes of one engine request.
const (
	ModeNormal = "normal" // one worker runs prefill and decode (no PD split)
	ModeSplit  = "split"  // prefill on P, decode on D, KV moves P -> D
)

// PD handshakes for split mode.
const (
	// HandshakePush: D registers and acknowledges with its KV receiver; P
	// is sent afterwards and pushes KV to D (Mooncake-style order).
	HandshakePush = "push"
	// HandshakeBootstrap: P and D get the request concurrently with a
	// shared room; D fetches KV from P's bootstrap server.
	HandshakeBootstrap = "bootstrap"
	// HandshakePull: P runs first and returns a KV handle that D uses to
	// pull the KV.
	HandshakePull = "pull"
)

// Peer identifies one side of a KV transfer.
type Peer struct {
	WorkerID      string `json:"worker_id"`
	Target        string `json:"target"` // its mwip target
	BootstrapAddr string `json:"bootstrap_addr,omitempty"`
	EngineID      string `json:"engine_id,omitempty"`
}

// KVReceiver is advertised by a registered decode worker (push).
type KVReceiver struct {
	WorkerID string `json:"worker_id"`
	URL      string `json:"url"`
	Room     string `json:"room"`
	// CachedTokens is how much of the prompt D already holds; P only has to
	// transfer KV after this point.
	CachedTokens int `json:"cached_tokens,omitempty"`
}

// KVHandle locates KV that P keeps for D (pull).
type KVHandle struct {
	WorkerID string `json:"worker_id"`
	URL      string `json:"url"`
	Room     string `json:"room"`
	Tokens   int    `json:"tokens"`
}

// Disagg describes the request's role in its execution.
type Disagg struct {
	Mode          string      `json:"mode"`           // normal | split
	Role          string      `json:"role,omitempty"` // prefill | decode (split)
	Handshake     string      `json:"handshake,omitempty"`
	Room          string      `json:"room,omitempty"`           // pairing token (== attempt id)
	BootstrapRoom int64       `json:"bootstrap_room,omitempty"` // numeric room, for KV connectors
	Prefill       *Peer       `json:"prefill,omitempty"`
	Decode        *Peer       `json:"decode,omitempty"`
	Receiver      *KVReceiver `json:"kv_receiver,omitempty"` // push: given to P
	Handle        *KVHandle   `json:"kv_handle,omitempty"`   // pull: given to D
}

// ---------------------------------------------------------------------------
// Generate: modelweave.mwip.v1.Engine/Generate
// ---------------------------------------------------------------------------

// GenerateRequest is the intermediate state with execution fields
// (mwipv1.GenerateRequest in memory). The engine parses its own output
// (reasoning, tool calls, stop sequences, structured output) with
// ParsingOptions and Conversation, and answers with a stream of events.
type GenerateRequest struct {
	RequestID        string           `json:"request_id"` // attempt id, shared by P and D
	Model            string           `json:"model,omitempty"`
	TokenIDs         []uint32         `json:"token_ids"`
	Multimodal       *Multimodal      `json:"multimodal,omitempty"`
	Conversation     Conversation     `json:"conversation"`
	InferenceOptions InferenceOptions `json:"inference_options"`
	ParsingOptions   ParsingOptions   `json:"parsing_options"`
	Disagg           Disagg           `json:"disagg"`
	TimeoutMS        int64            `json:"timeout_ms,omitempty"`
	SessionKey       string           `json:"session_key,omitempty"`
	Priority         string           `json:"priority,omitempty"`
}

// Event types of the engine output stream.
const (
	EventAccepted    = "accepted"     // D registered (push)
	EventPrefillDone = "prefill_done" // P finished
	EventReady       = "ready"        // output starts (gateway internal)
	EventDelta       = "delta"        // parsed output
	EventFinish      = "finish"
	EventError       = "error"
)

// GenerateEvent is one event of an engine output stream (a message of an
// mwip Generate stream).
type GenerateEvent struct {
	Type string `json:"type"`

	Receiver     *KVReceiver `json:"kv_receiver,omitempty"`
	Handle       *KVHandle   `json:"kv_handle,omitempty"`
	PromptTokens int         `json:"prompt_tokens,omitempty"`
	CachedTokens int         `json:"cached_tokens,omitempty"`

	// delta: parsed by the engine
	Content   string           `json:"content,omitempty"`
	Reasoning string           `json:"reasoning,omitempty"`
	ToolCalls []DeltaToolCall  `json:"tool_calls,omitempty"`
	TokenIDs  []uint32         `json:"token_ids,omitempty"`
	Logprobs  []LogprobContent `json:"logprobs,omitempty"`

	// finish
	Reason     string  `json:"reason,omitempty"`      // stop | length | tool_calls | content_filter
	StopReason string  `json:"stop_reason,omitempty"` // matched stop string
	Usage      *Usage  `json:"usage,omitempty"`
	Timing     *Timing `json:"timing,omitempty"`

	// error
	Code      string `json:"code,omitempty"`
	Message   string `json:"message,omitempty"`
	Retriable bool   `json:"retriable,omitempty"`
	WorkerID  string `json:"worker_id,omitempty"`
}

// ---------------------------------------------------------------------------
// Encode: modelweave.mwip.v1.Engine/Encode
// ---------------------------------------------------------------------------

// EncodeRequest asks an encoder to execute exactly one item's frozen plan.
type EncodeRequest struct {
	RequestID string      `json:"request_id"`
	AttemptID string      `json:"attempt_id"`
	Model     string      `json:"model"`
	Item      *MediaInput `json:"item"`
	TimeoutMS int64       `json:"timeout_ms,omitempty"`
}

// EncodeResponse returns one packed item embedding, not one result per frame.
type EncodeResponse struct {
	Embedding *EmbeddingLocation `json:"embedding"`
	EncodeMS  float64            `json:"encode_ms,omitempty"`
	WorkerID  string             `json:"worker_id,omitempty"`
}

// Error codes reported by engines (Error events, gRPC status details).
const (
	ErrCapacity       = "capacity_exhausted"   // no room now; retry elsewhere
	ErrNormalRejected = "normal_rejected"      // decode worker declines non-PD requests
	ErrKVTransfer     = "kv_transfer_failed"   // KV handoff failed
	ErrArtifact       = "artifact_unavailable" // embedding missing
	ErrBadRequest     = "bad_request"
	ErrTimeout        = "timeout"
	ErrAborted        = "aborted"
	ErrInternal       = "internal"
)

// ErrorBody is the JSON body of a failed HTTP call (the admin API, the
// mock engine's test endpoints).
type ErrorBody struct {
	Type      string `json:"type"` // always "error"
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retriable bool   `json:"retriable,omitempty"`
}

// KVEvents is a batch of KV cache events of one instance (block hashes per
// docs/protocol.md).
type KVEvents struct {
	Stored  []uint64 `json:"stored,omitempty"`
	Removed []uint64 `json:"removed,omitempty"`
	Cleared bool     `json:"cleared,omitempty"`
}
