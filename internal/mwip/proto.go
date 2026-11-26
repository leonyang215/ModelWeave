package mwip

import (
	"encoding/json"
	"fmt"

	"google.golang.org/protobuf/types/known/structpb"

	mwipv1 "github.com/leonyang215/ModelWeave/api/mwip/v1"
)

// Conversions between the in-memory types of this package and the wire
// messages of the mwip protocol (proto/modelweave/mwip/v1). The gateway
// builds requests and reads events in memory and sends them as proto; the
// mock engine does the reverse.

// enum maps the string values used in memory to a proto enum and back.
type enum[E ~int32] struct {
	to   map[string]E
	from map[E]string
}

func newEnum[E ~int32](pairs map[string]E) enum[E] {
	e := enum[E]{to: pairs, from: make(map[E]string, len(pairs))}
	for s, v := range pairs {
		e.from[v] = s
	}
	return e
}

var (
	roles = newEnum(map[string]mwipv1.Role{
		"encode": mwipv1.Role_ROLE_ENCODE, "prefill": mwipv1.Role_ROLE_PREFILL,
		"decode": mwipv1.Role_ROLE_DECODE, "unified": mwipv1.Role_ROLE_UNIFIED,
	})
	modalities = newEnum(map[string]mwipv1.Modality{
		"image": mwipv1.Modality_MODALITY_IMAGE, "video": mwipv1.Modality_MODALITY_VIDEO, "frame": mwipv1.Modality_MODALITY_FRAME,
	})
	modes = newEnum(map[string]mwipv1.DisaggMode{
		ModeNormal: mwipv1.DisaggMode_DISAGG_MODE_NORMAL, ModeSplit: mwipv1.DisaggMode_DISAGG_MODE_SPLIT,
	})
	handshakes = newEnum(map[string]mwipv1.Handshake{
		HandshakePush: mwipv1.Handshake_HANDSHAKE_PUSH, HandshakeBootstrap: mwipv1.Handshake_HANDSHAKE_BOOTSTRAP,
		HandshakePull: mwipv1.Handshake_HANDSHAKE_PULL,
	})
	stages = newEnum(map[string]mwipv1.ThinkingStage{
		"reasoning_start":        mwipv1.ThinkingStage_THINKING_STAGE_REASONING_START,
		"reasoning_continuation": mwipv1.ThinkingStage_THINKING_STAGE_REASONING_CONTINUATION,
		"content_continuation":   mwipv1.ThinkingStage_THINKING_STAGE_CONTENT_CONTINUATION,
	})
	constraints = newEnum(map[string]mwipv1.ConstraintKind{
		"json": mwipv1.ConstraintKind_CONSTRAINT_KIND_JSON, "regex": mwipv1.ConstraintKind_CONSTRAINT_KIND_REGEX,
		"tool_call":          mwipv1.ConstraintKind_CONSTRAINT_KIND_TOOL_CALL,
		"tool_call_required": mwipv1.ConstraintKind_CONSTRAINT_KIND_TOOL_CALL_REQUIRED,
	})
	finishReasons = newEnum(map[string]mwipv1.FinishReason{
		"stop": mwipv1.FinishReason_FINISH_REASON_STOP, "length": mwipv1.FinishReason_FINISH_REASON_LENGTH,
		"tool_calls":     mwipv1.FinishReason_FINISH_REASON_TOOL_CALLS,
		"content_filter": mwipv1.FinishReason_FINISH_REASON_CONTENT_FILTER,
	})
	errorCodes = newEnum(map[string]mwipv1.ErrorCode{
		ErrCapacity: mwipv1.ErrorCode_ERROR_CODE_CAPACITY_EXHAUSTED, ErrNormalRejected: mwipv1.ErrorCode_ERROR_CODE_NORMAL_REJECTED,
		ErrKVTransfer: mwipv1.ErrorCode_ERROR_CODE_KV_TRANSFER_FAILED, ErrArtifact: mwipv1.ErrorCode_ERROR_CODE_ARTIFACT_UNAVAILABLE,
		ErrBadRequest: mwipv1.ErrorCode_ERROR_CODE_BAD_REQUEST, ErrTimeout: mwipv1.ErrorCode_ERROR_CODE_TIMEOUT,
		ErrAborted: mwipv1.ErrorCode_ERROR_CODE_ABORTED, ErrInternal: mwipv1.ErrorCode_ERROR_CODE_INTERNAL,
	})
	priorities = newEnum(map[string]mwipv1.Priority{
		"low": mwipv1.Priority_PRIORITY_LOW, "normal": mwipv1.Priority_PRIORITY_NORMAL,
		"high": mwipv1.Priority_PRIORITY_HIGH, "critical": mwipv1.Priority_PRIORITY_CRITICAL,
	})
)

// RoleToProto converts a worker role name (encode, prefill, decode,
// unified).
func RoleToProto(s string) mwipv1.Role { return roles.to[s] }

// RoleFromProto names a role; "" for ROLE_UNSPECIFIED.
func RoleFromProto(r mwipv1.Role) string { return roles.from[r] }

// ErrorCodeToProto converts an error code; unknown codes are internal.
func ErrorCodeToProto(code string) mwipv1.ErrorCode {
	if c, ok := errorCodes.to[code]; ok {
		return c
	}
	return mwipv1.ErrorCode_ERROR_CODE_INTERNAL
}

// ErrorCodeFromProto names an error code; unspecified codes are internal.
func ErrorCodeFromProto(c mwipv1.ErrorCode) string {
	if s, ok := errorCodes.from[c]; ok {
		return s
	}
	return ErrInternal
}

func i32(p *int) *int32 {
	if p == nil {
		return nil
	}
	v := int32(*p)
	return &v
}

func intp(p *int32) *int {
	if p == nil {
		return nil
	}
	v := int(*p)
	return &v
}

// ---------------------------------------------------------------------------
// Generate requests
// ---------------------------------------------------------------------------

// RequestToProto converts a generate request.
func RequestToProto(r *GenerateRequest) (*mwipv1.GenerateRequest, error) {
	conv, err := conversationToProto(r.Conversation)
	if err != nil {
		return nil, err
	}
	return &mwipv1.GenerateRequest{
		RequestId: r.RequestID, Model: r.Model, TokenIds: r.TokenIDs,
		Multimodal: MultimodalToProto(r.Multimodal), Conversation: conv,
		InferenceOptions: inferenceToProto(r.InferenceOptions), ParsingOptions: parsingToProto(r.ParsingOptions),
		Disagg: disaggToProto(r.Disagg), TimeoutMs: r.TimeoutMS, SessionKey: r.SessionKey,
		Priority: priorities.to[r.Priority],
	}, nil
}

// RequestFromProto converts a received generate request.
func RequestFromProto(p *mwipv1.GenerateRequest) (*GenerateRequest, error) {
	conv, err := conversationFromProto(p.GetConversation())
	if err != nil {
		return nil, err
	}
	return &GenerateRequest{
		RequestID: p.GetRequestId(), Model: p.GetModel(), TokenIDs: p.GetTokenIds(),
		Multimodal: MultimodalFromProto(p.GetMultimodal()), Conversation: conv,
		InferenceOptions: inferenceFromProto(p.GetInferenceOptions()), ParsingOptions: parsingFromProto(p.GetParsingOptions()),
		Disagg: disaggFromProto(p.GetDisagg()), TimeoutMS: p.GetTimeoutMs(), SessionKey: p.GetSessionKey(),
		Priority: priorities.from[p.GetPriority()],
	}, nil
}

// openAITool is an OpenAI function tool as JSON.
type openAITool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters,omitempty"`
		Strict      *bool          `json:"strict,omitempty"`
	} `json:"function"`
}

func conversationToProto(c Conversation) (*mwipv1.Conversation, error) {
	out := &mwipv1.Conversation{
		ThinkingMode: c.ThinkingMode, ReasoningEffort: c.ReasoningEffort, ToolChoice: c.ToolChoice, ResponseFormat: c.ResponseFormat,
	}
	for i, raw := range c.Tools {
		var t openAITool
		if err := json.Unmarshal(raw, &t); err != nil {
			return nil, fmt.Errorf("tool %d: %w", i, err)
		}
		fn := &mwipv1.FunctionDefinition{Name: t.Function.Name, Description: t.Function.Description, Strict: t.Function.Strict}
		if t.Function.Parameters != nil {
			params, err := structpb.NewStruct(t.Function.Parameters)
			if err != nil {
				return nil, fmt.Errorf("tool %d parameters: %w", i, err)
			}
			fn.Parameters = params
		}
		if t.Type == "" {
			t.Type = "function"
		}
		out.Tools = append(out.Tools, &mwipv1.Tool{Type: t.Type, Function: fn})
	}
	if len(c.ChatTemplateKwargs) > 0 {
		kw, err := structpb.NewStruct(c.ChatTemplateKwargs)
		if err != nil {
			return nil, fmt.Errorf("chat_template_kwargs: %w", err)
		}
		out.ChatTemplateKwargs = kw
	}
	return out, nil
}

func conversationFromProto(c *mwipv1.Conversation) (Conversation, error) {
	out := Conversation{
		ThinkingMode: c.GetThinkingMode(), ReasoningEffort: c.GetReasoningEffort(), ToolChoice: c.GetToolChoice(),
		ResponseFormat: c.GetResponseFormat(),
	}
	for _, t := range c.GetTools() {
		var tool openAITool
		tool.Type = t.GetType()
		fn := t.GetFunction()
		tool.Function.Name, tool.Function.Description, tool.Function.Strict = fn.GetName(), fn.GetDescription(), fn.Strict
		if fn.GetParameters() != nil {
			tool.Function.Parameters = fn.GetParameters().AsMap()
		}
		raw, err := json.Marshal(tool)
		if err != nil {
			return Conversation{}, err
		}
		out.Tools = append(out.Tools, raw)
	}
	if kw := c.GetChatTemplateKwargs(); kw != nil && len(kw.GetFields()) > 0 {
		out.ChatTemplateKwargs = kw.AsMap()
	}
	return out, nil
}

func inferenceToProto(o InferenceOptions) *mwipv1.InferenceOptions {
	out := &mwipv1.InferenceOptions{
		MaxTokens: int32(o.MaxTokens), Temperature: o.Temperature, TopP: o.TopP, TopK: i32(o.TopK), MinP: o.MinP,
		PresencePenalty: o.PresencePenalty, FrequencyPenalty: o.FrequencyPenalty, RepetitionPenalty: o.RepetitionPenalty,
		Seed: o.Seed, Logprobs: o.Logprobs, TopLogprobs: i32(o.TopLogprobs), StopTokenIds: o.StopTokenIDs,
	}
	if c := o.OutputConstraint; c != nil {
		out.OutputConstraint = &mwipv1.OutputConstraint{Kind: constraints.to[c.Kind], Schema: c.Schema}
	}
	return out
}

func inferenceFromProto(o *mwipv1.InferenceOptions) InferenceOptions {
	out := InferenceOptions{
		MaxTokens: int(o.GetMaxTokens()), Temperature: o.Temperature, TopP: o.TopP, TopK: intp(o.TopK), MinP: o.MinP,
		PresencePenalty: o.PresencePenalty, FrequencyPenalty: o.FrequencyPenalty, RepetitionPenalty: o.RepetitionPenalty,
		Seed: o.Seed, Logprobs: o.GetLogprobs(), TopLogprobs: intp(o.TopLogprobs), StopTokenIDs: o.GetStopTokenIds(),
	}
	if c := o.GetOutputConstraint(); c != nil {
		out.OutputConstraint = &OutputConstraint{Kind: constraints.from[c.GetKind()], Schema: c.GetSchema()}
	}
	return out
}

func parsingToProto(p ParsingOptions) *mwipv1.ParsingOptions {
	return &mwipv1.ParsingOptions{
		ReasoningParser: p.ReasoningParser, ToolCallParser: p.ToolCallParser, ParseToolCalls: p.ParseToolCalls,
		StartsInToolCalls: p.StartsInToolCalls, ParseJsonOutput: p.ParseJSONOutput,
		ThinkingInitialStage: stages.to[p.ThinkingInitialStage], StopSequences: p.StopSequences,
	}
}

func parsingFromProto(p *mwipv1.ParsingOptions) ParsingOptions {
	return ParsingOptions{
		ReasoningParser: p.GetReasoningParser(), ToolCallParser: p.GetToolCallParser(), ParseToolCalls: p.GetParseToolCalls(),
		StartsInToolCalls: p.GetStartsInToolCalls(), ParseJSONOutput: p.GetParseJsonOutput(),
		ThinkingInitialStage: stages.from[p.GetThinkingInitialStage()], StopSequences: p.GetStopSequences(),
	}
}

func peerToProto(p *Peer) *mwipv1.Peer {
	if p == nil {
		return nil
	}
	return &mwipv1.Peer{InstanceId: p.WorkerID, Target: p.Target, BootstrapAddr: p.BootstrapAddr, EngineId: p.EngineID}
}

func peerFromProto(p *mwipv1.Peer) *Peer {
	if p == nil {
		return nil
	}
	return &Peer{WorkerID: p.GetInstanceId(), Target: p.GetTarget(), BootstrapAddr: p.GetBootstrapAddr(), EngineID: p.GetEngineId()}
}

// ReceiverToProto converts a push KV receiver.
func ReceiverToProto(r *KVReceiver) *mwipv1.KVReceiver {
	if r == nil {
		return nil
	}
	return &mwipv1.KVReceiver{InstanceId: r.WorkerID, Url: r.URL, Room: r.Room, CachedTokens: int32(r.CachedTokens)}
}

// ReceiverFromProto converts a push KV receiver.
func ReceiverFromProto(r *mwipv1.KVReceiver) *KVReceiver {
	if r == nil {
		return nil
	}
	return &KVReceiver{WorkerID: r.GetInstanceId(), URL: r.GetUrl(), Room: r.GetRoom(), CachedTokens: int(r.GetCachedTokens())}
}

func handleToProto(h *KVHandle) *mwipv1.KVHandle {
	if h == nil {
		return nil
	}
	return &mwipv1.KVHandle{InstanceId: h.WorkerID, Url: h.URL, Room: h.Room, Tokens: int32(h.Tokens)}
}

func handleFromProto(h *mwipv1.KVHandle) *KVHandle {
	if h == nil {
		return nil
	}
	return &KVHandle{WorkerID: h.GetInstanceId(), URL: h.GetUrl(), Room: h.GetRoom(), Tokens: int(h.GetTokens())}
}

func disaggToProto(d Disagg) *mwipv1.Disagg {
	return &mwipv1.Disagg{
		Mode: modes.to[d.Mode], Role: roles.to[d.Role], Handshake: handshakes.to[d.Handshake], Room: d.Room,
		BootstrapRoom: uint64(d.BootstrapRoom), Prefill: peerToProto(d.Prefill), Decode: peerToProto(d.Decode),
		KvReceiver: ReceiverToProto(d.Receiver), KvHandle: handleToProto(d.Handle),
	}
}

func disaggFromProto(d *mwipv1.Disagg) Disagg {
	mode := modes.from[d.GetMode()]
	if mode == "" {
		mode = ModeNormal
	}
	return Disagg{
		Mode: mode, Role: roles.from[d.GetRole()], Handshake: handshakes.from[d.GetHandshake()], Room: d.GetRoom(),
		BootstrapRoom: int64(d.GetBootstrapRoom()), Prefill: peerFromProto(d.GetPrefill()), Decode: peerFromProto(d.GetDecode()),
		Receiver: ReceiverFromProto(d.GetKvReceiver()), Handle: handleFromProto(d.GetKvHandle()),
	}
}

// ---------------------------------------------------------------------------
// Media
// ---------------------------------------------------------------------------

// MultimodalToProto converts the media of a prompt.
func MultimodalToProto(m *Multimodal) *mwipv1.Multimodal {
	if m == nil {
		return nil
	}
	return &mwipv1.Multimodal{PadShift: m.PadShift, PlaceholderTokenId: m.PlaceholderTokenID, Items: MediaItemsToProto(m.Items)}
}

// MultimodalFromProto converts the media of a prompt.
func MultimodalFromProto(m *mwipv1.Multimodal) *Multimodal {
	if m == nil {
		return nil
	}
	return &Multimodal{PadShift: m.GetPadShift(), PlaceholderTokenID: m.GetPlaceholderTokenId(), Items: MediaItemsFromProto(m.GetItems())}
}

func pixelToProto(p *PixelControlOptions) *mwipv1.PixelControlOptions {
	if p == nil {
		return nil
	}
	return &mwipv1.PixelControlOptions{MaxLongSide: i32(p.MaxLongSide), MaxTotalPixels: p.MaxTotalPixels}
}

func pixelFromProto(p *mwipv1.PixelControlOptions) *PixelControlOptions {
	if p == nil {
		return nil
	}
	return &PixelControlOptions{MaxLongSide: intp(p.MaxLongSide), MaxTotalPixels: p.MaxTotalPixels}
}

// MediaItemsToProto converts media items.
func MediaItemsToProto(items []MediaItem) []*mwipv1.MediaItem {
	if items == nil {
		return nil
	}
	out := make([]*mwipv1.MediaItem, len(items))
	for i := range items {
		it := &items[i]
		p := &mwipv1.MediaItem{
			Index: int32(it.Index), Modality: modalities.to[it.Modality], Digest: it.Digest,
			Source: &mwipv1.MediaSource{Kind: it.Source.Kind, Url: it.Source.URL, Mime: it.Source.MIME},
			Detail: it.Detail,
			Metadata: &mwipv1.MediaMetadata{
				Width: int32(it.Metadata.Width), Height: int32(it.Metadata.Height), DurationMs: it.Metadata.DurationMS,
				TotalFrames: it.Metadata.TotalFrames, VideoFps: it.Metadata.VideoFPS,
			},
			PixelControlOptions: pixelToProto(it.PixelControl),
			Plan: &mwipv1.MediaPlan{
				Width: int32(it.Plan.Width), Height: int32(it.Plan.Height), Tokens: int32(it.Plan.Tokens),
				Nframes: int32(it.Plan.NFrames), KeptFrames: int32(it.Plan.KeptFrames), Timestamps: it.Plan.Timestamps,
			},
			Span:               &mwipv1.Span{Offset: int32(it.Span.Offset), Length: int32(it.Span.Length)},
			Placeholder:        it.Placeholder,
			EmbeddingArtifacts: ArtifactsToProto(it.Embeddings),
		}
		if f := it.FrameExtract; f != nil {
			p.FrameExtractOptions = &mwipv1.FrameExtractOptions{Fps: f.FPS, Nframes: i32(f.NFrames), PixelControlOptions: pixelToProto(f.PixelControl)}
		}
		for _, pl := range it.Placements {
			p.Placements = append(p.Placements, &mwipv1.Placement{
				Offset: int32(pl.Offset), Length: int32(pl.Length), PadValue: pl.PadValue, Reference: pl.Reference,
				Modality: modalities.to[pl.Modality], Timestamp: pl.Timestamp,
			})
		}
		out[i] = p
	}
	return out
}

// MediaItemsFromProto converts media items.
func MediaItemsFromProto(items []*mwipv1.MediaItem) []MediaItem {
	if items == nil {
		return nil
	}
	out := make([]MediaItem, len(items))
	for i, p := range items {
		md, plan := p.GetMetadata(), p.GetPlan()
		it := MediaItem{
			Index: int(p.GetIndex()), Modality: modalities.from[p.GetModality()], Digest: p.GetDigest(),
			Source: MediaSource{Kind: p.GetSource().GetKind(), URL: p.GetSource().GetUrl(), MIME: p.GetSource().GetMime()},
			Detail: p.GetDetail(),
			Metadata: MediaMetadata{
				Width: int(md.GetWidth()), Height: int(md.GetHeight()), DurationMS: md.GetDurationMs(),
				TotalFrames: md.TotalFrames, VideoFPS: md.GetVideoFps(),
			},
			PixelControl: pixelFromProto(p.GetPixelControlOptions()),
			Plan: MediaPlan{
				Width: int(plan.GetWidth()), Height: int(plan.GetHeight()), Tokens: int(plan.GetTokens()),
				NFrames: int(plan.GetNframes()), KeptFrames: int(plan.GetKeptFrames()), Timestamps: plan.GetTimestamps(),
			},
			Span:        Span{Offset: int(p.GetSpan().GetOffset()), Length: int(p.GetSpan().GetLength())},
			Placeholder: p.GetPlaceholder(),
			Embeddings:  ArtifactsFromProto(p.GetEmbeddingArtifacts()),
		}
		if f := p.GetFrameExtractOptions(); f != nil {
			it.FrameExtract = &FrameExtractOptions{FPS: f.Fps, NFrames: intp(f.Nframes), PixelControl: pixelFromProto(f.GetPixelControlOptions())}
		}
		for _, pl := range p.GetPlacements() {
			it.Placements = append(it.Placements, Placement{
				Offset: int(pl.GetOffset()), Length: int(pl.GetLength()), PadValue: pl.GetPadValue(), Reference: pl.GetReference(),
				Modality: modalities.from[pl.GetModality()], Timestamp: pl.Timestamp,
			})
		}
		out[i] = it
	}
	return out
}

// ArtifactsToProto converts embedding artifacts.
func ArtifactsToProto(as []EmbeddingArtifact) []*mwipv1.EmbeddingArtifact {
	if as == nil {
		return nil
	}
	out := make([]*mwipv1.EmbeddingArtifact, len(as))
	for i, a := range as {
		out[i] = &mwipv1.EmbeddingArtifact{Reference: a.Reference, Uri: a.URI, Tokens: int32(a.Tokens), Encoder: a.Encoder}
	}
	return out
}

// ArtifactsFromProto converts embedding artifacts.
func ArtifactsFromProto(as []*mwipv1.EmbeddingArtifact) []EmbeddingArtifact {
	if as == nil {
		return nil
	}
	out := make([]EmbeddingArtifact, len(as))
	for i, a := range as {
		out[i] = EmbeddingArtifact{Reference: a.GetReference(), URI: a.GetUri(), Tokens: int(a.GetTokens()), Encoder: a.GetEncoder()}
	}
	return out
}

// ---------------------------------------------------------------------------
// Generate events
// ---------------------------------------------------------------------------

func timingToProto(t *Timing) *mwipv1.Timing {
	if t == nil {
		return nil
	}
	return &mwipv1.Timing{QueueMs: t.QueueMS, EncodeMs: t.EncodeMS, PrefillMs: t.PrefillMS, TransferMs: t.TransferMS, TtftMs: t.TTFTMS, DecodeMs: t.DecodeMS}
}

func timingFromProto(t *mwipv1.Timing) *Timing {
	if t == nil {
		return nil
	}
	return &Timing{QueueMS: t.GetQueueMs(), EncodeMS: t.GetEncodeMs(), PrefillMS: t.GetPrefillMs(), TransferMS: t.GetTransferMs(), TTFTMS: t.GetTtftMs(), DecodeMS: t.GetDecodeMs()}
}

func bytesToProto(b []int) []byte {
	if b == nil {
		return nil
	}
	out := make([]byte, len(b))
	for i, v := range b {
		out[i] = byte(v)
	}
	return out
}

func bytesFromProto(b []byte) []int {
	if b == nil {
		return nil
	}
	out := make([]int, len(b))
	for i, v := range b {
		out[i] = int(v)
	}
	return out
}

func usageToProto(u *Usage) *mwipv1.Usage {
	if u == nil {
		return nil
	}
	return &mwipv1.Usage{PromptTokens: int32(u.PromptTokens), CompletionTokens: int32(u.CompletionTokens),
		CachedTokens: int32(u.Cached()), ReasoningTokens: int32(u.Reasoning())}
}

func usageFromProto(u *mwipv1.Usage) *Usage {
	if u == nil {
		return nil
	}
	out := &Usage{PromptTokens: int(u.GetPromptTokens()), CompletionTokens: int(u.GetCompletionTokens()),
		TotalTokens: int(u.GetPromptTokens() + u.GetCompletionTokens())}
	if u.GetCachedTokens() > 0 {
		out.PromptTokensDetails = &PromptTokensDetails{CachedTokens: int(u.GetCachedTokens())}
	}
	if u.GetReasoningTokens() > 0 {
		out.CompletionTokensDetails = &CompletionTokensDetails{ReasoningTokens: int(u.GetReasoningTokens())}
	}
	return out
}

// EventToProto converts an engine event to a Generate stream message; nil
// for events that do not go on the wire.
func EventToProto(ev *GenerateEvent) *mwipv1.GenerateResponse {
	switch ev.Type {
	case EventAccepted:
		return &mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Accepted{Accepted: &mwipv1.Accepted{KvReceiver: ReceiverToProto(ev.Receiver)}}}
	case EventPrefillDone:
		return &mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_PrefillDone{PrefillDone: &mwipv1.PrefillDone{
			PromptTokens: int32(ev.PromptTokens), CachedTokens: int32(ev.CachedTokens), KvHandle: handleToProto(ev.Handle), Timing: timingToProto(ev.Timing),
		}}}
	case EventDelta:
		d := &mwipv1.Delta{Content: ev.Content, Reasoning: ev.Reasoning, TokenIds: ev.TokenIDs}
		for _, tc := range ev.ToolCalls {
			p := &mwipv1.ToolCallDelta{Index: int32(tc.Index), Id: tc.ID}
			if tc.Function != nil {
				p.Name, p.Arguments = tc.Function.Name, tc.Function.Arguments
			}
			d.ToolCalls = append(d.ToolCalls, p)
		}
		for _, lp := range ev.Logprobs {
			p := &mwipv1.TokenLogprob{Token: lp.Token, Logprob: lp.Logprob, Bytes: bytesToProto(lp.Bytes)}
			for _, top := range lp.TopLogprobs {
				p.TopLogprobs = append(p.TopLogprobs, &mwipv1.TopLogprob{Token: top.Token, Logprob: top.Logprob, Bytes: bytesToProto(top.Bytes)})
			}
			d.Logprobs = append(d.Logprobs, p)
		}
		return &mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Delta{Delta: d}}
	case EventFinish:
		return &mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Finish{Finish: &mwipv1.Finish{
			Reason: finishReasons.to[ev.Reason], StopSequence: ev.StopReason, Usage: usageToProto(ev.Usage), Timing: timingToProto(ev.Timing),
		}}}
	case EventError:
		return &mwipv1.GenerateResponse{Event: &mwipv1.GenerateResponse_Error{Error: &mwipv1.Error{
			Code: ErrorCodeToProto(ev.Code), Message: ev.Message, Retriable: ev.Retriable,
		}}}
	}
	return nil
}

// EventFromProto converts a Generate stream message; nil for an empty or
// unknown event.
func EventFromProto(m *mwipv1.GenerateResponse) *GenerateEvent {
	switch e := m.GetEvent().(type) {
	case *mwipv1.GenerateResponse_Accepted:
		ev := &GenerateEvent{Type: EventAccepted, Receiver: ReceiverFromProto(e.Accepted.GetKvReceiver())}
		if ev.Receiver != nil {
			ev.CachedTokens = ev.Receiver.CachedTokens
		}
		return ev
	case *mwipv1.GenerateResponse_PrefillDone:
		p := e.PrefillDone
		return &GenerateEvent{Type: EventPrefillDone, PromptTokens: int(p.GetPromptTokens()), CachedTokens: int(p.GetCachedTokens()),
			Handle: handleFromProto(p.GetKvHandle()), Timing: timingFromProto(p.GetTiming())}
	case *mwipv1.GenerateResponse_Delta:
		d := e.Delta
		ev := &GenerateEvent{Type: EventDelta, Content: d.GetContent(), Reasoning: d.GetReasoning(), TokenIDs: d.GetTokenIds()}
		for _, tc := range d.GetToolCalls() {
			call := DeltaToolCall{Index: int(tc.GetIndex()), ID: tc.GetId()}
			if call.ID != "" {
				call.Type = "function"
			}
			if tc.GetName() != "" || tc.GetArguments() != "" {
				call.Function = &DeltaFunction{Name: tc.GetName(), Arguments: tc.GetArguments()}
			}
			ev.ToolCalls = append(ev.ToolCalls, call)
		}
		for _, lp := range d.GetLogprobs() {
			c := LogprobContent{Token: lp.GetToken(), Logprob: lp.GetLogprob(), Bytes: bytesFromProto(lp.GetBytes()), TopLogprobs: []TopLogprob{}}
			for _, top := range lp.GetTopLogprobs() {
				c.TopLogprobs = append(c.TopLogprobs, TopLogprob{Token: top.GetToken(), Logprob: top.GetLogprob(), Bytes: bytesFromProto(top.GetBytes())})
			}
			ev.Logprobs = append(ev.Logprobs, c)
		}
		return ev
	case *mwipv1.GenerateResponse_Finish:
		f := e.Finish
		reason := finishReasons.from[f.GetReason()]
		if reason == "" {
			reason = "stop"
		}
		ev := &GenerateEvent{Type: EventFinish, Reason: reason, StopReason: f.GetStopSequence(), Usage: usageFromProto(f.GetUsage()), Timing: timingFromProto(f.GetTiming())}
		if ev.Usage != nil {
			ev.PromptTokens, ev.CachedTokens = ev.Usage.PromptTokens, ev.Usage.Cached()
		}
		return ev
	case *mwipv1.GenerateResponse_Error:
		return &GenerateEvent{Type: EventError, Code: ErrorCodeFromProto(e.Error.GetCode()), Message: e.Error.GetMessage(), Retriable: e.Error.GetRetriable()}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

// EncodeRequestToProto converts an encode request.
func EncodeRequestToProto(r *EncodeRequest) *mwipv1.EncodeRequest {
	return &mwipv1.EncodeRequest{RequestId: r.RequestID, AttemptId: r.AttemptID, Model: r.Model,
		Items: MediaItemsToProto(r.Items), TimeoutMs: r.TimeoutMS}
}

// EncodeRequestFromProto converts a received encode request.
func EncodeRequestFromProto(p *mwipv1.EncodeRequest) *EncodeRequest {
	return &EncodeRequest{RequestID: p.GetRequestId(), AttemptID: p.GetAttemptId(), Model: p.GetModel(),
		Items: MediaItemsFromProto(p.GetItems()), TimeoutMS: p.GetTimeoutMs()}
}

// ---------------------------------------------------------------------------
// KV events
// ---------------------------------------------------------------------------

// KVEventsToProto converts a KV event batch.
func KVEventsToProto(ev *KVEvents) *mwipv1.KVEvents {
	if ev == nil {
		return nil
	}
	return &mwipv1.KVEvents{Stored: ev.Stored, Removed: ev.Removed, Cleared: ev.Cleared}
}

// KVEventsFromProto converts a KV event batch.
func KVEventsFromProto(ev *mwipv1.KVEvents) *KVEvents {
	if ev == nil {
		return nil
	}
	return &KVEvents{Stored: ev.GetStored(), Removed: ev.GetRemoved(), Cleared: ev.GetCleared()}
}
