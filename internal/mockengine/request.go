package mockengine

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/wire"
)

// kvRole is a request's part in a disaggregated execution, read from the
// engine protocol's own fields.
type kvRole struct {
	kind      string // local | prefill | decode
	handshake string // push | bootstrap | pull
	room      string
	source    string           // decode: where to fetch KV (bootstrap, pull)
	receiver  *mwip.KVReceiver // prefill push: where to send KV
	// returnParams: the prefill leg returns a KV handle (pull).
	returnParams bool
}

// mediaRef is one media item of a request and how its embedding arrives.
type mediaRef struct {
	digest    string
	tokens    int      // positions in the expanded prompt
	artifacts []string // mwip: embedding artifact URIs (one per placement)
	ecURI     string   // vllm: where the EC connector loads the embedding from
}

// genReq is a generate call in protocol-neutral form.
type genReq struct {
	protocol         string // mwip | vllm | sglang
	stream           bool
	requestID        string
	tokens           []uint32 // expanded prompt
	maxTokens        int
	stop             []string
	stopTokenIDs     []uint32
	topLogprobs      int // -1: no logprobs
	includeUsage     bool
	outputMode       string // vllm: derender | text | tokens
	parse            bool   // engine-side parsing of reasoning and tool calls
	reasoning        bool   // output starts inside the reasoning
	openThink        bool   // the model opens the reasoning with <think> itself (Qwen3)
	reasoningParser  string // deepseek_v4 | qwen3
	toolParser       string // deepseek_v4 | hermes
	includeReasoning bool
	tools            []string
	toolChoice       string
	parseTools       bool
	callsOpened      bool // the prompt opened the tool-call block
	structured       bool
	media            []mediaRef
	timeoutMS        int64
	kv               kvRole
}

func truthy(v any) bool { b, _ := v.(bool); return b }

func str(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	return ""
}

func toolNames(tools []json.RawMessage) []string {
	var out []string
	for _, t := range tools {
		var tool struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		}
		if json.Unmarshal(t, &tool) == nil && tool.Function.Name != "" {
			out = append(out, tool.Function.Name)
		}
	}
	return out
}

// tail reports whether one of the last n tokens is t.
func tail(tokens []uint32, n int, t *uint32) bool {
	if t == nil {
		return false
	}
	for i := max(len(tokens)-n, 0); i < len(tokens); i++ {
		if tokens[i] == *t {
			return true
		}
	}
	return false
}

// promptThinks reads the thinking mode off the prompt, for requests that do
// not carry it (SGLang): a prompt ending in <think> thinks, one that closed
// the reasoning (</think>) does not, and a model that opens the reasoning
// itself (Qwen3) thinks unless it was switched off.
func promptThinks(tokens []uint32, info modelInfo) bool {
	switch {
	case tail(tokens, 4, info.ThinkEnd):
		return false
	case tail(tokens, 2, info.ThinkStart):
		return true
	}
	return info.ReasoningParser == mwip.ParserQwen3 && info.Thinking != "none"
}

// setReasoning records the output markup of a request and whether the model
// has to open the reasoning block itself.
func (g *genReq) setReasoning(reasoning bool, info modelInfo) {
	g.reasoning = reasoning
	g.openThink = reasoning && g.reasoningParser == mwip.ParserQwen3 && !tail(g.tokens, 2, info.ThinkStart)
}

// fromMWIP reads the native request: the intermediate state already holds
// the expanded prompt, the media with their embedding artifacts, and the
// parsing options the engine applies itself.
func fromMWIP(req *mwip.GenerateRequest, info modelInfo) *genReq {
	io, po := req.InferenceOptions, req.ParsingOptions
	g := &genReq{
		protocol: mwip.EngineMWIP, stream: req.Stream, requestID: req.RequestID, tokens: req.TokenIDs,
		maxTokens: io.MaxTokens, stop: po.StopSequences, stopTokenIDs: io.StopTokenIDs, topLogprobs: -1,
		includeUsage: true, parse: true, includeReasoning: true,
		reasoningParser: po.ReasoningParser, toolParser: po.ToolCallParser,
		tools: toolNames(req.Conversation.Tools), toolChoice: req.Conversation.ToolChoice,
		parseTools: po.ParseToolCalls, callsOpened: po.StartsInToolCalls,
		structured: io.OutputConstraint != nil && io.OutputConstraint.Kind == "json", timeoutMS: req.TimeoutMS,
	}
	if g.reasoningParser == "" && g.toolParser == "" {
		g.reasoningParser, g.toolParser = info.ReasoningParser, info.ToolCallParser
	}
	g.setReasoning(po.StartsInReasoning(), info)
	if io.Logprobs {
		g.topLogprobs = 0
		if io.TopLogprobs != nil {
			g.topLogprobs = *io.TopLogprobs
		}
	}
	if m := req.Multimodal; m != nil {
		for _, it := range m.Items {
			ref := mediaRef{digest: it.Digest, tokens: it.Span.Length}
			for _, a := range it.Embeddings {
				ref.artifacts = append(ref.artifacts, a.URI)
			}
			g.media = append(g.media, ref)
		}
	}
	d := req.Disagg
	g.kv = kvRole{kind: "local"}
	if d.Mode == mwip.ModeSplit {
		g.kv = kvRole{kind: d.Role, handshake: d.Handshake, room: d.Room, receiver: d.Receiver}
		switch {
		case d.Role == "prefill" && d.Handshake == mwip.HandshakePull:
			g.kv.returnParams = true
		case d.Role == "decode" && d.Handshake == mwip.HandshakePull && d.Handle != nil:
			g.kv.source, g.kv.room = d.Handle.URL, d.Handle.Room
		case d.Role == "decode" && d.Handshake == mwip.HandshakeBootstrap && d.Prefill != nil:
			g.kv.source = d.Prefill.BootstrapAddr
			if g.kv.source == "" {
				g.kv.source = d.Prefill.URL
			}
		}
	}
	return g
}

// expand restores the positions of collapsed media placeholders from the
// ModelWeave hints (a real engine computes them with its processor): each
// hint's placeholder, in order, becomes its expanded layout.
func expand(tokens []uint32, fallback uint32, hints []wire.MediaHint) ([]uint32, []mediaRef) {
	if len(hints) == 0 {
		return tokens, nil
	}
	var refs []mediaRef
	out := make([]uint32, 0, len(tokens))
	k := 0
	for i := 0; i < len(tokens); {
		if k < len(hints) {
			h := hints[k]
			ph := h.Placeholder
			if len(ph) == 0 {
				ph = []uint32{fallback}
			}
			if hasAt(tokens, i, ph) {
				out = append(out, expanded(h)...)
				refs = append(refs, mediaRef{digest: h.Digest, tokens: h.Tokens})
				i += len(ph)
				k++
				continue
			}
		}
		out = append(out, tokens[i])
		i++
	}
	return out, refs
}

func hasAt(tokens []uint32, i int, seq []uint32) bool {
	if i+len(seq) > len(tokens) {
		return false
	}
	for j, t := range seq {
		if tokens[i+j] != t {
			return false
		}
	}
	return true
}

// expanded is the token span of a hint: its run-length layout, or its pad
// value repeated.
func expanded(h wire.MediaHint) []uint32 {
	var out []uint32
	if len(h.Layout) > 0 {
		for _, run := range h.Layout {
			for n := uint32(0); n < run[1]; n++ {
				out = append(out, run[0])
			}
		}
		return out
	}
	for j := 0; j < max(h.Tokens, 1); j++ {
		out = append(out, h.PadValue)
	}
	return out
}

// fromVLLM reads vLLM's token-in GenerateRequest. Disaggregation comes from
// kv_transfer_params (bootstrap: MooncakeConnector transfer_id; pull:
// NixlConnector, P returns the parameters); push is the ModelWeave
// extension. Media arrive as content_parts, their embeddings through
// ec_transfer_params after an encode stage.
func fromVLLM(req *wire.VLLMGenerateRequest, placeholder uint32, info modelInfo) *genReq {
	sp := req.SamplingParams
	g := &genReq{
		protocol: mwip.EngineVLLM, stream: req.Stream, requestID: req.RequestID, tokens: req.TokenIDs,
		maxTokens: sp.MaxTokens, stop: sp.Stop, stopTokenIDs: sp.StopTokenIDs, topLogprobs: -1,
		outputMode: req.OutputMode, includeReasoning: true,
		reasoningParser: info.ReasoningParser, toolParser: info.ToolCallParser,
	}
	if g.outputMode == "" {
		g.outputMode = "tokens"
	}
	g.parse = g.outputMode == "derender"
	if sp.Logprobs != nil {
		g.topLogprobs = *sp.Logprobs
	}
	if req.StreamOptions != nil {
		g.includeUsage = req.StreamOptions.IncludeUsage
	}
	if pc := req.ParseContext; pc != nil {
		g.includeReasoning = pc.IncludeReasoning
		_ = json.Unmarshal(pc.ToolChoice, &g.toolChoice)
		g.tools = toolNames(pc.Tools)
	}
	g.parseTools = len(g.tools) > 0 && g.toolChoice != "none"
	g.callsOpened = g.parseTools && g.toolChoice == "required" && g.toolParser == mwip.ParserDeepseekV4
	if len(sp.StructuredOutputs) > 0 {
		var so struct {
			JSON       json.RawMessage `json:"json"`
			JSONObject bool            `json:"json_object"`
		}
		_ = json.Unmarshal(sp.StructuredOutputs, &so)
		g.structured = len(so.JSON) > 0 || so.JSONObject
	}
	var hints []wire.MediaHint
	var d *mwip.Disagg
	if req.ModelWeave != nil {
		hints, d, g.timeoutMS = req.ModelWeave.Media, req.ModelWeave.Disagg, req.ModelWeave.TimeoutMS
	}
	g.tokens, g.media = expand(req.TokenIDs, placeholder, hints)
	if req.ReasoningEnded != nil {
		g.setReasoning(!*req.ReasoningEnded, info)
	} else {
		g.setReasoning(promptThinks(g.tokens, info), info)
	}
	for i, p := range req.ContentParts {
		if i >= len(g.media) {
			g.media = append(g.media, mediaRef{digest: p.UUID, tokens: 1})
		}
		if h, ok := req.ECTransferParams[p.UUID].(map[string]any); ok {
			g.media[i].ecURI = str(h["uri"])
		}
	}
	kv := req.KVTransferParams
	switch {
	case d != nil && d.Mode == mwip.ModeSplit && d.Handshake == mwip.HandshakePush && d.Role == "decode":
		g.kv = kvRole{kind: "decode", handshake: mwip.HandshakePush, room: d.Room}
	case d != nil && d.Mode == mwip.ModeSplit && d.Handshake == mwip.HandshakePush && d.Role == "prefill":
		g.kv = kvRole{kind: "prefill", handshake: mwip.HandshakePush, room: d.Room, receiver: d.Receiver}
	case truthy(kv["do_remote_decode"]):
		if tid := str(kv["transfer_id"]); tid != "" {
			g.kv = kvRole{kind: "prefill", handshake: mwip.HandshakeBootstrap, room: tid}
		} else {
			g.kv = kvRole{kind: "prefill", handshake: mwip.HandshakePull, room: req.RequestID, returnParams: true}
		}
	case truthy(kv["do_remote_prefill"]):
		if addr := str(kv["remote_bootstrap_addr"]); addr != "" {
			g.kv = kvRole{kind: "decode", handshake: mwip.HandshakeBootstrap, source: addr, room: str(kv["transfer_id"])}
		} else {
			g.kv = kvRole{kind: "decode", handshake: mwip.HandshakePull, room: str(kv["remote_request_id"]),
				source: fmt.Sprintf("http://%s:%s", str(kv["remote_host"]), str(kv["remote_port"]))}
		}
	default:
		g.kv = kvRole{kind: "local"}
	}
	return g
}

// fromSGLang reads SGLang's /generate. Media arrive raw (image_data,
// video_data): SGLang encodes them itself; the bootstrap fields pair P and
// D. The native API does not parse output, so the thinking mode is read off
// the prompt only to generate model-like text.
func fromSGLang(req *wire.SGLangRequest, placeholder uint32, role string, info modelInfo) *genReq {
	sp := req.SamplingParams
	g := &genReq{
		protocol: mwip.EngineSGLang, stream: req.Stream, requestID: req.RID, maxTokens: 128,
		topLogprobs: -1, outputMode: "text", reasoningParser: info.ReasoningParser, toolParser: info.ToolCallParser,
	}
	if v, ok := sp["max_new_tokens"].(float64); ok {
		g.maxTokens = int(v)
	}
	switch s := sp["stop"].(type) {
	case string:
		g.stop = []string{s}
	case []any:
		for _, x := range s {
			if v, ok := x.(string); ok {
				g.stop = append(g.stop, v)
			}
		}
	}
	if ids, ok := sp["stop_token_ids"].([]any); ok {
		for _, x := range ids {
			if v, ok := x.(float64); ok {
				g.stopTokenIDs = append(g.stopTokenIDs, uint32(v))
			}
		}
	}
	_, g.structured = sp["json_schema"]
	if req.ReturnLogprob {
		g.topLogprobs = req.TopLogprobsNum
	}
	var hints []wire.MediaHint
	if req.ModelWeave != nil {
		hints, g.timeoutMS = req.ModelWeave.Media, req.ModelWeave.TimeoutMS
	}
	g.tokens, g.media = expand(req.InputIDs, placeholder, hints)
	g.setReasoning(promptThinks(g.tokens, info), info)
	for i := len(g.media); i < len(req.ImageData)+len(req.VideoData); i++ {
		g.media = append(g.media, mediaRef{tokens: 1})
	}
	g.kv = kvRole{kind: "local"}
	if req.BootstrapRoom != nil && req.BootstrapHost != "" {
		room := strconv.FormatInt(*req.BootstrapRoom, 10)
		port := 8998
		if req.BootstrapPort != nil {
			port = *req.BootstrapPort
		}
		// SGLang servers run in one disaggregation mode: the worker's role
		// decides the leg.
		if role == "prefill" {
			g.kv = kvRole{kind: "prefill", handshake: mwip.HandshakeBootstrap, room: room}
		} else {
			g.kv = kvRole{kind: "decode", handshake: mwip.HandshakeBootstrap, room: room, source: fmt.Sprintf("http://%s:%d", req.BootstrapHost, port)}
		}
	}
	return g
}
