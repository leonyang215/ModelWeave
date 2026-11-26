package mockengine

import (
	"encoding/json"

	"github.com/leonyang215/ModelWeave/internal/mwip"
)

// kvRole is a request's part in a disaggregated execution (its disagg).
type kvRole struct {
	kind      string // local | prefill | decode
	handshake string // push | bootstrap | pull
	room      string
	source    string           // decode: where to fetch KV (bootstrap, pull)
	receiver  *mwip.KVReceiver // prefill push: where to send KV
	// returnHandle: the prefill leg returns a KV handle (pull).
	returnHandle bool
}

// mediaRef is one media item of a request and its embedding artifacts.
type mediaRef struct {
	digest    string
	tokens    int      // positions in the prompt
	artifacts []string // embedding artifact URIs (one per placement)
}

// genReq is a Generate call as the mock runs it.
type genReq struct {
	requestID       string
	tokens          []uint32 // the prompt, media expanded to pad values
	maxTokens       int
	stop            []string
	stopTokenIDs    []uint32
	topLogprobs     int    // -1: no logprobs
	reasoning       bool   // output starts inside the reasoning
	openThink       bool   // the model opens the reasoning with <think> itself (Qwen3)
	reasoningParser string // deepseek_v4 | qwen3
	toolParser      string // deepseek_v4 | hermes
	tools           []string
	toolChoice      string
	parseTools      bool
	callsOpened     bool // the prompt opened the tool-call block
	structured      bool
	media           []mediaRef
	timeoutMS       int64
	kv              kvRole
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

// setReasoning records the output markup of a request and whether the model
// has to open the reasoning block itself.
func (g *genReq) setReasoning(reasoning bool, info modelInfo) {
	g.reasoning = reasoning
	g.openThink = reasoning && g.reasoningParser == mwip.ParserQwen3 && !tail(g.tokens, 2, info.ThinkStart)
}

// fromMWIP reads a Generate request: the intermediate state holds the
// expanded prompt, the media with their embedding artifacts, and the
// parsing options the engine applies itself.
func fromMWIP(req *mwip.GenerateRequest, info modelInfo) *genReq {
	io, po := req.InferenceOptions, req.ParsingOptions
	g := &genReq{
		requestID: req.RequestID, tokens: req.TokenIDs,
		maxTokens: io.MaxTokens, stop: po.StopSequences, stopTokenIDs: io.StopTokenIDs, topLogprobs: -1,
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
			g.kv.returnHandle = true
		case d.Role == "decode" && d.Handshake == mwip.HandshakePull && d.Handle != nil:
			g.kv.source, g.kv.room = d.Handle.URL, d.Handle.Room
		case d.Role == "decode" && d.Handshake == mwip.HandshakeBootstrap && d.Prefill != nil:
			g.kv.source = d.Prefill.BootstrapAddr
		}
	}
	return g
}
