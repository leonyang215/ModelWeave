package engine

import (
	"encoding/json"
	"fmt"
	"net"
	"strconv"
	"strings"

	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/wire"
)

// defaultBootstrapPort is the disaggregation bootstrap port of SGLang and
// vLLM's MooncakeConnector.
const defaultBootstrapPort = 8998

func ptr[T any](v T) *T { return &v }

// templateKwargs are the chat template variables the prompt was rendered
// with: as the tokenizer reports them, else DeepSeek's thinking switches.
func templateKwargs(c mwip.Conversation) map[string]any {
	if len(c.ChatTemplateKwargs) > 0 {
		kw := make(map[string]any, len(c.ChatTemplateKwargs))
		for k, v := range c.ChatTemplateKwargs {
			kw[k] = v
		}
		return kw
	}
	kw := map[string]any{"thinking": c.ThinkingMode}
	if c.ReasoningEffort != "" {
		kw["reasoning_effort"] = c.ReasoningEffort
	}
	return kw
}

// toVLLM adapts a stage to vLLM's token-in GenerateRequest:
//
//   - media are collapsed to one placeholder each and sent as content_parts
//     with uuid = digest (vLLM expands and processes them, or, after the
//     encode stage, loads the embeddings named by ec_transfer_params);
//   - the engine parses its output (output_mode derender) with a
//     parse_context built from the conversation;
//   - split execution sets kv_transfer_params: the prefill leg generates one
//     token with do_remote_decode, the decode leg gets do_remote_prefill and,
//     per handshake, P's returned parameters (pull, NixlConnector) or P's
//     bootstrap server and a shared transfer_id (bootstrap,
//     MooncakeConnector).
func toVLLM(st *Stage) (*wire.VLLMGenerateRequest, error) {
	r := st.Render
	tokens, items, err := collapse(r, st.media())
	if err != nil {
		return nil, err
	}
	io := r.InferenceOptions
	sp := wire.VLLMSamplingParams{
		MaxTokens: io.MaxTokens, Temperature: io.Temperature, TopP: io.TopP, TopK: io.TopK, MinP: io.MinP,
		PresencePenalty: io.PresencePenalty, FrequencyPenalty: io.FrequencyPenalty, RepetitionPenalty: io.RepetitionPenalty,
		Seed: io.Seed, Stop: r.ParsingOptions.StopSequences, StopTokenIDs: io.StopTokenIDs,
		// Template tokens (</think>, tool-call markup) stay in the text the
		// parsers read.
		SkipSpecialTokens: ptr(false), SpacesBetweenSpecialTokens: ptr(false), Detokenize: ptr(true),
	}
	if io.Logprobs {
		n := 0
		if io.TopLogprobs != nil {
			n = *io.TopLogprobs
		}
		sp.Logprobs = &n
	}
	if c := io.OutputConstraint; c != nil {
		switch {
		case c.Kind == "json" && c.Schema == "":
			sp.StructuredOutputs = json.RawMessage(`{"json_object":true}`)
		case c.Kind == "json":
			sp.StructuredOutputs, _ = json.Marshal(map[string]json.RawMessage{"json": json.RawMessage(c.Schema)})
		case c.Kind == "regex":
			sp.StructuredOutputs, _ = json.Marshal(map[string]string{"regex": c.Schema})
		}
		// Tool calls are parsed by the engine's tool parser, not constrained.
	}
	kw := templateKwargs(r.Conversation)
	choice, _ := json.Marshal(r.Conversation.ToolChoice)
	kwargs, _ := json.Marshal(map[string]any{"chat_template_kwargs": kw})
	out := &wire.VLLMGenerateRequest{
		RequestID: st.AttemptID, TokenIDs: tokens, SamplingParams: sp, Model: st.EngineModel,
		ReasoningEnded: ptr(!r.ParsingOptions.StartsInReasoning()), ReasoningParserKwargs: kwargs,
		OutputMode: "derender",
		ParseContext: &wire.VLLMParseContext{
			Tools: r.Conversation.Tools, ToolChoice: choice, ParallelToolCalls: ptr(true),
			IncludeReasoning: true, ChatTemplateKwargs: kw,
		},
		Stream: true, StreamOptions: &wire.VLLMStreamOptions{IncludeUsage: true},
		ECTransferParams: st.ECParams, ModelWeave: extensions(st, items),
	}
	for _, it := range items {
		out.ContentParts = append(out.ContentParts, contentPart(&it))
	}
	d := st.Disagg
	if d.Mode != mwip.ModeSplit {
		return out, nil
	}
	transferID := "xfer-" + d.Room
	switch d.Role {
	case "prefill":
		out.SamplingParams.MaxTokens = 1
		params := map[string]any{"do_remote_decode": true, "do_remote_prefill": false}
		if d.Handshake == mwip.HandshakePull {
			for _, k := range []string{"remote_engine_id", "remote_block_ids", "remote_host", "remote_port"} {
				params[k] = nil
			}
		} else {
			params["transfer_id"] = transferID
		}
		out.KVTransferParams = params
	case "decode":
		params := map[string]any{}
		switch d.Handshake {
		case mwip.HandshakePull:
			for k, v := range d.KVTransferParams {
				params[k] = v
			}
		case mwip.HandshakeBootstrap:
			params["transfer_id"] = transferID
			if p := d.Prefill; p != nil {
				params["remote_bootstrap_addr"] = bootstrapURL(p)
				params["remote_engine_id"] = p.EngineID
			}
		default:
			params["transfer_id"] = transferID
		}
		params["do_remote_prefill"] = true
		params["do_remote_decode"] = false
		out.KVTransferParams = params
	}
	return out, nil
}

// contentPart is a media item as vLLM content_parts: the raw source, with
// uuid = digest as vLLM's multimodal cache key.
func contentPart(it *mwip.MediaItem) wire.VLLMContentPart {
	return wire.VLLMContentPart{Type: it.Modality + "_url", URL: it.Source.URL, UUID: it.Digest}
}

// vllmEncodeRequest is the encode leg of vLLM's EPD: a non-streaming
// generate on an EC-producer instance whose prompt holds just the
// placeholders of the items; its response carries the ec_transfer_params
// the prefill and decode legs pass on.
func vllmEncodeRequest(st *EncodeStage) (*wire.VLLMGenerateRequest, error) {
	req := &wire.VLLMGenerateRequest{
		RequestID: st.AttemptID, Model: st.EngineModel, OutputMode: "tokens",
		SamplingParams: wire.VLLMSamplingParams{MaxTokens: 1},
	}
	mm := &mwip.Multimodal{PlaceholderTokenID: st.PlaceholderTokenID}
	items := make([]mwip.MediaItem, len(st.Items))
	for i := range st.Items {
		it := st.Items[i]
		it.Placeholder = placeholder(mm, &it)
		if len(it.Placeholder) == 0 {
			return nil, fmt.Errorf("the model has no placeholder for %s input: it needs the mwip engine protocol", it.Modality)
		}
		req.TokenIDs = append(req.TokenIDs, it.Placeholder...)
		req.ContentParts = append(req.ContentParts, contentPart(&it))
		items[i] = it
	}
	hints := mediaHints(nil, items)
	req.ModelWeave = &wire.Extensions{AttemptID: st.AttemptID, TimeoutMS: st.TimeoutMS, Media: hints}
	return req, nil
}

// bootstrapURL is a prefill worker's bootstrap server as a URL.
func bootstrapURL(p *mwip.Peer) string {
	addr := p.BootstrapAddr
	if addr == "" {
		host, _ := hostPort(p)
		return fmt.Sprintf("http://%s:%d", host, defaultBootstrapPort)
	}
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	return strings.TrimRight(addr, "/")
}

// hostPort locates a prefill worker's bootstrap server: its bootstrap
// address, else the host of its URL on the default bootstrap port.
func hostPort(p *mwip.Peer) (string, int) {
	addr, explicit := p.BootstrapAddr, true
	if addr == "" {
		addr, explicit = p.URL, false
	}
	addr = strings.TrimPrefix(strings.TrimPrefix(addr, "http://"), "https://")
	if i := strings.IndexByte(addr, '/'); i >= 0 {
		addr = addr[:i]
	}
	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return addr, defaultBootstrapPort
	}
	if !explicit {
		return host, defaultBootstrapPort
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return host, defaultBootstrapPort
	}
	return host, port
}
