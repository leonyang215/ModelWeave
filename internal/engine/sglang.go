package engine

import (
	"github.com/leonyang215/ModelWeave/internal/mwip"
	"github.com/leonyang215/ModelWeave/internal/wire"
)

// toSGLang adapts a stage to SGLang's /generate:
//
//   - media are collapsed to the model's placeholders and sent raw as
//     image_data / video_data; SGLang processes them, or with EPD its
//     language-only prefill server dispatches them to its own encoders
//     (--encoder-urls), so the gateway's encode stage does not run for
//     SGLang;
//   - split execution names P's bootstrap server and a shared room on both
//     legs (SGLang's PD protocol).
//
// SGLang's /generate does not parse reasoning or tool calls: output arrives
// as raw text.
func toSGLang(st *Stage) (*wire.SGLangRequest, error) {
	r := st.Render
	tokens, items, err := collapse(r, st.media())
	if err != nil {
		return nil, err
	}
	io := r.InferenceOptions
	params := map[string]any{
		"max_new_tokens": io.MaxTokens, "skip_special_tokens": false, "spaces_between_special_tokens": false,
	}
	if io.Temperature != nil {
		params["temperature"] = *io.Temperature
	}
	if io.TopP != nil {
		params["top_p"] = *io.TopP
	}
	if io.TopK != nil {
		params["top_k"] = *io.TopK
	}
	if io.MinP != nil {
		params["min_p"] = *io.MinP
	}
	if io.PresencePenalty != nil {
		params["presence_penalty"] = *io.PresencePenalty
	}
	if io.FrequencyPenalty != nil {
		params["frequency_penalty"] = *io.FrequencyPenalty
	}
	if io.RepetitionPenalty != nil {
		params["repetition_penalty"] = *io.RepetitionPenalty
	}
	if io.Seed != nil {
		params["sampling_seed"] = *io.Seed
	}
	if s := r.ParsingOptions.StopSequences; len(s) > 0 {
		params["stop"] = s
	}
	if len(io.StopTokenIDs) > 0 {
		params["stop_token_ids"] = io.StopTokenIDs
	}
	if c := io.OutputConstraint; c != nil {
		switch {
		case c.Kind == "json" && c.Schema == "":
			params["json_schema"] = `{"type": "object"}`
		case c.Kind == "json":
			params["json_schema"] = c.Schema
		case c.Kind == "regex":
			params["regex"] = c.Schema
		}
	}
	out := &wire.SGLangRequest{
		RID: st.AttemptID, InputIDs: tokens, SamplingParams: params, Stream: true,
		ModelWeave: extensions(st, items),
	}
	if io.Logprobs {
		out.ReturnLogprob, out.ReturnText = true, true
		if io.TopLogprobs != nil {
			out.TopLogprobsNum = *io.TopLogprobs
		}
	}
	for _, it := range items {
		if it.Modality == "video" {
			out.VideoData = append(out.VideoData, it.Source.URL)
		} else {
			out.ImageData = append(out.ImageData, it.Source.URL)
		}
	}
	if d := st.Disagg; d.Mode == mwip.ModeSplit && d.Handshake == mwip.HandshakeBootstrap && d.Prefill != nil {
		host, port := hostPort(d.Prefill)
		room := d.BootstrapRoom
		out.BootstrapHost, out.BootstrapPort, out.BootstrapRoom = host, &port, &room
	}
	return out, nil
}

// toMWIP is the native request: the intermediate state as is, with its
// media (and their embedding artifacts) and the execution fields.
func toMWIP(st *Stage) *mwip.GenerateRequest {
	r := st.Render
	return &mwip.GenerateRequest{
		RequestID: st.AttemptID, Model: st.EngineModel, TokenIDs: r.TokenIDs, Multimodal: st.media(),
		Conversation: r.Conversation, InferenceOptions: r.InferenceOptions, ParsingOptions: r.ParsingOptions,
		Stream: true, Disagg: st.Disagg, TimeoutMS: st.TimeoutMS, SessionKey: st.SessionKey, Priority: st.Priority,
	}
}
