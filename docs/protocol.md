# ModelWeave wire protocols

```
client ──OpenAI Chat / Responses, Anthropic Messages (stream or not)──▶ gateway
gateway ──POST /v1/render──────────────────────────────────────────────▶ tokenizer
         ◀── intermediate state: token list (media expanded to pad values),
             media plans and placements, conversation / inference / parsing options
gateway ──one engine protocol, chosen at startup (§2), every generate call streaming──▶ workers
           mwip    POST /mwip/v1/generate      the intermediate state as is (NDJSON events)
           vllm    POST /inference/v1/generate adapted: token in, output_mode derender (SSE)
           sglang  POST /generate              adapted: input_ids + image_data / video_data (SSE)
         ◀── parsed deltas: content / reasoning / tool calls, usage
gateway ──SSE or one JSON body, formatted per client protocol──▶ client
workers ──POST /mwip/v1/heartbeat, kv_events──▶ gateway   (control plane)
```

Each process has one job:

| Process | Does | Does not |
|---|---|---|
| tokenizer | for every registered model: protocol conversion (OpenAI / Anthropic → the model's conversation), the model's prompt format (deepseek-recipe encodings, Hugging Face chat templates), tokenization, **media planning**: the token count of every image and video from its metadata and pixel controls, expanded in the token list as pad values | know engines; touch engine output |
| engine | generation, stop strings, detokenization, **parsing** of reasoning and tool calls (mwip `parsing_options` name the parsers, vLLM `output_mode: "derender"`) | API formatting |
| gateway | auth, QoS, scheduling, normal / PD / EPD orchestration; adapts the intermediate state to the engine protocol; calls every engine **streaming**; formats the client response, streamed (SSE) or as one JSON body, for Chat Completions, Responses or Messages | parse model output |

---

## 1. Tokenizer: the intermediate state

### 1.1 `POST /v1/render`

```json
{
  "request_id": "mw-3f2a...",
  "protocol": "chat_completions",        // chat_completions | responses | messages
  "model": "DeepSeek-V4.1-Flash",        // tokenizer-side model (name or alias); default: body.model
  "body": {"model": "deepseek-v4.1", "messages": [...], "stream": true},   // the client's body, verbatim
  "context_length": 131072               // optional cap of the model's context window
}
```

The answer is the conversation rendered into one token list by the model's
prompt format, with the options engines need to generate and parse it. It
names no engine. For a DeepSeek V4.1 chat request with `"stream": true`,
`"stop": ["END"]`, a system prompt and one user message holding a text block,
a 1920×1080 remote image (`max_long_side: 1024`) and a 10 s video
(`nframes: 4`), the tokenizer of the local stack answers:

```json
{
  "request_id": "mw-demo",
  "protocol": "chat_completions",
  "model": "DeepSeek-V4.1-Flash",
  "model_version": "deepseek-v4.1",      // the model family that rendered the prompt
  "prompt_tokens": 1194,                 // == len(token_ids)
  "token_ids": [0, 128799, ..., 339, 1728996728, 1728996728, ...,   // image: 366 × its pad value
                271, 18, 16, 18, 9651, 1831243460, ...,             // "0.0 seconds" + frame 0
                1782006125, 128804, 128821],                        // last frame, Assistant, <think>
  "multimodal": { ... },                 // §1.2, absent without media
  "conversation": {"thinking_mode": true, "reasoning_effort": "high", "tool_choice": "auto",
                   "response_format": "text",           // text | json_object | json_schema | regex
                   "tools": [...],                      // OpenAI function tools, when given
                   "chat_template_kwargs": {"thinking": true, "reasoning_effort": "high"}},
  "inference_options": {"max_tokens": 65536,            // resolved against the context window
                        "temperature": 1.0, "top_p": 1.0, "logprobs": true,
                        "stop_token_ids": [1],          // the model's stop tokens
                        "seed": 7, "top_logprobs": 2,   // when given (also top_k, min_p, penalties)
                        "output_constraint": {"kind": "json", "schema": "{...}"}},   // json | regex | tool_call | tool_call_required
  "parsing_options": {"reasoning_parser": "deepseek_v4", "tool_call_parser": "deepseek_v4",
                      "parse_tool_calls": false, "starts_in_tool_calls": false, "parse_json_output": false,
                      "thinking_initial_stage": "reasoning_start",   // reasoning_start | reasoning_continuation | content_continuation
                      "stop_sequences": ["END"]},
  "stream": true,                        // what the client asked for
  "response_options": {"include_usage": false},   // stream_options.include_usage
  "render_artifact": {"version": 2, "encoding": "deepseek-v4.1",
                      "token_layout": {"cacheable_prefix": {"offset": 0, "length": 1193},
                                       "generation_primer": {"offset": 1193, "length": 1}},
                      "segments": [{"role": "system", "offset": 1, "length": 32},
                                   {"role": "user", "offset": 33, "length": 1159},
                                   {"role": "assistant", "offset": 1192, "length": 2}]},
  "timing_us": {"parse": 62, "convert": 14, "render": 8, "tokenize": 499, "media": 128, "total": 718}
}
```

* `conversation.chat_template_kwargs` are the template variables the prompt
  was rendered with (DeepSeek `thinking` / `reasoning_effort`, Qwen
  `enable_thinking`); adapters pass them to engines whose parsers read them.
* `parsing_options.reasoning_parser` / `tool_call_parser` name the markup of
  the output: `deepseek_v4` (reasoning ends at `</think>`, DSML tool calls),
  `qwen3` (`<think>…</think>`, the model may open it itself), `hermes`
  (`<tool_call>{"name": …, "arguments": …}</tool_call>`). Absent: no reasoning
  / no tool calls.
* `render_artifact.token_layout.cacheable_prefix` is stable when the
  conversation continues; `generation_primer` is the opened assistant turn;
  `segments` are the messages' token ranges.
* `context_length` caps the model's context window (the gateway's
  `models[].context_length`); by default the registry's window applies.

Errors use the HTTP status of the failure (400 invalid request, 404 unknown
model, 500 otherwise) with a body **shaped for the request's protocol**
(OpenAI `{"error": {...}}`, Anthropic `{"type": "error", "error": {...}}`), so
the gateway passes it to the client unchanged.

#### Model families

| family | prompt format | thinking | parsers | media |
|---|---|---|---|---|
| `deepseek-v4.1`, `deepseek-v4` | deepseek-recipe encodings | `thinking`, `reasoning_effort`; on by default | `deepseek_v4` | V4.1: images, videos |
| `qwen3` | the model's chat template (ChatML) | `chat_template_kwargs.enable_thinking`, `thinking.type`, effort `none`; on by default | `qwen3`, `hermes` | — |
| `qwen3-vl` | the model's chat template | — (Instruct; `thinking = "always"` for Thinking models) | `hermes` | images, videos |
| `chat-template` | any Hugging Face chat template | configured | configured | — |

DeepSeek requests follow recipe's rules: thinking mode forces
`temperature: 1.0` (and `top_p ≥ 0.95` on V4.1), `n` must be 1, presence /
frequency penalties and `parallel_tool_calls` are dropped, the default
`max_tokens` is 8192 (65536 thinking). Chat-template models keep the
request's sampling options, fill missing ones from `generation_config.json`
(Qwen3: `temperature` 0.6, `top_p` 0.95, `top_k` 20), take their stop tokens
from it (`<|im_end|>`, `<|endoftext|>`), and default `max_tokens` to the
family's value (Qwen3: 8192, 32768 thinking; Qwen3-VL: 16384).

A Qwen3 request with a tool renders the model's own chat template — the
tools in the system turn, thinking on by default (the prompt ends with
`<|im_start|>assistant\n` and the model opens `<think>` itself):

```json
{"model": "Qwen3-8B", "model_version": "qwen3", "prompt_tokens": 133,
 "token_ids": [151644, 8948, 198, 2, 13852, ..., 151644, 77091, 198],
 "conversation": {"thinking_mode": true, "tool_choice": "auto", "response_format": "text", "tools": [...],
                  "chat_template_kwargs": {"enable_thinking": true}},
 "inference_options": {"max_tokens": 32768, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                       "logprobs": false, "stop_token_ids": [151645, 151643],
                       "output_constraint": {"kind": "tool_call", "schema": "[...]"}},
 "parsing_options": {"reasoning_parser": "qwen3", "tool_call_parser": "hermes", "parse_tool_calls": true,
                     "starts_in_tool_calls": false, "parse_json_output": false,
                     "thinking_initial_stage": "reasoning_start"},
 "render_artifact": {"version": 2, "encoding": "qwen3",
                     "token_layout": {"cacheable_prefix": {"offset": 0, "length": 130},
                                      "generation_primer": {"offset": 130, "length": 3}},
                     "segments": [{"role": "system", "offset": 0, "length": 121},
                                  {"role": "user", "offset": 121, "length": 9},
                                  {"role": "assistant", "offset": 130, "length": 3}]}}
```

Chat templates are rendered like transformers' `apply_chat_template`: Jinja
with `trim_blocks` / `lstrip_blocks`, Python string methods, the `tojson`
filter of `json.dumps`, `raise_exception` (its message becomes a 400),
`strftime_now`. Messages are prepared like vLLM prepares them: Chat
Completions as sent (`developer` as `system`, `reasoning_content` kept,
tool-call arguments as objects), Anthropic tool results as `tool` messages and
thinking blocks as `reasoning_content`, Responses function calls and outputs
as assistant tool calls and `tool` messages; templates that read plain-string
content get text parts joined with newlines. A final assistant message is
continued (`continue_final_message`, Chat `prefix: true`, Anthropic prefill).
The special (control) tokens of the rendered prompt are only those the
template wrote: a `<|im_start|>` or `<|image_pad|>` a client spells stays
text.

### 1.2 Media: blocks, metadata and pixel controls

Media are content blocks of the client request: Chat `image_url`,
`video_url`, `file`; Messages `image`, `video` (also inside `tool_result`);
Responses `input_image`, `input_video` (in `input[].content` / `output`).
Besides the protocol's own fields a block may carry three ModelWeave
extensions, at block level:

```json
{"type": "image_url", "image_url": {"url": "https://...", "detail": "high"},
 "metadata": {"digest": "<sha256 hex>", "width": 1920, "height": 1080},
 "pixel_control_options": {"max_long_side": 1344, "max_total_pixels": 4000000}}

{"type": "video_url", "video_url": {"url": "https://..."},
 "metadata": {"digest": "<sha256 hex>", "width": 1280, "height": 720,
              "duration_ms": 10000, "total_frames": 300, "video_fps": 30},
 "frame_extract_options": {"fps": 2, "nframes": null,
                           "pixel_control_options": {"max_long_side": 672}}}
```

The tokenizer never fetches media. `metadata` holds what the caller measured:

| field | image | video |
|---|---|---|
| `digest` | sha256 of the bytes; **required** for remote URLs, computed for data URLs | same |
| `width`, `height` | **required** for remote URLs, read from the header of data URLs | required |
| `duration_ms`, `video_fps` | — | required |
| `total_frames` | — | optional (default `duration × fps`) |

**pad_value.** Every position of an item holds its pad value:
`pad_value(reference) = 2^30 + (BE_u64(sha256(reference)[0..8]) & (2^30 − 1))`.
The reference of an image is its digest; of a video's kept frame `k` it is
`"{digest}_kf_{k:04}"`. Equal media give equal tokens, so the prefix cache
works across requests; all pad values are ≥ `pad_shift` (2^30), outside any
vocabulary.

**Pixel controls** apply to every model: the long side is limited first —
`detail: "low"` → 512, `pixel_control_options.max_long_side` if given (for
video frames `frame_extract_options.pixel_control_options`) — then the
model's own resize runs; `max_total_pixels` is checked against the result
(defaults when pixel controls are present: 3584² for an image, 301,056,000
over a video's kept frames). Sources must be at most 8192 px a side (4096
with 15 or more media), with an aspect ratio of at most 200, and a request
takes at most 600 media.

**DeepSeek V4.1.** An image is sized by recipe's
`ImageTokenSpec::v41().calc_resize` (14 px patches, 3× downsample, at most
1024 tokens, at least 544² px) and takes `n_h × (n_w + 1) + 2` positions.
Video frames are sampled with qwen-vl-utils' `smart_nframes`
(`frame_extract_options.fps`, default 2.0, or a fixed `nframes`;
ties-to-even rounding; temporal patch size 1), limited to a long side of
672 px (pixel controls override) and planned like an image. A video renders
as its kept frames, each the timestamp text `"{t:.1} seconds"` followed by the
frame's pad region. The defaults are overridable per model in the tokenizer
config (`[models.vision]`: `video_temporal_patch_size`, `video_default_fps`,
`video_frame_max_long_side`).

**Qwen3-VL.** An image is sized by Qwen2-VL's `smart_resize` to multiples of
32 px (16 px patches, 2×2 merge) within the processor's pixel range
(65,536 – 16,777,216) and takes `(h/32) × (w/32)` positions between the
template's `<|vision_start|>` and `<|vision_end|>`. A video is sampled like
transformers' Qwen3-VL video processor: `n = ⌊total / video_fps × fps⌋`
frames (`fps` default 2, clamped to 4–768 and the frame count; or a fixed
`nframes`) at `linspace(0, total − 1, n)`; two frames form one temporal patch
(a last odd frame is repeated), whose timestamp is the mean of its frames'
times; frames are sized by the video `smart_resize` over all sampled frames
(pixel range 4,096 – 25,165,824, per-frame cap of 768 merged tokens as in
qwen-vl-utils). A video renders as one group per temporal patch:
`"<{t:.1} seconds>"` + `<|vision_start|>` + the patch's pad region +
`<|vision_end|>` — the layout vLLM expands Qwen3-VL videos to. The processor
values come from the model's `preprocessor_config.json` and
`video_preprocessor_config.json` (and `[models.vision]` overrides).

`multimodal` of the DeepSeek request above:

```json
{
  "pad_shift": 1073741824,
  "placeholder_token_id": 129264,       // <｜image｜>: one unexpanded image in the prompt format
  "items": [
    {"index": 0, "modality": "image",
     "digest": "55c64d0f...04e8",
     "source": {"kind": "url", "url": "https://img.example/photo.jpg"},
     "metadata": {"width": 1920, "height": 1080}, "pixel_control_options": {"max_long_side": 1024},
     "plan": {"width": 1036, "height": 588, "tokens": 366},          // 1920x1080 -> long side 1024 -> V4.1 resize
     "span": {"offset": 41, "length": 366},
     "placements": [{"offset": 41, "length": 366, "pad_value": 1728996728,
                     "reference": "55c64d0f...04e8", "modality": "image"}],
     "placeholder": [129264]},                                       // what engines that expand media get
    {"index": 1, "modality": "video",
     "digest": "67905ad3...6e6d",
     "source": {"kind": "url", "url": "https://video.example/clip.mp4"},
     "metadata": {"width": 1280, "height": 720, "duration_ms": 10000, "total_frames": 300, "video_fps": 30.0},
     "frame_extract_options": {"nframes": 4},
     "plan": {"width": 728, "height": 420, "tokens": 192,            // per kept frame (672 px long side, 544² floor)
              "nframes": 4, "kept_frames": 4, "timestamps": [0.0, 2.5, 5.0, 7.5]},
     "span": {"offset": 408, "length": 784},                         // 4 × ("{t:.1} seconds" 4 tokens + 192)
     "placements": [
       {"offset": 412, "length": 192, "pad_value": 1831243460, "reference": "67905ad3...6e6d_kf_0000", "modality": "frame", "timestamp": 0.0},
       {"offset": 608, "length": 192, "pad_value": 1997747760, "reference": "67905ad3...6e6d_kf_0001", "modality": "frame", "timestamp": 2.5},
       ...]}                                                         // no placeholder: V4.1 has none for videos
  ]
}
```

The same request to Qwen3-VL plans the image as 1024×576 = 576 positions
(`"placeholder": [151655]`, `<|image_pad|>`) and the video as two temporal
patches at 1.7 s and 8.3 s of 720 positions each (1152×640), its span from
the first `<1.7 seconds>` to the last `<|vision_end|>` and
`"placeholder": [151652, 151656, 151653]`
(`<|vision_start|><|video_pad|><|vision_end|>`).

`detail` (`"low"` / `"high"`), `pixel_control_options` and
`frame_extract_options` are echoed when the request had them.

Invalid media fail the render with a 400 that names the block
(`messages.0.content.1: metadata.digest (sha256 of the image bytes) is
required for remote media`, `... exceeds max_total_pixels 100000`, ...); a
text-only model rejects media (`model Qwen3-8B does not accept images or
videos`), and chat-template models accept media in user and tool messages.

### 1.3 Utility endpoints

* `POST /v1/tokenize_text` `{"model": "...", "text": "...", "parse_special": true}` → `{"token_ids": [...]}`
* `POST /v1/detokenize` `{"model": "...", "token_ids": [...], "skip_special": false, "per_token": true}` → `{"text": "...", "pieces": [...]}`
* `GET /v1/models` → every registered model: `name`, `aliases`, `family`,
  `context_length`, `eos_token_ids`, `reasoning_parser`, `tool_call_parser`,
  `thinking` (`toggle` / `always` / `none`), `think_start_token_id`,
  `think_end_token_id`, `modalities`
* `GET /healthz`, `GET /readyz`, `GET /metrics`

### 1.4 Registering models

The tokenizer reads a TOML file (`--config`, `MW_TOKENIZER_CONFIG`) with one
`[[models]]` entry per model; `configs/tokenizer.example.toml` documents
every option and registers the bundled DeepSeek V4.1, Qwen3-8B and
Qwen3-VL-8B-Instruct. An entry names the model and its aliases, its `family`,
and its Hugging Face files (`path` to a directory holding `tokenizer.json`,
`tokenizer_config.json` or `chat_template.json` / `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`,
`video_preprocessor_config.json`; single files can be set instead); optional
settings choose the context window, the thinking switch, the parsers, stop
tokens, the template's content format, default `max_tokens` and vision
overrides. Without `--config` the tokenizer serves one model from flags
(`--model`, `--aliases`, `--family`, `--model-path` or `--tokenizer-file`).

---

## 2. Engines

### 2.1 Engine protocol and execution modes

The gateway speaks **one engine protocol per model**:

| setting | scope |
|---|---|
| `engine.protocol` in the config (default `mwip`) | every model |
| startup option `modelweave-gateway -engine-protocol mwip\|vllm\|sglang` (env `MW_ENGINE_PROTOCOL`) | replaces `engine.protocol` |
| `models[].engine_protocol` | that model; wins over both |

The chosen protocol shows in the startup log and in
`modelweave_model_info{model, engine_protocol, mode, handshake, gateway_encode}`.

Worker roles: `encode` (E), `prefill` (P), `decode` (D), `unified`. Execution
modes:

| mode | what runs | |
|---|---|---|
| `normal` | one worker does prefill and decode: a unified worker, or a decode worker registered with `normal: true` | no PD split |
| `pd` | P, then D, with a KV hand-off (§2.5) | PD split |
| `epd_*` | the encode stage before either | when the gateway encodes (§2.6) |

`mode: auto` decides per request (normal when the decode worker would compute
at most `normal.max_uncached_tokens`, holds `normal.min_cache_ratio` of the
prompt, or the session is pinned to it; pd otherwise), `mode: normal` and
`mode: pd` force one.

Every call carries the **attempt** id (`<request id>-<n>`, shared by both
legs of a PD attempt) as its request id, the remaining first-token budget,
and, with `models[].engine_model`, the engine's served model name. Every
generate call is a **streaming** call, prefill legs included; non-streaming
clients are served by aggregating the stream in the gateway.

### 2.2 `mwip`: the intermediate state as is

`POST /mwip/v1/generate` — the tokenizer's result plus execution fields:

```json
{
  "request_id": "mw-3f2a...-1",
  "model": "",
  "token_ids": [...], "multimodal": {...},          // items carry embedding_artifacts after E
  "conversation": {...}, "inference_options": {...}, "parsing_options": {...},
  "stream": true,
  "disagg": {"mode": "split", "role": "decode", "handshake": "push", "room": "mw-3f2a...-1",
             "bootstrap_room": 4821930213,
             "prefill": {"worker_id": "prefill-0", "url": "...", "bootstrap_addr": "...", "engine_id": "..."},
             "decode": {...},
             "kv_receiver": {...},                   // push: given to P
             "kv_handle": {...}},                    // pull: given to D
  "timeout_ms": 29750, "session_key": "user-42", "priority": "normal"
}
```

`disagg.mode` is `normal` or `split`. The engine applies `stop_sequences`,
parses reasoning with `reasoning_parser` (from `thinking_initial_stage`),
tool calls with `tool_call_parser` (`parse_tool_calls`,
`starts_in_tool_calls`) and JSON output, and answers with NDJSON events:

```
{"type":"accepted","kv_receiver":{"worker_id":"decode-0","url":"...","room":"...","cached_tokens":896},"worker_id":"decode-0"}
{"type":"prefill_done","prompt_tokens":1183,"cached_tokens":0,"kv_handle":{...},"usage":{...},"timing":{...}}
{"type":"delta","reasoning":"The","token_ids":[671]}
{"type":"delta","tool_calls":[{"index":0,"id":"call_..","type":"function","function":{"name":"search_docs"}}],"token_ids":[3816]}
{"type":"delta","content":"Hello","token_ids":[19923],"logprobs":[{"token":"Hello","logprob":-0.05,"top_logprobs":[...]}]}
{"type":"finish","reason":"stop","stop_reason":"END","usage":{"prompt_tokens":1183,"completion_tokens":77,
 "prompt_tokens_details":{"cached_tokens":896},"completion_tokens_details":{"reasoning_tokens":17}},"timing":{...}}
{"type":"error","code":"kv_transfer_failed","message":"...","retriable":true}
```

`accepted` is the decode leg's registration (push), `prefill_done` ends a
prefill leg; `finish.reason` is `stop`, `length` or `tool_calls`.
Rejections before streaming use HTTP status codes with
`{"type": "error", "code": "capacity_exhausted", "message": "...", "retriable": true}`.

`POST /mwip/v1/encode` (unary) builds the embeddings of media items, one
artifact per placement:

```json
{"request_id": "...", "attempt_id": "...-enc0-1", "model": "deepseek-v4.1",
 "items": [{...item as rendered: plan and placements...}], "timeout_ms": 29000}
→ {"artifacts": [{"reference": "9c0d...", "uri": "s3://...", "tokens": 594, "encoder": "enc-0"}], "encode_ms": 41.2}
```

`POST /mwip/v1/abort {"request_id": "..."}` stops a request (best effort; the
gateway also closes the connection).

### 2.3 `vllm`: vLLM's token-in generate

`POST /inference/v1/generate` (`vllm/entrypoints/scale_out/token_in_token_out`):

| intermediate state | vLLM `GenerateRequest` |
|---|---|
| `token_ids` | `token_ids` with every media item **collapsed** back to its `placeholder` (Qwen-VL `<\|image_pad\|>`, `<\|vision_start\|><\|video_pad\|><\|vision_end\|>`; DeepSeek V4.1 `<｜image｜>`): vLLM's processor expands it |
| `multimodal.items[]` | `content_parts: [{"type": "image_url" \| "video_url", "url": source.url, "uuid": digest}]` (`uuid` is vLLM's multimodal cache key) |
| `inference_options` | `sampling_params`: `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, presence / frequency / repetition penalties, `seed`, `stop_token_ids`, `logprobs` (top-n), `structured_outputs` (`json` / `json_object` / `regex`), `skip_special_tokens: false`, `detokenize: true` |
| `parsing_options.stop_sequences` | `sampling_params.stop` |
| `conversation` | `parse_context` (`tools`, `tool_choice`, `include_reasoning`, `chat_template_kwargs` — the template variables the prompt was rendered with, e.g. Qwen `enable_thinking`), `reasoning_parser_kwargs` |
| `parsing_options.thinking_initial_stage` | `reasoning_ended` |
| — | `output_mode: "derender"` (vLLM parses its output), `stream: true`, `stream_options.include_usage` |

ModelWeave-only data travels in a `modelweave` object stock vLLM ignores:
`media` hints (`digest`, `modality`, `tokens`, `pad_value`, `placeholder`
and, for items with text inside their span such as Qwen3-VL video
timestamps, the expanded `layout` as run-length `[token, count]` pairs), the
push handshake's `disagg`, `timeout_ms`. The response is vLLM's
`GenerateDerenderStreamResponse` SSE (`delta` with `content`, `reasoning`,
`tool_calls`; a usage chunk; `data: [DONE]`); abort is
`POST /inference/v1/abort_requests {"request_ids": [...]}`.

### 2.4 `sglang`: SGLang's native generate

`POST /generate`:

| intermediate state | SGLang |
|---|---|
| attempt id | `rid` |
| `token_ids` (media collapsed to their placeholders) | `input_ids` |
| `multimodal.items[].source.url` | `image_data`, `video_data` (SGLang processes the media itself) |
| `max_tokens`, `seed` | `sampling_params.max_new_tokens`, `sampling_seed` |
| `temperature`, `top_p`, `top_k`, `min_p`, penalties, `stop_token_ids`, `stop_sequences` | the same names in `sampling_params`, `stop` |
| `output_constraint` | `json_schema` / `regex` |
| `logprobs`, `top_logprobs` | `return_logprob`, `top_logprobs_num`, `return_text_in_logprobs` |

The stream is `data: {"text", "output_ids", "meta_info": {"id", "finish_reason": {"type", "matched"}, "prompt_tokens", "completion_tokens", "cached_tokens"}}`
… `data: [DONE]`; cumulative and `--incremental-streaming-output` streams
are both understood. Abort: `POST /abort_request {"rid": ...}`.

**Limitation:** SGLang's `/generate` runs no reasoning or tool parser, so
output arrives as raw text (`content`, including `</think>`, DSML or
`<tool_call>` markup);
the gateway does not parse it. Use `mwip` or `vllm` for reasoning and tool
calls.

### 2.5 PD (prefill / decode split) per protocol

`pd.handshake` selects the rendezvous (default `push` for mwip, `bootstrap`
for vllm and sglang). D's first token decides success; a P failure before it
aborts D and re-runs the pair (`retry.whole_retries`).

| handshake | order | mwip (`disagg`) | vllm (`kv_transfer_params`) | sglang |
|---|---|---|---|---|
| `push` | D, then P | D answers `accepted` with its `kv_receiver` (and cached length); P gets `kv_receiver` and pushes the KV D lacks | P `{do_remote_decode, transfer_id}` + `modelweave.disagg.kv_receiver`; D `{do_remote_prefill, transfer_id}` (extension) | — |
| `bootstrap` | P and D concurrently | both get `room`, `bootstrap_room`; D gets `prefill.bootstrap_addr` | P `{do_remote_decode: true, transfer_id}`; D `{do_remote_prefill: true, transfer_id, remote_bootstrap_addr, remote_engine_id}` (MooncakeConnector) | both legs `bootstrap_host`, `bootstrap_port`, `bootstrap_room` of P's bootstrap server |
| `pull` | P, then D | P's `prefill_done` returns a `kv_handle`; D gets it | P `{do_remote_decode: true, remote_*: null}` returns `kv_transfer_params`; D gets them + `do_remote_prefill` (NixlConnector) | — |

Prefill legs of vllm and sglang generate one token (`max_tokens: 1`); a mwip
prefill leg ends with `prefill_done`. P's bootstrap address is
`bootstrap_addr` from its registration (default: its host on port 8998),
`remote_engine_id` its `engine_id` (default: the worker id).

vLLM's NixlConnector returns `kv_transfer_params` with the prefill response;
stock vLLM includes them only in non-streaming responses. ModelWeave keeps
every generate call streaming, so a vLLM prefill engine used with `pull` must
put them on its final stream chunk (`{"choices": [], "usage": {...},
"kv_transfer_params": {...}}`); prefer `bootstrap` with stock engines. The
KV transfer itself (RDMA, NIXL, Mooncake, ...) is engine-internal.

### 2.6 EPD (encode stage) per protocol

| protocol | encode stage | P / D receive |
|---|---|---|
| `mwip` | gateway → E worker `POST /mwip/v1/encode`; artifacts **cached** by digest and plan | `multimodal.items[].embedding_artifacts` (one per placement: an image, or each kept frame) |
| `vllm` | gateway → EC-producer instance: a non-streaming `POST /inference/v1/generate` whose prompt is just the items' placeholders, with their `content_parts` and `max_tokens: 1`; it answers `ec_transfer_params` keyed by `uuid` (vLLM's EPD proxy flow). Not cached | `content_parts` + `ec_transfer_params`: the engines' EC connector loads the embeddings |
| `sglang` | **engine-managed**: a language-only prefill server dispatches media to its own encoders (`--encoder-urls`) and overrides any router choice, so the gateway has no E stage | `image_data`, `video_data` |

Without encoders (or `encode.policy: never`) P / D encode media themselves.
`artifact_unavailable` before any output makes the gateway rebuild once.

**Media without a placeholder** need `mwip`: vLLM and SGLang expand media
themselves, from the model's placeholder. Qwen3-VL has one for images and
videos; DeepSeek V4.1 has none for videos, so the gateway answers 400 for
DeepSeek videos on vllm and sglang models before scheduling anything.

### 2.7 Support matrix

| | mwip | vllm | sglang |
|---|---|---|---|
| normal | ✓ | ✓ | ✓ |
| pd: push / bootstrap / pull | ✓ / ✓ / ✓ | ✓ (extension) / ✓ / ✓ (stream must carry `kv_transfer_params`) | — / ✓ / — |
| EPD, images | gateway E stage, cached artifacts | gateway E stage, `ec_transfer_params` | engine-managed |
| videos | ✓ (one artifact per kept frame) | with a video placeholder (Qwen3-VL); else 400 | same as vllm, engine-managed |
| engine-side parsing | ✓ `parsing_options` | ✓ `output_mode: derender` | raw text |

### 2.8 Error codes

| code | meaning | gateway reaction |
|---|---|---|
| `capacity_exhausted` | the worker is full (hard admission) | retry on another worker |
| `normal_rejected` | decode worker declines a normal (non-PD) request | fall back to pd |
| `kv_transfer_failed` | KV hand-off failed | re-run pd (whole retry) |
| `artifact_unavailable` | embedding artifact missing | rebuild via E, retry once |
| `bad_request` | invalid request | fail the request (400) |
| `timeout`, `internal`, `aborted` | | retry before the first token if retriable |

vLLM `{"error": {...}}` bodies and in-stream errors are mapped to these codes
(a ModelWeave code in `type` is used as is). The gateway never retries after
the first output reached the client. HTTP 5xx and transport errors count
against a worker's health.

---

## 3. Gateway output

The gateway turns the engine's deltas into the client's protocol
(`internal/format`). Streaming and non-streaming clients get the same output;
the JSON body is the aggregation of the stream.

| | Chat Completions | Responses | Messages |
|---|---|---|---|
| reasoning | `reasoning_content` | `reasoning` item, `response.reasoning_text.delta` | `thinking` block, `thinking_delta` |
| content | `content` | `message` item, `response.output_text.delta` | `text` block, `text_delta` |
| tool calls | `tool_calls` (streamed fragments) | `function_call` items, `response.function_call_arguments.delta` | `tool_use` blocks, `input_json_delta` |
| finish | `finish_reason` `stop` / `length` / `tool_calls` | `status` `completed` / `incomplete` (`max_output_tokens`) | `stop_reason` `end_turn` / `max_tokens` / `stop_sequence` (+ `stop_sequence`) / `tool_use` |
| usage | `prompt_tokens_details.cached_tokens`, `completion_tokens_details.reasoning_tokens`, DeepSeek `prompt_cache_hit_tokens` / `_miss_tokens` | `input_tokens_details.cached_tokens`, `output_tokens_details.reasoning_tokens` | `input_tokens` (uncached), `cache_read_input_tokens`, `output_tokens` |
| stream end | `data: [DONE]` (usage chunk first with `include_usage`) | `response.completed` / `response.incomplete` | `message_delta`, `message_stop` |

A stop after tool calls is reported as `tool_calls` / `tool_use`. Responses
echo request fields (`instructions`, `max_output_tokens`, `tools`,
`tool_choice`, `metadata`, ...). An error after output started ends the
stream with `data: {"error": ...}` (Chat), `event: error` (Responses,
Messages).

---

## 4. Control plane (worker → gateway)

### 4.1 `POST /mwip/v1/heartbeat`

Registers a worker (dynamic discovery) and reports its state. Send every
1–5 s; a worker is removed after `discovery.heartbeat_ttl` without
heartbeats.

```json
{
  "worker": {"id": "decode-0", "model": "deepseek-v4.1", "role": "decode", "url": "http://decode-0:9000",
             "normal": true, "kv_capacity_tokens": 262144, "max_running_requests": 32,
             "max_prefill_queue_tokens": 0, "block_size": 64,
             "bootstrap_addr": "", "engine_id": "", "labels": {}},
  "state": { ... EngineState ... },
  "kv_events": {"stored": [...], "removed": [...], "cleared": false},
  "draining": false
}
```

A worker does not declare an engine protocol: the gateway's configuration
decides how it is called. `normal: true` lets a decode worker run normal
requests. Answer: `{"ok": true, "ttl_ms": 15000}`. When the gateway sets
`server.control_token`, control-plane calls must carry it in
`X-ModelWeave-Token` (401 otherwise). `POST /mwip/v1/deregister {"worker_id": "..."}`
removes a worker on graceful shutdown. Workers can also be configured
statically (`discovery.workers`) with the same fields.

### 4.2 EngineState

```json
{"running_requests": 7, "waiting_requests": 1, "kv_used_tokens": 52000, "kv_capacity_tokens": 262144,
 "kv_usage": 0.2, "prefill_queue_tokens": 4096, "max_running_requests": 32,
 "decode_tokens_per_sec": 3100, "prefill_tokens_per_sec": 41000, "cached_blocks": 1800, "draining": false}
```

All fields are optional. The gateway merges reported usage with its own
reservations made since the report, so a stale report never over-commits a
worker. Engines without heartbeats are polled through Prometheus instead
(`engine_state.poll.format: prometheus` with the `vllm` or `sglang` preset).

### 4.3 KV events and block hashing

Prefix-cache routing compares **block hashes** of the token list as the
tokenizer returned it: media positions already hold their pad values, so the
list is hashed as is.

1. **Split** `token_ids` into blocks of `block_size` tokens (default 64); the
   partial tail is not hashed.
2. **Chain**: `h(-1) = xxh3_64(model_name)`;
   `h(i) = xxh3_64( le_u64(h(i-1)) || le_u32(tok[i*B]) || ... || le_u32(tok[i*B+B-1]) )`.

Engines that publish KV events (`kv_events` in heartbeats, or
`POST /mwip/v1/kv_events {"worker_id": ..., "stored": [...]}`) must hash the
same way over their expanded prompt. `stored` lists hashes of blocks that
became resident, `removed` those evicted, `cleared` drops everything known
about the worker.

### 4.4 External KV metadata services

`kv_index.type: remote`:
`POST {remote_url}/v1/prefix/match`
`{"model": "...", "block_size": 64, "block_hashes": [u64...], "candidates": ["decode-0", ...]}`
→ `{"matched_blocks": {"decode-0": 12, "decode-1": 0}}`.
On error or timeout the gateway falls back to its local approximate index.
The block hashes are those of §4.3 (media positions already hold their pad
values), so the service needs no knowledge of media or prompt formats.
