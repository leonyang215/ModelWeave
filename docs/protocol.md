# ModelWeave wire protocols

```
client ──OpenAI Chat / Responses, Anthropic Messages (stream or not)──▶ gateway
gateway ──POST /v1/render──────────────────────────────────────────────▶ tokenizer
         ◀── intermediate state: token list (media expanded to pad values),
             media plans and placements, conversation / inference / parsing options
gateway ──mwip: gRPC modelweave.mwip.v1.Engine/Generate (§2), the intermediate state as is, always streaming──▶ instances
         ◀── parsed deltas: content / reasoning / tool calls, usage
gateway ──SSE or one JSON body, formatted per client protocol──▶ client
instances ──gRPC modelweave.mwip.v1.Registry: one instance per heartbeat, load, KV events──▶ gateway   (§4)
```

Each process has one job:

| Process | Does | Does not |
|---|---|---|
| tokenizer | for every registered model: protocol conversion (OpenAI / Anthropic → the model's conversation), the model's prompt format (deepseek-recipe encodings, Hugging Face chat templates), tokenization, **media planning**: the token count of every image and video from its metadata and pixel controls, expanded in the token list as pad values | know engines; touch engine output |
| engine instance | generation, stop strings, detokenization, **parsing** of reasoning and tool calls (`parsing_options` name the parsers) | API formatting |
| gateway | auth, QoS, scheduling over the instances of each model and role, normal / PD / EPD orchestration; sends the intermediate state to the instances, every call **streaming**; formats the client response, streamed (SSE) or as one JSON body, for Chat Completions, Responses or Messages | parse model output |

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
  `enable_thinking`); engines whose parsers read them get them with the
  conversation.
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
     "placeholder": [129264]},                                       // the model's own placeholder of the item
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

## 2. Engines: mwip

Engines integrate through one protocol: **mwip**, the gRPC services defined
in [proto/modelweave/mwip/v1](../proto/modelweave/mwip/v1).
[mwip.md](mwip.md) describes it for engine implementers; the proto files are
its specification. There are no adapters for other engine APIs.

### 2.1 Instances and execution modes

Every model is served by its own **instances**, one model in one role each:
`encode` (E), `prefill` (P), `decode` (D), `unified`. An instance serves the
`Engine` service at its mwip `target` and registers with the gateway (§4.1).
Execution modes:

| mode | what runs | |
|---|---|---|
| `normal` | one instance does prefill and decode: a unified instance, or a decode instance registered with `normal: true` | no PD split |
| `pd` | P, then D, with a KV hand-off (§2.3) | PD split |
| `epd_*` | the encode stage before either | when the model has encode instances (§2.4) |

`mode: auto` decides per request (normal when the decode instance would
compute at most `normal.max_uncached_tokens`, holds `normal.min_cache_ratio`
of the prompt, or the session is pinned to it; pd otherwise), `mode: normal`
and `mode: pd` force one. The model's mode and handshake show in
`modelweave_model_info{model, mode, handshake}`.

Every call carries the **attempt** id (`<request id>-<n>`, shared by both
legs of a PD attempt) as its request id, the remaining first-token budget,
and, with `models[].engine_model`, the engine's served model name. Every
generate call is a **streaming** call, prefill legs included; non-streaming
clients are served by aggregating the stream in the gateway.

### 2.2 The Engine service: the intermediate state as is

| RPC | |
|---|---|
| `Generate(GenerateRequest) returns (stream GenerateResponse)` | one leg of a request |
| `Encode(EncodeRequest) returns (EncodeResponse)` | embeddings of media items, one artifact per placement |
| `Abort(AbortRequest) returns (AbortResponse)` | stop requests by id |
| `GetState(GetStateRequest) returns (GetStateResponse)` | the instance's load, for polling |

`GenerateRequest` is the tokenizer's result with the execution fields: the
same `token_ids`, `multimodal` (with the encode stage's
`embedding_artifacts`), `conversation`, `inference_options` and
`parsing_options` as §1.1, plus `request_id`, `model`, `disagg` (`mode`,
`role`, `handshake`, `room`, the `prefill` / `decode` peers, `kv_receiver`
for push, `kv_handle` for pull), `timeout_ms`, `session_key` and `priority`.
Values the JSON of §1.1 spells as strings are proto enums
(`THINKING_STAGE_REASONING_START`, `MODALITY_VIDEO`, `CONSTRAINT_KIND_JSON`,
...); tools are typed `Tool` messages with their JSON schema as a
`google.protobuf.Struct`.

The engine applies `stop_sequences`, parses reasoning with
`reasoning_parser` (from `thinking_initial_stage`), tool calls with
`tool_call_parser` (`parse_tool_calls`, `starts_in_tool_calls`) and JSON
output, and streams `GenerateResponse` messages, each one of:

| event | |
|---|---|
| `accepted` | the decode leg of a push handshake registered: its `kv_receiver` (address, room, cached prompt length); sent first |
| `delta` | parsed output: `reasoning`, `content`, `tool_calls` fragments, `token_ids`, `logprobs` |
| `finish` | end of a normal or decode leg: `reason` (`STOP`, `LENGTH`, `TOOL_CALLS`, `CONTENT_FILTER`), the matched `stop_sequence`, `usage` (prompt, cached, completion, reasoning tokens), `timing` |
| `prefill_done` | end of a prefill leg: prompt and cached tokens, the `kv_handle` of a pull handshake, `timing` |
| `error` | the leg failed: `code`, `message`, `retriable` (§2.5); ends the stream |

Canceling the call aborts the request; the gateway also sends `Abort` (best
effort). Unary calls fail with a gRPC status, optionally carrying an mwip
`Error` detail.

### 2.3 PD (prefill / decode split)

`pd.handshake` selects the rendezvous (default `push`). D's first token
decides success; a P failure before it aborts D and re-runs the pair
(`retry.whole_retries`).

| handshake | order | `disagg` |
|---|---|---|
| `push` | D, then P | D answers `accepted` with its `kv_receiver` (and cached length); P gets `kv_receiver` and pushes the KV D lacks |
| `bootstrap` | P and D concurrently | both get `room` and `bootstrap_room`; D fetches from `prefill.bootstrap_addr` (the address P registered) |
| `pull` | P, then D | P's `prefill_done` returns a `kv_handle`; D gets it and pulls the KV |

A prefill leg ends with `prefill_done`. The peers carry each instance's id,
target, `bootstrap_addr` and `engine_id` (its id in the engines' KV
connector, default the instance id). The KV transfer itself (RDMA, NIXL,
Mooncake, ...) is the engines' own.

### 2.4 EPD (encode stage)

The gateway sends each media item to an encode instance (`Engine.Encode`)
before P / D are reserved. The answer is one artifact per placement (an
image, or each kept video frame), **cached** by digest and plan. P / D get
them as `multimodal.items[].embedding_artifacts`. Without encode instances
(or with `encode.policy: never`) P / D encode media themselves from
`source`. `artifact_unavailable` before any output makes the gateway rebuild
the artifacts once.

### 2.5 Error codes

| code | meaning | gateway reaction |
|---|---|---|
| `capacity_exhausted` | the instance is full (hard admission) | retry on another instance |
| `normal_rejected` | decode instance declines a normal (non-PD) request | fall back to pd |
| `kv_transfer_failed` | KV hand-off failed | re-run pd (whole retry) |
| `artifact_unavailable` | embedding artifact missing | rebuild via E, retry once |
| `bad_request` | invalid request | fail the request (400) |
| `timeout`, `internal`, `aborted` | | retry before the first token if retriable |

Engines send them as `error` events (`ERROR_CODE_CAPACITY_EXHAUSTED`, ...) or
as gRPC status details; a plain gRPC status is classified by its code
(`UNAVAILABLE`: transport failure, `RESOURCE_EXHAUSTED`: capacity,
`INVALID_ARGUMENT` / `FAILED_PRECONDITION` / `OUT_OF_RANGE`: bad request,
`DEADLINE_EXCEEDED`: timeout, `CANCELLED` / `ABORTED`: aborted, others:
internal). The gateway never retries after the first output reached the
client. Transport failures and internal errors count against an instance's
health.

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

## 4. Control plane (instances → gateway)

### 4.1 The Registry (gRPC)

The gateway serves `modelweave.mwip.v1.Registry` on `server.grpc_listen`
(default `:8081`). Every instance registers itself with heartbeats and
reports its state; a heartbeat describes one instance.

| RPC | |
|---|---|
| `Heartbeat(HeartbeatRequest)` | register or refresh an instance, with its `EngineState`, KV events and draining flag; answers `registered` (new to the gateway) and the `ttl_ms` |
| `PublishKVEvents(PublishKVEventsRequest)` | KV events of an instance between heartbeats |
| `Deregister(DeregisterRequest)` | remove an instance at once (graceful shutdown) |

An instance is described by `Instance`: `id`, `model` (a model name or
alias), `role`, `target` (its Engine service), `normal`, `bootstrap_addr`,
`engine_id`, capacity hints (`kv_capacity_tokens`, `max_running_requests`,
`max_prefill_queue_tokens`), `block_size` and `labels`. A heartbeat without
an id, a role or a target fails with `INVALID_ARGUMENT`, one for a model the
gateway does not serve with `NOT_FOUND`; the status message says why. When
the gateway sets `server.control_token`, Registry calls carry it in the
`x-modelweave-token` metadata (`UNAUTHENTICATED` otherwise).

Lifecycle ([mwip.md](mwip.md#the-lifecycle-of-an-instance)): the first
heartbeat registers an instance in its model's pool for its role; heartbeats
every 1–5 s keep it registered; one without a heartbeat for
`discovery.heartbeat_ttl` expires; `draining: true` (or an operator's
`POST /admin/workers/{id}/drain`) stops new requests; `Deregister` removes it
at once; its next heartbeat registers it again.

Instances can also be listed statically in `discovery.workers` with the same
fields (`id`, `model`, `role`, `target`, ...). They send no heartbeats and
never expire.

### 4.2 EngineState

| field | |
|---|---|
| `running_requests`, `waiting_requests` | requests running and queued |
| `kv_used_tokens`, `kv_capacity_tokens`, `kv_usage` | KV cache occupancy (tokens, or a 0..1 ratio) |
| `prefill_queue_tokens` | prompt tokens waiting for prefill |
| `max_running_requests` | the instance's concurrency limit |
| `decode_tokens_per_sec`, `prefill_tokens_per_sec` | throughput |
| `cached_blocks`, `draining` | |

All fields are optional. The gateway merges reported usage with its own
reservations made since the report, so a stale report never over-commits an
instance. Heartbeats carry the state; with `engine_state.poll` the gateway
also asks every instance with `Engine.GetState` (static instances send no
heartbeats). Polls only update the load; registration lives on heartbeats.

### 4.3 KV events and block hashing

Prefix-cache routing compares **block hashes** of the token list as the
tokenizer returned it: media positions already hold their pad values, so the
list is hashed as is.

1. **Split** `token_ids` into blocks of `block_size` tokens (default 64); the
   partial tail is not hashed.
2. **Chain**: `h(-1) = xxh3_64(model_name)`;
   `h(i) = xxh3_64( le_u64(h(i-1)) || le_u32(tok[i*B]) || ... || le_u32(tok[i*B+B-1]) )`.

Engines that publish KV events (`kv_events` in heartbeats, or
`Registry.PublishKVEvents`) must hash the same way over the prompt they
run. `stored` lists hashes of blocks that became resident, `removed` those
evicted, `cleared` drops everything known about the instance.

### 4.4 External KV metadata services

`kv_index.type: remote`:
`POST {remote_url}/v1/prefix/match`
`{"model": "...", "block_size": 64, "block_hashes": [u64...], "candidates": ["decode-0", ...]}`
→ `{"matched_blocks": {"decode-0": 12, "decode-1": 0}}`.
On error or timeout the gateway falls back to its local approximate index.
The block hashes are those of §4.3 (media positions already hold their pad
values), so the service needs no knowledge of media or prompt formats.
