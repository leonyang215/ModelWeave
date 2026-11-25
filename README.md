# ModelWeave

An open-source gateway and scheduler for LLM serving.

ModelWeave sits between API clients and inference engines. It accepts OpenAI
(Chat Completions, Responses) and Anthropic (Messages) requests, streaming or
not, has them converted into one engine-neutral intermediate state, schedules
them onto encode / prefill / decode workers with prefix-cache and load
awareness, calls the engines in the protocol chosen at startup, orchestrates
disaggregated execution, and formats the engines' output as API responses.

It serves several model families side by side: **DeepSeek V4 / V4.1**,
**Qwen3** (hybrid thinking, tool calls), **Qwen3-VL** (images and videos),
and any model with a Hugging Face chat template.

It consists of two processes, with a strict division of labour:

* **`modelweave-tokenizer`** (Rust) — protocol conversion and tokenization
  only, for **every registered model at once**. It turns an API request into
  the **intermediate state**: the conversation rendered into one token list
  by the model's prompt format, with the options engines need to generate and
  parse. DeepSeek V4 / V4.1 prompts are rendered with
  [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe); Qwen3,
  Qwen3-VL and other models with their own Hugging Face chat template, rendered
  exactly like transformers' `apply_chat_template` (verified token for token).
  Control tokens only come from the template: a client cannot forge a turn or
  a media placeholder by spelling one. Images and videos are already expanded:
  from the caller's metadata (digest, size, duration, fps) and pixel controls
  the tokenizer computes how many positions each takes — following the
  model's own processor (DeepSeek V4.1's image spec, Qwen-VL's `smart_resize`
  and Qwen3-VL's timestamped temporal patches) — and fills them with pad
  values derived from the media digest (`2^30 + sha256(reference)`, per kept
  frame for videos).
* **`modelweave-gateway`** (Go) — API front, tenant QoS, admission queue and
  overload protection, worker registry, cache- and load-aware scheduling,
  normal / PD / EPD orchestration with retries and fallback, Prometheus
  metrics and access logs. It calls engines with one of three protocols — the
  intermediate state as is (`mwip`), adapted to vLLM, or adapted to SGLang —
  selected with `-engine-protocol`; calls **every engine generate streaming**;
  and **formats the client response** itself — SSE for streaming clients, one
  aggregated JSON body for the others.

**Engines parse their own output** (mwip `parsing_options` name the parsers —
`deepseek_v4`, `qwen3` reasoning, `hermes` tool calls; vLLM
`output_mode: "derender"`; SGLang's `/generate` returns raw text). See
[docs/protocol.md](docs/protocol.md). A mock engine (`modelweave-mockengine`)
serves all three protocols at once — prefix KV cache, KV hand-off, encode
stage, timing, faults and engine-side parsers of DeepSeek and Qwen output — so
the whole system runs locally without GPUs.

```
                                   ┌──────────── modelweave-gateway (Go) ─────────────┐
OpenAI / Anthropic clients ──────▶ │ auth · QoS · overload · queue                    │   -engine-protocol:
   (stream or not)          ◀───── │ scheduler (cache-aware / load-aware / affinity)  │──▶ mwip   /mwip/v1/generate
   SSE or JSON, formatted          │ orchestrator (normal · PD · EPD)                 │──▶ vllm   /inference/v1/generate
   by the gateway                  │ adapters: intermediate state → engine protocol   │──▶ sglang /generate
                                   │ formatter (chat · responses · messages)          │    E / P / D / unified workers,
                                   │ registry ◀─ heartbeats · KV events · polling      │    always streaming
                                   └──────────────┬───────────────────────────────────┘
                                                  │ POST /v1/render (unary)
                                   ┌──────────────▼───── modelweave-tokenizer (Rust) ──┐
                                   │ model registry: DeepSeek · Qwen3 · Qwen3-VL · ... │
                                   │ API request → token list (media → pad values)     │
                                   │ + media plans/placements + options                │
                                   └────────────────────────────────────────────────────┘
```

## Features

**Models**
* One tokenizer process serves every registered model; requests select one by
  name or alias. A model is registered from its Hugging Face files
  (`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, the
  processor configs of vision models) and a family:
  * `deepseek-v4.1`, `deepseek-v4` — deepseek-recipe prompt encodings, DSML
    tool calls, images and videos (V4.1);
  * `qwen3` — Qwen3 / Qwen3-2507 / QwQ: ChatML template, `enable_thinking`
    switch, `<think>` reasoning, hermes `<tool_call>` tool calls;
  * `qwen3-vl` — Qwen3-VL: the same plus images (`smart_resize`, 16 px
    patches, 2×2 merge) and videos (frames sampled like the Qwen3-VL processor,
    two per temporal patch, each patch `<t seconds>` + vision tokens);
  * `chat-template` — any other Hugging Face chat template, with parsers,
    stop tokens and the thinking switch from the configuration.
* Chat templates run on a Jinja engine with transformers' semantics
  (`trim_blocks` / `lstrip_blocks`, Python string methods, the `tojson`
  filter with `json.dumps` spacing, `raise_exception`, `strftime_now`),
  template content formats are detected like vLLM does, and tool-call
  arguments are passed to templates as objects.
* Per-model stop tokens and sampling defaults come from
  `generation_config.json`; context windows from the registry (the gateway
  can cap them).

**Protocols**
* OpenAI `/v1/chat/completions`, `/v1/responses`; Anthropic `/v1/messages`;
  streaming (SSE) and non-streaming; reasoning (`reasoning_content` / thinking
  blocks / reasoning items), tool calls, JSON / JSON-schema / regex output,
  stop sequences, logprobs, images and videos (with `metadata`,
  `pixel_control_options`, `frame_extract_options`, `detail`).
* Intermediate state: the token list with media expanded to pad values, the
  media plans and placements (and each item's placeholder for engines that
  expand media themselves), conversation / inference / parsing options —
  no engine in it.
* Engine protocols, one per model (`engine.protocol`, the
  `-engine-protocol` startup option, or `models[].engine_protocol`):
  `mwip` (the intermediate state as is, NDJSON events), `vllm`
  (token-in `/inference/v1/generate`, media collapsed to the model's
  placeholders + `content_parts`, `kv_transfer_params`, `ec_transfer_params`),
  `sglang` (`/generate` with `input_ids`, `image_data`, `video_data`,
  `bootstrap_*`).

**Execution modes**
* **normal** — one worker runs prefill and decode (no PD split): a unified
  worker, or a decode worker that accepts normal requests, when it caches
  most of the prompt, the uncached part is short, or the session lives there;
  falls back to PD on failure.
* **PD** — prefill and decode on separate workers with three KV handshakes,
  each in the protocol's own fields: `push` (decode registers, prefill pushes
  KV, Mooncake-style), `bootstrap` (concurrent; SGLang
  `bootstrap_host/port/room`, vLLM MooncakeConnector `kv_transfer_params`),
  `pull` (prefill returns a KV handle — vLLM NixlConnector).
* **EPD** — media are encoded before P/D: mwip on encode workers with
  content-addressed artifacts (one per image or video frame, reused), vLLM on
  an EC producer whose `ec_transfer_params` go to P/D; SGLang encodes inside
  the engine. Text-only requests skip E.
* `mode: auto` decides normal vs PD per request.

**Scheduling**
* Policies per role: `cache_aware` (SGLang-router style with load-imbalance
  guard), `weighted` (cache/load/affinity scores), `session_affinity`
  (rendezvous hashing with bounded load), `least_load`, `power_of_two`,
  `round_robin`, `random`.
* Cache awareness adapters: `approx` (learned from routing), `events` (engine
  KV events), `remote` (external KV metadata service), `none`. Media
  positions hold pad values derived from the media digest, so the chained
  block hashes of the token list cover multimodal prefixes too.
* Load awareness: engine state by push (MWIP heartbeats, dynamic registration)
  or pull (MWIP state endpoint, Prometheus scraping with vLLM / SGLang
  presets), merged with the gateway's own KV / prefill reservations.
* Passive health checks (ejection after consecutive failures), draining,
  retries on other workers before the first token, one absolute first-token
  deadline, abort propagation on disconnect / timeout; renders fail over to
  another tokenizer replica.

**QoS**
* Tenants by API key: priority class, fair-share weight, RPM / TPM token
  buckets (reconciled with real usage), concurrency caps — 429 + Retry-After.
* Admission queue: strict priorities, virtual-token-counter fairness between
  tenants, backfill, shedding of lower priorities, deadline — 503 + Retry-After.
* Utilization-level overload controller (L0–L4 with hysteresis) and a
  priority matrix; cache-hit requests are exempt from drops.

**Observability** — Prometheus metrics for every stage (TTFT and its
segments, TPOT, tokens incl. cached and reasoning, streaming vs non-streaming
clients, engine calls by protocol and role, the engine protocol of every
model, media items and their positions, finish reasons, scheduling reasons,
predicted cache-hit ratio, fallbacks, retries, aborts, queueing, overload
level, per-worker state; tokenizer renders per model) and one structured JSON
access-log line per request; see [docs/observability.md](docs/observability.md).

## Quick start (DeepSeek V4.1, Qwen3 and Qwen3-VL with mock engines)

Requirements: Docker with Compose v2.

```sh
make images      # builds modelweave/{tokenizer,gateway,mockengine,e2e}:dev
make up          # gateway :38080, second gateway :38081, tokenizer :38090
make e2e         # 44 end-to-end tests inside the compose network
```

Behind a proxy, export `http_proxy` / `https_proxy` (and `no_proxy`) before
`make images`; they are passed to the build as build arguments.

The stack (`deploy/compose`) runs one tokenizer with the real DeepSeek V4.1,
Qwen3 and Qwen3-VL tokenizers, two gateways (one calling engines over mwip,
one over vLLM's protocol) and 25 mock workers. Try it:

```sh
# Short prompt: runs normal on a decode worker (no PD split)
curl -s localhost:38080/v1/chat/completions -H 'Authorization: Bearer sk-team-a' \
  -H 'Content-Type: application/json' -D - \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"Hello"}]}'

# The same through Qwen3: <think> reasoning, then the answer; thinking can be
# switched off with chat_template_kwargs, thinking.type or reasoning effort "none"
curl -s localhost:38080/v1/chat/completions -H 'Authorization: Bearer sk-team-a' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello"}],
       "chat_template_kwargs":{"enable_thinking":false}}'

# The models the tokenizer serves, and what it returns: the intermediate state
curl -s localhost:38090/v1/models | jq '.models[] | {name, family, reasoning_parser, tool_call_parser, modalities}'
curl -s localhost:38090/v1/render -H 'Content-Type: application/json' -d '{"protocol":"chat_completions",
  "model":"qwen3-vl","body":{"model":"qwen3-vl","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"https://img.example/a.jpg"},
   "metadata":{"digest":"'$(printf a | sha256sum | cut -d" " -f1)'","width":1920,"height":1080}}]}]}}' \
  | jq '.prompt_tokens, (.multimodal.items[0] | {plan, placeholder})'

# Which worker ran it, and why
#   X-ModelWeave-Mode: normal | pd    X-ModelWeave-Reason: short_prompt | prefix_hit | long_prompt | ...
#   X-ModelWeave-Decode-Worker / -Prefill-Worker / -Cached-Tokens / -Attempt
curl -s -H 'X-ModelWeave-Token: local-control-token' localhost:38080/admin/workers
curl -s localhost:38080/metrics | grep modelweave_model_info
```

A long prompt runs PD (`prefill-*` → `decode-*`); sending it again is a
prefix-cache hit on the decode worker that received its KV, which then runs it
normal. Add `--profile monitoring` to start Prometheus on `:39090`.

Models in the local stack:

| Model | Tokenizer model | Engine protocol | Demonstrates |
|---|---|---|---|
| `deepseek-v4.1` | DeepSeek-V4.1-Flash | mwip | auto normal / PD (push), EPD with cached artifacts for images and videos, session affinity, cache-aware routing |
| `deepseek-v4.1-vllm` | DeepSeek-V4.1-Flash | vllm | normal on the decode worker, PD with MooncakeConnector-style bootstrap, EPD through an EC producer (`ec_transfer_params`) |
| `deepseek-v4.1-pull` | DeepSeek-V4.1-Flash | mwip | PD with the pull handshake (`kv_handle`) |
| `deepseek-v4.1-sglang` | DeepSeek-V4.1-Flash | sglang | `/generate`, bootstrap handshake, engine-managed encoding (the registered encoder is never called), raw-text output |
| `deepseek-v4.1-fallback` | DeepSeek-V4.1-Flash | mwip | normal rejected → PD fallback |
| `deepseek-v4.1-normal` | DeepSeek-V4.1-Flash | mwip | unified workers, session affinity, retry around and ejection of a broken worker |
| `deepseek-v4.1-tiny` | DeepSeek-V4.1-Flash | mwip | one slot: admission queue, queue-full rejection, client-disconnect abort; static worker polled via `/metrics`; context capped at 128K |
| `qwen3` | Qwen3-8B | mwip | hybrid thinking, hermes tool calls in every client protocol, auto normal / PD (push), prefix hits |
| `qwen3-vl` | Qwen3-VL-8B-Instruct | mwip | images and videos planned like the Qwen3-VL processor, EPD with one artifact per image and kept frame |
| `qwen3-vl-sglang` | Qwen3-VL-8B-Instruct | sglang | media collapsed to Qwen-VL placeholders, sent as `image_data` / `video_data` |

The second gateway (`:38081`) is started with `-engine-protocol vllm`: it
drives the same `deepseek-v4.1`, `qwen3` and `qwen3-vl` workers over vLLM's
protocol (pull, NixlConnector style; Qwen3-VL videos as `video_url` content
parts through the EC producer) and routes with engine KV events only.
Tenants: `sk-team-a` (high priority), `sk-team-b`, `sk-limited` (3 RPM),
`sk-batch` (low priority).

## Registering models

The tokenizer reads a TOML file with a `[[models]]` list
([configs/tokenizer.example.toml](configs/tokenizer.example.toml), baked into
the container image); every option is documented there:

```toml
[[models]]
name = "Qwen3-8B"                 # names the gateway may send
aliases = ["qwen3"]
family = "qwen3"                  # deepseek-v4.1 | deepseek-v4 | qwen3 | qwen3-vl | chat-template
path = "assets/tokenizers/qwen3"  # tokenizer.json, tokenizer_config.json, generation_config.json, ...
context_length = 40960

[[models]]
name = "my-model"                 # any Hugging Face chat template
family = "chat-template"
path = "/models/my-model"
thinking = "toggle"               # toggle | always | none
reasoning_parser = "qwen3"        # <think> markup
tool_call_parser = "hermes"
```

The gateway maps its served models to them (`models[].tokenizer_model`),
together with routing, engine protocol and QoS
([configs/gateway.example.yaml](configs/gateway.example.yaml)).

## Building from source

Go 1.26+ and Rust 1.88+ (edition 2024):

```sh
make build       # bin/modelweave-gateway, bin/modelweave-mockengine, bin/modelweave-tokenizer
make test        # go vet + race tests, cargo tests (incl. golden prompts from transformers)
make lint        # gofmt, rustfmt, clippy

bin/modelweave-tokenizer --config configs/tokenizer.example.toml --listen 127.0.0.1:8090
bin/modelweave-gateway -config configs/gateway.example.yaml -engine-protocol mwip   # or vllm, sglang
bin/modelweave-mockengine --id d0 --role decode --normal --listen :9000 \
  --advertise-url http://127.0.0.1:9000 --gateway-url http://127.0.0.1:8080 \
  --tokenizer-url http://127.0.0.1:8090 --model deepseek-v4.1 --tokenizer-model DeepSeek-V4.1-Flash
```

The tokenizer also runs a single model from flags or `MW_TOKENIZER_*`
variables (`--model`, `--family`, `--model-path`, `--context-length`).
`tokenizer/testdata/gen_qwen_golden.py` regenerates the reference prompts the
Qwen tests compare against (transformers' `apply_chat_template` and the
reference tokenizer).

## Connecting real engines

Pick the engine protocol at startup (`-engine-protocol`, or
`engine.protocol` / `models[].engine_protocol` in the config):

* **mwip** — the native protocol: engines receive the intermediate state
  (`POST /mwip/v1/generate`) with media already planned and expanded, the
  embedding artifacts of the encode stage, and the PD fields in `disagg`; they
  answer with parsed NDJSON events ([docs/protocol.md §2.2](docs/protocol.md)).
  It carries every media item, including DeepSeek V4.1 videos.
* **vllm** — `POST /inference/v1/generate` with `output_mode: "derender"`
  (start vLLM with the model's reasoning and tool parsers, e.g.
  `--reasoning-parser qwen3 --tool-call-parser hermes`) and
  `models[].engine_model` set to the served model name; the chat template
  variables the prompt was rendered with travel in `parse_context`. PD uses
  `kv_transfer_params`: `bootstrap` maps to the MooncakeConnector (default),
  `pull` to the NixlConnector — whose stock streaming responses do not carry
  the prefill's `kv_transfer_params` ([§2.5](docs/protocol.md)). EPD sends
  media to an EC-producer instance registered as an encode worker and passes
  its `ec_transfer_params` on.
* **sglang** — `POST /generate` (`input_ids`, `image_data`, `video_data`,
  `bootstrap_*` for PD; bootstrap only); cumulative and incremental streams.
  Encoding stays inside SGLang (`--encoder-urls` on the prefill server). The
  native API does not parse reasoning or tool calls, so output is raw text —
  use mwip or vllm for those.
* Media reach vLLM and SGLang as the model's placeholders (Qwen-VL
  `<|image_pad|>` and `<|vision_start|><|video_pad|><|vision_end|>`, DeepSeek
  V4.1 `<｜image｜>`); a model without a placeholder for a modality (DeepSeek
  V4.1 videos) is served over mwip only.
* Workers announce themselves with heartbeats (`POST /mwip/v1/heartbeat`;
  `normal: true` lets a decode worker run normal requests) or are listed
  statically; engine load is read from heartbeats or scraped from vLLM /
  SGLang Prometheus metrics without any engine change.

## Related projects

ModelWeave builds on ideas of open-source serving systems:
[vLLM](https://github.com/vllm-project/vllm) (token-in generate, KV and EC
connectors, chat template content formats),
[SGLang](https://github.com/sgl-project/sglang) and its model gateway
(cache-aware routing, PD bootstrap, multimodal pad values, Rust chat
templates), [llm-d](https://github.com/llm-d/llm-d) (scorer composition),
[Mooncake](https://github.com/kvcache-ai/Mooncake) (KV transfer),
[transformers](https://github.com/huggingface/transformers) (chat templates,
Qwen-VL processors), [qwen-vl-utils](https://github.com/QwenLM/Qwen3-VL)
(video sampling) and
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) (DeepSeek
prompt encodings).

## Documentation

* [docs/architecture.md](docs/architecture.md) — components, request lifecycle, model registry, modes, scheduling, QoS
* [docs/protocol.md](docs/protocol.md) — the intermediate state and media planning per model family, engine protocols (mwip / vLLM / SGLang), PD and EPD per protocol, output formatting, control plane, block hashing
* [docs/observability.md](docs/observability.md) — metrics, logs, queries
* [README.zh.md](README.zh.md) — 中文说明

## Layout

```
cmd/                 gateway and mock-engine binaries
internal/            Go packages (gateway, format, orchestrator, scheduler, kvindex, registry,
                     enginestate, qos, tokenizer client, engine adapters, mwip, wire, mockengine, obs)
tokenizer/           Rust tokenizer service (model registry, DeepSeek and chat-template renderers,
                     protocol conversion, media planning); testdata/ holds the Qwen golden prompts
assets/tokenizers/   bundled tokenizers: DeepSeek V4.1 (MIT), Qwen3 and Qwen3-VL (Apache-2.0)
configs/             reference gateway and tokenizer configurations
deploy/compose/      local stack: tokenizer, two gateways, mock engines, Prometheus, e2e runner
test/e2e/            end-to-end tests
docs/                architecture, protocol, observability
```

## License

Apache License 2.0 ([LICENSE](LICENSE)). The bundled DeepSeek tokenizer files
are MIT licensed ([assets/tokenizers/deepseek-v4.1/LICENSE](assets/tokenizers/deepseek-v4.1/LICENSE));
the bundled Qwen tokenizer and processor files are Apache-2.0 licensed
([assets/tokenizers/qwen3/LICENSE](assets/tokenizers/qwen3/LICENSE)).
