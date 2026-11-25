# Architecture

ModelWeave is two processes plus the engines they drive:

| Process | Language | Role |
|---|---|---|
| `modelweave-gateway` | Go | API front (OpenAI Chat Completions / Responses, Anthropic Messages, streaming or not), auth and tenant QoS, admission queue and overload shedding, worker registry, cache- and load-aware scheduling, normal / PD / EPD orchestration, **engine protocol adapters** (mwip, vLLM, SGLang), retries and fallback, **response formatting**, metrics and access logs |
| `modelweave-tokenizer` | Rust | **protocol conversion and tokenization only**, for every registered model at once: API request → the intermediate state (the conversation rendered into one token list by the model's prompt format, images and videos already expanded to their pad-value positions, generation and parsing options) |
| workers | any | engines in roles encode (E), prefill (P), decode (D) or unified, called with one of three protocols: the intermediate state as is (`mwip`), vLLM's token-in generate, or SGLang's `/generate`. They detokenize and **parse** their output. `modelweave-mockengine` simulates them (all three protocols at once) for tests |

The tokenizer serves several model families side by side (§ Tokenizer
internals): DeepSeek V4 / V4.1 prompts are rendered with
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe); Qwen3,
Qwen3-VL and any other model with its Hugging Face chat template, rendered
like transformers' `apply_chat_template`. Media planning (pad values, pixel
controls, frame sampling) follows each model's own processor.

```
                 ┌──────────────────────────────── modelweave-gateway ────────────────────────────────┐
 client ──HTTP──▶│ api ─▶ render ─▶ QoS ─▶ overload ─▶ encode ─▶ plan+queue ─▶ execute ──────▶ format   │──▶ workers
 (stream or not) │          │      limiter  L0–L4        (E)   scheduler   (orchestrator)    (SSE|JSON) │   mwip | vllm | sglang
                 │          ▼                                     ▲              │                     │   (always streaming)
                 │   modelweave-tokenizer         registry ◀── heartbeats / pollers                    │
                 │   (unary /v1/render)           kvindex  ◀── routing / KV events                     │
                 │                                                engine adapters: intermediate state  │
                 │                                                → mwip / vLLM / SGLang request       │
                 └────────────────────────────────────────────────────────────────────────────────────┘
```

Who does what:

* the **tokenizer** converts and tokenizes, and plans media: it knows how many
  positions every image and video takes, from the caller's metadata and pixel
  controls, but nothing about engines. It is out of the output path
  (stateless, unary);
* the **gateway** chooses how to call engines — the engine protocol is a
  startup option — and adapts the intermediate state to it, with PD and EPD
  expressed in each protocol's own fields (protocol §2);
* the **engine** applies stop strings, detokenizes and parses reasoning and
  tool calls, streaming parsed deltas;
* the **gateway** never parses: it relays the deltas through a formatter that
  writes the client's protocol, as SSE for streaming clients or as one JSON
  body for the others. Every engine generate call streams, whatever the
  client asked.

## Request lifecycle

1. **Front** (`internal/gateway/api.go`). Read the body, authenticate the API
   key to a tenant, resolve the served model, pick the session key
   (`X-Session-Id`, `prompt_cache_key`, `user`, `metadata.user_id`).
2. **Render.** One unary `POST /v1/render` to any tokenizer replica with the
   request verbatim. The answer is the intermediate state: the token list
   (every media position holding its pad value), the media items with their
   plans and placements, and the conversation, inference and parsing options
   (protocol §1). Invalid requests come back already shaped as the protocol's
   error body and are returned unchanged. Media a model has no placeholder
   for (DeepSeek V4.1 videos) are rejected here (400) when the model is served
   over vLLM or SGLang, which expand media themselves.
3. **Prefix hashes.** The token list is hashed as is into chained 64-token
   block hashes (`internal/kvindex`, protocol §4.3).
4. **Tenant QoS.** RPM / TPM token buckets and a concurrency cap per tenant
   (`internal/qos/limiter.go`). TPM is charged up front with prompt + reserved
   output and reconciled with the real usage at the end. Violations return
   **429 + Retry-After**.
5. **Overload shedding.** An overload controller per model maps pool
   utilization (KV usage, prefill queue, admission-queue fill) to levels
   L0–L4 with hysteresis; a priority matrix admits, drops probabilistically or
   rejects (**503 + Retry-After**). Requests whose prefix is mostly cached are
   exempt from drops (they are cheap).
6. **Encode (EPD).** If the prompt has media, the model has encoders and the
   engine protocol encodes outside the engine (mwip, vllm), every item is
   encoded on an E worker (parallel, retried on other encoders): mwip builds
   content-addressed artifacts (cached; one per placement), vLLM's EC producer
   returns `ec_transfer_params`. No P/D capacity is held meanwhile. SGLang
   encodes inside the engine.
7. **Plan + admission queue** (`internal/orchestrator/plan.go`). The planner
   picks the execution mode and reserves workers. When no worker has capacity
   the request waits in the model's admission queue (strict priority classes,
   VTC fair sharing between tenants, backfill, shedding of lower priorities,
   deadline → 503).
8. **Execute** (`internal/orchestrator/execute.go`). Run the plan. The engine
   adapters (`internal/engine`) turn each stage into a streaming call in the
   model's engine protocol and translate its stream (mwip NDJSON, vLLM or
   SGLang SSE) into events (accepted, prefill done, delta, finish). Deltas go
   through the formatter (`internal/format`) straight to the client: SSE
   chunks for streaming clients, one JSON body at the end for the others.
9. **Finish.** Release reservations, reconcile TPM, record the session's decode
   worker, emit metrics and one structured access-log line.

## Engine protocols

`engine.protocol` (default `mwip`), the startup option
`-engine-protocol` / `MW_ENGINE_PROTOCOL` (replaces it) and
`models[].engine_protocol` (wins) select how a model's workers are called:

| | `mwip` | `vllm` | `sglang` |
|---|---|---|---|
| endpoint | `POST /mwip/v1/generate` | `POST /inference/v1/generate` | `POST /generate` |
| request | the intermediate state + `disagg` | token in, media collapsed to the model's placeholders + `content_parts`, `output_mode: derender` | `input_ids`, `image_data`, `video_data` |
| PD handshakes | push, bootstrap, pull | push (extension), bootstrap (Mooncake), pull (Nixl) | bootstrap |
| encode stage | gateway, `/mwip/v1/encode`, cached artifacts | gateway, EC producer → `ec_transfer_params` | inside the engine |
| videos | yes | with a video placeholder (Qwen3-VL); DeepSeek V4.1: 400 | same as vllm |
| output | parsed by the engine | parsed by the engine | raw text |

Workers do not declare a protocol; the mock engine serves all three, so the
same workers can be driven by gateways started with different options
(`deploy/compose`: `gateway` speaks mwip, `gateway-events` runs with
`-engine-protocol vllm`).

## Execution modes

| Mode | Workers | Chosen when |
|---|---|---|
| `normal` | one worker does prefill and decode: a unified worker, or a decode worker registered with `normal: true` | only unified workers serve the model, `mode: normal`, or in `auto`: the decode worker caches most of the prompt (`normal.min_cache_ratio`), the uncached part is short (`normal.max_uncached_tokens`), or the session is pinned to it |
| `pd` | P + D | the uncached part of the prompt is long, or `mode: pd` |
| `epd_*` | E, then either of the above (metric label) | the prompt contains media and the gateway runs the encode stage |

`normal` runs a request without PD split. In `mode: auto` the decision is per
request (`planAuto`):

1. session affinity: a follow-up turn runs normal on the decode worker that
   served the session (it holds the conversation's KV);
2. otherwise the decode worker is chosen cache-aware among workers that run
   normal requests, and the request runs normal if that worker would compute
   at most `max_uncached_tokens` tokens or holds at least `min_cache_ratio`
   of the prompt;
3. otherwise the same decode worker is kept and a prefill worker is chosen
   cache-aware for a `pd` run.

### PD handshakes

The orchestrator implements three rendezvous styles (protocol §2.5), selected
per model with `pd.handshake` (default push for mwip, bootstrap for vllm and
sglang), and expressed in each protocol's own fields:

* **bootstrap**: P and D are sent concurrently with a shared room; D fetches
  KV from P's bootstrap server — mwip `disagg.prefill.bootstrap_addr`, SGLang
  `bootstrap_host/port/room`, vLLM MooncakeConnector `kv_transfer_params`
  (`transfer_id`, `remote_bootstrap_addr`, `remote_engine_id`).
* **pull**: P runs first and returns a KV handle — mwip `prefill_done` with a
  `kv_handle`, vLLM NixlConnector `kv_transfer_params` on P's stream — which D
  uses to pull the KV.
* **push** (Mooncake-style): D registers and acknowledges with its KV receiver
  and the length of the prefix it already caches; only then is P sent, and P
  pushes the missing KV to D.

In all three, D's first token decides success: once D streams, P's completion
no longer matters. A P failure before D's first token aborts D and re-runs the
pair (`retry.whole_retries`).

### Retries, fallback and deadlines

* One **absolute first-token deadline** per request
  (`server.first_token_timeout`) bounds every stage and retry; each stage gets
  the remaining budget as `timeout_ms`.
* Before the first token reaches the client:
  * a failed decode / normal / unified attempt is retried on another worker
    (`retry.stage_attempts`);
  * a failed **normal** attempt on a decode worker falls back to **pd** once
    (not charged to the retry budget), e.g. on `normal_rejected` or capacity
    errors (`normal.fallback_to_pd`);
  * a missing embedding artifact (`artifact_unavailable`) triggers one rebuild
    through E.
* After the first token, errors end the stream (an SSE error event); output is
  never replayed.
* Each attempt has its own id, used as the engine request id and as the
  pairing token of P and D.
* Client disconnects and timeouts abort the engines (connection close plus a
  best-effort abort in the engine's protocol: `/mwip/v1/abort`,
  `/inference/v1/abort_requests` or SGLang `/abort_request`). Stop strings are
  the engine's job: it stops itself.

### Reservations

The gateway keeps an optimistic ledger per worker
(`internal/registry/worker.go`):

* decode / unified / normal: KV tokens = prompt + reserved output
  (`min(max_tokens, reservation.output_tokens)`: engines grow longer outputs
  themselves), rounded to blocks;
* prefill / normal: uncached prompt tokens, optionally attention-weighted
  `u·(1+β(cached+u/2))` (`reservation.uncached_beta`), released when prefill
  completes.

Engine-reported usage is the truth: the effective usage of a worker is
`max(own reservations, reported usage + reservations made since the report)`,
so stale reports never over-commit a worker and several gateway replicas can
share workers. Workers still do the hard check and answer
`capacity_exhausted`, which the gateway treats as "retry elsewhere".

## Scheduling

`internal/scheduler` runs one pass per stage:

1. snapshot the candidates of the model and role;
2. filter: healthy (passive health ejects workers after consecutive transport
   errors or 5xx), not draining, able to run normal requests when required,
   capacity (KV, running requests, prefill queue);
3. prefix-cache lookup through the `kvindex.Index` adapter;
4. load per candidate (queued prefill tokens for P, running requests and KV
   utilization for D);
5. the policy picks; the reservation is taken under the same lock.

Policies (per role, `routing.<role>.policy`):

| Policy | Behaviour |
|---|---|
| `cache_aware` (default) | SGLang-router style: when load is balanced, the longest cached prefix wins if it covers `cache_threshold` of the prompt; otherwise, and whenever load is imbalanced (`balance_abs_threshold` and `balance_rel_threshold`), the least loaded worker wins. A miss seeds the chosen worker's cache, so a shared prefix follows it. |
| `weighted` | score = `cache_weight·cache_ratio − load_weight·norm_load + affinity_weight·[session match]` (llm-d style scorer composition) |
| `session_affinity` | rendezvous hashing of the session key with bounded load; falls back to another policy for keyless requests |
| `least_load`, `power_of_two`, `round_robin`, `random` | classic load balancing |

### Cache-awareness adapters (`kv_index.type`)

| Type | Source of truth |
|---|---|
| `approx` | learned from routing decisions, bounded per worker by its KV capacity, with TTL |
| `events` | engine KV events (`stored` / `removed` / `cleared`) from heartbeats or `POST /mwip/v1/kv_events`; short optimistic entries bridge the gap until the event arrives |
| `remote` | an external KV metadata service (`POST /v1/prefix/match`), falling back to `approx` on error |
| `none` | cache-blind |

Since media positions hold pad values derived from the media digest, the same
image or video frame yields the same blocks in every request: prefix caching
covers multimodal prompts without extra steps.

### Engine-state adapters

* **push:** MWIP heartbeats carry `EngineState` (running/waiting requests, KV
  used/capacity, prefill queue, throughput) and register workers dynamically.
* **pull:** `engine_state.poll` scrapes each worker: MWIP `GET /mwip/v1/state`, or
  Prometheus `/metrics` mapped with the `vllm`, `sglang` or `modelweave`
  presets (or explicit metric names), so engines without MWIP heartbeats still
  inform the scheduler.

State older than `engine_state.stale_after` is ignored; the gateway's own
reservations always apply.

## Tokenizer internals

The tokenizer is a model registry (`tokenizer/src/model.rs`): every
`[[models]]` entry of its TOML file is loaded at startup — the model's
Hugging Face files and its family — and a render picks one by name or alias.
Two renderers produce the same intermediate state:

* **DeepSeek V4 / V4.1** (`deepseek.rs`) wraps deepseek-recipe: its protocol
  conversion (`ProtocolRequest::convert`), `DeepseekV41Encoding` /
  `DeepseekV4Encoding::render_conversation` and `PromptTokenizer::encode_ids`
  (special tokens only from the encoding, never from user text);
* **chat-template models** (`hf.rs`: Qwen3, Qwen3-VL, any Hugging Face chat
  template):
  * `chat.rs` converts Chat Completions, Responses and Messages requests
    into OpenAI-style messages, tools and sampling options the way vLLM and
    SGLang prepare requests for chat templates (developer → system, Anthropic
    tool results → tool messages, Responses function calls → assistant tool
    calls, tool-call arguments as objects, reasoning as
    `reasoning_content`, named tool choices narrowed to their tool);
  * `template.rs` renders the model's Jinja chat template with transformers'
    semantics (`trim_blocks`, `lstrip_blocks`, Python string methods, the
    `tojson` filter of `json.dumps`, `raise_exception`, `strftime_now`,
    Python spellings of printed values) and detects the template's content
    format (plain strings or typed parts) like vLLM;
  * every string of the request is escaped before rendering, so the special
    (control) tokens of the rendered prompt — `<|im_start|>`,
    `<|image_pad|>`, ... — are exactly those the template wrote; the text
    between them is tokenized with special tokens disabled, so a spelled
    control token stays text. Non-special added tokens (`<think>`,
    `<tool_call>`) are parsed as Hugging Face tokenizers parse them;
  * the thinking switch (`enable_thinking`), parsers (`qwen3`, `hermes`),
    stop tokens and sampling defaults come from the family and
    `generation_config.json`.

Media (`media.rs`, `vision.rs`) are shared: content blocks are extracted
(with their `metadata`, `pixel_control_options`, `frame_extract_options`) and
replaced by numbered placeholders before conversion; digests come from the
caller or the inline bytes; every item is planned by the model's vision spec:

* DeepSeek V4.1: recipe's `ImageTokenSpec` (14 px patches, 3× downsample,
  `n_h·(n_w+1)+2` tokens); videos sampled with qwen-vl-utils' `smart_nframes`,
  every kept frame a `"{t:.1} seconds"` text and its pad region;
* Qwen3-VL: Qwen2-VL `smart_resize` within the processor's pixel range
  (`(h/32)·(w/32)` tokens); videos sampled like the Qwen3-VL video processor
  (`fps` 2, 4–768 frames, per-frame pixel cap of qwen-vl-utils), two frames
  per temporal patch, every patch `"<{t:.1} seconds>"` + `<|vision_start|>`
  + its pad region + `<|vision_end|>`, as vLLM expands Qwen3-VL videos;
* the long-side limits (`detail: low`, `max_long_side`), `max_total_pixels`
  and source limits apply to every model.

Each rendered placeholder is then replaced by its item's positions, and the
item records its collapsed form (the model's placeholder) for engines that
expand media themselves. `max_tokens` is resolved against the context window.

Golden tests compare the token IDs of the chat-template renderer with
transformers' `apply_chat_template` and the reference tokenizer
(`tokenizer/testdata`). The service is stateless: rendering runs on a
blocking thread pool, and any replica serves any request (HTTP/1.1 and h2c on
one port).

## Gateway output (`internal/format`)

The formatter consumes the engine's parsed deltas and writes the client's
protocol (protocol §3): Chat Completions chunks with `reasoning_content` and
streamed `tool_calls`; Responses items and events (reasoning, message,
function calls); Anthropic content blocks (thinking, text, tool_use). The
same code builds the non-streaming body by aggregation, so stream and JSON
answers never diverge. It normalizes finish reasons (a stop after tool calls
is `tool_calls` / `tool_use`, a matched stop string is Anthropic's
`stop_sequence`), fills usage from the engine's usage (cached and reasoning
tokens), and ends broken streams with the protocol's error event.

## Package map

| Path | Content |
|---|---|
| `cmd/modelweave-gateway` | gateway binary (`-config`, `-engine-protocol`) |
| `cmd/modelweave-mockengine` | mock worker binary |
| `internal/gateway` | HTTP handlers, request lifecycle, control plane, metrics collectors |
| `internal/format` | client response formatting (Chat Completions, Responses, Messages; SSE or JSON) |
| `internal/orchestrator` | planning, execution, PD handshakes, encode stage, artifact cache, session map |
| `internal/scheduler` | scheduling passes and policies |
| `internal/kvindex` | block hashing, approximate / event / remote prefix indexes |
| `internal/registry` | workers, engine state, reservations, passive health |
| `internal/enginestate` | engine-state pollers (MWIP, Prometheus presets) |
| `internal/qos` | token buckets, tenant limiter, admission queue, overload controller |
| `internal/tokenizer` | tokenizer client (unary render, failover between replicas) |
| `internal/engine` | engine protocol adapters: mwip, vLLM `/inference/v1/generate`, SGLang `/generate`; always streaming; PD and EPD fields; encode, abort |
| `internal/mwip` | the intermediate state, the native engine protocol, the control plane |
| `internal/wire` | vLLM and SGLang wire types and the `modelweave` extension |
| `internal/mockengine` | simulated worker serving all three protocols, generating and parsing the markup of its model (DeepSeek DSML, Qwen `<think>` and hermes tool calls) |
| `internal/obs` | logging and Prometheus metrics |
| `tokenizer/` | Rust tokenizer service: model registry, DeepSeek and chat-template renderers, protocol conversion, media planning |
| `deploy/compose` | local DeepSeek V4.1 / Qwen3 / Qwen3-VL stack with mock engines, Prometheus, e2e runner |
| `test/e2e` | end-to-end tests |
