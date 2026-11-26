# Observability

Every process exposes Prometheus metrics on `GET /metrics` and writes JSON logs
to stderr. `deploy/compose` ships a Prometheus that scrapes the gateways, the
tokenizer and all mock engines:

```sh
docker compose -f deploy/compose/docker-compose.yaml --profile monitoring up -d prometheus
open http://127.0.0.1:39090
```

## Gateway metrics

### Requests and latency

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `modelweave_requests_total` | counter | model, api, stream, mode, code | requests by API (`chat_completions`, `responses`, `messages`), client streaming (`true` / `false` — engines are called streaming either way), executed mode (`normal`, `pd`, `epd_normal`, `epd_pd`, `none`) and HTTP status (499 = client closed) |
| `modelweave_finish_reasons_total` | counter | model, api, reason | completed generations by the finish reason sent to the client (`stop`, `length`, `tool_calls`) |
| `modelweave_request_duration_seconds` | histogram | model, api, mode | end-to-end latency |
| `modelweave_ttft_seconds` | histogram | model, mode | arrival → first token written to the client |
| `modelweave_ttft_segment_seconds` | histogram | model, mode, segment | TTFT split into `gateway` (tokenize + QoS + queue + schedule), `encode`, `engine` (dispatch → first token) |
| `modelweave_time_per_output_token_seconds` | histogram | model, mode | mean inter-token latency after the first token |
| `modelweave_stage_duration_seconds` | histogram | model, stage | `tokenize`, `admission_queue`, `encode`, `decode_register`, `prefill`, `first_token`, `decode` |
| `modelweave_tokens_total` | counter | model, tenant, kind | `prompt`, `cached` (prompt tokens served from KV cache, engine reported), `completion`, `reasoning` |
| `modelweave_inflight_requests` | gauge | model | requests being processed, including queued ones |
| `modelweave_tokenizer_client_duration_seconds` | histogram | outcome | render latency seen by the gateway (`ok`, `rejected`, `error`) |
| `modelweave_tokenizer_unreachable_total` | counter | endpoint | tokenizer replicas that could not be reached; the render moved on to the next replica, which is skipped for 5 s |
| `modelweave_media_items_total` | counter | model, modality | images and videos in rendered requests (`image`, `video`) |
| `modelweave_media_tokens_total` | counter | model, modality | prompt positions taken by media: pad-value tokens, plus timestamp text for videos |
| `modelweave_model_info` | gauge | model, mode, handshake | 1 per served model: its mode and PD handshake |

### Scheduling and execution

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `modelweave_schedule_decisions_total` | counter | model, role, policy, reason | worker picks; reason is `cache_hit`, `least_load`, `rebalance`, `affinity`, `affinity_spill`, `score_cache`, ... role `decode_normal` marks picks among decode workers that run normal requests |
| `modelweave_schedule_failures_total` | counter | model, role, error | passes without a worker (`no_workers`, `no_capacity`) |
| `modelweave_schedule_duration_seconds` | histogram | model, role | latency of one pass |
| `modelweave_prefix_cache_match_ratio` | histogram | model, role | predicted cache-hit ratio of the selected worker |
| `modelweave_kv_index_lookup_seconds` | histogram | kind, outcome | prefix-index lookups (`approx`, `events`, `remote_*`) |
| `modelweave_execution_plans_total` | counter | model, mode, reason | plans by mode (`normal`, `pd`; `epd_` prefix after a gateway encode stage) and reason (`short_prompt`, `prefix_hit`, `session`, `unified`, `long_prompt`, `partial_hit`, `fallback`, `forced`, ...) |
| `modelweave_fallbacks_total` | counter | model, from, to, reason | normal → pd fallbacks (`reason` is the error code, e.g. `normal_rejected`) |
| `modelweave_retries_total` | counter | model, stage, reason | retries before the first token (`decode`, `prefill`, `encode`) |
| `modelweave_engine_requests_total` | counter | model, role | mwip calls sent to instances by role (`encode`, `prefill`, `decode`, `unified`); Generate calls always stream, Encode calls are unary |
| `modelweave_upstream_errors_total` | counter | model, role, code | worker errors by error code |
| `modelweave_aborts_total` | counter | model, reason | engine requests aborted by the gateway (`client_gone`, `timeout`, `stopped`, `error`); stop strings are handled inside engines and do not abort |
| `modelweave_embedding_artifact_lookups_total` | counter | model, result | E-stage artifact cache (`hit`, `miss`, `invalidated`) |

### Admission and QoS

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `modelweave_admission_total` | counter | model, tenant, priority, result | `admitted`, `queued`, `rejected_rate_limit`, `rejected_overload`, `rejected_queue_full`, `rejected_timeout`, `shed` |
| `modelweave_rate_limited_total` | counter | tenant, reason | tenant limit rejections (`rpm`, `tpm`, `concurrency`) |
| `modelweave_queue_depth` | gauge | model, priority | requests waiting for capacity |
| `modelweave_queue_wait_seconds` | histogram | model, priority, outcome | time in the queue (`granted`, `timeout`, `shed`, `full`, `canceled`) |
| `modelweave_overload_level` | gauge | model | overload level 0–4 (L0 admits everything; each higher level drops or rejects more priority classes) |
| `modelweave_pool_utilization` | gauge | model, role | utilization fed to the overload controller |

### Workers and control plane

Per-worker series (labels `worker`, `model`, `role`) are produced at scrape
time from the registry:

With `state.backend: redis`, worker capacity/load/reservation gauges are
**global shared values**: do not sum them over gateway replicas. Worker
completion/error counters, request/rejection counters and latency remain local
to each gateway. Tenant balances/concurrency and queue-depth gauges are shared
in Redis mode too: do not sum these gauges across replicas. `/admin/stats`
includes `state_backend`, `qos_backend`, global tenant/queue state and the
active scorer names/weights for every model/role.

| Metric | Meaning |
|---|---|
| `modelweave_worker_up` | 1 when healthy and not draining |
| `modelweave_worker_inflight_requests` | requests the gateway has assigned |
| `modelweave_worker_reserved_kv_tokens` | KV reserved by the gateway |
| `modelweave_worker_kv_used_tokens`, `modelweave_worker_kv_capacity_tokens`, `modelweave_worker_kv_utilization` | effective KV usage (reports merged with unreported reservations) |
| `modelweave_worker_queued_prefill_tokens` | prefill backlog |
| `modelweave_worker_reported_running_requests`, `modelweave_worker_reported_waiting_requests` | engine-reported load |
| `modelweave_worker_state_age_seconds` | age of the last state report |
| `modelweave_worker_completed_total`, `modelweave_worker_errors_total` | lifetime counters |

Also: `modelweave_heartbeats_total{result}` (mwip Registry heartbeats:
`registered`, `updated`, `rejected`),
`modelweave_instance_removals_total{model,role,reason}` (instances leaving the
registry: `deregistered` by `Registry.Deregister`, `expired` without a
heartbeat for `discovery.heartbeat_ttl`), `modelweave_kv_events_total{kind}`,
`modelweave_engine_state_polls_total{outcome}` (`Engine.GetState`),
`modelweave_build_info`, and the Go runtime/process collectors.

## Tokenizer metrics

The tokenizer only renders, so its metrics describe conversion, media
planning and tokenization, per registered model (`model` is the registered
name, whichever alias the request used); output metrics live in the gateway
and the engines.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `modelweave_tokenizer_requests_total` | counter | endpoint, model, protocol, code | requests by endpoint (`render`, `tokenize_text`, `detokenize`), model, protocol and status |
| `modelweave_tokenizer_render_duration_seconds` | histogram | model, protocol, stage | `parse`, `media` (extraction and planning), `convert`, `render` (prompt format: recipe encoding or chat template), `tokenize` (including media expansion), `total` |
| `modelweave_tokenizer_prompt_tokens` | histogram | model, protocol | prompt lengths (media expanded) |
| `modelweave_tokenizer_media_total` | counter | model, modality | images and videos rendered into token lists |
| `modelweave_tokenizer_errors_total` | counter | model, stage | `parse`, `media` (missing metadata, pixel limits, media on a text-only model, ...), `convert`, `render` (template errors, `raise_exception`), `tokenize` |
| `modelweave_tokenizer_model_info` | gauge | model, family, reasoning_parser, tool_call_parser | one series per registered model (`none` for a missing parser) |
| `modelweave_tokenizer_build_info` | gauge | version, models | build information and the registered model names (comma-separated) |

The mock engine (one instance) exposes `mwip_engine_*` gauges (running,
waiting, KV used and capacity, prefill queue, cached blocks) and
`mock_engine_generate_requests_total`; `GET /mock/stats` returns its counters
(Generate and Encode calls, embedding artifacts verified, media it encoded
itself) and the last Generate request (in its JSON form, with proto field
names). `GET /mock/registration` returns its registration, and
`POST /mock/registration/{pause,leave,resume,drain,undrain}` steers it for
tests.

## Logs

The gateway writes one `request` line per API request (slog JSON):

```json
{"time":"...","level":"INFO","msg":"request","request_id":"mw-6e2e298b36c69a997d2ba3aa",
 "api":"chat_completions","model":"deepseek-v4.1","tenant":"team-a","priority":"high","stream":false,
 "status":200,"mode":"epd_pd","reason":"long_prompt","decode_worker":"decode-0","prefill_worker":"prefill-1",
 "attempts":1,"fallback":"","prompt_tokens":1970,"completion_tokens":25,"reasoning_tokens":0,
 "predicted_cached_tokens":0,"engine_cached_tokens":0,"finish_reason":"stop",
 "admission":"admitted","queued":false,"tokenize_ms":0.927,"queue_ms":0,
 "images":0,"videos":1,"encode_ms":70.792,"encoded_items":1,"artifact_hits":0,
 "register_ms":0.765,"prefill_ms":102.745,"ttft_ms":175.28,"e2e_ms":251.574}
```

`predicted_cached_tokens` is the scheduler's estimate; `engine_cached_tokens`
is what the engine reported — comparing the two shows how accurate the prefix
index is. Failed requests add `error_kind` and `error`. Status 5xx and
mid-stream errors log at WARN. Other gateway logs cover every model at
startup (`model configured`), the mwip Registry address (`mwip_registry`),
instances joining and leaving the registry (`worker added`, `worker removed`,
`instance expired: no heartbeat`), ejections, retries and fallbacks; all
carry `request_id` or `worker` where relevant.

The tokenizer logs (tracing JSON) one `model loaded` line per registered
model at startup (name, aliases, family, context window, parsers,
modalities) and one `rendered` line per request with `request_id`, model,
protocol, prompt tokens, resolved `max_tokens`, client streaming, thinking,
tool count, image and video counts and render time; render failures log at
WARN with the model, status and message. The gateway logs a WARN line for
each tokenizer replica it could not reach.
Every client gets the `X-Request-Id` header (generated or propagated), so a
request can be followed across the gateway, tokenizer and engines.

## Useful queries

```promql
# Requests per second by execution mode, and streaming vs non-streaming clients
sum by (mode) (rate(modelweave_requests_total[1m]))
sum by (stream) (rate(modelweave_requests_total[1m]))

# Engine calls by model and role (Generate calls all streaming)
sum by (model, role) (rate(modelweave_engine_requests_total[1m]))

# Instances that expired (lost heartbeats) in the last hour, per model and role
sum by (model, role) (increase(modelweave_instance_removals_total{reason="expired"}[1h]))

# The mode and PD handshake of each model
modelweave_model_info

# What the tokenizer serves, and its load and latency per model
modelweave_tokenizer_model_info
sum by (model) (rate(modelweave_tokenizer_requests_total{endpoint="render"}[1m]))
histogram_quantile(0.99, sum by (le, model) (rate(modelweave_tokenizer_render_duration_seconds_bucket{stage="total"}[5m])))
sum by (model, stage) (rate(modelweave_tokenizer_errors_total[5m]))

# Multimodal load: media per second and positions they take
sum by (model, modality) (rate(modelweave_media_items_total[5m]))
sum by (model, modality) (rate(modelweave_media_tokens_total[5m]))

# How generations end
sum by (reason) (rate(modelweave_finish_reasons_total[5m]))

# TTFT p95 per mode, and its engine share
histogram_quantile(0.95, sum by (le, mode) (rate(modelweave_ttft_seconds_bucket[5m])))
histogram_quantile(0.95, sum by (le, segment) (rate(modelweave_ttft_segment_seconds_bucket[5m])))

# Prefix-cache effectiveness: share of prompt tokens served from KV cache
sum(rate(modelweave_tokens_total{kind="cached"}[5m])) / sum(rate(modelweave_tokens_total{kind="prompt"}[5m]))

# Why workers were chosen
sum by (role, reason) (rate(modelweave_schedule_decisions_total[5m]))

# normal vs pd decisions and fallbacks
sum by (mode, reason) (rate(modelweave_execution_plans_total[5m]))
sum by (reason) (rate(modelweave_fallbacks_total[5m]))

# Overload and queueing
max by (model) (modelweave_overload_level)
sum by (model, result) (rate(modelweave_admission_total[5m]))
histogram_quantile(0.99, sum by (le, model) (rate(modelweave_queue_wait_seconds_bucket[5m])))

# Unhealthy or saturated workers
modelweave_worker_up == 0
topk(5, modelweave_worker_kv_utilization)
```
