# mwip: the engine protocol

mwip (ModelWeave Inference Protocol) is how inference engines plug into
ModelWeave, and the only way they do. It is a gRPC protocol defined in
[proto/modelweave/mwip/v1](../proto/modelweave/mwip/v1); the proto files are
the normative specification, and this page explains how to implement it.

- Engines implement the **Engine** service. An Engine server is one
  *instance*: one model in one role. The gateway sends it each request as
  the *intermediate state*, which means the prompt already rendered into
  token IDs, media already planned and expanded, and the generation and
  parsing options. The engine streams back its output, already parsed into
  reasoning, content and tool calls.
- The gateway implements the **Registry** service. Instances register there
  with heartbeats, which carry their load and KV cache events, and they stay
  registered while the heartbeats keep coming.

There are no adapters for other engine APIs. An engine built on another
serving framework integrates by serving the Engine service in front of it.

## The proto

| File | Defines |
|---|---|
| [types.proto](../proto/modelweave/mwip/v1/types.proto) | the intermediate state: `Multimodal` (media items, placements, pad values, embedding artifacts), `Conversation`, `InferenceOptions`, `ParsingOptions`; `Role`, `EngineState` |
| [engine.proto](../proto/modelweave/mwip/v1/engine.proto) | the `Engine` service: `Generate`, `Encode`, `Abort`, `GetState` |
| [registry.proto](../proto/modelweave/mwip/v1/registry.proto) | the `Registry` service: `Heartbeat`, `PublishKVEvents`, `Deregister` |

The package is `modelweave.mwip.v1`. Go code is generated into
[api/mwip/v1](../api/mwip/v1) (`make proto`) and committed. For other
languages, generate stubs from the same files:

```sh
# Python (grpcio-tools)
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/modelweave/mwip/v1/*.proto
# C++ / Rust: protoc with grpc_cpp_plugin, or tonic-build, on the same files
```

## Models, roles and instances

The gateway serves many models (`models[]` in its configuration), and every
model is served by its own instances. An instance runs in one role:

| Role | Runs |
|---|---|
| `ROLE_ENCODE` | `Encode`: the embeddings of images and videos (the E stage) |
| `ROLE_PREFILL` | the prefill leg of a split request: computes the prompt's KV and hands it to a decode instance |
| `ROLE_DECODE` | the decode leg of a split request, and whole (normal) requests when registered with `normal: true` |
| `ROLE_UNIFIED` | whole requests only |

The gateway schedules each request over the instances of its model, per
role. A normal request goes to a decode instance with `normal: true` or to a
unified instance. A split (PD) request goes to a prefill instance and a
decode instance. Media first go through an encode instance when the model has
any. Within a role the gateway picks by prefix cache and load.

## The Engine service

The gateway is the client of every instance:

| RPC | What the engine does |
|---|---|
| `Generate(GenerateRequest) returns (stream GenerateResponse)` | runs one leg of a request and streams its parsed output |
| `Encode(EncodeRequest) returns (EncodeResponse)` | builds the embeddings of media items, one artifact per placement (encode instances) |
| `Abort(AbortRequest) returns (AbortResponse)` | stops requests by id (best effort; the gateway also cancels the call) |
| `GetState(GetStateRequest) returns (GetStateResponse)` | reports the instance's load, for gateways that poll |

### GenerateRequest

| Field | Meaning |
|---|---|
| `request_id` | the attempt id; both legs of a split request share it, a retry gets a new one |
| `model` | the engine-side model name (the gateway's `models[].engine_model`), may be empty |
| `token_ids` | **the prompt to run as is**: the conversation rendered by the model's prompt format, every media position holding its pad value |
| `multimodal` | the images and videos: plan, `span`, one `placement` per image or kept video frame (`offset`, `length`, `pad_value`, `reference`), the model's `placeholder`, and the `embedding_artifacts` of the encode stage when it ran |
| `conversation` | what parsers need: `thinking_mode`, `reasoning_effort`, `tools`, `tool_choice`, `response_format`, `chat_template_kwargs` |
| `inference_options` | sampling: `max_tokens` (resolved), `temperature`, `top_p`, `top_k`, `min_p`, penalties, `seed`, `logprobs` / `top_logprobs`, the model's `stop_token_ids`, an `output_constraint` (JSON, regex, tool calls) |
| `parsing_options` | how to parse the output: `reasoning_parser`, `tool_call_parser`, `parse_tool_calls`, `starts_in_tool_calls`, `parse_json_output`, `thinking_initial_stage`, `stop_sequences` |
| `disagg` | the part the call plays: `mode` (normal or split), `role`, `handshake`, `room`, the `prefill` / `decode` peers, `kv_receiver` (push), `kv_handle` (pull) |
| `timeout_ms`, `session_key`, `priority` | time left for the first output, the session, the tenant's priority |

Pad values are `2^30 + (BE_u64(sha256(reference)[0..8]) & (2^30 − 1))`, so
they never collide with vocabulary tokens, and equal media produce equal
tokens. A prefix cache keyed on the token list therefore covers multimodal
prompts as is. An engine fills the positions of each placement with the
embedding named by its artifact. Without artifacts, it encodes the item
itself from `source`.

### The Generate stream

Every call ends with exactly one final message: `Finish` for normal and
decode legs, `PrefillDone` for prefill legs, or `Error` for any leg.

| Call (`disagg`) | Messages |
|---|---|
| normal (`mode` NORMAL or unset) | `Delta`… then `Finish` |
| prefill, push | compute, push the KV the decode instance lacks to `kv_receiver` (after its `cached_tokens`), then `PrefillDone` |
| prefill, bootstrap | compute, make the KV available at the instance's `bootstrap_addr` for `room`, then `PrefillDone` |
| prefill, pull | compute, keep the KV, then `PrefillDone` with a `kv_handle` |
| decode, push | `Accepted` with a `kv_receiver` (its address, `room`, `cached_tokens`) first; once the KV arrived, `Delta`… then `Finish` |
| decode, bootstrap | fetch the KV from `disagg.prefill.bootstrap_addr` for `room`, then `Delta`… then `Finish` |
| decode, pull | fetch the KV with `disagg.kv_handle`, then `Delta`… then `Finish` |

How KV moves between instances (RDMA, NIXL, Mooncake, …) is up to the
engines: the addresses in `kv_receiver`, `kv_handle` and `bootstrap_addr`
are engine-defined.

`Delta` carries output **parsed by the engine**:

- `reasoning` and `content` follow `parsing_options.reasoning_parser`, and
  the output starts in the reasoning when `thinking_initial_stage` says so;
- `tool_calls` follow `tool_call_parser`, with OpenAI streaming semantics:
  the first fragment of a call has its `id` and `name`, and later fragments
  append to its `arguments`;
- stop strings (`stop_sequences`) are applied by the engine and never
  output; the one that matched goes into `Finish.stop_sequence`;
- `token_ids` are the delta's output tokens, and `logprobs` holds one entry
  per token when `inference_options.logprobs` is set.

`Finish` carries the `reason` (`STOP`, `LENGTH`, `TOOL_CALLS`,
`CONTENT_FILTER`), the `usage` (prompt tokens, cached ones included;
completion tokens, reasoning included) and optional engine-side `timing`.
The gateway turns the stream into the client's protocol (Chat Completions,
Responses or Messages, as SSE or one JSON body), so engines never format API
responses.

### Errors

In a `Generate` stream, a failure is an `Error` message (`code`, `message`,
`retriable`), sent last; the call then ends with an OK status. Unary calls
fail with a gRPC status and may attach an mwip `Error` as a status detail to
set the code. The gateway reacts per code:

| Code | Meaning | Gateway |
|---|---|---|
| `CAPACITY_EXHAUSTED` | the instance is full | retries on another instance |
| `NORMAL_REJECTED` | a decode instance declines a whole request | falls back to a split execution |
| `KV_TRANSFER_FAILED` | the KV hand-off failed | re-runs both legs |
| `ARTIFACT_UNAVAILABLE` | an embedding artifact is missing | rebuilds it through the encode stage once |
| `BAD_REQUEST` | the request is invalid | fails it (400) |
| `TIMEOUT`, `ABORTED`, `INTERNAL` | | retries before the first output if `retriable` |

A status without an mwip detail is classified by its code: `UNAVAILABLE` is
a transport failure (retried elsewhere, and it counts against the instance's
health), `RESOURCE_EXHAUSTED` is capacity, `INVALID_ARGUMENT`,
`FAILED_PRECONDITION` and `OUT_OF_RANGE` are bad requests,
`DEADLINE_EXCEEDED` is a timeout, `CANCELLED` and `ABORTED` are aborts, and
anything else is internal. Nothing is retried once output reached the client.

### Cancellation, limits, transport

- The gateway cancels a `Generate` call when the client disconnects, the
  first-token deadline passes, or the sibling leg of a split request failed,
  and it also sends `Abort`. The engine stops the request and frees its KV
  cache.
- Messages can be large: a 1M-token prompt is about 5 MiB. Engines must
  accept messages of at least 64 MiB.
- Plaintext HTTP/2 by default. An instance registered with an `https://`
  target is called over TLS.

## The Registry service

The gateway serves the Registry on `server.grpc_listen` (default `:8081`).
When the gateway has a `server.control_token`, every call carries it in the
`x-modelweave-token` metadata (otherwise `UNAUTHENTICATED`).

| RPC | |
|---|---|
| `Heartbeat(HeartbeatRequest)` | registers or refreshes one instance; answers whether it was `registered` by this call and the `ttl_ms` to send the next one within |
| `PublishKVEvents(PublishKVEventsRequest)` | KV cache events of an instance between heartbeats |
| `Deregister(DeregisterRequest)` | removes an instance at once (graceful shutdown); answers whether it was registered |

A `HeartbeatRequest` describes the `instance`, its current `state`
(`EngineState`, optional), its `kv_events` (optional) and whether it is
`draining`:

| `Instance` field | |
|---|---|
| `id` | unique among the gateway's instances |
| `model` | the gateway's model name or one of its aliases |
| `role` | `ROLE_ENCODE`, `ROLE_PREFILL`, `ROLE_DECODE` or `ROLE_UNIFIED` |
| `target` | gRPC target of the instance's Engine service (`host:port`; `https://host:port` for TLS) |
| `normal` | a decode instance that also runs whole requests |
| `bootstrap_addr`, `engine_id` | where decode instances fetch this prefill instance's KV, and its KV connector id |
| `kv_capacity_tokens`, `max_running_requests`, `max_prefill_queue_tokens`, `block_size`, `labels` | static hints; a reported `EngineState` wins |

A heartbeat the gateway cannot accept fails with the reason in the status
message: `INVALID_ARGUMENT` without an `id`, a `role` or a `target`,
`NOT_FOUND` for a model the gateway does not serve.

## The lifecycle of an instance

| Step | How | Gateway |
|---|---|---|
| register | the first accepted `Heartbeat` | the instance joins its model's pool for its role; the answer has `registered: true` |
| refresh | a `Heartbeat` within every `ttl_ms` (`discovery.heartbeat_ttl`, 15s by default) | records the load (`state`), the KV events and `draining` |
| drain | `draining: true` in the heartbeats | no new requests; running ones finish |
| leave | `Deregister` | removed at once |
| expire | no heartbeat for `ttl_ms` (a crashed or unreachable engine) | removed |
| return | the next `Heartbeat` (after a restart, or once reachable again) | registered again, `registered: true` |

- The gateway forgets everything it knew about a removed instance: its
  prefix-cache index entries and its session affinities. Requests already
  running there are not canceled. If the engine is gone, their streams fail
  and are retried elsewhere before the first output. Removals are counted in
  `modelweave_instance_removals_total` by model, role and reason
  (`deregistered`, `expired`).
- An instance that publishes KV events sends its whole cache once after
  every heartbeat answered with `registered: true`, as `cleared: true` with
  every resident block in `stored`. The gateway's index of a newly
  registered instance is empty.
- Operators drain an instance with `POST /admin/workers/{id}/drain` and undo
  it with `?undo=1`. The heartbeats' `draining` does not override an
  operator's drain.
- An instance whose calls fail `health.failure_threshold` times in a row
  (transport or `INTERNAL` errors) is ejected for `health.cooldown`
  (passive health checking). It stays registered and gets requests again
  afterwards.
- Instances can also be listed in `discovery.workers` instead of
  heartbeating. Static instances never expire, and the gateway learns their
  load by polling `GetState` (`engine_state.poll`).

KV events name blocks by the chained block hashes of the token list
([protocol.md, KV events and block hashing](protocol.md#43-kv-events-and-block-hashing)).
They feed the gateway's prefix-cache routing when it runs with
`kv_index.type: events`.

## Implementing an engine

1. Generate stubs from the proto files.
2. Serve the Engine service and accept messages of at least 64 MiB.
3. `Generate`: run `token_ids` with `inference_options`; parse the output
   with the parsers in `parsing_options`; stream `Delta`s and end with
   `Finish` and its usage.
4. Implement the split legs your deployment uses (push, bootstrap or pull),
   and `Encode` if you run encode instances.
5. Treat a canceled call or an `Abort` as the end of the request.
6. Register with `Heartbeat` at least every `ttl_ms / 3`, report `draining`
   before shutting down, and `Deregister` on shutdown.
7. Optionally publish KV events (prefix-cache routing; resend the whole
   cache after `registered: true`) and answer `GetState`.

Reference implementations:

- [examples/python-engine](../examples/python-engine): a minimal Python
  engine (grpcio), one instance per process, that streams parsed output;
- [test/e2e/mwip_test.go](../test/e2e/mwip_test.go): an engine written
  against the generated Go code inside the end-to-end suite, which checks
  what the gateway sends, how it formats the answer, and the registration
  lifecycle;
- [internal/mockengine](../internal/mockengine): the mock engine of the local
  stack, covering every RPC, every PD handshake, the encode stage, KV events
  and the registration lifecycle
  ([test/e2e/lifecycle_test.go](../test/e2e/lifecycle_test.go) drives it).
