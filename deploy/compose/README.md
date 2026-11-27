# Local stack

`deploy/compose` runs ModelWeave on one machine without GPUs. It starts the
real tokenizer, with the bundled DeepSeek V4.1, Qwen3 and Qwen3-VL
tokenizers, two gateways and 18 mock engines (`modelweave-mockengine`:
prefix KV cache, KV hand-off, encoders, timing, faults, and the output markup
of DeepSeek and Qwen). Every mock engine is one instance, one model in one
role: it implements mwip, the engine protocol
([docs/mwip.md](../../docs/mwip.md)), on gRPC port 9001, and registers with
the gateways' mwip Registry by heartbeat. The end-to-end tests run against
it.

```sh
make images   # modelweave/{tokenizer,gateway,mockengine,e2e}:dev
make up       # start the stack
make e2e      # end-to-end tests inside this existing compose network
make down     # stop and remove it
```

For a non-invasive regression run instead of reusing a running stack:

```sh
make test-e2e        # fresh project/network/images, no host ports, automatic cleanup
make test-e2e-redis  # real Redis + extra gateway replica; same full suite
```

These targets require Compose with `!reset` override support (2.24.4+), use the
real Rust tokenizer and 18 GPU-free mock engines, and preserve the default
`modelweave` compose project. `MW_E2E_KEEP=1` keeps only the new test project for
debugging; logs/config are saved under `tmp/modelweave-e2e-*/`. To reuse images,
set `MW_E2E_BUILD=0 MW_E2E_IMAGE_TAG=<already-built-tag>`; all four
`modelweave/{gateway,tokenizer,mockengine,e2e}:<tag>` images must exist. Set `GO`
to the Go 1.26+ executable if it is not the default on the host. These are
**ModelWeave functional tests**, not Kubernetes cluster tests.

| Service | Host port | What it is |
|---|---|---|
| `tokenizer` | 38090 (`MW_TOKENIZER_PORT`) | `modelweave-tokenizer` with [configs/tokenizer.example.toml](../../configs/tokenizer.example.toml) |
| `gateway` | 38080 (`MW_GATEWAY_PORT`), Registry 38082 (`MW_GATEWAY_GRPC_PORT`) | `modelweave-gateway` with [gateway.yaml](gateway.yaml): every model below; instances register at `gateway:8081` (inside the stack) or `127.0.0.1:38082` (from the host) |
| `gateway-events` | 38081 (`MW_GATEWAY_EVENTS_PORT`) | a second gateway with [gateway-events.yaml](gateway-events.yaml): the same `deepseek-v4.1`, `qwen3` and `qwen3-vl` instances (they register with both gateways), pull handshake, KV-event routing |
| 18 mock engines | | one instance each: mwip (gRPC :9001); KV hand-off, metrics, counters and registration controls (HTTP :9000) |
| `prometheus` | 39090 (`MW_PROMETHEUS_PORT`) | optional: `docker compose -f deploy/compose/docker-compose.yaml --profile monitoring up -d prometheus` |

The host ports move with the variables, e.g. `MW_GATEWAY_PORT=48080 make up`.
The control-plane token is `local-control-token` (`MW_CONTROL_TOKEN`).

The gateway embeds the **[admin console](../../docs/admin.md)** at
http://localhost:38080/admin/ (or `MW_GATEWAY_PORT`). Enter the control-plane
token to inspect models/instances/queues/tenant quotas, drain/resume, or
block/unblock an instance ID. No extra frontend service is needed. This local
stack uses memory backends: changes affect each gateway separately and disappear
when it restarts. Use Redis for shared/persistent isolation, and a distinct
`server.admin_token` plus HTTPS/network restrictions outside a local demo.

## Models

| Model (gateway :38080) | Tokenizer model | Instances by role | Demonstrates |
|---|---|---|---|
| `deepseek-v4.1` | DeepSeek-V4.1-Flash | encode enc-0; prefill prefill-0/1; decode decode-0/1 | auto normal / PD (push), EPD with cached artifacts for images and videos, session affinity, cache-aware routing; the registration lifecycle in the e2e suite |
| `deepseek-v4.1-pull` | DeepSeek-V4.1-Flash | prefill pp-0; decode dp-0 | PD with the pull handshake (`kv_handle`) |
| `deepseek-v4.1-fallback` | DeepSeek-V4.1-Flash | prefill pf-0; decode df-0 | normal rejected → PD fallback |
| `deepseek-v4.1-normal` | DeepSeek-V4.1-Flash | unified normal-0/1, normal-bad | normal on unified instances, session affinity, retry around and ejection of a broken instance |
| `deepseek-v4.1-tiny` | DeepSeek-V4.1-Flash | unified tiny-0 | one slot: admission queue, queue-full rejection, client-disconnect abort; a static instance (no heartbeats), polled with `Engine.GetState`; context capped at 128K |
| `qwen3` | Qwen3-8B | prefill qwen-p0; decode qwen-d0 | hybrid thinking, hermes tool calls in every client protocol, auto normal / PD (push), prefix hits |
| `qwen3-vl` | Qwen3-VL-8B-Instruct | encode qvl-enc-0; prefill qvl-p0; decode qvl-d0 | images and videos planned like the Qwen3-VL processor, EPD with one artifact per image and kept frame |
| `external-engine` | Qwen3-8B | none in the stack | engines outside the stack register here: the e2e suite's engine written against the proto, and [examples/python-engine](../../examples/python-engine) |

The second gateway (`:38081`) runs the `deepseek-v4.1`, `qwen3` and
`qwen3-vl` instances with the pull handshake, and routes on engine KV events
only.

Tenants (API keys, sent as `Authorization: Bearer ...` or `x-api-key`):
`sk-team-a` (high priority), `sk-team-b`, `sk-limited` (3 requests per
minute) and `sk-batch` (low priority).

## Requests to try

```sh
# A short prompt runs normal on a decode worker (no PD split). The response
# headers tell which worker ran it and why:
#   X-ModelWeave-Mode: normal | pd    X-ModelWeave-Reason: short_prompt | prefix_hit | long_prompt | ...
#   X-ModelWeave-Decode-Worker / -Prefill-Worker / -Cached-Tokens / -Attempt
curl -s localhost:38080/v1/chat/completions -H 'Authorization: Bearer sk-team-a' \
  -H 'Content-Type: application/json' -D - \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"Hello"}]}'

# Qwen3 thinks first by default; chat_template_kwargs, thinking.type or
# reasoning effort "none" switch thinking off
curl -s localhost:38080/v1/chat/completions -H 'Authorization: Bearer sk-team-a' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"Hello"}],
       "chat_template_kwargs":{"enable_thinking":false}}'

# The models the tokenizer serves, and the intermediate state it returns
curl -s localhost:38090/v1/models | jq '.models[] | {name, family, reasoning_parser, tool_call_parser, modalities}'
curl -s localhost:38090/v1/render -H 'Content-Type: application/json' -d '{"protocol":"chat_completions",
  "model":"qwen3-vl","body":{"model":"qwen3-vl","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"https://img.example/a.jpg"},
   "metadata":{"digest":"'$(printf a | sha256sum | cut -d" " -f1)'","width":1920,"height":1080}}]}]}}' \
  | jq '.prompt_tokens, (.multimodal.items[0] | {plan, placeholder})'

# Instances by model and role: load, drain, heartbeat age
curl -s -H 'X-ModelWeave-Token: local-control-token' localhost:38080/admin/workers

# The registration lifecycle: stop decode-1's heartbeats (it expires after the
# 6s TTL), let it leave (removed at once), and bring it back
docker compose -f deploy/compose/docker-compose.yaml exec decode-1 \
  curl -s -X POST localhost:9000/mock/registration/pause    # then: leave, resume, drain, undrain
curl -s localhost:38080/metrics | grep modelweave_instance_removals_total
```

A long prompt runs PD (`prefill-*` → `decode-*`). Sent again, it is a
prefix-cache hit on the decode instance that received its KV, which then
runs it normal.

To plug in an engine of your own, run
[examples/python-engine](../../examples/python-engine): each process is one
instance of `external-engine`, registered through `127.0.0.1:38082`.
