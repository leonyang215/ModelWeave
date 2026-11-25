# Local stack

`deploy/compose` runs ModelWeave on one machine without GPUs. It starts the
real tokenizer, with the bundled DeepSeek V4.1, Qwen3 and Qwen3-VL
tokenizers, two gateways and 25 mock workers that simulate inference
(`modelweave-mockengine`: prefix KV cache, KV hand-off, encoders, timing,
faults, and the output markup of DeepSeek and Qwen). The end-to-end tests run
against it.

```sh
make images   # modelweave/{tokenizer,gateway,mockengine,e2e}:dev
make up       # start the stack
make e2e      # 44 end-to-end tests inside the compose network
make down     # stop and remove it
```

| Service | Host port | What it is |
|---|---|---|
| `tokenizer` | 38090 (`MW_TOKENIZER_PORT`) | `modelweave-tokenizer` with [configs/tokenizer.example.toml](../../configs/tokenizer.example.toml) |
| `gateway` | 38080 (`MW_GATEWAY_PORT`) | `modelweave-gateway` with [gateway.yaml](gateway.yaml): every model below, with its own engine protocol |
| `gateway-events` | 38081 (`MW_GATEWAY_EVENTS_PORT`) | a second gateway with [gateway-events.yaml](gateway-events.yaml), started with `-engine-protocol vllm` |
| 25 mock workers | | one per role and model; each serves mwip, vLLM and SGLang at once |
| `prometheus` | 39090 (`MW_PROMETHEUS_PORT`) | optional: `docker compose -f deploy/compose/docker-compose.yaml --profile monitoring up -d prometheus` |

The host ports move with the variables, e.g. `MW_GATEWAY_PORT=48080 make up`.
The control-plane token is `local-control-token` (`MW_CONTROL_TOKEN`).

## Models

| Model (gateway :38080) | Tokenizer model | Engine protocol | Workers | Demonstrates |
|---|---|---|---|---|
| `deepseek-v4.1` | DeepSeek-V4.1-Flash | mwip | enc-0, prefill-0/1, decode-0/1 | auto normal / PD (push), EPD with cached artifacts for images and videos, session affinity, cache-aware routing |
| `deepseek-v4.1-vllm` | DeepSeek-V4.1-Flash | vllm | ev-0, pv-0, dv-0 | normal on the decode worker, PD with MooncakeConnector-style bootstrap, EPD through an EC producer (`ec_transfer_params`) |
| `deepseek-v4.1-pull` | DeepSeek-V4.1-Flash | mwip | pp-0, dp-0 | PD with the pull handshake (`kv_handle`) |
| `deepseek-v4.1-sglang` | DeepSeek-V4.1-Flash | sglang | es-0, ps-0, ds-0 | `/generate`, bootstrap handshake, engine-managed encoding (the registered encoder is never called), raw-text output |
| `deepseek-v4.1-fallback` | DeepSeek-V4.1-Flash | mwip | pf-0, df-0 | normal rejected → PD fallback |
| `deepseek-v4.1-normal` | DeepSeek-V4.1-Flash | mwip | normal-0/1, normal-bad | unified workers, session affinity, retry around and ejection of a broken worker |
| `deepseek-v4.1-tiny` | DeepSeek-V4.1-Flash | mwip | tiny-0 | one slot: admission queue, queue-full rejection, client-disconnect abort; static worker polled via `/metrics`; context capped at 128K |
| `qwen3` | Qwen3-8B | mwip | qwen-p0, qwen-d0 | hybrid thinking, hermes tool calls in every client protocol, auto normal / PD (push), prefix hits |
| `qwen3-vl` | Qwen3-VL-8B-Instruct | mwip | qvl-enc-0, qvl-p0, qvl-d0 | images and videos planned like the Qwen3-VL processor, EPD with one artifact per image and kept frame |
| `qwen3-vl-sglang` | Qwen3-VL-8B-Instruct | sglang | qvl-sg-0 | media collapsed to Qwen-VL placeholders, sent as `image_data` / `video_data` |

The second gateway (`:38081`) drives the `deepseek-v4.1`, `qwen3` and
`qwen3-vl` workers over vLLM's protocol (pull, NixlConnector style; Qwen3-VL
videos go as `video_url` content parts through the EC producer) and routes on
engine KV events only.

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

# Workers and their state, and the engine protocol of every model
curl -s -H 'X-ModelWeave-Token: local-control-token' localhost:38080/admin/workers
curl -s localhost:38080/metrics | grep modelweave_model_info
```

A long prompt runs PD (`prefill-*` → `decode-*`). Sent again, it is a
prefix-cache hit on the decode worker that received its KV, which then runs
it normal.
