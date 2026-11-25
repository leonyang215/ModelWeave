# ModelWeave

开源的大模型推理网关 + 调度器。

ModelWeave 位于 API 调用方与推理引擎之间。它接收 OpenAI（Chat Completions、Responses）和 Anthropic（Messages）协议的请求，流式和非流式都支持。请求先交给 tokenizer 转换成一份与引擎无关的中间态；网关结合前缀缓存和负载，把请求调度到 encode / prefill / decode 各角色的 worker 上，按启动时选定的协议调用引擎并编排分离式执行，最后把引擎的输出格式化成客户端协议的响应。

它可以同时服务多个模型家族：**DeepSeek V4 / V4.1**、**Qwen3**（混合思考、工具调用）、**Qwen3-VL**（图片和视频），以及任何带 Hugging Face chat template 的模型。

由两个进程组成，分工明确：

* **`modelweave-tokenizer`（Rust）**：只做协议转换和 tokenize，**同时服务所有注册的模型**。它把协议请求转换成**中间态**：按模型自己的 prompt 格式把对话渲染成一个 token list，并附上引擎生成和解析所需的选项。
  * DeepSeek V4 / V4.1 的渲染基于 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)；Qwen3、Qwen3-VL 等模型用它们自己的 Hugging Face chat template 渲染，结果与 transformers 的 `apply_chat_template` 逐 token 一致（有 golden 测试）。
  * 控制 token 只能来自模板：客户端在文本里拼出 `<|im_start|>`、`<|image_pad|>` 之类的写法，既伪造不了对话轮次，也伪造不了媒体占位符。
  * 图片和视频在这一步就已展开：tokenizer 根据调用方提供的元信息（digest、宽高、时长、fps）和 pixels 控制参数，按模型自己的 processor 规则（DeepSeek V4.1 的图片规格、Qwen-VL 的 `smart_resize`、Qwen3-VL 带时间戳的 temporal patch）算出每个媒体占多少个位置，并用由内容 digest 推导出的 pad_value（`2^30 + sha256(reference)`，视频按每个保留帧计算）填满这些位置。
* **`modelweave-gateway`（Go）**：负责协议接入、租户 QoS、准入排队与过载保护、worker 注册、cache 感知 / load 感知调度、normal / PD / EPD 编排（含重试与回退），以及 Prometheus 指标和访问日志。
  * 调用引擎时有三种协议，通过启动选项 `-engine-protocol` 选择：直接发中间态（`mwip`）、适配到 vLLM、适配到 SGLang。
  * 对推理实例的 generate **一律走流式**。
  * **客户端响应由网关输出**：客户端要流式就回 SSE，要非流式就把流聚合成一个 JSON。

**输出解析交给引擎。** mwip 的 `parsing_options` 写明了用哪种解析器（`deepseek_v4`、`qwen3` 推理块、`hermes` 工具调用），vLLM 用 `output_mode: "derender"`；SGLang 的 `/generate` 只返回原始文本。详见 [docs/protocol.md](docs/protocol.md)。

仓库里的 `modelweave-mockengine` 同时提供三种协议的接口，模拟了前缀 KV cache、KV 交接、encode 阶段、时延、故障注入，以及引擎侧对 DeepSeek 和 Qwen 输出的解析。整套系统不需要 GPU 就能在本地运行。

## 功能

**模型**
* 一个 tokenizer 进程服务所有注册的模型，请求按模型名或别名选择。注册一个模型只需要它的 Hugging Face 文件（`tokenizer.json`、`tokenizer_config.json`、`generation_config.json`，视觉模型再加 processor 配置）和家族（family）：
  * `deepseek-v4.1`、`deepseek-v4`：deepseek-recipe 的 prompt 编码，DSML 工具调用，支持图片和视频（V4.1）；
  * `qwen3`：Qwen3 / Qwen3-2507 / QwQ，ChatML 模板、`enable_thinking` 开关、`<think>` 推理块、hermes 风格的 `<tool_call>` 工具调用；
  * `qwen3-vl`：Qwen3-VL，在上面基础上支持图片（`smart_resize`，16 px patch，2×2 合并）和视频（按 Qwen3-VL processor 的规则抽帧，每两帧一个 temporal patch，每个 patch 带 `<t seconds>` 时间戳和 vision token）；
  * `chat-template`：其他任何 Hugging Face chat template，解析器、stop token、思考开关由配置指定。
* chat template 运行在与 transformers 语义一致的 Jinja 引擎上（`trim_blocks` / `lstrip_blocks`、Python 字符串方法、`json.dumps` 风格的 `tojson`、`raise_exception`、`strftime_now`），像 vLLM 一样自动识别模板的 content 格式，工具调用参数以对象形式传给模板。
* 每个模型的 stop token 和采样默认值来自 `generation_config.json`，上下文窗口来自注册表（网关可以再压低）。

**协议**
* OpenAI `/v1/chat/completions`、`/v1/responses`，Anthropic `/v1/messages`。
* 支持流式（SSE）与非流式，推理内容（`reasoning_content` / thinking block / reasoning item）、工具调用、JSON / JSON schema / regex 输出、stop 序列、logprobs。
* 支持图片和视频，可携带 `metadata`、`pixel_control_options`、`frame_extract_options`、`detail`。
* 中间态包含：媒体已展开成 pad_value 的 token list、媒体规划（plan）与 placement（以及每个媒体给"自己展开媒体的引擎"用的占位符）、conversation / inference / parsing 选项。中间态里不涉及任何引擎。
* 引擎协议按模型确定，三个来源依次为：配置项 `engine.protocol`、启动选项 `-engine-protocol`（覆盖前者）、`models[].engine_protocol`（优先级最高）。
  * `mwip`：直接发送中间态，引擎返回 NDJSON 事件。
  * `vllm`：token-in 的 `/inference/v1/generate`。媒体折叠回模型自己的占位符，另附 `content_parts`；PD 用 `kv_transfer_params`，EPD 用 `ec_transfer_params`。
  * `sglang`：`/generate`，携带 `input_ids`、`image_data`、`video_data`、`bootstrap_*`。

**执行形态**
* **normal**（非 PD 分离）：由一个 worker 完成 prefill + decode，可以是 unified worker，也可以是接受 normal 请求的 decode worker。在 D 上已缓存大部分 prompt、未命中部分很短、或会话就在该 D 上时选择 normal，失败时回退到 PD。
* **PD**：prefill 与 decode 分离，支持三种 KV 握手，均使用各协议自己的字段：
  * `push`：D 先注册，P 推 KV（Mooncake 风格）。
  * `bootstrap`：P、D 并发；对应 SGLang 的 `bootstrap_host/port/room` 和 vLLM MooncakeConnector 的 `kv_transfer_params`。
  * `pull`：P 先算完、返回 KV 句柄，对应 vLLM NixlConnector。
* **EPD**：媒体在 P/D 之前先编码。
  * mwip：在 encode worker 上生成按内容寻址的 artifact，每张图片或每个视频帧一个，可跨请求复用。
  * vLLM：交给 EC producer，它返回的 `ec_transfer_params` 再传给 P/D。
  * SGLang：由引擎内部自己编码。
  * 纯文本请求跳过 E。
* `mode: auto` 按请求在 normal 与 PD 之间决策。

**调度**
* 按角色配置策略：`cache_aware`（SGLang router 风格，带负载失衡保护）、`weighted`（cache / load / 亲和打分）、`session_affinity`（带负载上界的 rendezvous hash）、`least_load`、`power_of_two`、`round_robin`、`random`。
* cache 感知适配层：
  * `approx`：从路由决策中学习。
  * `events`：引擎上报的 KV 事件。
  * `remote`：外部的 KV 元信息服务。
  * `none`：不做 cache 感知。
* 媒体位置填的是由内容 digest 推导出的 pad_value，所以对 token list 做链式 block hash 时，多模态前缀也能命中缓存。
* load 感知适配层（引擎状态接入）：
  * push：MWIP 心跳，同时完成动态注册。
  * pull：MWIP state 接口，或 Prometheus 抓取（内置 vLLM / SGLang 预设）。
  * 两者都会与网关自己记录的 KV / prefill 预留合并，避免状态滞后导致超卖。
* 被动健康检查（连续失败后摘除）、drain、首 token 前换 worker 重试、整请求唯一的首 token 截止时间、断连 / 超时时向引擎传播 abort；tokenizer 副本不可达时自动换下一个副本。

**QoS**
* 按 API key 区分租户：优先级、公平份额权重、RPM / TPM 令牌桶（结束后按真实用量对账）、并发上限。超限返回 429 + Retry-After。
* 准入队列：严格优先级、租户间 VTC（virtual token counter）公平排队、backfill、队满时挤掉低优先级请求，超时返回 503 + Retry-After。
* 按利用率分级的过载控制（L0–L4，带回滞）+ 优先级矩阵；cache 命中率高的请求不会被丢弃。

**可观测性**：每个阶段都有 Prometheus 指标，包括：
* TTFT 及其分段、TPOT、含 cached / reasoning 的 token 计数；
* 流式与非流式客户端请求数、按协议和角色统计的引擎调用、每个模型实际使用的引擎协议；
* 媒体条目数及其占用的位置数；
* finish reason、调度原因、预测命中率、回退、重试、abort、排队、过载等级、每个 worker 的状态；
* tokenizer 按模型统计的渲染请求、耗时和 prompt 长度。

每个请求还会输出一条 JSON 访问日志。详见 [docs/observability.md](docs/observability.md)。

## 快速开始（DeepSeek V4.1、Qwen3、Qwen3-VL + mock 引擎）

依赖：Docker 与 Compose v2。

```sh
make images      # 构建 modelweave/{tokenizer,gateway,mockengine,e2e}:dev
make up          # 网关 :38080、第二个网关 :38081、tokenizer :38090
make e2e         # 在 compose 网络内跑 44 个端到端用例
```

在代理环境下，先 export `http_proxy` / `https_proxy`（以及 `no_proxy`）再执行 `make images`，它们会作为 build arg 传给构建过程。

本地栈（`deploy/compose`）里只有一个 tokenizer，同时加载真实的 DeepSeek V4.1、Qwen3、Qwen3-VL tokenizer；两个网关（一个用 mwip 协议调用引擎，一个用 vLLM 协议），后面是 25 个 mock worker。

```sh
# 短 prompt：在 decode worker 上以 normal 执行（不做 PD 分离）
curl -s localhost:38080/v1/chat/completions -H 'Authorization: Bearer sk-team-a' \
  -H 'Content-Type: application/json' -D - \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"你好"}]}'

# 同样的请求发给 Qwen3：先输出 <think> 推理再给答案；可以用 chat_template_kwargs、
# thinking.type 或 reasoning effort "none" 关掉思考
curl -s localhost:38080/v1/chat/completions -H 'Authorization: Bearer sk-team-a' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"你好"}],
       "chat_template_kwargs":{"enable_thinking":false}}'

# tokenizer 服务哪些模型，以及它返回的中间态
curl -s localhost:38090/v1/models | jq '.models[] | {name, family, reasoning_parser, tool_call_parser, modalities}'
curl -s localhost:38090/v1/render -H 'Content-Type: application/json' -d '{"protocol":"chat_completions",
  "model":"qwen3-vl","body":{"model":"qwen3-vl","messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"https://img.example/a.jpg"},
   "metadata":{"digest":"'$(printf a | sha256sum | cut -d" " -f1)'","width":1920,"height":1080}}]}]}}' \
  | jq '.prompt_tokens, (.multimodal.items[0] | {plan, placeholder})'

# 响应头说明了由谁执行、为什么：
#   X-ModelWeave-Mode: normal | pd   X-ModelWeave-Reason: short_prompt | prefix_hit | long_prompt | ...
#   X-ModelWeave-Decode-Worker / -Prefill-Worker / -Cached-Tokens / -Attempt
curl -s -H 'X-ModelWeave-Token: local-control-token' localhost:38080/admin/workers
curl -s localhost:38080/metrics | grep modelweave_model_info
```

长 prompt 会走 PD（`prefill-*` → `decode-*`）。再次发送同样的 prompt 时，接收过它 KV 的 decode worker 会前缀命中，直接以 normal 执行。加上 `--profile monitoring` 可以启动 Prometheus（`:39090`）。

本地栈里的模型：

| 模型 | tokenizer 模型 | 引擎协议 | 演示内容 |
|---|---|---|---|
| `deepseek-v4.1` | DeepSeek-V4.1-Flash | mwip | auto normal / PD（push）、图片和视频的 EPD（artifact 缓存）、会话亲和、cache 感知路由 |
| `deepseek-v4.1-vllm` | DeepSeek-V4.1-Flash | vllm | decode worker 上的 normal、MooncakeConnector 风格的 bootstrap PD、经 EC producer 的 EPD（`ec_transfer_params`） |
| `deepseek-v4.1-pull` | DeepSeek-V4.1-Flash | mwip | pull 握手的 PD（`kv_handle`） |
| `deepseek-v4.1-sglang` | DeepSeek-V4.1-Flash | sglang | `/generate`、bootstrap 握手、引擎内部编码（注册了 encoder 但网关不会调用）、原始文本输出 |
| `deepseek-v4.1-fallback` | DeepSeek-V4.1-Flash | mwip | normal 被拒 → 回退 PD |
| `deepseek-v4.1-normal` | DeepSeek-V4.1-Flash | mwip | unified worker、会话亲和、绕开并摘除故障 worker |
| `deepseek-v4.1-tiny` | DeepSeek-V4.1-Flash | mwip | 单槽位：排队、队满拒绝、客户端断连 abort；静态注册，通过 `/metrics` 轮询状态；上下文压到 128K |
| `qwen3` | Qwen3-8B | mwip | 混合思考、各种客户端协议下的 hermes 工具调用、auto normal / PD（push）、前缀命中 |
| `qwen3-vl` | Qwen3-VL-8B-Instruct | mwip | 按 Qwen3-VL processor 规划图片和视频，EPD 时每张图片、每个保留帧各一个 artifact |
| `qwen3-vl-sglang` | Qwen3-VL-8B-Instruct | sglang | 媒体折叠为 Qwen-VL 占位符，以 `image_data` / `video_data` 发送 |

第二个网关（`:38081`）以 `-engine-protocol vllm` 启动。它用 vLLM 协议（pull，NixlConnector 风格）驱动同一批 `deepseek-v4.1`、`qwen3`、`qwen3-vl` worker（Qwen3-VL 的视频以 `video_url` content part 经 EC producer 编码），并且只依靠引擎上报的 KV 事件做前缀路由。

租户：`sk-team-a`（高优先级）、`sk-team-b`、`sk-limited`（3 RPM）、`sk-batch`（低优先级）。

## 注册模型

tokenizer 读取一个带 `[[models]]` 列表的 TOML 文件（[configs/tokenizer.example.toml](configs/tokenizer.example.toml)，已打进镜像），每个选项在文件里都有说明：

```toml
[[models]]
name = "Qwen3-8B"                 # 网关可以使用的模型名
aliases = ["qwen3"]
family = "qwen3"                  # deepseek-v4.1 | deepseek-v4 | qwen3 | qwen3-vl | chat-template
path = "assets/tokenizers/qwen3"  # tokenizer.json、tokenizer_config.json、generation_config.json 等
context_length = 40960

[[models]]
name = "my-model"                 # 任意 Hugging Face chat template
family = "chat-template"
path = "/models/my-model"
thinking = "toggle"               # toggle | always | none
reasoning_parser = "qwen3"        # <think> 推理块
tool_call_parser = "hermes"
```

网关通过 `models[].tokenizer_model` 把自己对外提供的模型映射到这些模型上，同时配置路由、引擎协议和 QoS（[configs/gateway.example.yaml](configs/gateway.example.yaml)）。

## 从源码构建

需要 Go 1.26+、Rust 1.88+（edition 2024）：

```sh
make build       # bin/modelweave-gateway、bin/modelweave-mockengine、bin/modelweave-tokenizer
make test        # go vet + race 测试、cargo 测试（含与 transformers 对齐的 golden prompt）
make lint        # gofmt、rustfmt、clippy

bin/modelweave-tokenizer --config configs/tokenizer.example.toml --listen 127.0.0.1:8090
bin/modelweave-gateway -config configs/gateway.example.yaml -engine-protocol mwip   # 或 vllm、sglang
```

tokenizer 也可以只用命令行参数或 `MW_TOKENIZER_*` 环境变量跑单个模型（`--model`、`--family`、`--model-path`、`--context-length`）。`tokenizer/testdata/gen_qwen_golden.py` 用 transformers 的 `apply_chat_template` 和官方 tokenizer 重新生成 Qwen 测试对照的 golden prompt。

## 接入真实引擎

引擎协议在启动时选定：`-engine-protocol`，或配置里的 `engine.protocol` / `models[].engine_protocol`。

* **mwip**（原生协议）
  * 引擎直接收到中间态（`POST /mwip/v1/generate`）：媒体已规划好、已展开，并带上 encode 阶段产出的 embedding artifact，PD 字段放在 `disagg` 里。
  * 引擎返回解析好的 NDJSON 事件，见 [docs/protocol.md §2.2](docs/protocol.md)。
  * 能承载所有媒体，包括 DeepSeek V4.1 的视频。
* **vllm**
  * 使用 `POST /inference/v1/generate`，`output_mode: "derender"`。需要按模型开启 vLLM 的 reasoning / tool 解析器（例如 `--reasoning-parser qwen3 --tool-call-parser hermes`），并把 `models[].engine_model` 设成引擎的 served model name；渲染 prompt 时用的 chat template 变量放在 `parse_context` 里一起传给 vLLM。
  * PD 使用 `kv_transfer_params`：`bootstrap`（默认）对应 MooncakeConnector，`pull` 对应 NixlConnector。原生 vLLM 的流式响应不带 prefill 的 `kv_transfer_params`，见 [§2.5](docs/protocol.md)。
  * EPD：把媒体发给注册为 encode worker 的 EC producer 实例，再把它返回的 `ec_transfer_params` 传给 P/D。
* **sglang**
  * 使用 `POST /generate`，携带 `input_ids`、`image_data`、`video_data`，PD 时带 `bootstrap_*`。只支持 bootstrap 握手。
  * 兼容累积式和增量式两种流。
  * 编码留在 SGLang 内部（prefill server 上配置 `--encoder-urls`）。
  * 原生接口不解析 reasoning / tool，输出是原始文本；需要这些能力时请用 mwip 或 vllm。
* 媒体以模型自己的占位符发给 vLLM 和 SGLang（Qwen-VL 的 `<|image_pad|>` 和 `<|vision_start|><|video_pad|><|vision_end|>`，DeepSeek V4.1 的 `<｜image｜>`）；某种模态没有占位符的模型（如 DeepSeek V4.1 的视频）只能走 mwip。
* **worker 注册**
  * 通过心跳（`POST /mwip/v1/heartbeat`）注册，或静态配置。`normal: true` 表示该 decode worker 接受 normal 请求。
  * 引擎负载来自心跳，或直接抓取 vLLM / SGLang 的 Prometheus 指标，不需要改动引擎。

## 相关项目

ModelWeave 借鉴了这些开源推理服务项目的设计：[vLLM](https://github.com/vllm-project/vllm)（token-in generate、KV / EC connector、chat template content 格式）、[SGLang](https://github.com/sgl-project/sglang) 及其 model gateway（cache 感知路由、PD bootstrap、多模态 pad value、Rust 版 chat template）、[llm-d](https://github.com/llm-d/llm-d)（scorer 组合）、[Mooncake](https://github.com/kvcache-ai/Mooncake)（KV 传输）、[transformers](https://github.com/huggingface/transformers)（chat template、Qwen-VL processor）、[qwen-vl-utils](https://github.com/QwenLM/Qwen3-VL)（视频抽帧），以及 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)（DeepSeek prompt 编码）。

## 文档

* [docs/architecture.md](docs/architecture.md)：组件、请求生命周期、模型注册表、引擎协议、执行形态、调度、QoS
* [docs/protocol.md](docs/protocol.md)：中间态与各模型家族的媒体规划、引擎协议（mwip / vLLM / SGLang）、各协议下的 PD 与 EPD、输出格式化、控制面、block hash
* [docs/observability.md](docs/observability.md)：指标、日志、查询示例

## 许可证

Apache License 2.0（[LICENSE](LICENSE)）。内置的 DeepSeek tokenizer 文件为 MIT 许可（[assets/tokenizers/deepseek-v4.1/LICENSE](assets/tokenizers/deepseek-v4.1/LICENSE)），内置的 Qwen tokenizer 与 processor 配置文件为 Apache-2.0 许可（[assets/tokenizers/qwen3/LICENSE](assets/tokenizers/qwen3/LICENSE)）。
