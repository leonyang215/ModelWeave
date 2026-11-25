# Qwen3 tokenizer

The tokenizer, chat template and generation defaults of
[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), unmodified. They serve
every Qwen3 model with this vocabulary (Qwen3 dense and MoE, Qwen3-2507) and
Qwen3-VL (`../qwen3-vl`).

- Upstream revision: `b968826d9c46dd6066d109eabc6255188de91218`
- `tokenizer.json` SHA-256: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
  (the upstream LFS object); `tokenizer_config.json`, `generation_config.json`
  and `LICENSE` are byte-identical to the upstream files
- License: Apache License 2.0, Copyright 2024 Alibaba Cloud — see [LICENSE](LICENSE)

Special tokens used by ModelWeave: `<|endoftext|>` 151643 and `<|im_end|>`
151645 (stop tokens, from `generation_config.json`), `<|im_start|>` 151644,
`<tool_call>` 151657 / `</tool_call>` 151658 (hermes tool calls), `<think>`
151667 / `</think>` 151668 (reasoning).
