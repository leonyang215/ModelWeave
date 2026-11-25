# Qwen3-VL tokenizer configuration and processors

The chat template, generation defaults and image / video processor settings
of [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
unmodified.

- Upstream revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- `tokenizer_config.json`, `generation_config.json`,
  `preprocessor_config.json` and `video_preprocessor_config.json` are
  byte-identical to the upstream files. The upstream `chat_template.json`
  holds the same template as `tokenizer_config.json`.
- License: Apache License 2.0 (declared by the model card; the repository
  ships no license file, so [LICENSE](LICENSE) is the Apache 2.0 text as
  distributed with Qwen3-8B)

The vocabulary is Qwen3's: the upstream `tokenizer.json` has the same
vocabulary, merges, added tokens, normalizer, pre-tokenizer and decoder as
`../qwen3/tokenizer.json` (it only omits `ignore_merges`, whose default is the
same `false`), so the tokenizer configuration loads that file
(`tokenizer_file`) instead of bundling a second copy.

Tokens used for media: `<|vision_start|>` 151652, `<|vision_end|>` 151653,
`<|image_pad|>` 151655, `<|video_pad|>` 151656.
