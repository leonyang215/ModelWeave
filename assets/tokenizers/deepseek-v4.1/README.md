# DeepSeek V4.1 tokenizer

`tokenizer.json` is the V4.1 tokenizer bundled with
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)
(`static/tokenizers/v41`), itself a copy of the Hugging Face file
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/tokenizer.json` with one change: token
129264 is `<｜image｜>` (special), which the V4.1 prompt encoding uses as the
image placeholder.

- Upstream comparison revision: `6821d6ad3681a4b137b066b76094fa82ebd0a380`
- SHA-256: `81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`
- License: MIT, Copyright (c) 2023 DeepSeek — see [LICENSE](LICENSE)

Special tokens used by ModelWeave: BOS 0, EOS (stop) 1, `<think>` 128821,
`</think>` 128822, `｜DSML｜` 128825, `<｜image｜>` 129264.
