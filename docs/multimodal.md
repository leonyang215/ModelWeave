# Multimodal execution contract

**One frozen plan → one item Encode call → one packed embedding location.**

The tokenizer is the only planner. An encoder executes that plan; P/D read
its result and inject the exact tensor slices into the planned token ranges.
No engine reconstructs a resize or sampling decision from user hints.

## Ownership

| Stage | Input → output |
|---|---|
| Tokenizer | User metadata/pixel/frame controls → `plan`, `plan_id`, expanded `token_ids`, `placements` |
| `Engine.Encode` | **One** `MediaInput` → **one** `EmbeddingLocation` |
| Gateway | Validate plan ID/row count/location, cache per item, attach `items[].embedding` to both P/D calls |
| P/D | Read the location, verify its tensor metadata, scatter the declared row slices, run the model |

`detail`, caller `metadata`, `pixel_control_options`, `frame_extract_options`
and collapsed placeholders exist only before planning, not in engine input.
The Rust renderer keeps span/placeholder diagnostics internally for its
transformers golden tests; they are not serialized into the execution contract.

## Single-item encoder RPC

This is a video from the checked-in Rust-rendered fixture. Qwen3-VL samples
source frames **0, 20, 39, 59**, grouped in pairs. It emits two temporal groups
of 60 rows: one packed tensor with **120 rows**, not four unrelated artifacts.
`nframes` is explicit: image inputs declare 1; video inputs declare the exact
index count **including explicitly repeated temporal-padding indices**.
For example, five selected video frames padded to three pairs have
`nframes: 6`, six indices (the last repeated), and three output groups.

`EncodeRequest` (ProtoJSON with proto field names):

```json
{
  "request_id": "mw-demo",
  "attempt_id": "mw-demo-enc1-1",
  "model": "Qwen3-VL-8B-Instruct",
  "item": {
    "modality": "MODALITY_VIDEO",
    "digest": "2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6",
    "source": {"kind": "url", "url": "https://v.example/c.mp4"},
    "plan_id": "62dc7ca164da94b670bbc3de7f12d4b757944a3efe110d410be797d393f0b259",
    "plan": {
      "processor": "qwen_vl.v1",
      "width": 320,
      "height": 192,
      "tokens": 60,
      "nframes": 4,
      "patch_size": 16,
      "spatial_merge_size": 2,
      "temporal_patch_size": 2,
      "frame_indices": ["0", "20", "39", "59"],
      "timestamps": [0.3333333333333333, 1.6333333333333333]
    }
  },
  "timeout_ms": "30000"
}
```

The encoder must decode those exact frame indices (including any explicit
repeated-tail padding), resize to exactly 320×192, and execute the declared
processor/patch/merge/temporal contract. It must not run `smart_resize` or
`smart_nframes` again. For videos `nframes == len(frame_indices)` and
`nframes % temporal_patch_size == 0` are required; the encoder may not sample,
drop, add or implicitly pad any frames. Normalization/model execution follow the versioned
processor contract. If the actual media, engine or output shape cannot honor
the plan, fail the RPC instead of silently substituting another plan.

`EncodeResponse` (location URI is illustrative, transport is engine-defined):

```json
{
  "embedding": {
    "uri": "file:///shared/embeddings/62dc7ca164da94b670bbc3de7f12d4b757944a3efe110d410be797d393f0b259.tensor",
    "tokens": 120,
    "plan_id": "62dc7ca164da94b670bbc3de7f12d4b757944a3efe110d410be797d393f0b259",
    "encoder": "enc-0"
  },
  "encode_ms": 12.5
}
```

The location identifies the item tensor after the visual processing needed
to produce `plan.tokens` rows per output group, including a model's merger
when it changes the sequence length. It is not an arbitrary raw patch tensor.
URI payloads must carry (or have an agreed descriptor for) the plan ID, row
count, dtype/shape and layout so P/D can verify and read them. File/shared
storage, RDMA and other URI schemes are engine implementation choices.

## Binding the result to P/D input

The gateway does not change the plan or token positions after Encode. It
only attaches the returned location to `GenerateRequest.multimodal.items[]`.
`placements` give a **prompt offset** and an independent **tensor row offset**:

```json
{
  "placements": [
    {
      "offset": 207,
      "length": 60,
      "embedding_offset": 0,
      "modality": "MODALITY_FRAME",
      "timestamp": 0.3333333333333333,
      "reference": "62dc7ca164da94b670bbc3de7f12d4b757944a3efe110d410be797d393f0b259_kf_0000",
      "pad_value": 1439182217
    },
    {
      "offset": 275,
      "length": 60,
      "embedding_offset": 60,
      "modality": "MODALITY_FRAME",
      "timestamp": 1.6333333333333333,
      "reference": "62dc7ca164da94b670bbc3de7f12d4b757944a3efe110d410be797d393f0b259_kf_0001",
      "pad_value": 1317755157
    }
  ]
}
```

For every placement, P/D performs this binding (ranges are zero-based,
left-closed/right-open):

```text
tensor[embedding_offset : embedding_offset + length]
    → model_input[offset : offset + length]

tensor[ 0: 60] → token positions [207:267]
tensor[60:120] → token positions [275:335]
```

The gaps are timestamp/control/text tokens, not missing embedding rows.
`pad_value` is a content+plan-derived scheduling/hash marker, **not** a text
vocabulary ID to look up in the language model's embedding table.

Images have one group and `embedding_offset: 0`. Videos have one placement
per output temporal group, with row offsets `k * plan.tokens`. Both P and D
receive the same location/plan/map. A split decode leg that already receives
KV need not re-encode the prompt; the location remains available for normal
execution, fallback and whichever computation that engine actually needs.

Without a separate encode instance, P/D's local encoder must execute this
same frozen input plan. It must not infer a different plan from defaults.

## Identity, caching and validation

`plan_id` binds **content digest + complete execution plan**, including exact
frame indices and temporal grouping. Source URLs and absolute prompt offsets
are excluded: signed URLs may rotate, and a cached tensor may be reused at a
different conversation position. The cache is scoped per served model.
Changing visual weights under one model name requires normal model/cache
version isolation, just as changing weights invalidates KV prefix caches.

The fingerprint is SHA-256 of this language-neutral byte sequence, in order:

1. ASCII `mwip-media-plan-v2` followed by a NUL byte.
2. Strings: modality (`image`/`video`), lowercase content digest, processor.
   Each is UTF-8 prefixed by its **uint64 little-endian byte length**.
3. Seven uint64 little-endian values: width, height, tokens per group, nframes,
   patch size, spatial merge size, temporal patch size.
4. Uint64 little-endian frame-index count, then each uint64 index.
5. Uint64 little-endian timestamp count, then each timestamp's IEEE-754
   binary64 bits in little-endian order.

Image placement references equal `plan_id`; video references are
`{plan_id}_kf_{group:04}`. Pad values remain
`2^30 + (BE_u64(sha256(reference)[0:8]) & (2^30 - 1))`. Thus two plans with
equal output counts but different frames/geometry no longer collide merely
because they use the same media bytes.

Validation is fail-closed:

- Validate processor geometry, explicit frame count/index agreement, exact group count, plan ID, non-overlapping
  prompt ranges, row offsets, timestamps and every pad token before scheduling.
- Encode receives no batch, prompt offsets or original sampling/resize hints.
- Reject a missing URI, wrong plan ID or wrong row count from the encoder;
  do not cache or dispatch it. A bad encoder response can retry another encoder.
- Validate cached locations against the item before reuse; stale/bad entries
  are misses. New prompt positions bind the same item-local rows, never cached
  absolute offsets.
- P/D validate the map again and verify fetched tensor metadata before use.
  Missing/stale locations use the existing `artifact_unavailable` rebuild path.

## Compatibility and verification

This intentionally replaces the old multi-item Encode/per-placement artifact
contract. Regenerate stubs and upgrade tokenizer, gateway and multimodal
engines together; do not mix old multimodal engines with this contract.
Retired wire field numbers/names are reserved. Text-only request fields and
the Registry service are unchanged. No legacy batch/re-planning fallback is
provided. Old advisory cache keys are not reused by the new plan-ID keys.

```sh
make proto
make test-go
cd tokenizer && cargo test --locked
# Intentional renderer/contract changes: regenerate the Rust→Go fixtures.
MW_UPDATE_MWIP_FIXTURES=1 cargo test --locked mwip_fixtures_match
```

Fixtures: `internal/mwip/testdata/`. Go checks the Rust-emitted plan IDs, pads
and ranges, and exercises Encode→P/D plus cache reuse at shifted prompt offsets.
Mock tensor metadata verifies plan/row identity; it is not a real GPU ViT
implementation. Production engines must implement the stated processor/tensor
contract and verify the actual fetched media bytes against their digest.
