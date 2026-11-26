# Architecture artwork

The architecture diagram is rendered with **Graphviz**, not a screenshot or
an external image service. Both READMEs and the architecture guide share it.

| File | Purpose |
|---|---|
| `architecture.dot` | Editable source: layout, labels and palette |
| `architecture.svg` | Vector artwork used in the documentation |
| `architecture.png` | High-resolution raster export for slides and sharing |

To update the image, edit the `.dot` file and run from the repository root:

```sh
make diagrams              # Python 3 + Graphviz (`dot` on PATH)
```

The source uses fixed Graphviz coordinates and explicit splines (`neato -n2`)
to keep control-plane links away from the main request path.

Preview the PNG before committing; check labels, arrow directions and
readability at README width. Blue lines carry requests/output, purple is
the tokenizer call, and dashed teal lines carry control/state updates.
The state box is a backend abstraction: memory is in-process; Redis is the
shared service. Redis stores metadata, not model weights or KV tensors.
