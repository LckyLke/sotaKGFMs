# Partial SEMMA dump, run WITHOUT flash-attn

12 of 41 graphs, GPU, 2026-08-26. Aborted deliberately, not failed.

## Why it is here and not in `ranks/semma/`

SEMMA embeds relation descriptions with `jina-embeddings-v3`. This run used
PyTorch native attention, because `flash_attn` is absent from the container.
See `docs/report_notes.md`, section "SEMMA runs without flash-attn".

The next run will install flash-attn. Flash attention is exact, so the
mathematics does not change, but the summation order does. Embeddings then
differ in the low-order float bits. SEMMA keeps every relation pair above a
cosine similarity of 0.8, so a pair on that boundary can move, and the semantic
relation graph with it.

That makes these ranks a different measurement from a flash-attn run. Mixing the
two in one directory would produce a group mean corresponding to neither, and no
check downstream would catch it -- the same failure the CPU/GPU split guards
against.

## What to do with it

* flash-attn works tomorrow -> re-run all 41 into a clean `ranks/semma/`.
  Keep this directory until the new run is complete, then decide whether to
  report the pair as a measured sensitivity check.
* flash-attn does not work -> move this back to `ranks/semma/`, restore
  `ranks/.claims-semma` from the 12 graph ids here, and resume.

Matching model-side CSVs are in `results/semma-noflash/`.
