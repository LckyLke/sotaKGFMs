# Stack identity check: the CUDA 12.8 incite image against the cu118 dump

Date: 2026-09-02. Question: does `containers/incite/Dockerfile.cu128`
(torch 2.8.0+cu128, PyG 2.4.0, rspmm for sm_120, RTX 5070 Laptop) reproduce
what the pinned cu118 image (torch 2.1.0, RTX 4070 Ti SUPER) produced from
the same checkpoint? Same checkpoint (`checkpoints/incite-4g-last-step20k.pth`),
same config, same data, same rank definition; only the stack and the GPU
differ. Two DEV10 graphs, test splits, `scripts/compare_dumps.py`, joined
on (dataset, query id, direction).

A = ranks/incite-4g-last (NVIDIA GeForce RTX 4070 Ti SUPER, 610.57.04, torch None, cuda None)
B = ranks/incite-4g-last-bw (NVIDIA GeForce RTX 5070 Laptop GPU, 610.43.03, torch 2.8.0+cu128, cuda 12.8)

| graph | MRR A | MRR B | delta | H@10 A | H@10 B | identical ranks | max |rank move| | rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NELLInductive:v1 | 0.8085 | 0.8111 | +0.0026 | 0.9204 | 0.9204 | 97.3% | 6 | 402 |
| Metafam | 0.5101 | 0.5101 | +0.0000 | 0.9810 | 0.9810 | 100.0% | 0 | 368 |

## Reading

Metafam is identical rank for rank. On NELLInductive:v1, 11 of 402 rows
move by at most 6 ranks and the graph MRR by +0.0026: float32 kernels of a
different toolkit on a different GPU flip near-ties, which is exactly why
the runners refuse to mix devices in one ranks directory. Both dumps carry
their stack in PROVENANCE.json (image, torch, cuda, gpu). Rule: dumps from
this stack live in their own directories (suffix `-bw`) and are never
merged with cu118 dumps; a cross-stack comparison of two MODELS states the
stack of each row.

What it took to get here (all on the branch): a build-time architecture
check that cannot work without a GPU (moved to run time in
`scripts/plan_lite.sh`); PyG pinned back to TRIX's 2.4.0 because 2.5+
renamed the MessagePassing internals TRIX's `propagate` calls;
`torch.load(weights_only=False)` in our own loaders and, as
`patches/trix/0005`, in the pinned TRIX tree (torch 2.6 flipped the
default; the value is what torch 2.1 used); and the legacy nvidia runtime
(`KGFM_GPU_ARGS`) because this daemon's CDI mode leaves torch with "CUDA
unknown error" although nvidia-smi works. The in-image test suite passes
(96 passed, 1 skipped, layer gate against the pinned TRIX layer included).

Timings in this dump's TIMINGS.jsonl (85 s and 8 s) include the first-touch
dataset processing and a run-time rspmm compile for the prepared work tree
and are not comparable with the workstation's steady-state numbers.
