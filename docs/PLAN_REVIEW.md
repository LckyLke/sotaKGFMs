# Review of docs/KGFM_PLAN.md against this repository (2026-09-02)

Verdict: the plan as written is not worth testing. Three of its eight
phases re-specify work that is done and validated here, its central model
bet rests on a mechanism this project has falsified twice, and its compute
assumptions are about ten times what either machine has. Two of its four
hypotheses are worth testing, inside the harness, at low cost. That is the
branch `claude/plan-lite`.

## Phase by phase

| plan phase | status in this repository |
| --- | --- |
| 0 registry, dev10 | Done. `shared/suite.py` (54 graphs, families, tail-only); DEV10 shares 7 of the plan's 10 graphs. |
| 1 harness | Done. One metric implementation, bit-exact on Hits against ULTRA; half-link strata under Gregucci's names; paired graph bootstrap. Missing: Wilcoxon, sign test, Holm, leak split, trivial baselines. |
| 2 baselines | Done on the 41 inductive graphs: ULTRA 3g and 4g, MOTIF, TRIX (entity and relation), SEMMA, KG-ICL, FLOCK 40 of 41, KGPFN 25 of 41. MOTIF and TRIX reproduce within 0.0012; ULTRA's 0.004 gap is documented. The plan's K0, passed. |
| 3 generator, K2 | Partial. Rules prior in `incite/synth.py` (four rule families, typed signatures, Zipf degrees, confidences, incompleteness); the 100 percent synthetic pilot P1 is queued on the first machine. Missing: fidelity loop, export, sibling and exclusion rules. |
| 4 corpus, scaling | 3 to 4 graphs done (+0.005 ind_er). 64-graph corpora at 3M to 30M parameters: not done, not feasible on 16 GB (activation memory of labeling-trick encoders scales with hidden x nodes x batch; dim 32 already holds 13 GB at batch 32). |
| 5 Model A | Context set: CREST (PFN-style context transformer as a residual on TRIX) and INCITE 2.2 (one-head cross-attention over retrieved support rows) both ended metric-neutral. Joint head: done. Negatives: the TRIX recipe already. |
| 6 walk encoder | Done (walks plus synthetic supervision, PETALS 82 to 94 percent, no MRR gain). FLOCK measured. |
| 7 diagnostics | PETALS, half-link, reachability, re-ranking done. ConnectHub and rule recovery not. |
| 8 Model B | Is the current INCITE 4g plus joint head. |

## KGPFN on the 25 graphs every model has (single seed)

| ind_e, 9 graphs | MRR | ind_er, 16 graphs | MRR |
| --- | --- | --- | --- |
| ULTRA 3g | 0.5119 | ULTRA 3g | 0.3126 |
| ULTRA 4g | 0.5681 | ULTRA 4g | 0.3105 |
| KGPFN | 0.5735 | KGPFN | 0.3240 |
| TRIX | 0.5811 | TRIX | 0.3406 |
| INCITE 4g last | 0.5823 | INCITE 4g last | 0.3550 |

KGPFN minus ULTRA 3g: +0.062 [+0.032, +0.093] on ind_e, +0.011 [−0.002,
+0.032] on ind_er (paired graph bootstrap, 20,000 resamples). KGPFN minus
INCITE 4g last on ind_er: −0.031 [−0.048, −0.017], 0 wins of 16. Per
half-link scenario on the same graphs (ind_er): KGPFN wins only SQSA
against TRIX (0.3674 vs 0.3343) and is furthest behind on SQUA (0.1222 vs
0.1683). The plan's H2 predicts the largest context gains in the stratum
without a seen half-link; the measured in-context model has its only gain
in the seen-answer stratum. Cost: 18.9 GPU-hours for these 25 graphs.

## What is worth testing (this branch)

1. H4, measurement: leak split over the committed dumps (family in the
   pretraining set or not), Wilcoxon, sign test at 0.005, Holm, the two
   trivial baselines as rank dumps. No GPU.
2. H1, the synthetic prior: gated on P1 (first machine, about 5 Sep). Then
   the plan's fidelity statistics and an ULTRA-format export so the
   unchanged ULTRA architecture can train on the same graphs (the K2
   protocol, rescaled: 82 percent of measured ULTRA-3g is 0.34 / 0.28).
3. The context question, settled before anyone builds Model A: the
   context-necessity diagnostic (`diagnostics/context_necessity.py`) and,
   on the first machine, the plan's K3 test on the released KGPFN
   (shuffled labels, minimal context).

Not worth testing: Model A at plan scale, 64-graph corpora at 3M to 30M
parameters, Protocol H proper (every baseline retrained twice), phase 6,
Model B.

## Second machine

RTX 5070 Laptop, 8 GB, Blackwell, CUDA 13.3 driver. The pinned CUDA 11.8
images cannot run on it; `containers/incite/Dockerfile.cu128` is the same
recipe on torch 2.8 + CUDA 12.8 (`KGFM_STACK=cu128`). Dumps from the two
stacks never share a ranks directory. CPU-only work (the diagnostic, the
generator, statistics over committed dumps) runs here without docker.
