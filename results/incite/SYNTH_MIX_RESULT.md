# Synthetic rules-prior mixing (MX1): the first lever that moves ind_e

Date: 2026-09-03. Setup: the 4-graph last checkpoint continued for 10k
steps with linear decay (the L1 recipe) where 30 percent of the steps
are synthetic union batches from the rules prior
(configs/incite_phase1_4g_synth30.yaml; incite/synth.py, prior "rules").
Paired against L1 (decay only, same start, same schedule). Training took
4.7 h; synthetic steps run 7x faster than real ones.

## Benchmark (41 graphs, test splits; ranks/incite-4g-synth30-last)

| model | ind_e MRR | ind_er MRR | ind_e H@10 | ind_er H@10 |
| --- | --- | --- | --- | --- |
| L1 decay last (reference) | 0.4560 | 0.3852 | 0.5973 | 0.5634 |
| G1 unary last | 0.4571 | 0.3874 | 0.5973 | 0.5650 |
| MX1 synthetic mix 30%, last | 0.4606 | 0.3851 | 0.6000 | 0.5645 |
| TRIX | 0.4562 | 0.3679 | 0.5931 | 0.5409 |

MX1 minus reference: ind_e +0.0046 [+0.0002, +0.0092], 11 of 18 graphs;
ind_er −0.0002 [−0.006, +0.006]. Versus TRIX: ind_e +0.0044 [−0.004,
+0.014], ind_er +0.0172 [+0.007, +0.032], 20 of 23 graphs. Best single
model so far. DEV10 scalar 0.4337, the highest of the program; DEV10
ind_e 0.5214, the highest ever.

## Per-scenario: the complement lands where predicted

| ind_e | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| reference | 0.4929 | 0.2856 | 0.6221 | 0.3187 |
| synthetic mix | 0.4805 | 0.3051 | 0.6048 | 0.3864 |

| ind_er | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| reference | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| synthetic mix | 0.3942 | 0.1876 | 0.5762 | 0.2684 |

Unseen-answer cells: SQUA +0.020 / +0.015, UQUA +0.068 / +0.065 (a
third more). Seen-answer cells: −0.012 to −0.017. The same trade-off
masking exposed, but here the unseen-answer side gains far more than
the seen side loses, so the net is positive on ind_e and neutral on
ind_er. Realistic unseen-answer supervision with certain labels
(rule-derived missing facts, ordinary targets) does what stripping real
edges could not (MASKING_RESULT.md).

## Standing

The paper's mechanism candidate. Queued next (plan v8): MXG1 = the same
mix plus the unary channel (both moved the unseen cells; unary at no
seen-answer cost), and MX15 = the mix at 15 percent (does the
seen-answer cost shrink faster than the unseen gain?). Then, if MXG1
holds, a from-scratch 4-graph run with the mix as THE recipe, and seed
repeats of that. Risks: single seed; the trade-off might reverse at
other fractions; relation-state crosstalk in union batches (RULES_PRIOR.md).
