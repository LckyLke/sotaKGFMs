# Synthetic mix plus the unary channel (MXG1): the unary channel is redundant

Date: 2026-09-03. Setup: the 4-graph last checkpoint warm-started into
the unary model (`configs/incite_phase1_4g_unary_synth30.yaml`), 10k
steps with warmup 500 and linear decay, 30 percent synthetic rules-prior
steps (the MX1 mix). Paired against L1 (decay only), MX1 (mix only) and
G1 (unary only). Run directory `output/incite-pretrain-4g-unary-synth30`
(last checkpoint at step 10,000 of the continuation).

## Benchmark (41 graphs, test splits; ranks/incite-4g-unary-synth30-last)

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| L1 decay last (reference) | 0.4560 | 0.3852 |
| G1 unary last | 0.4571 | 0.3874 |
| MX1 mix last | 0.4606 | 0.3851 |
| MXG1 mix + unary last | 0.4593 | 0.3852 |
| TRIX | 0.4562 | 0.3679 |

Paired graph bootstrap (20,000 resamples):

* MXG1 minus L1: ind_e +0.0034 [−0.0012, +0.0080], 11 of 18 graphs;
  ind_er −0.0000 [−0.0033, +0.0039].
* MXG1 minus MX1: ind_e −0.0013 [−0.0025, +0.0001], 5 of 18 graphs;
  ind_er +0.0002 [−0.0026, +0.0038].
* MXG1 minus TRIX: ind_e +0.0031 [−0.0057, +0.0130]; ind_er +0.0174
  [+0.0058, +0.0350], 19 of 23 graphs.

## Per-scenario (scripts/halflink_report.py)

| ind_e | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| L1 | 0.4929 | 0.2856 | 0.6221 | 0.3187 |
| G1 | 0.4908 | 0.2908 | 0.6218 | 0.3221 |
| MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| MXG1 | 0.4802 | 0.3046 | 0.6023 | 0.3848 |

| ind_er | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| L1 | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| G1 | 0.4038 | 0.1821 | 0.5899 | 0.2059 |
| MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |
| MXG1 | 0.3949 | 0.1873 | 0.5746 | 0.2631 |

MXG1's profile is MX1's profile, cell for cell, within 0.005. The unary
channel's own gains (SQUA +0.005 / +0.010, UQUA +0.003 / +0.003 over L1)
are a strict subset of what the synthetic mix already delivers on the
same cells (+0.020 / +0.015 and +0.068 / +0.065). Two levers aimed at the
unseen-answer cells; the mix covers the unary channel's contribution and
more, and stacking them buys nothing.

## Verdict

The unary channel leaves the recipe. It stays a diagnostic result
(UNARY_RESULT.md): a query-independent entity state helps exactly the
cells that realistic unseen-answer supervision helps more. The paper
recipe is the MX1 family; MX2 (the generator-side fixes) decides its
synthetic steps next. MXG1 remains a candidate in the plan's automatic
winner pick, where it loses to MX1 on the mean group MRR.
