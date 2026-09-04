# SC1: the scenario-conditioned readout on MX1's recipe (2026-09-04)

Stage SC1 of plan v15 (`configs/incite_phase1_4g_synth30_scenario.yaml`):
MX1's recipe (the 4-graph last checkpoint, 10k decay steps, 30 percent
synthetic rules-prior steps, warm start with `synth.step_offset 20000` so
the synthetic stream is MX1's) plus the scenario-conditioned readout
(`incite/model.py`: `scenario_features`, four half-link indicators per
candidate on the message graph, and `scenario_mlp`, Linear(2d+4, d), ReLU,
Linear(d, 1) with its last layer zero-initialized, added to the score).
The review's first direction: the per-query oracle that takes the
seen-answer cells from L1 and the unseen-answer cells from MX1 scores
+0.0076 / +0.0079 over MX1, so a readout that calibrates the two
candidate populations separately had visible headroom. Paired against
MX1. LAST checkpoint, 41 test graphs, seed 1024. First candidate under
the recipe rule (dev gain over MX1 of at least 0.003 on the stratified
dev suite AND a paired test interval above zero on one group with the
other group's point estimate above −0.002).

## Verdict: MX1 within noise on every cell; not accepted

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| MX1 (`ranks/incite-4g-synth30-last`) | 0.4606 | 0.3851 |
| SC1 (`ranks/incite-4g-synth30-scenario-last`) | 0.4613 | 0.3873 |
| TRIX (released) | 0.4562 | 0.3679 |

Paired graph bootstrap (20,000 resamples):

| pair | ind_e | ind_er |
| --- | --- | --- |
| SC1 − MX1 | +0.0007 [−0.0003, +0.0018], 11 of 18 graphs | +0.0023 [−0.0009, +0.0068], 14 of 23 |
| SC1 − TRIX | +0.0051 [−0.0031, +0.0141], 11 of 18 | +0.0195 [+0.0080, +0.0377], 21 of 23 |

Both intervals against MX1 include zero, so the test leg of the recipe
rule fails whatever the dev suite says. Per-scenario MRR:

| group | model | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | L1 | 0.4929 | 0.2856 | 0.6221 | 0.3187 |
| ind_e | MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| ind_e | SC1 | 0.4827 | 0.3036 | 0.6064 | 0.3835 |
| ind_er | L1 | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| ind_er | MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |
| ind_er | SC1 | 0.3947 | 0.1908 | 0.5765 | 0.2654 |

SC1 is MX1 within 0.003 in every cell. The seen-answer cells, where the
headroom was (L1 is 0.012 to 0.017 above MX1 there), did not move.

## Why: the head stayed near its zero start

The last layer of `scenario_mlp` after 10k steps: weight norm 0.03
(largest entry 0.015), bias 0.003. Its output is a score shift of a few
hundredths of a logit: the readout is effectively off. The first layer
moved (weight norm 3.2), so the gradient reached the head, but the
signal that would separate the populations is weak on the pretraining
graphs: a training query's negatives are sampled entities, and whether
a candidate's answer half is present in the (dense, transductive)
training graph correlates little with its label. On the sparse
inference graphs of the test suite that indicator is what distinguishes
the populations. A larger learning rate for the head, a longer run, or a
training signal built from the synthetic instances (where the generator
controls the mix, `synth.unseen_answer_share`) are the untried
variants; the oracle bound is also between two different models, and how
much of it a per-candidate scalar shift on one trunk can recover is
unknown. Nothing here is queued: MXS9 (share 0.9) is the closest queued
lever.

## The dev suite

SC1 is the first file under the stratified protocol
(`results/incite/dev/SC1.json`, `protocol: stratified_v2`): benchmark-
weighted mean 0.3031, the graphs' own mix 0.3318 (MX1's uniform number
was 0.3324). The references' stratified numbers come from stage D0W (plan
v16, running after SC1); the comparison is added below when they land.
