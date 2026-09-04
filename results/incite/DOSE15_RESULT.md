# MX15: the mix at 15 percent (2026-09-04)

Stage MX15 of plan v16 (`configs/incite_phase1_4g_synth15.yaml`): MX1's
recipe with `synth.fraction 0.15` instead of 0.30, resume-style (steps
20001..30000, the coin nested in MX1's: a step that is synthetic at 15
percent is synthetic at 30 percent too). Paired against MX1 and L1. LAST
checkpoint, 41 test graphs, seed 1024. A recipe candidate.

## Numbers

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| L1, 4g + decay, no mix | 0.4560 | 0.3852 |
| MX1, mix 30 percent | 0.4606 | 0.3851 |
| MX15, mix 15 percent (`ranks/incite-4g-synth15-last`) | 0.4621 | 0.3893 |
| TRIX (released) | 0.4562 | 0.3679 |

Paired graph bootstrap (20,000 resamples):

| pair | ind_e | ind_er |
| --- | --- | --- |
| MX15 − MX1 | +0.0016 [−0.0005, +0.0037], 12 of 18 | +0.0042 [+0.0011, +0.0078], 15 of 23 |
| MX15 − L1 | +0.0062 [+0.0018, +0.0109], 11 of 18 | +0.0040 [+0.0009, +0.0078], 14 of 23 |
| MX15 − TRIX | +0.0060 [−0.0033, +0.0159], 10 of 18 | +0.0214 [+0.0101, +0.0385], 22 of 23 |

Per-scenario MRR:

| group | model | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | L1 | 0.4929 | 0.2856 | 0.6221 | 0.3187 |
| ind_e | MX15 | 0.4856 | 0.3020 | 0.6132 | 0.3756 |
| ind_e | MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| ind_er | L1 | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| ind_er | MX15 | 0.4013 | 0.1892 | 0.5827 | 0.2518 |
| ind_er | MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |

## Reading

Half the dose keeps most of the unseen-answer gain (UQUA +0.057 / +0.049
over L1 against MX1's +0.068 / +0.065) at half the seen-answer cost (SQSA
−0.007 / −0.007 against −0.012 / −0.014, UQSA −0.009 / −0.007 against
−0.017 / −0.013). The net is the first mix result with intervals above
zero on BOTH groups against the no-mix baseline: +0.0062 / +0.0040 over
L1. Against MX1 the ind_er interval is above zero and the ind_e point is
positive, so the test leg of the recipe rule passes.

## The dev suite (stratified, `results/incite/dev/MX15.json`)

| model | dev (benchmark-weighted) | graphs' own mix |
| --- | --- | --- |
| L1 | 0.3082 | 0.3378 |
| MX1 | 0.3014 | 0.3290 |
| MX15 | 0.3050 | 0.3360 |

MX15 − MX1: +0.0036 on the dev suite, 7 of 8 graphs: the dev leg passes
too (the rule asks for +0.003 on at least six common graphs). MX15 is
0.0032 below L1 there, half of MX1's 0.0068: the dose curve moves the
held-out cost the same way as the benchmark cost. As of this landing
MX15 is the accepted modification with the largest dev gain; MX45, MXS9,
MX2a, RR2 and PG3 still land before the decision.
