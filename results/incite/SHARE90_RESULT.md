# MXS9: MX1 with 90 percent unseen-answer synthetic queries (2026-09-04)

Stage MXS9 of plan v17 (`configs/incite_phase1_4g_synth30_share90.yaml`):
MX1's recipe with `synth.unseen_answer_share 0.9` (the generator targets
queries whose answer has no half-link in the instance; the natural draw
is about 0.47), resume-style, the same coin and instances as MX1 up to
the share. The review's suggestion (the 0.37 share of MXS was inside
noise). Paired against MX1. LAST checkpoint, 41 test graphs, seed 1024.
A recipe candidate.

## Verdict: rejected on both gates; the trade pushed further

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| L1, no mix | 0.4560 | 0.3852 |
| MX15, 15 percent | 0.4621 | 0.3893 |
| MX1, 30 percent | 0.4606 | 0.3851 |
| MXS9 (`ranks/incite-4g-synth30-s90-last`) | 0.4553 | 0.3853 |

| pair | ind_e | ind_er |
| --- | --- | --- |
| MXS9 − MX1 | −0.0052 [−0.0074, −0.0034], 0 of 18 | +0.0003 [−0.0056, +0.0097], 5 of 23 |
| MXS9 − MX15 | −0.0068 [−0.0095, −0.0043], 2 of 18 | −0.0039 [−0.0090, +0.0034], 2 of 23 |
| MXS9 − L1 | −0.0006 [−0.0042, +0.0031], 8 of 18 | +0.0001 [−0.0047, +0.0064], 6 of 23 |

Per-scenario MRR:

| group | model | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | L1 | 0.4929 | 0.2856 | 0.6221 | 0.3187 |
| ind_e | MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| ind_e | MXS9 | 0.4711 | 0.3075 | 0.5947 | 0.3895 |
| ind_er | L1 | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| ind_er | MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |
| ind_er | MXS9 | 0.3866 | 0.1958 | 0.5681 | 0.2724 |

Steering nine of ten synthetic queries to unseen answers moves the model
further along the same trade: the unseen-answer cells gain 0.003 to
0.008 over MX1, the seen-answer cells lose 0.009 to 0.010, net −0.005
on ind_e with every graph below MX1. The dose curve already showed the
unseen-answer gain saturating; the share knob confirms that the prior's
gain is not limited by how many unseen-answer queries it sees, and that
the seen-answer cost scales with how much of the synthetic signal
targets them.

## The dev suite (stratified, `results/incite/dev/MXS9.json`)

| model | dev | graphs' own mix |
| --- | --- | --- |
| L1 | 0.3082 | 0.3378 |
| MX1 | 0.3014 | 0.3290 |
| MXS9 | 0.2979 | 0.3262 |

MXS9 − MX1: −0.0035 (3 of 8 graphs). Fails the dev gate too.
