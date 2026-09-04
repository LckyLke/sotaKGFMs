# MX2a: MX1 plus isolated relation blocks only, bisection step one (2026-09-04)

Stage MX2a of plan v17 (`configs/incite_phase1_4g_synth30_iso.yaml`): MX1's
recipe with `synth.isolate_relations` on (every instance of the union
gets its own relation ids, so the union carries about 1,200 relations
with about 20 facts each instead of a shared vocabulary), nothing else
of the MX2 bundle. Resume-style, MX1's coin and instances. The first
bisection step of MX2's loss (results/incite/SYNTH_V2_RESULT.md), and the
base RR2 needs. Paired against MX1. LAST checkpoint, 41 test graphs,
seed 1024. A recipe candidate.

## Verdict: about half of MX2's loss is the isolation; rejected

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| MX1, the shared vocabulary | 0.4606 | 0.3851 |
| MX2a, isolated blocks (`ranks/incite-4g-synth30-iso-last`) | 0.4556 | 0.3797 |
| MX2, the full bundle | 0.4512 | 0.3717 |

| pair | ind_e | ind_er |
| --- | --- | --- |
| MX2a − MX1 | −0.0049 [−0.0087, −0.0018], 6 of 18 | −0.0053 [−0.0123, +0.0015], 8 of 23 |
| MX2a − MX15 | −0.0065 [−0.0105, −0.0030], 4 of 18 | −0.0096 [−0.0155, −0.0044], 5 of 23 |
| MX2 − MX1 (from SYNTH_V2_RESULT.md) | −0.0094 [−0.0134, −0.0056] | −0.0134 [−0.0219, −0.0059] |

Per-scenario MRR:

| group | model | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| ind_e | MX2a | 0.4742 | 0.3035 | 0.5979 | 0.3818 |
| ind_e | MX2 | 0.4708 | 0.2952 | 0.5976 | 0.3696 |
| ind_er | MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |
| ind_er | MX2a | 0.3821 | 0.1912 | 0.5597 | 0.2735 |
| ind_er | MX2 | 0.3809 | 0.1764 | 0.5552 | 0.2634 |

Isolation alone costs about half of what the bundle cost, on the
seen-answer cells mostly (SQSA −0.006 / −0.012, UQSA −0.007 / −0.017)
with the unseen-answer cells within noise. The review's reading holds:
a union of 1,200 relations with 20 facts each is a regime no real graph
has, and the relation encoder pays for it. The other half of MX2's loss
belongs to the certified and hard negatives and the multi-positive rows;
MX2H (the bundle at half dose with the negative mask) is queued after
the paper-model runs.

## The dev suite (stratified, `results/incite/dev/MX2a.json`)

| model | dev | graphs' own mix |
| --- | --- | --- |
| L1 | 0.3082 | 0.3378 |
| MX1 | 0.3014 | 0.3290 |
| MX2a | 0.3065 | 0.3356 |

MX2a − MX1: +0.0051 (7 of 8 graphs): the dev gate passes while the test
gate fails on both groups. The pattern is the dev suite's known one: on
the held-out graphs the prior's cost shrinks with anything that weakens
its effect, and isolation weakens it. Not accepted (the rule needs both
gates); an instructive disagreement to keep in the paper's dev table.
