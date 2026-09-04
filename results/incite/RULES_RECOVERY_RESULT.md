# RR2: rule recovery at weight 0.2 on MX2a's recipe (2026-09-04)

Stage RR2 of plan v18 (`configs/incite_phase1_4g_synth30_iso_rules.yaml`):
MX2a's recipe (MX1 plus isolated relation blocks, which the rule head
needs) plus the `RuleHead` on the relation states, trained on synthetic
steps to score rule hypotheses with certain labels (hierarchy, inversion,
symmetry, composition; results/incite/GATE_RULES_DESIGN.md) at
`synth.rule_weight 0.2` (the review: 1.0 was a fivefold dose). Warm start
with MX1's synthetic stream (`step_offset 20000`). Paired against MX2a
and MX1. LAST checkpoint, 41 test graphs, seed 1024. A recipe candidate.

## Verdict: the head learns the rules and the link predictor pays; rejected

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| MX1 | 0.4606 | 0.3851 |
| MX2a (isolated blocks) | 0.4556 | 0.3797 |
| RR2 (`ranks/incite-4g-synth30-iso-rules-last`) | 0.4521 | 0.3773 |

| pair | ind_e | ind_er |
| --- | --- | --- |
| RR2 − MX2a (the rule head alone) | −0.0035 [−0.0047, −0.0023], 2 of 18 | −0.0024 [−0.0059, +0.0013], 6 of 23 |
| RR2 − MX1 | −0.0084 [−0.0129, −0.0047], 1 of 18 | −0.0077 [−0.0159, +0.0011], 8 of 23 |
| RR2 − MX15 | −0.0100 [−0.0147, −0.0059], 3 of 18 | −0.0119 [−0.0189, −0.0052], 3 of 23 |

Per-scenario MRR:

| group | model | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| ind_e | MX2a | 0.4742 | 0.3035 | 0.5979 | 0.3818 |
| ind_e | RR2 | 0.4676 | 0.3020 | 0.5914 | 0.3841 |
| ind_er | MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |
| ind_er | MX2a | 0.3821 | 0.1912 | 0.5597 | 0.2735 |
| ind_er | RR2 | 0.3747 | 0.1967 | 0.5508 | 0.2775 |

The rule-recovery loss is learned: the synthetic-step loss falls from
0.96 in the first quarter of the run to 0.19 in the last (the task part
is about 0.14, so the head's BCE reaches about 0.05). What it costs is
the seen-answer cells again (SQSA −0.007 / −0.007, UQSA −0.007 / −0.009
against MX2a), with the unseen-answer cells flat. Making the relational
algebra of the synthetic vocabulary linearly readable from the relation
states does not transfer to link prediction, at weight 0.2 and on top of
isolation. On the dev suite RR2 is 0.3019, +0.0005 over MX1 and −0.0046
under MX2a: rejected on the test gate (and short of the dev gate).

The direction's ledger: the generator's latent rule system has now been
used three ways beyond the plain mix, as proof edges for a gate (PG2
inert, PG3 running), as rule hypotheses for a head (RR2, negative), and
through the generator-side bundle (MX2, negative). The plain mix at a
low dose remains the only use that helps on the benchmark.
