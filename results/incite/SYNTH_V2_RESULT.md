# The generator-side bundle (MX2): NEGATIVE on both groups

Date: 2026-09-03. Setup: MX1's recipe (the 4-graph last checkpoint, 10k
steps with linear decay, 30 percent synthetic rules-prior steps) with the
four generator-side changes of SYNTH_V2_DESIGN.md: 64 certified negatives
per row with half of them within two hops of the head, per-instance
relation blocks in the union batch, up to four full-closure positives
per row. Query draw natural. Paired against MX1 (the same run without
the bundle) and L1. Run directory `output/incite-pretrain-4g-synth30v2`;
training took 3.8 hours.

## Benchmark (41 graphs, test splits; ranks/incite-4g-synth30v2-last)

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| L1 decay last (reference) | 0.4560 | 0.3852 |
| MX1 mix last | 0.4606 | 0.3851 |
| MX2 bundle last | 0.4512 | 0.3717 |
| TRIX | 0.4562 | 0.3679 |

Paired graph bootstrap (20,000 resamples):

* MX2 minus MX1: ind_e −0.0094 [−0.0134, −0.0056], 3 of 18 graphs;
  ind_er −0.0134 [−0.0219, −0.0059], 5 of 23 graphs.
* MX2 minus L1: ind_e −0.0048 [−0.0083, −0.0015]; ind_er −0.0136
  [−0.0232, −0.0060].
* MX2 minus TRIX: ind_e −0.0050 [−0.0133, +0.0041]; ind_er +0.0038
  [−0.0062, +0.0157].

## Per-scenario (scripts/halflink_report.py)

| ind_e | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| L1 | 0.4929 | 0.2856 | 0.6221 | 0.3187 |
| MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| MX2 | 0.4708 | 0.2952 | 0.5976 | 0.3696 |

| ind_er | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| L1 | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |
| MX2 | 0.3809 | 0.1764 | 0.5552 | 0.2634 |

Every cell is below MX1, the unseen-answer cells included. The bundle
did not move the model along MX1's trade-off; it lowered the whole
profile. The seen-answer cells lost the most (SQSA −0.010 / −0.013,
UQSA −0.007 / −0.021).

## The one number that explains the direction

| run | real-step loss | synthetic-step loss |
| --- | --- | --- |
| MX1 | 0.381 | 0.143 |
| MX2 | 0.379 | 0.266 |

The real steps are unchanged. The synthetic steps carry almost twice
the loss, so the synthetic gradient roughly doubled at the same
fraction of steps: the bundle raised the DOSE of the prior, not only its
quality. MX1's gain came with a seen-answer cost that grew with the
dose (SYNTH_MIX_RESULT.md); a doubled dose with the same trade-off
predicts what the table shows on the seen cells. It does not explain
the unseen cells, which also fell. Two of the four changes act on the
distribution and not only on the dose: the neighborhood negatives teach
"structurally close is not the answer", against the locality prior the
benchmark rewards, and the relation blocks replace one busy shared
vocabulary (16 rule systems in 64 ids, dense like a real graph's
relations) with about 1,000 sparse relations of some 20 facts each,
unlike any real graph the model meets.

## What the design note got wrong

SYNTH_V2_DESIGN.md called the four changes objective fixes because they
remove mismatches between the synthetic objective and the real one. The
real objective is the wrong target: the prior's job in MX1 is a mild
complementary signal (a 0.14 loss, mostly solved), and "as informative as
a real step" is a different, larger dose. Objective fixes to a signal
whose value came from being weak are not fixes.

## Standing

Dead as a bundle. Bisection queued in plan v12 (both paired against MX1,
MX1's recipe plus one change each): MX2a = relation blocks alone (also
the base the rule head needs), MX2b = 64 uniform certified negatives plus
four positives, no neighborhood bias, no blocks. If neither hurts, the
neighborhood negatives are the culprit (untested alone; the bundle's
loss doubling makes them the first suspect). The three hypotheses (gate,
rule recovery, scenario share) were rebased on MX1's recipe: PG2, RR2 (on
MX2a), MXS2. The v2 configs stay in the repository for the record.
