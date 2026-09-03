# Test-time and checkpoint levers on the existing runs (2026-09-01, E-stages)

All numbers: 41 inductive graphs, test splits, shared rank definition,
seed 1024, single run each. Intervals are paired graph bootstraps
(`scripts/paired_bootstrap.py`, 20,000 resamples); they cover graph
variation, not seed variation.

## E1: last checkpoint versus DEV10-selected best (4-graph run)

| checkpoint | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| best by DEV10 (step 17k) | 0.4542 | 0.3791 |
| last (step 20k) | 0.4534 | 0.3825 |

Last minus best: ind_e −0.0007 [−0.003, +0.002], ind_er +0.0034
[+0.000, +0.007]. The benchmark-graph selection bought nothing. From now
on the LAST checkpoint is the reported one and DEV10 is diagnostic only.
Last versus TRIX on ind_er: +0.0146 [+0.004, +0.031], 17 of 23 graphs.

## E2: weight soup of the floor family

Average of four checkpoints warm-started from the phase-1 floor (floor
best, walks+synth, support, joint), evaluated with the floor config.

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| floor | 0.4553 | 0.3740 |
| soup | 0.4571 | 0.3775 |

Soup minus floor: ind_e +0.0018 [+0.0002, +0.0034], ind_er +0.0035
[−0.004, +0.014]. Soup minus TRIX on ind_er: +0.0096 [+0.003, +0.019].
Free at inference. Consequence: every decay run gets a last-5 snapshot
soup (`scripts/snapshot_watcher.sh` on the baseline branch).

## E3: bidirectional re-ranking, DEV10 valid splits (500 queries per graph)

4-graph best checkpoint; selection scalar = mean of the three group means.

| k | weight | selection | ind_e (3) | ind_er (6) | transductive (1) |
| --- | --- | --- | --- | --- | --- |
| 0 | – | 0.4244 | 0.5089 | 0.3675 | 0.3967 |
| 4 | 0.5 | 0.4274 | 0.5138 | 0.3644 | 0.4041 |
| 8 | 0.5 | 0.4284 | 0.5156 | 0.3645 | 0.4052 |
| 16 | 0.5 | 0.4285 | 0.5157 | 0.3648 | 0.4051 |
| 16 | 1.0 | 0.4268 | 0.5117 | 0.3611 | 0.4077 |

Verdict: passes the stop rule (+0.004 >= 0.002) but the gain is small
and split: entity-inductive graphs gain 0.005 to 0.008, unseen-relation
graphs lose up to 0.013 (WikiTopicsMT1:tax, Metafam). Weight 0.5 beats
1.0 everywhere. k=8 equals k=16 within 0.0001 at half the cost, so the
41-graph evals (E4/E5) run at k=8, weight 0.5, and are labelled with the
cost: about 3.6x the base evaluation. A scenario-aware weight (reverse
score only where the answer half is unseen) is the natural refinement
if the lever is kept at all; the mechanism levers (masking, unary) rank
above it.

## Pending

E4/E5 (re-ranked 41-graph dumps) and E6 (score ensemble of four trunks)
run after the lever and seed stages of plan v3.

## Snapshot soups of the decay runs (2026-09-03)

Average of the last five kept checkpoints (26k to 30k) of each decay
continuation, evaluated with the run's config.

| run | last | last-5 soup | delta |
| --- | --- | --- | --- |
| L1 (4g decay) | 0.4560 / 0.3852 | 0.4566 / 0.3858 | +0.0006 / +0.0006 |
| G1 (4g unary) | 0.4571 / 0.3874 | 0.4562 / 0.3882 | −0.0009 / +0.0008 |

Nothing: a linear decay to zero already averages the trajectory. The
floor-family soup's +0.002 / +0.0035 came from averaging DIFFERENT
constant-lr runs, not from snapshots of one decayed run. Keep the
watcher for completeness; expect no gain from it.
