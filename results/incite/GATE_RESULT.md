# PG2: the proof-guided propagation gate on MX1's recipe (2026-09-03)

Stage PG2 of plan v12 (`configs/incite_phase1_4g_synth30_gate.yaml`): MX1's
recipe (the 4-graph last checkpoint, 10k decay steps, 30 percent synthetic
rules-prior steps) plus one `EdgeGate` per round, trained one-sidedly open
on the generator's proof edges (`synth.proof_weight 1.0`). Design and code:
`results/incite/GATE_RULES_DESIGN.md`, `incite/model.py`. Paired against
MX1. LAST checkpoint, 41 test graphs, seed 1024.

## Verdict: inert, as the independent review predicted

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| MX1 (`ranks/incite-4g-synth30-last`) | 0.4606 | 0.3851 |
| PG2 (`ranks/incite-4g-synth30-gate-last`) | 0.4588 | 0.3869 |
| TRIX (released) | 0.4562 | 0.3679 |

Paired graph bootstrap (20,000 resamples):

| pair | ind_e | ind_er |
| --- | --- | --- |
| PG2 − MX1 | −0.0018 [−0.0033, −0.0005], 4 of 18 graphs | +0.0018 [−0.0019, +0.0073], 11 of 23 |
| PG2 − TRIX | +0.0026 [−0.0056, +0.0117], 11 of 18 | +0.0190 [+0.0065, +0.0385], 19 of 23 |

Per-scenario MRR (`scripts/halflink_report.py`; SQ/UQ = the query half is
seen/unseen, SA/UA = the answer half is seen/unseen):

| group | model | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| ind_e | PG2 | 0.4798 | 0.3045 | 0.6026 | 0.3809 |
| ind_er | MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |
| ind_er | PG2 | 0.3959 | 0.1909 | 0.5742 | 0.2631 |

The profile is MX1's cell for cell. The only cell that moved beyond 0.003
is UQUA on both groups (−0.0055 / −0.0053), the cell the synthetic steps
feed.

## What the gate did during training

The gate started as the exact identity (bias 6.0, weights zero, scales
divided by sigmoid(6)). After 10k steps (`incite_last.pth`):

| round | node bias | rel bias | node weight norm | node query norm | rel weight norm | rel query norm |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 6.09 | 6.08 | 0.00 | 1.30 | 0.42 | 1.29 |
| 1 | 6.01 | 5.98 | 0.68 | 0.91 | 0.62 | 0.74 |
| 2 | 5.94 | 5.92 | 0.50 | 0.53 | 0.35 | 0.40 |
| 3 | 5.92 | 5.91 | 0.42 | 0.42 | 0.30 | 0.32 |
| 4 | 5.90 | 5.89 | 0.33 | 0.31 | 0.23 | 0.23 |
| 5 | 5.90 | 5.89 | 0.27 | 0.26 | 0.20 | 0.20 |

No bias moved by more than 0.11. Every gradient through the gate is
multiplied by sigmoid'(6) = 0.0025 and the proof loss only pushes proof
edges open, so nothing asked the gate to close: the review's prediction.
The synthetic-step loss averaged 0.150 against MX1's 0.143. At an open
gate the proof term is −log sigmoid(6) = 0.0025 per gate, two gates per
round, about 0.005 per synthetic step: the whole difference. The task
part of the synthetic loss is MX1's.

Whether the learned logits prune anything useful at inference is the
question of PG2P (the pruning curve with the realized kept fraction and a
random control), recorded below when it lands.

## A pairing flaw found here: warm starts count steps from 1

PG2 warm-started (`--init_from`, new parameters need `strict=False`), so
its step counter ran 1..10000. MX1 resumed (`--resume`), so its counter
ran 20001..30000. The synthetic coin and the instance seed are pure
functions of the step number. Consequences, read from the two
`pretrain.jsonl` files (one row per 50 steps):

| run | synthetic rows | FB15k237 | CoDExMedium | NELL995 | WN18RR |
| --- | --- | --- | --- | --- | --- |
| MX1 | 58 | 51 | 36 | 33 | 22 |
| PG2 | 46 | 64 | 39 | 30 | 21 |

152 of the 200 logged positions show a different graph, and the two runs
saw different synthetic instances at every synthetic step. The PG2 − MX1
difference above therefore contains data-order noise of unknown size on
top of the gate's effect. The same held for G1 and MXG1 (warm starts) and
would have held for SC1 and RR2.

The fix (2026-09-03, `incite/synth.py`, `incite/pretrain.py`): the knob
`synth.step_offset` (default 0) is added to the step number for the coin,
for `start_step` and for the instance seed. The warm-started levers still
to run set `step_offset: 20000` (`incite_phase1_4g_synth30_scenario.yaml`
for SC1, `incite_phase1_4g_synth30_iso_rules.yaml` for RR2): they now see
MX1's coin sequence and MX1's instances position for position, and the
real-graph draws fall on the same positions too (the real draw generator
restarts fresh in both cases). `incite/tests/test_step_offset.py` pins
this. The resumed levers (MX15, MX45, MXS9, MX2a, MX2H, FMX) never had the
problem. PG2 is not repeated: the gate's mechanism is settled by the
parameter drift above and the pruning curve below.
