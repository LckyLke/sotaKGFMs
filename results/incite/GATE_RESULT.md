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

## PG2P: the pruning curve (DEV10 valid splits, 500 queries per graph)

`diagnostics/gate_prune_dev.py` on `incite_last.pth`, batch 4, seed 1024;
the cells are in `results/incite/gate_prune.json`. At inference every gate
product is 1.00 to two decimals on all ten graphs (share below 0.5:
0.000; 10th percentile 1.000): the gate closes nothing on its own.

Pruning keeps, per query and round, the edges whose gate product is at or
above the (1 − x) quantile; ties at the threshold keep more. The random
control replaces the products with seeded uniforms. A gate point counts
only against the random curve at the same REALIZED kept fraction (below:
linear interpolation between the random points, with the unpruned model
as the point at kept 1.0).

| requested x | realized kept | gate ind_e | random ind_e | gate ind_er | random ind_er | gate transductive | random transductive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.000 | 0.5169 | 0.5169 | 0.3774 | 0.3774 | 0.4028 | 0.4028 |
| 0.2 | 0.934 | 0.5152 | 0.5010 | 0.3776 | 0.3660 | 0.3946 | 0.3942 |
| 0.4 | 0.845 | 0.5117 | 0.4796 | 0.3701 | 0.3508 | 0.3816 | 0.3827 |
| 0.6 | 0.726 | 0.5074 | 0.4403 | 0.3802 | 0.3321 | 0.3778 | 0.3673 |
| 0.8 | 0.529 | 0.4938 | 0.3606 | 0.3561 | 0.2959 | 0.3519 | 0.3370 |
| 0.9 | 0.394 | 0.4759 | 0.3006 | 0.3304 | 0.2620 | 0.3232 | 0.3099 |
| 0.95 | 0.307 | 0.4521 | 0.2562 | 0.3057 | 0.2334 | 0.2767 | 0.2855 |

Four readings.

1. The gate's logits rank edges far above chance. At half the edges
   kept, ind_e loses 0.023 where random pruning at the same fraction
   loses 0.156. The one-sided proof loss taught the gate which edges
   matter, although the scales it emits at inference are all 1.
2. Every pruning step costs accuracy on ind_e and on the transductive
   graph, monotonically: −0.002 at 93 percent kept, −0.005 at 85, −0.010
   at 73, −0.023 at 53. ind_er wobbles (+0.003 at 73 percent, inside the
   noise of 500 queries) and then falls. No fraction is free.
3. The requested fraction is never realized (0.95 requested, 0.31 kept).
   Every node the propagation has not reached carries the same state, so
   its gate logit is the bias plus the query term, shared by all unreached
   sources of a relation: blocks of tied edges sit at the threshold, and
   the pruning removes edges from unreached sources first, whose messages
   carry no path information. A random control does not know that, which
   inflates the gate's margin over it. The right second control is
   pruning by reachability (unreached sources first, random among the
   rest). It was not run, because it cannot change the verdict.
4. The measurement zeroes edge weights inside the same kernel and saves
   no time. A speedup needs a per-query compacted edge list, which the
   fused kernel does not have.

## Verdict for the direction

The gate makes the model neither more accurate (PG2: MX1 within noise)
nor cheaper at equal accuracy (PG2P: every fraction costs). The
sparse-propagation direction is closed; the code stays as a measurement
tool. What the idea did show: the generator's proof edges train an
edge-importance ranking that beats chance by a wide margin. That is the
piece to reuse if a compacting kernel ever exists.

## PG3, queued 4 Sep: the gate that can close, and a probe of its labels

PG3 (plan v17, `configs/incite_phase1_4g_synth30_gate3.yaml`) is MX1's
recipe plus the gate at start bias 2 (still the exact identity, forty
times the gradient, room to close), the two-sided proof loss on
synthetic steps (proof edges pushed open, two non-proof edges of the
same instance per proof edge pushed shut through the gate product,
weight 0.29 so that the identity is a stationary point), `proof_weight`
0.02 (the term sums over six rounds and starts near 6.0), soft weights at
inference, warm start with MX1's synthetic stream. A recipe candidate.

Are the labels enough? About 22 proof pairs and 45 negatives per
synthetic step, 65,000 and 130,000 over the run, for 780 gate
parameters. A CPU probe (`diagnostics/gate_label_probe.py`, trunk
frozen, gates trained by the proof loss alone for 150 steps of 8
instances) answers on held-out instances, AUC of the gate product
between a query's proof edges and the other edges of its instance:

| round | any2 (PG3) | any8 | near2 (hard negatives) |
| --- | --- | --- | --- |
| 0 | 0.52 | 0.52 | 0.52 |
| 1 | 0.80 | 0.80 | 0.80 |
| 2 | 0.92 | 0.92 | 0.93 |
| 3 | 0.94 | 0.94 | 0.93 |
| 4 | 0.94 | 0.94 | 0.94 |
| 5 | 0.94 | 0.94 | 0.94 |

Against the near non-proof edges only (within two hops of the head) the
AUC is 0.88 to 0.90 from round 2 on, for all three. Round 0 carries no
messages and cannot learn. The gate learns the concept from today's
labels in about 150 of its 2,900 synthetic steps, and neither more
negatives nor hard negatives change what it learns. PG3 runs as
configured. What the probe cannot say is whether the concept transfers
to real graphs; that is PG3's question.

## PG3 result (01:53, 5 Sep): the gate learned, the accuracy did not move

Stage PG3 of plan v17/v18: 0.4596 / 0.3857 (`ranks/incite-4g-synth30-gate3-last`).

| pair | ind_e | ind_er |
| --- | --- | --- |
| PG3 − MX1 | −0.0010 [−0.0022, +0.0003], 5 of 18 | +0.0006 [−0.0007, +0.0023], 10 of 23 |
| PG3 − PG2 | +0.0008 [−0.0003, +0.0023], 8 of 18 | −0.0012 [−0.0065, +0.0027], 12 of 23 |
| PG3 − MX15 | −0.0026 [−0.0045, −0.0006], 6 of 18 | −0.0036 [−0.0068, −0.0008], 7 of 23 |

Per-scenario MRR: PG3 is MX1 within 0.002 in every cell (ind_e SQSA
0.4796, SQUA 0.3039, UQSA 0.6031, UQUA 0.3864; ind_er 0.3935, 0.1898,
0.5738, 0.2693). Dev suite: 0.3014, MX1's number exactly (5 of 8 graphs).
Not the recipe modification.

This time the gate did move. After 10k steps the biases sit at 1.93 to
2.00 (no uniform drift: the 0.29 negative weight did what it was meant
to), the node weights grew to norms 0.44 to 0.73, and the synthetic-step
loss ended at 0.146 against MX1's 0.143: the proof term was learned
almost to zero. On held-out synthetic instances the trained gate
separates a query's proof edges from the rest with AUC 0.95 from round 3
on (proof products 0.92, near non-proof products 0.49 to 0.58). And on
REAL graphs it closes most of the messages: in the last round, 90 to 95
percent of the gate products are below 0.5 on FB15k237Inductive v1,
NELLInductive v1, WN18RRInductive v1 and CoDExSmall, with mean products
0.13 to 0.27 and a 10th percentile of 0.02 to 0.12.

So an edge-importance weighting learned from the generator's proof
structure, applied as a soft multiplier on 90 percent of the messages,
changes the accuracy by nothing: neither up on the benchmark nor on the
dev suite, nor down. The trunk's scoring is invariant to it. Two
readings are possible and a cheap measurement separates them: either the
gate's ranking on real graphs is as good as on synthetic ones and the
open 5 to 10 percent of the edges carry the answer (then hard pruning at
inference costs nothing up to that fraction, and a kernel that skips
closed edges would make the trunk several times cheaper), or the closed
edges still carry their share through the sum and nothing is separable.
PG2's hard-pruning curve lost 0.023 at half the edges kept; the same
curve on PG3's checkpoint (CPU, four DEV10 graphs, 100 queries) runs
now and is recorded below when it lands.

## PG3's hard-pruning curve (CPU, four DEV10 graphs, 100 queries each, 5 Sep)

`diagnostics/gate_prune_dev.py` on PG3's last checkpoint, FB15k237Inductive
v1, NELLInductive v1, WN18RRInductive v1, CoDExSmall (`results/incite/gate3_prune_cpu.json`).
Mean MRR over the four graphs; the random control interpolated at the
gate's realized kept fraction.

| requested x | realized kept | gate | random at that kept | loss against unpruned |
| --- | --- | --- | --- | --- |
| 0 | 1.000 | 0.4871 | 0.4871 | |
| 0.5 | 0.608 | 0.4612 | 0.3740 | −0.026 |
| 0.7 | 0.437 | 0.4499 | 0.3146 | −0.037 |
| 0.8 | 0.332 | 0.4387 | 0.2675 | −0.048 |
| 0.9 | 0.222 | 0.4053 | 0.2138 | −0.082 |
| 0.95 | 0.150 | 0.3860 | 0.1671 | −0.101 |

The gate that learned (AUC 0.95 on synthetic proof edges, 90 percent of
real-graph messages below 0.5) prunes no better than PG2's inert one:
about 5 percent of the MRR lost at 60 percent of the edges kept, 8
percent at 44 percent kept, against PG2's 4.5 percent at 53 percent and
8 percent at 39 percent (different graph sets and query counts, the same
shape). Far above random pruning, and never free. The closed edges
carry information the trunk uses through the sum even at a weight of
0.2, so the soft weighting leaves the accuracy unchanged and the hard cut
removes it. One graph is the exception: WN18RRInductive v1 keeps 0.557 of
0.575 with 17 percent of its edges, a sparse hierarchy where few edges
matter; FB, NELL and CoDEx lose 0.04 to 0.11 at the same fraction.

## Verdict on the gate direction, final

Three experiments, one answer. A gate that cannot close (PG2) is MX1
within noise; a gate that can and does close (PG3) is MX1 within noise
in every cell, on the benchmark and on the dev suite; and neither gate's
ranking allows a free hard cut of the edges. Weighting or removing edges
by the generator's proof structure changes neither the accuracy nor,
without a kernel that skips edges and a tolerance for a 5 percent loss,
the cost. The one durable finding is that the proof labels are learnable
and transfer as a ranking to real graphs (90 percent of messages gated
below 0.5 with no harm), which says the trunk is invariant to strong
per-edge rescaling, not that it needs fewer edges.
