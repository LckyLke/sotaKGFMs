# Unary channel (G1): small gain, in the predicted direction

Date: 2026-09-02. Setup: `model.unary: yes` (incite/model.py): a
query-independent state per entity from the unlabeled pass, read at the
head and at every candidate, scored with the query relation state by a
second head added to the path score. Warm start from the 4-graph last
checkpoint (unary head fresh), 10k steps, linear decay, warmup 500. The
paired baseline is the decay-only continuation (L1, DECAY_RESULT.md).

## Benchmark (41 graphs, test splits; ranks/incite-4g-unary-last)

| model | ind_e MRR | ind_er MRR | ind_e H@10 | ind_er H@10 |
| --- | --- | --- | --- | --- |
| L1 decay last (reference) | 0.4560 | 0.3852 | 0.5973 | 0.5634 |
| G1 unary last | 0.4571 | 0.3874 | 0.5973 | 0.5650 |
| TRIX | 0.4562 | 0.3679 | 0.5931 | 0.5409 |

Unary minus decay: ind_e +0.0011 [−0.000, +0.002], 14 of 18 graphs;
ind_er +0.0022 [−0.001, +0.006], 10 of 23. Free at inference (one cached
unlabeled pass per graph). DEV10 best equals last (step 10k).

## Per-scenario (ind_er; ind_e the same pattern)

| | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| decay | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| unary | 0.4038 | 0.1821 | 0.5899 | 0.2059 |

The gain sits where the mechanism predicts: unseen-answer scenarios
(+0.010 SQUA, +0.003 UQUA on ind_er; +0.005 / +0.003 on ind_e), at a
small cost on SQSA. It did NOT help the unreachable-heavy graphs
(HM:1k −0.004, HM:3k/5k flat): the pretraining graphs are well connected,
so the unary head never sees a query whose answer has no path, and it
learns nothing for that case. Same lesson as the walks lever: a
capability needs its own supervision.

## Standing

Kept: the best single model so far (0.4571 / 0.3874; +0.001 / +0.020
over TRIX). Not the step change. Next for this channel: training with
explicit no-path rows (drop the head's k-hop ball for a fraction of rows)
or a from-scratch run, both untested. Checkpoint:
checkpoints/incite-4g-unary-last-step10k.pth.
