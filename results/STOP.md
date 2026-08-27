# Phase 2 stop, per plan rule 5

CREST v1 -- the TRIX encoder plus an in-context bank readout -- does not improve
zero-shot transfer. Three independently trained checkpoints, all evaluated over
the full 41-graph suite under the shared rank definition:

| checkpoint | seen-graph val gain | zero-shot delta (ind_e / ind_er) | ranks moved vs TRIX |
| --- | --- | --- | --- |
| stage A, frozen encoder, step 3000 | +0.017 | +0.0001 / +0.0004 | 36.9% |
| stage B, encoder lr 5e-5, step 1000 | +0.021 | +0.0001 / -0.0000 | 32.6% |
| stage B, encoder lr 1e-4, step 5000 | +0.019 (peak) | -0.0000 / -0.0000 | **0.2%** |

The three lines say more together than any alone.

1. With the encoder frozen, the readout learns real calibration -- worth +0.017
   MRR on the graphs it trained on -- that is relation-specific: zero-shot it
   still moves over a third of all ranks, and the moves cancel to nothing.
2. A lightly trained encoder changes neither half of that.
3. Given full freedom, the optimiser drives the readout to silence: after 5000
   full-rate steps the model differs from TRIX on 0.2% of ranks. End-to-end
   training examined the mechanism and discarded it.

## Most likely cause

The bank rows are leave-one-out snapshots of the encoder's own output, and the
readout can only recombine them. On the pretraining graphs that supports
per-relation memorisation, which the validation gains measure. On unseen
relations the rows carry no information the encoder's own score s_v0 does not
already carry -- both are functions of the same representations -- so the
readout has nothing to add, and when trained jointly the loss prefers s = s_v0
exactly. The self-adversarial objective is on TRIX's own optimum; every
deviation the readout can express costs loss on hard negatives.

## What this does not condemn

Track A (the order-sensitive message) and track B (the random channel) are
architecturally independent of the bank readout and remain untested hypotheses.
They were sequenced after phase 2 by the plan, and phase 2's failure is a
failure of the readout, not of them. Running them would start from TRIX rather
than from a phase-2 checkpoint.

## What stands regardless

Phase 0's identity harness (185,870-rank bitwise gate), phase 1's relation
baseline (criterion A 123/123, per-graph paper agreement to the third decimal),
the batched TRIX-recipe trainer, the chunked bank builder, and the measured
disqualification of pretraining-mix validation as a transfer signal -- all
committed, all reproducible.

## Numbers behind this file

results/crest/phase0_identity.json, phase2_bank_smoke.json,
phase2_stageA_zeroshot.json, phase2_stageB_halflr_zeroshot.json, and
ranks/crest/ at each evaluated checkpoint (git history).
