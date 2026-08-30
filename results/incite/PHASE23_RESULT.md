# Phase 2.3 result: one network, both tasks -- the first lever with substance

Date: 2026-08-30. Setup: relation loss at lambda 1.0 beside the entity
loss (design D), warm start from the floor, 20k steps.

## Entity task (ranks/incite-joint/)

ind_e 0.4509 (floor 0.4553, delta -0.0044), ind_er 0.3709 (floor
0.3740, delta -0.0031). Kill switch was "entity must not drop more than
0.005": -0.0044 passes, barely. Report the margin, not just the pass.

## Relation task (ranks-relation/incite-joint/, unfiltered protocol)

ind_e 0.7286 (TRIX's dedicated relation model: 0.7564),
ind_er 0.8222 (TRIX: 0.8415).

## Standing

ONE checkpoint now does both tasks: entity within 0.005 of the
specialist floor, relation within 0.02-0.03 of TRIX's separate
relation model -- which itself trained a full dedicated budget. No
model in the comparison set ships one network for both tasks (TRIX
ships two checkpoints; FLOCK ships two). This is the first lever whose
value survives measurement, with the honest caveats: single seed, 20k
budget, and the relation gap to the specialist is real. FLOCK's
relation baseline (not yet measured here) remains the missing
comparison point.
