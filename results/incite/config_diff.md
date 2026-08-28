# INCITE: config vs plan diffs

Mandatory ledger (docs/INCITE_PLAN.md lesson 4): every hyperparameter that
deviates from docs/INCITE_DESIGN.md / docs/INCITE_PLAN.md, or that those
documents leave unspecified, is recorded here THE DAY it happens. The CREST
stage-B lr ran at half spec until a human asked; that does not repeat.

| date | parameter | plan value | actual value | reason |
| --- | --- | --- | --- | --- |
| 2026-08-28 | relation.lambda | unspecified (design D: "L = L_entity + lambda*L_relation") | 1.0 (v1), 0.0 (phase1) | design fixes the mechanism, not the number; 1.0 = equal weighting, the simplest defensible default. Phase 1 runs entity-only by definition (PLAN phase 2.3 adds the lever). |
| 2026-08-28 | support.class_prior | unspecified (design B: "down-weight them with the class prior") | 0.1 | no prior estimate exists in-harness; 0.1 down-weights negatives to 0.9. Revisit when the usage probe (PLAN phase 2.2) runs. |
| 2026-08-28 | support.neg_per_pos | implied 4 (design 4: "at most 80 support rows" with K=16 -> 16*(1+4)=80) | 4 | back-derived from the 80-row budget; the design never states it directly. |
| 2026-08-28 | support.ball_cap | unspecified (design B estimates ~100 entities per 3-hop ball on WN18RR) | 1024 | dense graphs (Hetionet-like) explode the 3-hop ball; the cap bounds support building. Seeded subsample beyond the cap. |
| 2026-08-28 | support.build_batch_size | not a plan parameter | 16 | memory/time chunking only (crest/bank.py precedent: sampling is drawn chunk-size-invariantly). |
| 2026-08-28 | train.val_samples | unspecified | 500 per DEV10 graph per validation | TRIX's own fast_test=500 convention; full DEV10 eval at every val_interval would dominate wall clock. |
| 2026-08-28 | DEV10 selection scalar | PLAN: "one mean per suite group, never one number" | per-group means logged at every validation; checkpoints ORDERED by the unweighted mean of the three group means | selection needs a total order; the reported quantities remain the group means (incite/pretrain.py logs all three per checkpoint). |
| 2026-08-28 | DEV10 validation split | PLAN: "zero-shot DEV10" | each DEV10 graph's VALID split, never test | test-split selection would leak the benchmark into checkpoint choice. |
| 2026-08-28 | gradient flow through support rows | PLAN lesson 3 (specified deviation) | rows no_grad + detached, refresh_interval 500, cost gate 0.20 | recorded per the PLAN's own instruction: known deviation, first suspect if support underperforms. |
| 2026-08-28 | walk token / pooling details | design C: "small GRU", "pool the outputs into the states of the visited entities and relations" | per-step token = anon-relation emb + anon-entity emb; GRU outputs mean-pooled per visited entity / traversed relation, linear-projected, ADDED to initial states | the design fixes protocol and pooling targets, not the token layout; incite/walks.py documents the exact construction. |
| 2026-08-28 | relation-step direction | design A writes m(r_j) = sum over incoming | implemented in TRIX's EXECUTED direction (rspmm aggregates into edge_index[0]; ht/th are each other's transposes) | the layer gate compares against the pinned layer as it actually runs; see incite/layers.py docstring. Learned per-channel weights make the labeling difference immaterial. |
| 2026-08-28 | model.dim / TRIX recipe dims | TRIX ships 32 with per-block hidden lists | dim 32 everywhere, 6 alternating rounds (one entity layer + one relation step per round) | design A/4 fixes 6 rounds of the alternation; TRIX's own 3+2+4 block split does not map onto it. |
| 2026-08-28 | train.batch_size / train.accum_steps | recipe batch 32 | micro-batch 16, accum_steps 2 (effective 32) | measured OOM at batch 32 on the 15.6 GiB GPU (15.0 GiB allocated, first step). Micro-batches come from the same drawn graph and are loss-averaged before the optimizer step, so the objective per update matches the recipe up to negative-sample draws. |
| 2026-08-28 | (diagnosis correction) | — | — | the batch-8 commit blamed rspmm's saved sorted pair lists for the OOM. Re-reading the hot path: pair_sum folds batch into features, edge tensors are ~13 MB per call. The memory is ordinary retained (batch, V, d) activations across 6 rounds x ~9 module calls. Fix path: per-round activation checkpointing, then batch 32 in one piece. |
