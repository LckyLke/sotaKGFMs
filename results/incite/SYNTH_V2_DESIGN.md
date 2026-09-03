# The rules prior, second version: what only the generator knows (MX2)

Date: 2026-09-03. Code: `incite/synth.py` (knobs in `SYNTH_DEFAULTS`,
`_draw_negatives`, `union_batch(isolate_relations=...)`),
`incite/train.py::multi_positive_nll`, config
`configs/incite_phase1_4g_synth30_v2.yaml`, tests
`incite/tests/test_synth_rules2.py`. Run: stage MX2 of
`scripts/research_plan_v9.sh` (baseline branch), the MX1 recipe with this
config, paired against MX1 and L1.

## Why

MX1 moved ind_e (+0.0046 over L1, SYNTH_MIX_RESULT.md) with synthetic
steps that scored ONE positive against ONE type-consistent negative. A
real step scores one positive against 512 strict negatives. The generator
also holds the full rule system, the full closure and every entity type,
and none of that reached the loss. The defaults of every knob below
reproduce the MX1 draw sequence exactly (pinned by test (a)).

## Measured on the prior before choosing the knobs (64 instances, seed 2048)

| quantity | value |
| --- | --- |
| generation time per instance | 8 ms (MX1 path), 5 ms (v2 path) |
| direct edges per instance, median | 737 |
| relations per instance, median | 33 (sum over 16 instances: 501) |
| certified negative pool per query: min / p10 / median | 17 / 35 / 136 |
| pool within 2 hops of the head: min / p10 / median | 0 / 2 / 37 |
| derivable-but-absent tails per query (h, r): median / p75 / p90 | 1 / 2 / 8 (65 percent have exactly one) |
| unseen-answer share of the natural query draw | 0.47 (benchmark test mix: 0.37; real training positives: 0.10) |

## The four changes in the v2 config

1. **Certified negatives, many of them.** `neg_per_pos_rules: 64` (was 1).
   Every negative is a type-consistent participating tail that no rule
   derives from the observed graph or from the full base closure, so no
   negative is a true fact whose evidence was dropped. Real KGs cannot
   offer that. A pool shorter than 64 is sampled with replacement instead
   of retried (retries biased the instance draw toward hub-heavy graphs).
   The self-adversarial weighting now applies as in real training. Extra
   candidates cost only readout: the propagation is per (head, relation).
2. **Hard negatives.** `hard_neg_frac: 0.5`: half of the negatives come
   from the head's 1-2 hop neighborhood first, the candidates a path
   model confuses; the rest are uniform over the pool. KMAS (arXiv
   2605.27023) reports +0.005 to +0.009 from structurally close negatives
   on real graphs, where they carry false negatives; here they are
   certified.
3. **Per-instance relation blocks.** `isolate_relations: yes`. The union
   batch used one shared relation vocabulary, so the factorized relation
   step mixed the bulk terms of 16 unrelated rule systems into every
   instance's relation states (the crosstalk risk in RULES_PRIOR.md).
   Each instance now owns a disjoint block (about 1,000 relation ids per
   union, inverses at + num_direct), and its relation states are computed
   from its own facts only, as for a real graph. ind_er is where this
   should matter; MX1 was flat there.
4. **Full-closure positives.** `num_positive_rules: 4`: up to four
   derivable-but-absent tails of the query (h, r) share the row as masked
   positive slots (padding repeats the first positive and is masked out).
   `multi_positive_nll` takes the masked MEAN over the slots, so a row
   weighs what it weighs in TRIX's loss; with one slot it equals
   `self_adversarial_nll` (test (e)). Free extra supervision: same head,
   same relation, no extra propagation.

Not changed: the query draw (`unseen_answer_share: -1`, natural). The
natural share is already above the benchmark's, so targeting is a
hypothesis about the seen/unseen trade-off, not a fix; it is a knob for a
later paired run.

Batch shape at 16 instances: about 21k edges (inverses included), 1,232
relation ids, queries `[16, 68, 3]`, mask `[16, 4]`.

## What MX2 can and cannot say

MX2 bundles the four changes. If it beats MX1, the bundle is the recipe
and a per-change ablation is optional. If it loses, bisect: blocks off
first (change 3 is the only one that alters the relation states), then
hard negatives (change 2 is the only one that alters the negative
distribution rather than its size).
