# Context necessity: does a from-scratch in-context scorer use its context?

Date: 2026-09-02, second machine (CPU). Code: `diagnostics/context_necessity.py`,
`incite/context.py`, `incite/synth.py::create_context_instance`,
`diagnostics/context_knn.py`; config `configs/context_necessity.yaml`; per-run
files under `output/context-necessity/<name>/`; the generated table is
`results/incite/context_necessity_table.md`.

## The question

Three in-context readouts had been measured in this harness and none moved a
zero-shot number: CREST's PFN-style transformer as a residual on TRIX
(`results/STOP.md`, crest branch), INCITE 2.2's cross-attention over retrieved
support rows (`PHASE22_RESULT.md`), and the released KGPFN (25 graphs, at or
below TRIX, `docs/PLAN_REVIEW.md`). Each was a residual on a trained encoder's
own score with detached rows on real graphs, where a structural encoder can
already answer most queries. `docs/KGFM_PLAN.md` proposes the same mechanism
trained from scratch and end to end (Model A). This diagnostic removes every
excuse the earlier negatives left open:

* synthetic rules-family graphs in which the query relation has NO edge in the
  message graph (`withhold 1.0`): its facts exist only in the context, so a
  structural model cannot learn which rule derives it and context is
  necessary by construction;
* labels certain by construction (query derivable from the message graph,
  negatives not derivable from anything), half of the negatives from the
  head's 3-hop ball;
* three scorers on one INCITE trunk (dim 32, 6 rounds), each trained from
  scratch on the same stream: `floor` (the trunk's MLP), `context_only` (a
  PFN-style transformer, context self-attention plus query cross-attention,
  width 64, depth 2, 4 heads, is the ONLY scorer), `residual` (MLP plus that
  transformer, the design that died twice); gradients flow through the
  context rows unless `detached rows`;
* evaluation on 100 held-out worlds (seed 4096; mean 206 nodes, 452 edges,
  96.6 candidates per query of which 56.2 inside the 3-hop ball) under the
  external plan's K3 conditions: full context, shuffled labels, no context.
  Pass, by the plan's own rule: full > shuffled > none and full minus none
  at least 0.02 MRR.

Recipe: 2000 steps, 4 instances per step, 8 context positives with 4
negatives each, 32 training negatives, AdamW 5e-4, self-adversarial BCE
(the TRIX loss). About 40 CPU-minutes per run.

## Result

MRR on the held-out set, final step, mean and spread over seeds 1024, 1337, 7
where three seeds exist:

| scorer | withhold | steps | full | shuffled | none | full − none | K3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor (MLP) | 1.0 | 2000 | 0.6456 ± 0.0175 | same | same | – | – |
| context_only | 1.0 | 2000 | 0.6480 ± 0.0076 | 0.6450 ± 0.0091 | 0.6494 ± 0.0052 | −0.0014 | fails, 3 of 3 |
| residual | 1.0 | 2000 | 0.6461 ± 0.0056 | 0.6478 ± 0.0041 | 0.6459 ± 0.0052 | +0.0002 | fails |
| context_only, detached rows | 1.0 | 2000 | 0.6446 | 0.6497 | 0.6515 | −0.0069 | fails |
| residual, detached rows | 1.0 | 2000 | 0.6550 | 0.6457 | 0.6575 | −0.0025 | fails |
| context_only, seed 1024 | 1.0 | 6000 | 0.6718 | 0.6486 | 0.6752 | −0.0034 | fails |
| floor (MLP) | 0.0 | 2000 | 0.8070 | same | same | – | – |
| context_only | 0.0 | 2000 | 0.8896 | 0.8901 | 0.8779 | +0.0117 | fails |

The 6000-step curve (full / shuffled / none every 500 steps): 0.607/0.606/0.604,
0.634/0.632/0.635, 0.669/0.678/0.646, 0.666/0.664/0.663, 0.665/0.663/0.644,
0.659/0.654/0.636, 0.673/0.682/0.670, 0.654/0.643/0.639, 0.670/0.659/0.672,
0.672/0.631/0.669, 0.677/0.654/0.662, 0.672/0.649/0.675. The full-minus-none
gap wanders between −0.01 and +0.02 with no trend; from step 5000 on,
shuffled labels cost 0.02 to 0.04 while removing the context costs nothing.
Training loss at steps 1800 to 2000: context_only 0.464, floor 0.469,
residual 0.470.

## The control: the signal is in the rows

`diagnostics/context_knn.py` scores each candidate by the mean cosine
similarity of its row to the context positives minus the mean to the
negatives, over exactly the rows the learned scorer reads, with no
parameters. Same held-out set, withheld regime:

| trunk from | kNN, real labels | kNN, shuffled | the trunk's MLP |
| --- | --- | --- | --- |
| floor, seed 1024 | 0.6815 | 0.3394 | 0.6692 |
| floor, seed 1337 | 0.6556 | 0.3340 | 0.6272 |
| floor, seed 7 | 0.6639 | 0.3119 | 0.6405 |
| context_only, seeds 1024 / 1337 / 7 | 0.6727 / 0.6572 / 0.6488 | 0.34 / 0.31 / 0.33 | (untrained) |
| residual, seed 7 | 0.6684 | 0.3437 | 0.6382 |

Visible regime (withhold 0.0, one seed): kNN over the floor trunk 0.7957
against its MLP 0.8070; over the context-trained trunk 0.8915, next to the
learned scorer's 0.8896 and its no-context score 0.8779.

## Reading

1. **The learned attention scorer does not use its context, even here.** Full
   minus none is within ±0.007 of zero on every configuration, three seeds,
   both designs, rows attached or detached, 2000 or 6000 steps. Shuffling the
   labels costs nothing at 2000 steps and 0.02 to 0.04 late in the long run,
   which means the late model reads labels but the query path alone reaches
   the same score without them. This is the CREST and INCITE 2.2 end state
   reproduced from scratch, with no warm start, no residual shortcut, no
   detached rows and no real-graph redundancy to blame.
2. **The information is present in the rows.** A parameter-free nearest-
   neighbour readout over the same rows beats the trained MLP on all three
   floor trunks, by +0.012, +0.028 and +0.023, and drops to 0.31 to 0.34 with
   shuffled labels. The representation carries the context signal; the
   trained readout is what fails to extract it within these budgets.
3. **What context training buys in the visible regime is a better trunk, not
   context use.** With the relation visible, the context-trained model scores
   0.89 against the floor's 0.81, but its no-context score is already 0.88
   and shuffled labels change nothing; single seed, so the size is uncertain.
4. **Consequence for docs/KGFM_PLAN.md phase 5.** Model A's premise, that an
   end-to-end context transformer will learn to exploit labelled context, fails
   the plan's own K3 test at a scale where the signal is provably present.
   Building it at 3M to 30M parameters on a 64-graph corpus would test
   optimisation budget, not the mechanism. It stays out.
5. **The lever that does use context is non-parametric.** A kNN or prototype
   term over support rows, the rows `incite/support.py` already builds on
   real graphs, is an evaluation-time addition with no training. Its natural
   next test is DEV10 valid splits on the 4g-last checkpoint (the stop rule
   of the re-ranking sweep: +0.002 on the selection scalar), then the 41
   graphs with the scenario table. That is the only in-context direction this
   result leaves open.

## Caveats

Synthetic worlds of about 200 nodes and a small scorer; one label-embedding
scheme; K3 as an ordering test on 100 instances swings by ±0.01 between
evaluations, so a gap of 0.02 is the smallest thing it can see, which is
also the plan's bar. Three seeds for the main rows, one for the variants and
the long run. Nothing here measures real graphs; the kNN follow-up does.
