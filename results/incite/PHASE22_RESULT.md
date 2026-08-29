# Phase 2.2 result: the support lever is FUNCTIONALLY DEAD

Date: 2026-08-29. Lever: retrieval support sets with hard negatives and
PU labels (design B), warm-started from the phase-1 floor, 20k steps,
DEV10-selected best at step 6000. Eval: 41 graphs, stores built and
attached on every one (41x "support ready" in the orchestrator log).

## Benchmark (ranks/incite-support/)

| group | support | floor | delta |
| --- | --- | --- | --- |
| ind_e MRR | 0.4553 | 0.4553 | -0.0001 |
| ind_er MRR | 0.3741 | 0.3740 | +0.0001 |

Identical to the floor to the fourth decimal, with live stores.

## Usage probe (results/incite/support_probe.json)

Permuted support labels change SOME ranking positions on 65-81% of
probed queries (mean |score delta| 0.002-0.005) -- the readout is not
architecturally silenced the way CREST's was (0.2% there). But MRR under
permuted labels equals MRR under real labels to four decimals on all
three probe graphs. The label-derived signal has amplitude without task
information: twitching, not thinking.

The probe script's automatic "ALIVE" verdict used a too-coarse threshold
(any ranking change > 5%); the metric-invariance is the decisive
measurement. Keep both numbers in mind when reading the JSON.

## Verdict

KILLED. Two independent mechanisms (CREST's random bank + residual
readout on a frozen-then-tuned TRIX; INCITE's retrieved hard-negative
store + cross-attention readout trained warm from its own trunk) now
reach the same end state: the encoder's own representations already
carry whatever the same-graph support rows could add, and the optimizer
routes the readout to metric-neutrality. This is now a REPLICATED
negative result about same-encoder in-context readouts at this scale,
not an implementation accident. Whatever KGPFN's +0.044 comes from
(their eval, when we run it, will say -- protocol, scale, or a
mechanism difference), it does not reproduce as a bolt-on here.

Detached rows + periodic refresh (the memory-forced deviation) remains
a possible contributor and is recorded as the standing suspect
(config_diff.md, PLAN lesson 3).

Phase 2.1b (walks revival, synthetic supervision) launched next by the
orchestrator; phase 2.3 (joint relation loss) follows.
