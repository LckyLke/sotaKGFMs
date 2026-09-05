# Takeover — 2026-09-05, 12:55 (day 9)

Goal: a novel, publishable KGFM that beats TRIX and FLOCK on the 41-graph
zero-shot inductive suite, with seed spread and matched pretraining data.
This file is the single entry point for any agent, on this machine or
another. Read it fully before touching anything.

## Repositories and branches (all pushed to origin, 2026-09-02)

* `claude/gpu-multi-model-baseline` — the benchmark harness, seven
  baselines, all queue scripts, all baseline rank dumps. Worktree here:
  `~/Dokumente/GitHub/sotaKGFMs`.
* `incite` — our model (INCITE) and every experiment on it: code
  (`incite/`), configs, diagnostics, results (`results/incite/*.md`), the
  deviation ledger (`results/incite/config_diff.md`), INCITE rank dumps,
  and the key checkpoints (`checkpoints/`, see its README). Worktree here:
  `~/Dokumente/GitHub/sotaKGFMs-incite`.
* `crest` — a falsified predecessor, kept for the record.
* `main` — the pre-GPU state; do not work there.

The incite worktree symlinks `output/` and `data/roots` to the main
worktree's copies. `output/` (training dirs, logs) and `data/` are not in
git. Checkpoints that matter are in `checkpoints/` on the incite branch.

## Live state right now (2026-09-05, 13:26) and how the session operates

* **PAUSED (13:26, 5 Sep, on Luke's request).** Plan v18 and the R0
  training container are stopped; the GPU is free. R0's last saved
  checkpoint is step 27,000 of 30,000 (`output/incite-pretrain/incite_last.pth`,
  STAGE = R0); about 950 steps, 25 minutes, are lost on resume. To resume
  everything exactly where it stopped: `./RESTART.sh` in the baseline
  worktree (it launches v18, which keeps R0's run directory and resumes
  from the last checkpoint with the same absolute-step schedule; R1, the
  seeds and the rest follow as before). Nothing else needs to be done
  first; the queue's markers are intact. Until then, no GPU job runs.

* **Best single model:** MX15, the 4-graph backbone continued for 10k
  decay steps with 15 percent synthetic rules-prior steps: 0.4621 /
  0.3893 (ind_e / ind_er); TRIX 0.4562 / 0.3679; the no-mix baseline L1
  0.4560 / 0.3852. Single seed, a continuation. Its recipe is THE RECIPE
  (`output/research-plan/RECIPE` = `mx1f15`, decided 01:53, 5 Sep, by the
  recorded rule below): the paper model R1 is that recipe trained from
  scratch (30k steps, the mix in the decay phase only), its control R0
  the same without the mix, at seeds 1024, 1337 and 7.
* **What we know (the week's verdicts, each with a note under
  `results/incite/`):**
  * The synthetic rules prior lifts the unseen-answer cells everywhere
    it was measured (benchmark, dev suite, carved inductive splits) and
    pays on the seen-answer cells everywhere too. On the benchmark the
    trade is net positive at 15 percent dose (MX15 over L1 +0.0062 /
    +0.0040, both intervals above zero), flat at 30 (MX1), negative at 45
    (MX45: DOSE_CURVE.md). Outside the benchmark it is net negative at
    every dose (DEV_SUITE.md), and on the benchmark the gain sits on the
    diet's own Freebase and NELL families. Until R0 versus R1 at three
    seeds lands, the honest description is a diet-family effect with a
    seen-answer cost, not a general structural prior.
  * A model trained on the prior alone reaches 86 / 82 percent of
    ULTRA-3g with no real graph (P1). At matched 3-graph diet the mix
    brings INCITE exactly to TRIX on ind_e (FMX). The ind_er margin over
    TRIX is the fourth diet graph plus the decay (L1 has it).
  * Closed directions, each MX1 within noise or worse: the proof-guided
    gate three ways (PG2 inert by construction; PG3 learned the proof
    concept at AUC 0.95 and gated 90 percent of real-graph messages below
    half weight with NO change in any cell; neither gate's ranking allows
    a free hard cut of the edges: GATE_RESULT.md); the scenario-
    conditioned readout (SC1, the head stayed at zero); the rule-recovery
    head (RR2, learns the rules, costs the seen-answer cells); isolated
    relation blocks (MX2a, half of MX2's loss); the 90 percent unseen-
    answer share (MXS9); the unary channel on top of the mix (MXG1);
    half-link masking (M1, M2); every score-level or weight-level
    combination of L1 with MX1 or MX15 (COMBINATION_PROBE.md, SWAP_SOUP_RESULT.md:
    the swap keeps L1's seen-answer cells and a third of the unseen gain,
    below MX15; the 0.5 soup ties MX15 at one pass); same-encoder
    in-context readouts (phase 2.2). The in-context family (KG-ICL, KGPFN)
    sits below MX15 on every scenario cell of the benchmark and wins per
    graph family, not per scenario.
  * The independent review's minimum set for a defensible claim: a
    disjoint dev suite (done, stratified), R0/R1 at three seeds (running),
    a matched-diet comparison (FMX, done), X1/X2 (queued), the mechanism
    figure (dose curve, scenario tables, family tables, P1: done).
* **What runs (unattended):** `scripts/research_plan_v18.sh` (pid 140885,
  launched 15:05 on 4 Sep, took over from v17 at MX2a's boundary). It ran
  SW1, RR2 and PG3, wrote the recipe, and trains R0 (from scratch, no
  mix, seed 1024) since 01:53. Then R1 (the paper model), X1/X2 (TRIX at
  20k with their code, the matched-budget test), MX2H (the MX2 bundle at
  half dose with the negative mask), the seeds R1S2/R0S2/R1S3/R0S3, the
  chores E4/E5/E6/F0, R1L (the recipe at 60k). Expected end: about 10
  Sep (the 60k run ends last), counted from the resume. The plan table below has the ETAs: continuations ran about 1.5
  times faster than the ETAs written on 3 Sep, but a from-scratch run
  goes at about 0.7 steps per second here, half a continuation's speed
  (R0: 26,600 steps in 11 hours), so the seeds take about 13 hours each. Every plan version since
  v15 was read by an independent verifier agent before it ran (v15, the
  stratified dev suite and v16, PG3 and v17); v18 added an evaluation-only
  stage. Nothing else runs on the GPU; CPU studies (the combination
  probe, the gate probes) run in containers on the CPU only.
* **Protocol:** every lever got a DEV-SUITE verdict first
  (`diagnostics/dev_eval.py`: valid splits of eight transductive graphs
  outside the diet and the 41 test graphs; `results/incite/dev/<stage>.json`;
  `protocol: stratified_v2`: up to 300 queries per (direction, half-link
  scenario) cell, the eight cell MRRs combined with the benchmark's
  (direction, scenario) weights, cells under 10 queries left out, every
  cell's MRR and count in the file, the graph's own mix beside it as
  `graphs_natural`; DEV_SUITE.md has the weights and the history: the
  uniform sample of 3 Sep was 80 to 90 percent seen-answer queries and
  ranked MX1 below L1, the stratified one still does, because the
  prior's unseen-answer gains do not transfer while its costs do), then
  its 41-graph dump. `--split inductive` (protocol `inductive_v3`,
  analysis only) carves sparse inference splits out of the dev graphs.
  The from-scratch runs R0 and R1 and their seeds get dev numbers too.
* **The recipe rule (recorded in the plan, applied 01:53, 5 Sep):** MX1's
  recipe, the mix in the decay phase only (`synth.start_step 20001`),
  plus AT MOST ONE modification, accepted if its stratified dev number
  beats MX1's by at least 0.003 on the graphs both have (at least six)
  AND its paired 41-graph interval against MX1 is above zero on one group
  with the other group's point estimate above −0.002; the accepted
  candidate with the largest dev gain wins. Outcome: `mx1f15` (MX15, dev
  +0.0036, test ind_er interval above zero). SC1 dev +0.0017, MX45
  −0.0011, MXS9 −0.0035, RR2 +0.0005, PG3 −0.0001 fell short of the dev
  gate; MX2a (dev +0.0051) failed the test gate. The dev gate favours
  candidates that reduce the prior's cost over ones that intensify it;
  disclosed and kept, because a test-set argmax is the alternative the
  review ruled out.
* **How to intervene:** `touch output/research-plan/SEEDS_HOLD` holds
  the seed stages R1S2/R0S2/R1S3/R0S3 and R1L (R0 and R1 still run).
  `output/research-plan/RECIPE_HOLD` holds the recipe decision (moot now:
  the decision is written; delete `output/research-plan/RECIPE` to redo
  it). A stage is skipped by its marker in `output/research-plan/`;
  delete `<stage>.failed` to retry it on a relaunch (`./RESTART.sh`
  launches only the plan, currently v18). Never edit a running plan
  script: write the next version with the boundary takeover and launch
  it; a version launched while the predecessor is between stages waits
  through the predecessor's next stage (v18 did, harmlessly).
* **Per-stage ritual** (what the session does when a stage lands, in
  this order): read `results/incite/dev/<stage>.json` against its paired
  reference's dev number FIRST; then `scripts/paired_bootstrap.py
  <new-last> <reference>` (R1: R0 and TRIX and MX15; the seed repeats:
  their own R0 and the seed-1024 pair; X1/X2: L1 and the 4g-20k
  checkpoint; MX2H: MX15 and MX1) and versus `ranks/trix`;
  `scripts/halflink_report.py --labels
  ../sotaKGFMs-incite/results/incite/halflink_labels.json name=dir ...`
  for the per-scenario table; a result note under `results/incite/`; a
  checkpoint into `checkpoints/` with a README row for R1 and its seeds;
  the state table and the plan table here; commit and push both branches
  (`git push origin incite` from the incite worktree; the remote push URL
  is SSH). For R1 against R0 at three seeds, also the family table
  (FB/NELL/WK/WN/other) and the carved inductive dev number
  (`dev_eval.py --split inductive`, CPU), the two views that decide the
  paper's framing.
* **Open decision for Luke:** whether the paper model keeps the mix at
  all, once R0 versus R1 at three seeds is in (benchmark, dev suite,
  carved splits). The queue produces both; the framing follows the
  result. Luke's framing of the field (4 Sep): "structural" = the
  propagation family (TRIX, ULTRA, our trunk), "in-context" = the
  prompt/example family (KG-ICL, KGPFN); he believes the right fusion is
  still unfound. The combination probe's designs and the scenario-
  conditioned trunk with a training signal are the candidates for after
  the seeds (section "Directions NOT covered").
* **Watchers alive on this machine:** only the plan and a Monitor on
  `output/research-plan/log.txt` that wakes the session on DONE/FAILED
  lines. Do not start another GPU job.
* **Memory notes for the session** live in
  `~/.claude/projects/-home-lukef-Dokumente-GitHub-sotaKGFMs/memory/`
  (validate before building; never edit running scripts; idempotent
  queues; objective fixes get built, hypotheses get a paired run; every
  recipe candidate before the paper-model runs; ask before reprioritizing).

## The state in one table (41 inductive graphs, test splits, seed 1024, LAST checkpoints unless stated)

| model | ind_e MRR | ind_er MRR | note |
| --- | --- | --- | --- |
| ULTRA 3g | 0.4158 | 0.3421 | released checkpoint |
| ULTRA 4g | 0.4454 | 0.3460 | released; the same diet as INCITE-4g |
| MOTIF | 0.4361 | 0.3491 | released |
| SEMMA | 0.4496 | 0.3520 | released |
| KG-ICL | 0.4240 | 0.3722 | matched convention |
| TRIX | 0.4562 | 0.3679 | released, about 100k steps |
| FLOCK | 0.4558 | 0.3674 | 40 of 41 graphs (ind_er over 22) |
| KGPFN | 29 of 41 | | at TRIX level on the graphs done; the suite is stopped |
| INCITE floor (3g, 20k steps) | 0.4553 | 0.3740 | DEV10-best checkpoint (17k); the last checkpoint (20k) is 0.4533 / 0.3749 |
| INCITE floor-family soup | 0.4571 | 0.3775 | average of four floor descendants; the best matched-diet (3g) row |
| INCITE floor + 10k decay (L2), last | 0.4510 | 0.3745 | no gain at 3 graphs (results/incite/DECAY_RESULT.md) |
| INCITE 4g (20k) DEV10-best | 0.4542 | 0.3791 | |
| INCITE 4g (20k) last | 0.4534 | 0.3825 | beats ULTRA-4g by +0.008 / +0.036; TRIX by −0.003 / +0.015 (ind_er interval excludes zero) |
| **INCITE 4g + 10k decay (L1), last** | **0.4560** | **0.3852** | THE REFERENCE (2026-09-02, `checkpoints/incite-4g-decay-last-step30k.pth`): ties TRIX on ind_e, +0.017 on ind_er [+0.005, +0.036], 18 of 23 graphs; see results/incite/DECAY_RESULT.md |
| INCITE 4g + unary channel (G1), last | 0.4571 | 0.3874 | +0.001 / +0.002 over L1, gains in the unseen-answer cells; results/incite/UNARY_RESULT.md |
| **INCITE 4g + 30 percent synthetic mix (MX1), last** | **0.4606** | **0.3851** | best single model (2026-09-03): ind_e +0.0046 over L1 with interval [+0.0002, +0.0092]; both-unseen cell +0.065 on both groups; results/incite/SYNTH_MIX_RESULT.md |
| INCITE 4g + synthetic mix + unary channel (MXG1), last | 0.4593 | 0.3852 | the unary channel is redundant with the mix: −0.0013 [−0.0025, +0.0001] / +0.0002 versus MX1, the same scenario profile cell for cell; the DEV10-best checkpoint (9k of 10k) scores 0.4610 / 0.3870, +0.002 on both groups over last, the checkpoint-level noise at the end of the decay; results/incite/UNARY_SYNTH_RESULT.md |
| INCITE 4g + synthetic mix with the generator-side bundle (MX2), last | 0.4512 | 0.3717 | NEGATIVE: −0.0094 [−0.0134, −0.0056] / −0.0134 [−0.0219, −0.0059] versus MX1, every scenario cell down; the synthetic-step loss doubled (0.14 to 0.27), the dose rose; bisection MX2a/MX2b queued; results/incite/SYNTH_V2_RESULT.md |
| INCITE 4g + synthetic mix + proof-guided gate (PG2), last | 0.4588 | 0.3869 | inert, as the review predicted: −0.0018 [−0.0033, −0.0005] / +0.0018 [−0.0019, +0.0073] versus MX1, the same scenario profile cell for cell, no gate bias moved by more than 0.11; results/incite/GATE_RESULT.md |
| INCITE 4g + synthetic mix + scenario-conditioned readout (SC1), last | 0.4613 | 0.3873 | MX1 within noise: +0.0007 [−0.0003, +0.0018] / +0.0023 [−0.0009, +0.0068], every scenario cell within 0.003; the head's last layer stayed near zero (weight norm 0.03); not accepted as the recipe modification (the test interval includes zero); results/incite/SCENARIO_RESULT.md |
| INCITE 3g floor + 10k decay + 30 percent mix (FMX), last | 0.4563 | 0.3720 | the matched-diet test: ties TRIX on ind_e (+0.0001 [−0.0061, +0.0061]) and +0.0041 [−0.0057, +0.0165] on ind_er; the mix over L2 is +0.0053 [−0.0013, +0.0125] / −0.0025, the same scenario trade as MX1's; the fourth graph is +0.004 / +0.013 (MX1 − FMX); results/incite/FLOOR_MIX_RESULT.md |
| **INCITE 4g + 15 percent synthetic mix (MX15), last** | **0.4621** | **0.3893** | the first mix result above the no-mix baseline on BOTH groups with intervals above zero: +0.0062 [+0.0018, +0.0109] / +0.0040 [+0.0009, +0.0078] over L1; +0.0016 [−0.0005, +0.0037] / +0.0042 [+0.0011, +0.0078] over MX1; half the seen-answer cost, most of the unseen-answer gain; dev +0.0036 over MX1 (7 of 8): clears both gates of the recipe rule (results/incite/DOSE15_RESULT.md) |
| INCITE 4g + 45 percent synthetic mix (MX45), last | 0.4585 | 0.3820 | the dose curve's high point: below MX1 (−0.0021 [−0.0034, −0.0007] / −0.0030 [−0.0060, −0.0006]) and MX15; the unseen-answer gain saturates by 30 percent, the seen-answer cost keeps growing; results/incite/DOSE_CURVE.md |
| INCITE 4g + 30 percent mix, 90 percent unseen-answer queries (MXS9), last | 0.4553 | 0.3853 | the trade pushed further: −0.0052 [−0.0074, −0.0034] / +0.0003 versus MX1, every ind_e graph below MX1; rejected on both gates; results/incite/SHARE90_RESULT.md |
| INCITE 4g + 30 percent mix with isolated relation blocks (MX2a), last | 0.4556 | 0.3797 | bisection of MX2: isolation alone is about half of the bundle's loss, −0.0049 [−0.0087, −0.0018] / −0.0053 [−0.0123, +0.0015] versus MX1, on the seen-answer cells; dev +0.0051 over MX1 (7 of 8) but the test gate fails: rejected; results/incite/ISOLATION_RESULT.md |
| MX2a + rule recovery head at weight 0.2 (RR2), last | 0.4521 | 0.3773 | the head learns the rules (synthetic loss 0.96 → 0.19) and the seen-answer cells pay: −0.0035 [−0.0047, −0.0023] / −0.0024 versus MX2a, −0.0084 / −0.0077 versus MX1; rejected; results/incite/RULES_RECOVERY_RESULT.md |
| INCITE 4g + synthetic mix + the gate that can close (PG3), last | 0.4596 | 0.3857 | the gate learned this time (AUC 0.95 on synthetic proof edges; 90 to 95 percent of real-graph messages gated below 0.5) and the accuracy did not move: −0.0010 [−0.0022, +0.0003] / +0.0006 versus MX1, dev equal to MX1's; not the recipe modification; results/incite/GATE_RESULT.md |
| round-wise swap: L1's rounds 0-2 and heads, MX15's rounds 3-5 (one pass) | 0.4580 | 0.3868 | keeps L1's seen-answer cells and a third of the unseen-answer gain: +0.0021 / +0.0016 over L1 (intervals through zero), −0.0041 [−0.0070, −0.0012] / −0.0025 below MX15; results/incite/SWAP_SOUP_RESULT.md |
| 0.5 parameter soup of L1 and MX15 (one pass) | 0.4609 | 0.3880 | MX15 at one pass with a milder profile: +0.0049 [+0.0016, +0.0086] / +0.0027 [+0.0005, +0.0055] over L1, tied with MX15 (−0.0012 / −0.0013, intervals through zero) |
| INCITE 4g + masked continuation, dose 1 (M1) | 0.4420 | 0.3604 | NEGATIVE, see below |
| INCITE 4g + masked continuation, dose 2 (M2, cap 10) | 0.4384 | 0.3629 | NEGATIVE; after 1,000 masked steps the unseen-answer cells rose +0.03 at a −0.035 SQSA cost, then inverted |
| INCITE v1 composite (walks+synth+joint) | 0.4500 | 0.3659 | below TRIX on entity; relation ind_er 0.8484 beats TRIX's specialist (0.8415) |
| **INCITE trained on the synthetic rules prior ONLY** (P1, 10k steps, 41 min) | **0.3593** | **0.2795** | no real KG seen; 86 / 82 percent of ULTRA-3g; strong on unseen-answer cells, weak on seen-answer ones; results/incite/SYNTH_PILOT_RESULT.md |

The dev suite so far (eight transductive valid splits outside the diet
and the benchmark; results/incite/DEV_SUITE.md):

| model | uniform (D0) | stratified, benchmark-weighted (D0W) | graphs won against L1 (stratified) |
| --- | --- | --- | --- |
| L1 (4g + decay) | 0.3408 | 0.3082 | |
| MX1 (4g + decay + 30 percent mix) | 0.3324 | 0.3014 | 1 of 8 |
| G1 (4g + decay + unary) | 0.3399 | 0.3088 | 4 of 8 |
| L2 (3g floor + decay) | 0.3407 | 0.3029 | 2 of 8 |
| SC1 (MX1 + scenario readout) | | 0.3031 | 1 of 8 |
| FMX (3g floor + decay + mix) | | 0.2997 | 2 of 8 against L2 |
| MX15 (4g + decay + 15 percent mix) | | 0.3050 | 7 of 8 against MX1, 1 of 8 against L1 |
| MX45 (4g + decay + 45 percent mix) | | 0.3003 | 3 of 8 against MX1 |
| MXS9 (MX1 with 90 percent unseen-answer queries) | | 0.2979 | 3 of 8 against MX1 |
| MX2a (MX1 with isolated relation blocks) | | 0.3065 | 7 of 8 against MX1; the test gate fails |
| RR2 (MX2a + rule head) | | 0.3019 | 5 of 8 against MX1 |
| PG3 (MX1 + the gate that can close) | | 0.3014 | 5 of 8 against MX1 |
| swap (L1 rounds 0-2, MX15 rounds 3-5) | | 0.3065 | 4 of 8 against MX15 |
| soup (0.5 L1 + 0.5 MX15) | | 0.3060 | 5 of 8 against MX15 |

THE FINDING OF 4 SEP: the prior's benchmark gain does not transfer.
Outside the benchmark MX1 is 0.007 to 0.009 below L1 whichever way the
queries are weighted, and 0.014 below (2 wins of 16) on sparse inductive
splits carved from the same graphs (`dev_eval.py --split inductive`,
`results/incite/dev/ind_*.json`), where it also ends below its own 20k
start while the plain decay gains 0.007. Its seen-answer costs
(SQSA −0.011, UQSA −0.017) transfer from the benchmark; its unseen-answer
gains (there +0.020 / +0.068) do not (+0.007 / +0.020). On the benchmark
the gain sits on the diet's own families: FB15k237-derived +0.0066
(ind_e) / +0.0047 (ind_er), NELL-derived +0.0084 / +0.0121, Wikidata-
and WordNet-derived −0.003 to −0.001, Metafam −0.042. The decisive
experiment is queued (R0 against R1 at three seeds, benchmark and dev
suite); until it lands, the prior is a diet-family effect with a
seen-answer cost, not a general structural prior. THE COMBINATION PROBE (4 Sep, CPU, results/incite/COMBINATION_PROBE.md):
every score-level combination of L1 with MX1 or MX15 lands within
±0.001 of L1 on held-out carved splits against an oracle headroom of
+0.0065; the states carry the answer-half indicator with AUC 1.0 but no
head uses it; KG-ICL is below MX15 on every scenario cell of the 41
graphs and wins per graph family, not per scenario; the one lead, a
round-wise swap (L1's early rounds, MX15's late rounds), +0.0024 over
L1 on the carved splits, came out 0.004 below MX15 on the benchmark
(stage SW1), and the 0.5 soup of L1 and MX15 ties MX15 at one pass:
no combination of the two checkpoints is a recipe. OPEN DECISION FOR
LUKE: whether the paper model keeps the mix at all. The queue needs no
change for it (R0 is the no-mix control at every seed, with dev
numbers), but the framing does. The fourth diet graph adds nothing on
the transductive valid splits (L2 equals L1) and 0.020 on the carved
inductive splits.

Relation task (unfiltered protocol): TRIX 0.7564 / 0.8415, INCITE joint
0.7286 / 0.8222, v1 composite ind_er 0.8484. PETALS: v1 94.6 / 98.6
percent; deterministic models 50 percent.

Tools: `scripts/paired_bootstrap.py A B` (graph-level interval of a
margin), `scripts/halflink_report.py --labels
../sotaKGFMs-incite/results/incite/halflink_labels.json name=dir ...`
(per-scenario table), `scripts/make_summary.py`, `scripts/complementarity.py`.

## The independent review (2026-09-03, `docs/REVIEW_2026-09-03.md`)

An independent Fable 5.1 agent reviewed every note, recomputed the
headline numbers from the parquets, and judged the direction. Its
verdict, all of it verified where it says so: the SOTA claim does not
hold as stated. At matched diet INCITE ties TRIX on both groups; the
4-graph ind_er margin (+0.017) comes from the diet and the decay (L1
already has it), not from the prior; the prior's own contribution is
+0.0046 on ind_e, single seed, chosen from 32 test-set evaluations, and a
test-set argmax picked the recipe by 0.0006. The defensible paper is
about what a synthetic structural prior teaches a KGFM (the synthetic-only
result, the per-scenario mechanism, a scaling curve), positioned against
GraphPFN, PluRel and RDB-PFN. Findings acted on: the gate was inert by
construction (saturated init, no closing pressure) and its pruning
measurement could not tell "nothing pruned" from "flat" (fixed); rule
recovery at weight 1.0 was a fivefold dose (now 0.2); the scenario share
0.37 moved the training mixture by three points (now 0.9); MX2b confounded
many negatives with duplicates (replaced by the bundle at half dose, with
a negative mask); R1 lacked a no-mix control and applied the mix through
the constant phase (R0 added, mix in the decay phase only); the winner
pick was a test-set argmax (replaced by a recorded rule and a dev suite).
Its ranked new directions: (1) a scenario-conditioned readout (upper
bound from our dumps +0.0076 / +0.0079 over MX1; built as SC1), (2) the
prior as the pretraining stage with a scaling curve, (3) disconnected-
answer supervision for the unreachable 17 percent. Its minimum set for a
defensible claim: a disjoint dev suite, R0/R1 at three seeds, a
matched-diet comparison (FMX), X1/X2, the mechanism figure.

## Findings that shape the next steps

1. **INCITE ties TRIX and FLOCK at 20 percent of TRIX's steps.** The
   backbone is TRIX's alternation with an exact O(E) relation step (gate
   test at 1e-5). Whether parity at 20k is sample efficiency is the TRIX
   matched-budget A/B (plan stages X1/X2, pending).
2. **Where the MRR is lost** (`results/incite/halflink.json`,
   `reachability.json`, `results/halflink_report.md` on the baseline
   branch). Scenario shares of Gregucci et al. (arXiv 2606.18001)
   reproduce on our suite. Every KGFM scores 0.11 to 0.29 MRR when the
   answer has NO edge of the query relation (SQUA, 28 percent of queries)
   against 0.54 to 0.62 when it has one. 17 percent of ind_e answers lie
   beyond six hops and score below 0.01 (HM:1k/3k/5k are 73 to 83
   percent unreachable).
3. **Half-link masking, dose 1, is negative and inverted**
   (`results/incite/MASKING_RESULT.md`). Masking the target's other
   query-relation edges at p 0.3 (plus query half 0.3) during a 10k
   decay continuation: −0.011 / −0.022 MRR, seen-answer scenarios UP,
   unseen-answer scenarios DOWN. Verified on FB15k237 that the masking
   does what it says; the culprit is hub stripping (one target lost 484
   edges), which teaches a popularity prior. Dose 2 (answer only, only
   targets with at most 10 query-relation edges, p 0.5) is queued (M2).
   The paired decay-only baseline (L1) went UP at its first validation
   (DEV10 scalar 0.4233 to 0.4288), so the optimizer restart is not the
   cause. L1's per-scenario table shows plain continued training already
   drifts toward seen-answer candidates (SQSA/UQSA up, SQUA/UQUA down);
   masking amplified that drift. Read every continuation lever against
   the L1 row (results/incite/DECAY_RESULT.md). Dose 2 (cap 10, answer
   only) is negative too: −0.018 / −0.022. Its 1,000-step checkpoint shows
   the lever is a scenario KNOB: unseen-answer cells up 0.03 to 0.04,
   SQSA down 0.035, net negative under the test mix. Masking is dead as a
   net lever at both doses; keep it as a diagnostic figure.
4. **Checkpoint selection on DEV10 buys nothing** (E1): the unselected
   last checkpoint equals or beats the selected one. Report LAST
   checkpoints. DEV10 is diagnostic only.
5. **Weight soups of different constant-lr runs help a little** (E2:
   +0.002 / +0.0035 over the floor). Snapshot soups of one decayed run
   add nothing (L1, G1: within ±0.001); the decay already averages.
6. **Bidirectional re-ranking is marginal** (`results/incite/TESTTIME_LEVERS.md`):
   +0.004 on the DEV10 scalar at k=8, weight 0.5, gains on ind_e graphs,
   losses on ind_er graphs, at 3.6x eval cost. Queued late (E4/E5).
7. **Dead levers, closed:** same-encoder in-context support readouts
   (twice), unsupervised walks, KGPFN-style in-context inference as a
   bolt-on (KGPFN itself sits at TRIX level here on 13 graphs); and,
   this week, the proof-guided gate (PG2, PG3 and both pruning curves),
   the scenario-conditioned readout (SC1), the rule-recovery head (RR2),
   isolated relation blocks (MX2a), the 90 percent unseen-answer share
   (MXS9), the unary channel on top of the mix (MXG1), and every
   inference-time or weight-level combination of L1 with MX1 or MX15
   (the combination probe and SW1).
8. **Hard negatives are published** (KMAS, arXiv 2605.27023, +0.005 to
   +0.009 for every KGFM): a citation, not a contribution.
9. **The synthetic rules prior: what it does and where.** A model
   trained on it alone reaches 0.3593 / 0.2795 on the 41 real graphs (P1,
   86 / 82 percent of ULTRA-3g), strong on the unseen-answer cells, weak
   on the seen-answer ones. Mixed into the 4-graph continuation it lifts
   the unseen-answer cells (+0.06 on the both-unseen cell at 30 percent)
   and costs the seen-answer cells, and the net depends on the dose and
   the graphs: 15 percent is the optimum on the benchmark (MX15 over L1
   +0.0062 / +0.0040, both intervals above zero; DOSE_CURVE.md), 30 is
   flat, 45 negative; outside the benchmark every dose is net negative
   (DEV_SUITE.md), and on the benchmark the gain sits on the diet's own
   Freebase and NELL families. Everything built on top of the plain mix
   lost or tied: the generator-side bundle (MX2) and its halves (MX2a;
   MX2H queued), the unseen-answer share (MXS9), the gate (PG2, PG3), the
   rule head (RR2), the readout (SC1). The prior's value comes from being
   a mild complementary signal. The paper's mechanism figure is the dose
   curve, the scenario tables, the family table and P1; the paper's
   claim waits for R0 versus R1 at three seeds.
10. **Seed spread is the missing number.** Same-architecture runs swing by
   up to 0.10 MRR on single graphs; the only statistics we have are
   graph-level bootstraps. Three seeds of the winner are in the plan.

## Running on this machine: `scripts/research_plan_v18.sh` (markers in output/research-plan/)

Stages are idempotent (v4 fixed a relaunch that wiped a finished run; a
`trained` guard skips a finished training). `./RESTART.sh` relaunches the
plan after a pause. The seed stages run AUTOMATICALLY (the user's
2026-09-03 instruction) unless `output/research-plan/SEEDS_HOLD` exists.
Order and ETAs (continuations: 10k steps from the 4g last checkpoint,
linear lr decay, warmup 500, kept snapshots every 1000; from-scratch runs:
30k steps, constant to 20k, then warmup and decay, the mix in the decay
phase only):

| stage | what | output (incite worktree) | ETA |
| --- | --- | --- | --- |
| L1 | decay only, the paired baseline: DONE, 0.4560 / 0.3852 | ranks/incite-4g-decay(-last) | done |
| G1 | unary channel: DONE, 0.4571 / 0.3874 | ranks/incite-4g-unary(-last) | done |
| M2 | masking dose 2: DONE, negative (0.4384 / 0.3629) | ranks/incite-4g-mask2(-last) | done |
| P1 | synthetic rules-prior 100 percent pilot: DONE, 0.3593 / 0.2795 | ranks/incite-synth100-pilot | done |
| L2 | floor (3-graph) decay: DONE, no gain (0.4510 / 0.3745) | ranks/incite-decay(-last) | done |
| MX1 | 4g continuation with 30 percent synthetic steps: DONE, 0.4606 / 0.3851 | ranks/incite-4g-synth30(-last) | done |
| MXG1 | synthetic mix 30 percent + unary channel: DONE, 0.4593 / 0.3852, the unary channel adds nothing on top of the mix (results/incite/UNARY_SYNTH_RESULT.md); the DEV10-best dump finishes under v11 | ranks/incite-4g-unary-synth30(-last) | done |
| MX2 | MX1 plus the generator-side bundle: DONE, NEGATIVE (0.4512 / 0.3717; results/incite/SYNTH_V2_RESULT.md) | ranks/incite-4g-synth30v2-last | done |
| PG2 | MX1 plus the proof-guided propagation gate: DONE, inert (0.4588 / 0.3869, MX1 within noise; results/incite/GATE_RESULT.md) | ranks/incite-4g-synth30-gate-last | done |
| PG2P | the gate's pruning curve: DONE. The gate ranks edges far above chance at the same realized kept fraction, but every fraction costs accuracy (ind_e −0.002 at 93 percent kept, −0.023 at 53); at inference every gate product is 1.00; the sparse-propagation direction is closed (results/incite/GATE_RESULT.md) | results/incite/gate_prune.json | done |
| D0 | dev-suite numbers of L1, MX1, G1, L2 under the UNIFORM protocol: DONE (22:48, 3 Sep): 0.3408 / 0.3324 / 0.3399 / 0.3407, MX1 below L1 on 7 of 8 graphs, the seen-answer skew of transductive valid splits (see Protocol) | results/incite/dev/{L1,MX1,G1,L2}.uniform.json after D0W | done |
| SC1 | MX1 plus the scenario-conditioned readout: DONE (02:04, 4 Sep), MX1 within noise (0.4613 / 0.3873), the head stayed near zero, NOT the recipe modification; stratified dev 0.3031 (the references come with D0W); results/incite/SCENARIO_RESULT.md | results/incite/dev/SC1.json, ranks/incite-4g-synth30-scenario-last | done |
| D0W | the four reference dev numbers under the stratified protocol: DONE (02:30, 4 Sep): L1 0.3082, MX1 0.3014, G1 0.3088, L2 0.3029; the expectation failed, the prior does not transfer (results/incite/DEV_SUITE.md) | results/incite/dev/{L1,MX1,G1,L2}.json (+ .uniform.json) | done |
| FMX | the 3-graph floor plus the mix, the matched-diet test: DONE (04:58, 4 Sep), 0.4563 / 0.3720, ties TRIX at matched diet, the mix's trade is the same at 3 graphs, dev −0.0032 versus L2 (results/incite/FLOOR_MIX_RESULT.md) | ranks/incite-synth30-last, results/incite/dev/FMX.json | done |
| MX15 | the mix at 15 percent: DONE (08:41, 4 Sep), 0.4621 / 0.3893, clears both gates, the leading candidate (results/incite/DOSE15_RESULT.md) | ranks/incite-4g-synth15-last, results/incite/dev/MX15.json | done |
| MX45 | the mix at 45 percent: DONE (11:43, 4 Sep), 0.4585 / 0.3820, below MX1 on both groups; the dose curve 0/15/30/45 peaks at 15 (results/incite/DOSE_CURVE.md) | ranks/incite-4g-synth45-last, results/incite/dev/MX45.json | done |
| MXS9 | MX1 with 90 percent unseen-answer synthetic queries: DONE (15:15, 4 Sep), 0.4553 / 0.3853, rejected on both gates (results/incite/SHARE90_RESULT.md) | ranks/incite-4g-synth30-s90-last, results/incite/dev/MXS9.json | done |
| SW1 | (plan v18, evaluation only) the combination probe's round-wise swap (L1 rounds 0-2 + MX15 rounds 3-5) and the 0.5 soup of L1 and MX15: dev suite for both, the 41 graphs for one within 0.002 of MX15's dev 0.3050. Expectation recorded by the probe: near MX15, not +0.003 above it | results/incite/dev/SW1{swap,soup}.json, ranks/incite-{swap,soup}-l1mx15-last | done (19:11, 4 Sep): both within 0.002 of MX15 on dev, both evaluated; the swap 0.4580 / 0.3868, the soup 0.4609 / 0.3880, neither above MX15 (results/incite/SWAP_SOUP_RESULT.md) |
| MX2a | relation blocks alone: DONE (18:35, 4 Sep), 0.4556 / 0.3797, half of MX2's loss, rejected (dev gate passes, test gate fails; results/incite/ISOLATION_RESULT.md) | ranks/incite-4g-synth30-iso-last, results/incite/dev/MX2a.json | done |
| RR2 | rule recovery at weight 0.2: DONE (22:28, 4 Sep), 0.4521 / 0.3773, rejected (results/incite/RULES_RECOVERY_RESULT.md) | ranks/incite-4g-synth30-iso-rules-last, results/incite/dev/RR2.json | done |
| PG3 | the gate that can close: DONE (01:53, 5 Sep), 0.4596 / 0.3857, MX1 within noise with a gate that did learn; its hard-pruning curve is PG2's shape (−0.026 at 61 percent of the edges kept): the gate direction is closed for good (results/incite/GATE_RESULT.md) | ranks/incite-4g-synth30-gate3-last, results/incite/dev/PG3.json | done |
| recipe | DECIDED (01:53, 5 Sep) by the recorded rule: `mx1f15` (MX15: dev +0.0036 over MX1, test ind_er interval above zero, ind_e point +0.0016). The others: SC1 dev +0.0017, MX45 −0.0011, MXS9 −0.0035, RR2 +0.0005, PG3 −0.0001 (all short of the dev gate or, MX2a, failing the test gate) | output/research-plan/RECIPE | done |
| R0 | from scratch, 30k steps, no mix, seed 1024 (the control); trained 01:53 to 13:26, 5 Sep, PAUSED at step 27,000 (last checkpoint); resumes with `./RESTART.sh` | ranks/incite-nomix-mx1f15-seed1024-last, results/incite/dev/R0.json | about 3 h after the resume |
| R1 | THE PAPER MODEL: from scratch, 30k steps, the recipe `configs/incite_recipe_mx1_f15.yaml` (the 15 percent mix in the decay phase), seed 1024 | ranks/incite-recipe-mx1f15-seed1024-last, results/incite/dev/R1.json | 05:30, 6 Sep |
| X1/X2 | TRIX@20k with their code, best epoch and last (the matched-budget test) | ranks/trix-20k-best/-last | 12:30, 6 Sep |
| MX2H | the MX2 bundle at half dose with the negative mask (`configs/incite_phase1_4g_synth15_v2.yaml`): dose or distribution; paired against MX15 and MX1 | ranks/incite-4g-synth15v2-last | 18:00, 6 Sep |
| R1S2, R0S2, R1S3, R0S3 | the recipe and its control at seeds 1337 and 7 (each with its own synthetic seed, `--synth_seed`); `touch output/research-plan/SEEDS_HOLD` holds them | ranks/incite-{recipe,nomix}-mx1f15-seed{1337,7}-last | 00:00, 9 Sep |
| E4/E5/E6, F0 | re-ranked dumps (k=8), score ensemble of four trunks, FLOCK FBIngram:25 | ranks/incite-*-rerank, incite-ens4, flock 41/41 | 06:00, 9 Sep |
| R1L | the recipe at 60k steps (40k constant, 20k decay), one seed; held by SEEDS_HOLD | ranks/incite-recipe-mx1f15-60k-seed1024-last | 08:00, 10 Sep |

Nothing else runs on this GPU. Do NOT start another training run on it.
Evals of 1 to 3 GB are fine, but only after two low memory readings three
minutes apart: a training container ramps to 13 GB within minutes of its
start and an eval that starts in the ramp OOMs (it happened).

## Directions NOT covered by the plan (for a second agent or machine)

Ranked by my expected value. Each is independent of the running plan.
Use distinct ranks dir names (for example a `-m2` suffix) so dumps merge
without conflicts.

1. **Label-free test-time training.** Fine-tune the 4g-last checkpoint on
   each inference graph's own edges for a few hundred steps before
   scoring. No test labels. Goes on its own table row, never the
   zero-shot headline. Expected +0.01 to +0.02. Cheap: minutes per graph
   on the 4g-last checkpoint (`checkpoints/`).
2. **Scenario-aware re-ranking.** The reverse query of an unseen-answer
   case is a seen-answer case from the other side. Apply the reverse
   logit only where the candidate lacks an edge of the query relation
   (weight by that indicator) instead of everywhere. Eval-only,
   `incite/rerank.py` is the starting point.
3. **Relation-vocabulary augmentation in pretraining.** Split a relation
   by edge hashing, merge two, or drop some, on a fraction of steps, to
   manufacture the relation diversity that the fourth graph supplied
   (+0.005 ind_er). Needs one 20k run per setting.
4. **8-graph pretraining and hidden size 64.** The transductive graphs
   YAGO310, CoDExLarge, Hetionet, ConceptNet100k, DBpedia100k, AristoV4
   are available in the roots; NELL995 needed batch 16x2, larger graphs
   need 8x4. Compare against ULTRA-4g, never against 3g rows.
5. **Ensemble distillation into one trunk.** TRIX, FLOCK and INCITE
   disagree at Hits@10 on 6 to 8 percent of queries. Distil the score
   ensemble (E6 gives its realistic gain) into a single INCITE model on
   the pretraining graphs.
6. **After the seeds, the two directions the week's negatives point
   at** (neither is queued; both need a training run):
   * *A scenario-conditioned trunk with a training signal in which the
     indicator matters.* The combination probe showed the final node
     states carry the answer-half indicator with AUC 1.0 and that no head
     on top of frozen trunks uses it (SC1 failed for lack of contrast in
     the training graphs, not for lack of signal). Half-link masking or
     the generator's unseen-answer share would supply the contrast; the
     scenario features already exist (`scenario_features`).
   * *Propagation plus in-context relation evidence, design B of
     COMBINATION_PROBE.md*: prompt-graph rule-body evidence fed into the
     relation step. Its premise test (a CPU half-day) is specified there
     and not yet run. Do not rebuild a same-encoder support readout
     (phase 2.2) or a per-relation calibration from pseudo-queries
     (design A, negative).
   * *The prior as the pretraining stage with a scaling curve* (the
     review's direction 2): P1 at 30k and 60k with decay, dim 64, larger
     instances; the positioning against GraphPFN and RDB-PFN.
   * Closed for good, do not reopen without new evidence: sparse
     propagation through the gate (three experiments, GATE_RESULT.md),
     the rule head, the generator-side bundle's isolation.
7. **FLOCK's relation baseline** (`flock_relation.pth`) for the joint-head
   story: FLOCK reports 0.881 on 54 graphs; our joint model has 0.8484
   on ind_er. Its entity eval took 23 hours here; budget accordingly.
8. **Rule-mining hybrid.** Mine length-2/3 path rules on the inference
   graph and fuse confidences with the model score. Strong on WordNet
   and sparse graphs where 12 to 23 percent of answers are unreachable.

Do not repeat: support readouts, unsupervised walks, KGPFN bolt-ons,
uncapped half-link masking, DEV10 checkpoint selection, the proof-guided
gate, the rule head, score-level combinations of L1 with a mix model.

## Bootstrapping on a new machine

1. Clone the repo twice (or use worktrees): branch
   `claude/gpu-multi-model-baseline` as `sotaKGFMs`, branch `incite` as
   `sotaKGFMs-incite` beside it. Symlink `sotaKGFMs-incite/output` to
   `sotaKGFMs/output` and `sotaKGFMs-incite/data/roots` to
   `sotaKGFMs/data/roots` (the scripts assume the two directories are
   siblings).
2. `scripts/clone_repos.sh` (pinned SHAs in `repos/PINS.json`), then
   `scripts/mirror_data.py`, `scripts/prefetch_raw.py`,
   `scripts/fetch_pretrain_graphs.py`. NELL995's raw files moved
   upstream; `results/incite/config_diff.md` (2026-08-30) says where they
   are pre-seeded from. Some dataset URLs 404: pre-seed raw files.
3. Build the containers: `docker build -f containers/<model>/Dockerfile -t
   kgfm/<model>:<first 8 chars of the pin> .` for each model you need
   (`scripts/docker_run.sh` prints the expected tag). The incite image is
   `kgfm/incite:incite01` (its Dockerfile compiles TRIX's rspmm kernel
   for sm_89; set `TORCH_CUDA_ARCH_LIST` for another GPU).
4. Tests: `scripts/test_incite.sh` in the incite worktree (85 to 87 pass
   depending on GPU kernel gates).
5. Eval a checkpoint: `INCITE_CKPT=/kgfm-src/checkpoints/incite-4g-last.pth
   INCITE_CONFIG=/kgfm-src/configs/incite_phase1.yaml
   INCITE_RANKS=/kgfm-src/ranks/incite-4g-m2 INCITE_SUPPORT=skip
   scripts/docker_run.sh incite bash -c '/kgfm-src/scripts/run_incite.sh
   ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"'`
   (13 minutes on a 4070 Ti SUPER). Then `python3 shared/analyse.py` or
   `scripts/paired_bootstrap.py`.
6. Train: `scripts/train_incite.sh` through `docker_run.sh` with
   `INCITE_CONFIG`, `INCITE_SEED`, `INCITE_TRAIN_GRAPHS`, `INCITE_RESUME` or
   `INCITE_INIT_FROM`, `INCITE_TRAIN_STEPS`, `INCITE_TRAIN_EXTRA_ARGS`
   (lr schedule, masking, keep_every). See `scripts/research_plan_v18.sh`
   for every exact invocation. 16 GB fits batch 32x1 on the 3-graph mix
   and 16x2 on the 4-graph mix with activation checkpointing.
7. Every rank dump goes into its own `ranks/<name>/` with a
   PROVENANCE.json; never mix devices in one dir; commit dumps.

## Traps (each cost real time — do not relearn)

* **A from-scratch run with `--schedule_start` ran at a NEGATIVE lr before
  the schedule start** (the warmup formula with a negative step offset).
  Fixed on 2026-09-03: steps before the start run at the base lr
  (`incite/pretrain.py::make_lr_schedule`, `test_lr_schedule.py`). The
  recipe runs R0/R1 and their seed repeats rely on it.
* **A container stopped mid-start leaves torch's extension build lock
  behind** (`output/.torch_extensions-incite/rspmm/lock`), and the next
  container waits on it forever at "Load rspmm extension". One hour lost
  on 2026-09-03. `scripts/in_container.sh` now deletes a lock older than
  ten minutes at every container start; by hand: remove the lock file,
  the waiting process continues on its own.
* **`$ALLOC scripts/docker_run.sh ...` runs the variable's value as a
  command** (exit 127, "command not found"): bash decides what is an
  assignment before expansion. Always `env $ALLOC scripts/docker_run.sh`.
  Plan v14's dev-suite call had it; the second verifier caught it before
  a single stage ran.
* **A warm-start continuation without `--schedule_start` re-anchors its
  decay on a crash-resume** (a second warmup and a fresh decay from the
  resumed step). `UDECAY` now carries `--schedule_start 1`. PG2 ran under
  the old flags: if its train.log shows more than one attempt, record it.
* **A boundary detector that counts every marker fires on the snapshot
  watcher's markers.** v10 waited for "a new marker" while MXG1's evals
  ran; `snapshot-MX1.done` arrived, v10 stopped the eval container and
  started MXG1 from scratch. Killed after one minute, the finished run
  was intact, v11 ignores `snapshot-*` markers and guards every stage.
* **A takeover between a finished training and its marker used to
  retrain the stage** (the run directory was already moved, so the stage
  seeded a fresh one). Plan v10's `trained <suffix>` guard skips the
  training when the moved directory carries "checkpoint reload OK".

* **A warm start counts its steps from 1, a resumed continuation from
  20001, and the synthetic coin and the instance seed are functions of
  the step number.** PG2 (warm start) and MX1 (resumed) saw different
  data orders: 152 of 200 logged positions show a different graph, so a
  warm-started lever's difference to MX1 carried data-order noise. Since
  2026-09-03 the knob `synth.step_offset` (20000 in SC1's and RR2's
  configs) gives a warm start MX1's exact stream
  (`incite/tests/test_step_offset.py`). The resumed levers (MX15, MX45,
  MXS9, MX2a, MX2H, FMX) never had the problem.

* **A uniform sample of a transductive valid split is 80 to 90 percent
  seen-answer queries.** It ranked MX1 0.0084 below L1 (D0) while the
  benchmark ranks it above, because the prior trades seen-answer MRR for
  unseen-answer MRR and the benchmark is 61 / 66 percent seen-answer.
  The dev number is stratified by scenario and benchmark-weighted since
  then (`protocol: stratified_v2`); never compare a uniform file with a
  stratified one (plan v16's decision code refuses).

* Never edit a bash script while an instance runs, not even a comment
  line: bash reads by byte offset. Write a new file and relaunch.
* Queue scripts must be idempotent. Never clear a run directory before
  the completion check (v3 wiped a finished 4.8-hour run on relaunch).
* A documented env knob is not a read env knob. Grep the consumer script.
  Config files can lie about a run: PROVENANCE.json and train.log are the
  truth.
* docker_run.sh forwards a FIXED env list; unforwarded vars are silently
  unset in containers. Runners execute the prepared work tree, never the
  baked image copy. Two concurrent runners must not prepare the same
  work tree (`INCITE_WORKDIR`).
* A stopped container leaves a claim without a parquet; run_incite.sh now
  reclaims those at start. FLOCK's loop exits 0 on per-graph failures;
  TIMINGS status lines are the truth.
* `pkill -f` with a pattern that appears in your own command line kills
  your own shell. Anchor patterns (`^bash scripts/...`).
* Training holds about 13 GB; never start a second GPU job beside it
  beyond a 1 to 3 GB eval. The container has no pandas: use pyarrow to
  numpy.
* TRIX's `output_dir` in its yaml decides where checkpoints go; an
  unmounted path loses them with the container.
* Upstream drifts: dataset URLs 404, Wikidata labels change. Pre-seed raw
  files; pinned repos stay pristine via patches/ only.

## Conventions

Ranks: 1-based, pessimistic ties, strict filtering (relation task:
unfiltered by design; state it on tables). Unweighted means over graphs.
Seed 1024 for the first run of everything; `INCITE_SEED` for repeats.
Report LAST checkpoints; DEV10 (valid splits of ten benchmark graphs) is
diagnostic only. Every deviation goes in `results/incite/config_diff.md`
the day it happens. Stop rules are honored: dead levers get a result
file, not tuning. Re-ranked and ensembled rows state k, weight, member
count and eval cost. Compare 4-graph rows against ULTRA-4g, 3-graph rows
against the 3-graph baselines, never across.
