# Takeover — 2026-09-03, 09:40 (day 7)

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

## Live state right now (2026-09-03, 09:40) and how the session operates

* **Best single model:** the 4-graph backbone continued with 30 percent
  synthetic rules-prior steps (MX1): 0.4606 / 0.3851, ties or beats
  TRIX on ind_e (+0.004) and leads on ind_er by +0.017 with a graph
  bootstrap interval of +0.007 to +0.032. Checkpoint:
  `checkpoints/incite-4g-synth30-last-step30k.pth`. Single seed.
* **Best matched-diet (3-graph) model:** the floor-family soup,
  0.4571 / 0.3775. No 3-graph lever beat it (decay adds nothing there).
* **Training now:** MXG1 = synthetic mix 30 percent PLUS the unary
  channel, warm start from the 4-graph last checkpoint, 10k steps with
  decay (started 09:14, evals done about 14:00). Then MX2 (12:30 update):
  MX1's recipe with the generator-side fixes of the rules prior, the
  synthetic steps now use what only the generator knows (64 certified
  negatives per row, half of them from the head's 1-2 hop neighborhood;
  every instance in its own relation block; up to 4 full-closure
  positives per row; `configs/incite_phase1_4g_synth30_v2.yaml`,
  `results/incite/SYNTH_V2_DESIGN.md`). Plan v9 takes over from v8 at the
  MXG1 marker and runs MX2 before MX15 (mix at 15 percent), then the TRIX
  budget A/B, the test-time levers, FLOCK's last graph. Seed repeats stay
  gated behind `output/research-plan/SEEDS_GO` until the paper model is
  chosen.
* **The decision ahead:** the paper recipe is the best of MX1, MXG1 and
  MX2 by the paired bootstrap against L1 (MX2 lands about 20:00 on 3
  Sep; if it beats MX1, its synthetic steps replace MX1's in every later
  run, unary included). Then a from-scratch 20k+10k run with the mix as
  THE recipe (not a continuation), then two more seeds of it. Release
  the seed stages only for that recipe (rewrite B2/B3/C2/C3 first: they
  still assume the continuation form).
* **Per-stage ritual** (what the session does when a stage lands, in
  this order): `scripts/paired_bootstrap.py <new-last> <reference>` and
  versus `ranks/trix`; `scripts/halflink_report.py --labels
  ../sotaKGFMs-incite/results/incite/halflink_labels.json ...` for the
  per-scenario table; a result note under `results/incite/`; the
  checkpoint into `checkpoints/` with a README row; this table; commit
  and push both branches (`git push origin incite` from the incite
  worktree; the remote push URL is SSH). The reference for every
  continuation lever is L1 (`ranks/incite-4g-decay-last`), never the 20k
  start.
* **Watchers alive on this machine:** `scripts/research_plan_v8.sh` (the
  queue, running MXG1) and `scripts/research_plan_v9.sh` (waiting for the
  MXG1 marker, then it stops v8 and v8's freshly started MX15 container
  and continues the same marker-based list with MX2 inserted; a Monitor
  on `output/research-plan/log.txt` wakes the session on DONE/FAILED
  lines), `scripts/snapshot_watcher.sh` (last-5 snapshot soups; v2 with a
  double memory check is armed to replace it after the current soup), the
  KGPFN suite (29 of 41 graphs; the small-graph retry finished).
* **Memory notes for the session** live in
  `~/.claude/projects/-home-lukef-Dokumente-GitHub-sotaKGFMs/memory/`
  (validate before building; never edit running scripts; idempotent
  queues; objective fixes get built and queued without a question,
  hypotheses get a paired experiment).

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
| KGPFN | 13 of 41 | | at TRIX level on the graphs done |
| INCITE floor (3g, 20k steps) | 0.4553 | 0.3740 | DEV10-best checkpoint (17k); the last checkpoint (20k) is 0.4533 / 0.3749 |
| INCITE floor-family soup | 0.4571 | 0.3775 | average of four floor descendants; the best matched-diet (3g) row |
| INCITE floor + 10k decay (L2), last | 0.4510 | 0.3745 | no gain at 3 graphs (results/incite/DECAY_RESULT.md) |
| INCITE 4g (20k) DEV10-best | 0.4542 | 0.3791 | |
| INCITE 4g (20k) last | 0.4534 | 0.3825 | beats ULTRA-4g by +0.008 / +0.036; TRIX by −0.003 / +0.015 (ind_er interval excludes zero) |
| **INCITE 4g + 10k decay (L1), last** | **0.4560** | **0.3852** | THE REFERENCE (2026-09-02, `checkpoints/incite-4g-decay-last-step30k.pth`): ties TRIX on ind_e, +0.017 on ind_er [+0.005, +0.036], 18 of 23 graphs; see results/incite/DECAY_RESULT.md |
| INCITE 4g + unary channel (G1), last | 0.4571 | 0.3874 | +0.001 / +0.002 over L1, gains in the unseen-answer cells; results/incite/UNARY_RESULT.md |
| **INCITE 4g + 30 percent synthetic mix (MX1), last** | **0.4606** | **0.3851** | best single model (2026-09-03): ind_e +0.0046 over L1 with interval [+0.0002, +0.0092]; both-unseen cell +0.065 on both groups; results/incite/SYNTH_MIX_RESULT.md |
| INCITE 4g + masked continuation, dose 1 (M1) | 0.4420 | 0.3604 | NEGATIVE, see below |
| INCITE 4g + masked continuation, dose 2 (M2, cap 10) | 0.4384 | 0.3629 | NEGATIVE; after 1,000 masked steps the unseen-answer cells rose +0.03 at a −0.035 SQSA cost, then inverted |
| INCITE v1 composite (walks+synth+joint) | 0.4500 | 0.3659 | below TRIX on entity; relation ind_er 0.8484 beats TRIX's specialist (0.8415) |
| **INCITE trained on the synthetic rules prior ONLY** (P1, 10k steps, 41 min) | **0.3593** | **0.2795** | no real KG seen; 86 / 82 percent of ULTRA-3g; strong on unseen-answer cells, weak on seen-answer ones; results/incite/SYNTH_PILOT_RESULT.md |

Relation task (unfiltered protocol): TRIX 0.7564 / 0.8415, INCITE joint
0.7286 / 0.8222, v1 composite ind_er 0.8484. PETALS: v1 94.6 / 98.6
percent; deterministic models 50 percent.

Tools: `scripts/paired_bootstrap.py A B` (graph-level interval of a
margin), `scripts/halflink_report.py --labels
../sotaKGFMs-incite/results/incite/halflink_labels.json name=dir ...`
(per-scenario table), `scripts/make_summary.py`, `scripts/complementarity.py`.

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
   bolt-on (KGPFN itself sits at TRIX level here on 13 graphs).
8. **Hard negatives are published** (KMAS, arXiv 2605.27023, +0.005 to
   +0.009 for every KGFM): a citation, not a contribution.
9. **The synthetic rules prior transfers** (P1): a model trained on no
   real KG reaches 0.3593 / 0.2795 on the 41 real graphs, at one seventh
   of the per-step cost, and its scenario profile complements real data
   (matches or beats ULTRA-3g on the unseen-answer cells, trails by 0.08
   to 0.13 on seen-answer cells). The mixing run MX1 tests whether that
   complement adds to the reference; the full 25/75/100 sweep waits for
   it. MX1 (the 4-graph continuation with 30 percent synthetic steps)
   confirmed it: ind_e +0.0046 [+0.0002, +0.0092] over the reference,
   both-unseen cell +0.065 on both groups, seen-answer cells −0.012 to
   −0.017, ind_er net flat. THE mechanism candidate for the paper:
   realistic unseen-answer supervision from the prior. Next: mix + unary
   (MXG1), the generator-side fixes (MX2: MX1's synthetic steps scored
   one positive against ONE negative while real steps score 512; now 64
   certified negatives, half of them structurally close, per-instance
   relation blocks, full-closure positives), the 15 percent dose (MX15),
   then a from-scratch run with the mix as the recipe and its seeds.
   Measured on the prior (`results/incite/SYNTH_V2_DESIGN.md`): the
   natural query draw is already 47 percent unseen-answer, above the
   benchmark's 37 percent, so scenario targeting is a knob
   (`unseen_answer_share`), not a fix.
10. **Seed spread is the missing number.** Same-architecture runs swing by
   up to 0.10 MRR on single graphs; the only statistics we have are
   graph-level bootstraps. Three seeds of the winner are in the plan.

## Running on this machine: `scripts/research_plan_v9.sh` (markers in output/research-plan/)

Stages are idempotent (v4 fixed a relaunch that wiped a finished run).
`./RESTART.sh` relaunches everything after a pause. Seed repeats are
DEFERRED (user decision, 2026-09-02): they run only after someone
creates `output/research-plan/SEEDS_GO`, once the paper model is known.
Order and ETAs (continuations: 10k steps from the 4g last checkpoint,
linear lr decay, warmup 500, kept snapshots every 1000):

| stage | what | output (incite worktree) | ETA |
| --- | --- | --- | --- |
| L1 | decay only, the paired baseline: DONE, 0.4560 / 0.3852 | ranks/incite-4g-decay(-last) | done |
| G1 | unary channel: DONE, 0.4571 / 0.3874 | ranks/incite-4g-unary(-last) | done |
| M2 | masking dose 2: DONE, negative (0.4384 / 0.3629) | ranks/incite-4g-mask2(-last) | done |
| P1 | synthetic rules-prior 100 percent pilot: DONE, 0.3593 / 0.2795 | ranks/incite-synth100-pilot | done |
| L2 | floor (3-graph) decay: DONE, no gain (0.4510 / 0.3745) | ranks/incite-decay(-last) | done |
| MX1 | 4g continuation with 30 percent synthetic steps: DONE, 0.4606 / 0.3851 | ranks/incite-4g-synth30(-last) | done |
| MXG1 | synthetic mix 30 percent + unary channel, warm start, 10k decay (`configs/incite_phase1_4g_unary_synth30.yaml`); TRAINING NOW since 09:14 (v8) | ranks/incite-4g-unary-synth30(-last) | 14:00, 3 Sep |
| MX2 | MX1 plus the generator-side fixes of the rules prior (`configs/incite_phase1_4g_synth30_v2.yaml`); first stage of v9, paired against MX1 and L1 | ranks/incite-4g-synth30v2-last | 20:00, 3 Sep |
| MX15 | synthetic mix at 15 percent (`configs/incite_phase1_4g_synth15.yaml`) | ranks/incite-4g-synth15-last | 02:00, 4 Sep |
| X1/X2 | TRIX@20k with their code, best epoch and last | ranks/trix-20k-best/-last | 10:00, 4 Sep |
| E4/E5/E6 | re-ranked dumps (k=8), score ensemble of four trunks | ranks/incite-*-rerank, incite-ens4 | 14:00, 4 Sep |
| F0 | FLOCK FBIngram:25 (patch 0005) | ranks/flock 41/41 | 16:00, 4 Sep |
| B2/B3, C2/C3 | backbone seeds 1337 and 7 (20k), then the winner continuation (auto-picked among L1/M1/G1/M2/MX1/MXG1/MX15/MX2 by mean group MRR) on each | gated by SEEDS_GO | on release, about 26 h; rewrite first if the paper recipe is a from-scratch run |

Also running: the KGPFN suite in the background (1.7 GB; 29 of 41; the
small-graph retry finished). Do NOT start another training run on this
GPU. Evals of 1 to 3 GB are fine, but only after two low memory readings
three minutes apart: a training container ramps to 13 GB within minutes
of its start and an eval that starts in the ramp OOMs (it happened).

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
6. **Use more of what the generator knows** (the MX2 fixes are the first
   installment; these are the hypotheses that still need a paired run):
   * *Scenario mix.* The knob exists (`synth.unseen_answer_share`; the
     natural draw is 47 percent unseen-answer, the benchmark 37). Test
     whether matching the benchmark share shrinks the seen-answer cost of
     the mix faster than it shrinks the unseen-answer gain.
   * *Rule-recovery auxiliary loss.* Every instance carries its latent
     rules (`inst.rules`). A bilinear head on the relation states that
     predicts hierarchy, inversion, symmetry and composition pairs with
     certain labels trains the relation encoder to read the relational
     algebra of an unseen vocabulary. No KGFM does this. Cost: a day plus
     one 5 GPU-hour continuation paired against MX2.
   * *Proof-guided propagation.* The chainer can record the premises of
     every derived query, so the observed edges that support the proof
     are known per query (a small change to `forward_chain`, which does
     not record derivations today). Step 1: a scalar per-edge gate in the
     entity step, sigmoid of a linear map of source state, relation state
     and query, with a one-sided auxiliary loss on synthetic steps that
     pushes proof edges to gate 1 (non-proof edges unpenalized: type
     context flows through them). Step 2, measurement only: at eval,
     zero the messages of the lowest-gate x percent of edges per query
     and plot MRR against x. If the curve is flat to 60 to 80 percent,
     step 3 is A*Net-style sparse propagation (per-query edge subsets,
     custom kernels, a week) for larger batches or more rounds on 16 GB.
     Do not start step 3 before the curve exists.
   * *Synthetic dev set.* A fixed held-out instance pool at another seed
     replaces DEV10 for every selection or calibration choice, so no
     benchmark graph is touched at all. DEV10 selection buys nothing, so
     this costs nothing.
7. **FLOCK's relation baseline** (`flock_relation.pth`) for the joint-head
   story: FLOCK reports 0.881 on 54 graphs; our joint model has 0.8484
   on ind_er. Its entity eval took 23 hours here; budget accordingly.
8. **Rule-mining hybrid.** Mine length-2/3 path rules on the inference
   graph and fuse confidences with the model score. Strong on WordNet
   and sparse graphs where 12 to 23 percent of answers are unreachable.

Do not repeat: support readouts, unsupervised walks, KGPFN bolt-ons,
uncapped half-link masking, DEV10 checkpoint selection.

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
   (lr schedule, masking, keep_every). See `scripts/research_plan_v9.sh`
   for every exact invocation. 16 GB fits batch 32x1 on the 3-graph mix
   and 16x2 on the 4-graph mix with activation checkpointing.
7. Every rank dump goes into its own `ranks/<name>/` with a
   PROVENANCE.json; never mix devices in one dir; commit dumps.

## Traps (each cost real time — do not relearn)

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
