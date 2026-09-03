# Takeover — 2026-09-02, 10:50 (day 6)

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
| INCITE floor (3g, 20k steps) | 0.4553 | 0.3740 | DEV10-best checkpoint |
| INCITE floor-family soup | 0.4571 | 0.3775 | average of four floor descendants; the best matched-diet (3g) row |
| INCITE floor + 10k decay (L2), last | 0.4510 | 0.3745 | no gain at 3 graphs (results/incite/DECAY_RESULT.md) |
| INCITE 4g (20k) DEV10-best | 0.4542 | 0.3791 | |
| INCITE 4g (20k) last | 0.4534 | 0.3825 | beats ULTRA-4g by +0.008 / +0.036; TRIX by −0.003 / +0.015 (ind_er interval excludes zero) |
| **INCITE 4g + 10k decay (L1), last** | **0.4560** | **0.3852** | THE REFERENCE (2026-09-02, `checkpoints/incite-4g-decay-last-step30k.pth`): ties TRIX on ind_e, +0.017 on ind_er [+0.005, +0.036], 18 of 23 graphs; see results/incite/DECAY_RESULT.md |
| **INCITE 4g + unary channel (G1), last** | **0.4571** | **0.3874** | best single model (2026-09-02, `checkpoints/incite-4g-unary-last-step10k.pth`): +0.001 / +0.002 over L1, gains in the unseen-answer cells; results/incite/UNARY_RESULT.md |
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
5. **Weight soups help a little** (E2: +0.002 / +0.0035 over the floor).
   `scripts/snapshot_watcher.sh` averages the last five kept snapshots of
   each decay run automatically.
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
   it. This is the most promising open thread for the paper.
10. **Seed spread is the missing number.** Same-architecture runs swing by
   up to 0.10 MRR on single graphs; the only statistics we have are
   graph-level bootstraps. Three seeds of the winner are in the plan.

## Running on this machine: `scripts/research_plan_v7.sh` (markers in output/research-plan/)

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
| MX1 | 4g continuation with 30 percent synthetic rules-prior steps, paired against L1 (`configs/incite_phase1_4g_synth30.yaml`), training now | ranks/incite-4g-synth30(-last) | 11:00, 3 Sep |
| X1/X2 | TRIX@20k with their code, best epoch and last | ranks/trix-20k-best/-last | 19:00, 3 Sep |
| E4/E5/E6 | re-ranked dumps (k=8), score ensemble of four trunks | ranks/incite-*-rerank, incite-ens4 | 23:00, 3 Sep |
| F0 | FLOCK FBIngram:25 (patch 0005) | ranks/flock 41/41 | 01:00, 4 Sep |
| B2/B3, C2/C3 | backbone seeds 1337 and 7 (20k), then the winner continuation (auto-picked among L1/M1/G1/M2) on each | gated by SEEDS_GO | on release, about 26 h |

Also running: the KGPFN suite in the background (1.7 GB, days), and
`scripts/kgpfn_small_retry.sh` for its small failed graphs. Do NOT start
another training run on this GPU; evals of 1 to 3 GB are fine.

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
6. **From-scratch masking with the cap, if M2 is positive.** A 10k
   continuation may be too short to change what the model relies on.
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
   (lr schedule, masking, keep_every). See `scripts/research_plan_v5.sh`
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
