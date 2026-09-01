# Takeover — 2026-09-01, evening (day 5, second session)

One GPU (RTX 4070 Ti SUPER, 16 GB), one shared rank definition, seven
published KGFMs plus our own model. Two worktrees:

* `~/Dokumente/GitHub/sotaKGFMs` — branch `claude/gpu-multi-model-baseline`:
  the benchmark harness, all baselines, and the queue scripts.
* `~/Dokumente/GitHub/sotaKGFMs-incite` — branch `incite`: our model
  (INCITE) and every experiment on it. (`crest` branch: a falsified
  predecessor, kept for the record.)

After any pause or crash: `./RESTART.sh` in the main worktree relaunches
the two live queues; both are marker-based and resume where they stopped.
Nothing is pushed anywhere; all work is local commits.

## Read this first: the validation pass of 2026-09-01

`results/incite/VALIDATION_2026-09-01.md` (incite branch) re-derived every
number from the rank files and read the code. Summary:

* The code is a faithful TRIX re-implementation with an exact O(|E|)
  relation step; the eval protocol matches the baselines. Numbers in the
  phase result files reproduce.
* INCITE ties TRIX and FLOCK. It does not beat them: floor 0.4553 / 0.3740,
  4-graph 0.4542 / 0.3791, TRIX 0.4562 / 0.3679, FLOCK 0.4558 / 0.3674.
  Per-graph swings between same-architecture runs reach 0.10 MRR, so
  single-seed margins of 0.006 mean nothing.
* The v1 composite (0.4500 / 0.3659) is BELOW TRIX on entity prediction.
  Its "seed 1337" repeat trained at seed 1024 (the seed knob was never
  read). Killed, ledgered, fixed. Seed repeats now wait for the winner of
  the plan below.
* KGPFN sits at TRIX level on the 11 graphs measured; the suite needs days
  more. Background only, nothing gated on it.
* Research chain 2 would have failed (TRIX stage wrote checkpoints to an
  unmounted path). Rewritten inside the plan.

## Baseline state (41 inductive graphs, test splits, seed 1024)

| model | entity MRR ind_e / ind_er | status |
| --- | --- | --- |
| ULTRA | 0.4158 / 0.3421 | done |
| ULTRA 4g (FB15k237, WN18RR, CoDExMedium, NELL995; the INCITE-4g diet) | 0.4454 / 0.3460 | done 2026-09-01, ranks/ultra-4g; INCITE-4g-last beats it by +0.008 / +0.036 (graph-bootstrap intervals exclude zero, 23/23 ind_er graphs) |
| INCITE floor-family weight soup | 0.4571 / 0.3775 | done (E2); +0.002 / +0.0035 over the floor |
| MOTIF | 0.4361 / 0.3491 | done |
| TRIX | 0.4562 / 0.3679 | done; relation task (0.7564/0.8415 UNFILTERED); transductive 54/54 |
| SEMMA | 0.4496 / 0.3520 | done |
| KG-ICL | 0.4240 / 0.3722 | done (matched convention) |
| FLOCK | 0.4558 / 0.3674 (22 of 23 ind_er) | 40/41; FBIngram:25 retry is plan stage F0 |
| KGPFN | 11/41 | background pass running; small failed graphs retried by `scripts/kgpfn_small_retry.sh` |

Published targets live in `shared/published.json`. Group means regenerate
via `scripts/make_summary.py`.

## Where the MRR is lost (diagnostics of 2026-09-01, incite branch)

`results/incite/halflink.json` (scenarios of Gregucci et al., arXiv
2606.18001) and `results/incite/reachability.json`:

* 28 percent of test queries in both groups have an answer WITHOUT any
  edge of the query relation (SQUA). INCITE-4g scores 0.29 (ind_e) and
  0.17 (ind_er) MRR there, against 0.61 / 0.58 when the answer half is
  seen. Pretraining positives almost always carry a seen answer half, so
  the model learns that shortcut.
* 17 percent of ind_e answers (6 percent ind_er) lie beyond six hops of
  the query entity and score MRR below 0.01. HM:1k/3k/5k are 73 to 83
  percent unreachable, which is why every model sits near 0.06 there.
* E1 showed the unselected LAST checkpoint of the 4-graph run is at least
  as good as the DEV10-selected one (0.4534 / 0.3825). Report last
  checkpoints; DEV10 is diagnostic only.

Two remedies are implemented and tested (86 tests): half-link masking
during pretraining (`--mask_answer_p/--mask_query_p`) and the unary
channel (`model.unary`). KMAS (arXiv 2605.27023) already published hard
negatives for KGFMs (+0.005 to +0.009), so that lever is a citation, not
a contribution.

## The plan now running: `scripts/research_plan_v2.sh` (markers in output/research-plan/)

v2 took over from v1 after the cheap E-stages (same markers). Every
stage writes its own ranks dir on the incite worktree.

| stage | what | output |
| --- | --- | --- |
| E1 | 4-graph LAST checkpoint (done: 0.4534 / 0.3825) | ranks/incite-4g-last |
| E2 | weight soup of the floor family | ranks/incite-soup |
| E3 | DEV10 (valid splits) sweep of bidirectional re-ranking | results/incite/rerank_dev.json |
| E4/E5 | 41-graph re-ranking evals, only if E3 passes its stop rule | ranks/incite-4g-rerank, ranks/incite-rerank |
| E6 | score ensemble of four trunks | ranks/incite-ens4 |
| L1 | 4g continuation 20k -> 30k, linear decay (the paired baseline) | ranks/incite-4g-decay(-last) |
| M1 | same continuation WITH half-link masking 0.3 / 0.3 | ranks/incite-4g-mask(-last) |
| G1 | unary channel, warm start from 4g last, 10k steps with decay | ranks/incite-4g-unary(-last) |
| F0 | FLOCK FBIngram:25 (patch 0005) | ranks/flock 41/41 |
| L2 | floor continuation 20k -> 30k with decay | ranks/incite-decay(-last) |
| X1/X2 | TRIX@20k with their code (fixed output dir); eval best epoch and last | ranks/trix-20k-best, ranks/trix-20k-last |
| P1 | synthetic-prior 100 percent pilot, 10k steps | ranks/incite-synth100-pilot |

After M1 and G1 land: `python3 scripts/halflink_report.py --labels
../sotaKGFMs-incite/results/incite/halflink_labels.json name=dir ...`
gives the per-scenario table, and `scripts/paired_bootstrap.py` the
graph-level intervals. The ULTRA 4-graph checkpoint (INCITE-4g's exact
diet) is being evaluated into ranks/ultra-4g for the scaled table.
The KGPFN eval may share the GPU (1.7 GB); nothing else may.

## Decisions after the plan

* **Headline candidate**: the best of {4g-decay-last, decay-last, their
  re-ranked or ensembled variants} on the 41-graph test, with the
  matched-diet (3-graph) row always beside the 4-graph row.
* **Seed repeats** (real seeds now: `INCITE_SEED` is read) go to that
  candidate only. Three seeds before any "best" claim.
* **Re-ranking**: dead if E3 fails its stop rule; then record and drop.
* **Synthetic prior**: the full 25/75/100 sweep only if P1's DEV10 curve
  reaches a meaningful fraction of the real-data curve at 10k steps.
* **TRIX A/B**: if TRIX@20k is well below TRIX@100k, INCITE's parity at
  20k is a sample-efficiency result and INCITE at TRIX's budget is the
  next run.

## INCITE levers and verdicts (unchanged from the first session)

1. Floor: TRIX's layer algebra recomposed, 20k steps. Ties TRIX.
2. Support lever: dead twice (CREST, INCITE). Closed.
3. Walks: dead unsupervised; synthetic automorphic supervision revives the
   PETALS capability (94.6 / 98.6 percent in v1) at no benchmark gain.
4. Joint relation head: one checkpoint, both tasks; entity −0.004,
   relation ind_er 0.8484 beats the TRIX specialist (0.8415), ind_e 0.7286
   below it (0.7564). FLOCK's relation baseline is not measured here.
5. 4-graph mix: +0.005 ind_er, diet-caveated.

## Traps (each cost real time — do not relearn)

* A documented env knob is not a read env knob. Grep the consumer script
  for the variable, not only docker_run.sh's forwarding list.
* Config files can lie about a run: PROVENANCE.json and the train.log are
  the truth (the 4-graph config lists three graphs).
* TRIX's `output_dir` in its yaml decides where checkpoints go; an
  unmounted path loses them with the container.
* docker_run.sh forwards a FIXED env list; unforwarded vars are silently
  unset in containers.
* Runners must execute the prepared work tree, never the baked image copy.
* Never edit a bash script while an instance runs. Never write into a data
  root a live runner is reading.
* `pkill -f` with a pattern that appears in your own command line kills
  your own shell. Anchor patterns (`^bash scripts/...`).
* Exit codes lie (FLOCK loop exits 0 on per-graph failures); TIMINGS
  status lines are the truth.
* Upstream drifts: dataset URLs 404, Wikidata labels change. Pre-seed raw
  files; pinned repos stay pristine via patches/ only.

## Conventions

Ranks: 1-based, pessimistic ties, strict filtering (relation task:
unfiltered by design — state it on tables). One processed root and one
claims dir per model; never mix devices in a rank dir. Seed 1024 for the
first run of everything. Checkpoint selection: DEV10 valid splits, one
mean per group, reported beside the LAST checkpoint from now on. Every
deviation goes in the ledger the day it happens. Stop rules are honored:
dead levers get a result file, not tuning. Re-ranked and ensembled rows
state k, weight, member count and eval cost.
