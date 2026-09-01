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
| MOTIF | 0.4361 / 0.3491 | done |
| TRIX | 0.4562 / 0.3679 | done; relation task (0.7564/0.8415 UNFILTERED); transductive 54/54 |
| SEMMA | 0.4496 / 0.3520 | done |
| KG-ICL | 0.4240 / 0.3722 | done (matched convention) |
| FLOCK | 0.4558 / 0.3674 (22 of 23 ind_er) | 40/41; FBIngram:25 retry is plan stage F0 |
| KGPFN | 11/41 | background pass running; small failed graphs retried by `scripts/kgpfn_small_retry.sh` |

Published targets live in `shared/published.json`. Group means regenerate
via `scripts/make_summary.py`.

## The plan now running: `scripts/research_plan.sh` (markers in output/research-plan/)

Order, by expected gain per GPU hour. Every stage writes its own ranks dir
on the incite worktree; nothing overwrites an earlier dump.

| stage | what | output |
| --- | --- | --- |
| E1 | 4-graph LAST checkpoint (selection-protocol check) | ranks/incite-4g-last |
| E2 | weight soup of the floor family | ranks/incite-soup |
| E3 | DEV10 (valid splits) sweep of bidirectional re-ranking, k in 4/8/16, weight 0.5/1.0 | results/incite/rerank_dev.json |
| E4/E5 | 41-graph re-ranking evals of the 4g and floor bests, IF E3 lifts the selection scalar by >= 0.002 | ranks/incite-4g-rerank, ranks/incite-rerank |
| E6 | score ensemble of four trunks (floor, 4g, joint, support) | ranks/incite-ens4 |
| F0 | FLOCK FBIngram:25 (patch 0005) | ranks/flock 41/41 |
| L1/L2 | continue the 4g and floor runs 20k -> 30k with linear lr decay; eval best AND last | ranks/incite-4g-decay(-last), ranks/incite-decay(-last) |
| X1/X2 | TRIX@20k with their code (fixed output dir); eval best epoch and last | ranks/trix-20k-best, ranks/trix-20k-last |
| P1 | synthetic-prior 100 percent pilot, 10k steps | ranks/incite-synth100-pilot |

Rough wall clock: E-stages 4 to 6 h, F0 1 to 2 h, L1+L2 10 h, X1+X2 12 h,
P1 5 h. The KGPFN eval may share the GPU (1.7 GB); nothing else may.

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
