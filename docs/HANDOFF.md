# Takeover — 2026-09-01 (day 5)

One GPU (RTX 4070 Ti SUPER, 16 GB), one shared rank definition, seven
published KGFMs plus our own model. Two worktrees:

* `~/Dokumente/GitHub/sotaKGFMs` — branch `claude/gpu-multi-model-baseline`:
  the benchmark harness and all baselines.
* `~/Dokumente/GitHub/sotaKGFMs-incite` — branch `incite`: our model
  (INCITE) and every experiment on it. (`crest` branch: a falsified
  predecessor, kept for the record.)

After any pause or crash: `./RESTART.sh` in the main worktree relaunches
every queue; all queues are marker-based and resume where they stopped.
Nothing is pushed anywhere; all work is local commits.

## Baseline state (41 inductive graphs, test splits, seed 1024)

| model | entity MRR ind_e / ind_er | status |
| --- | --- | --- |
| ULTRA | 0.4158 / 0.3421 | done, criterion A/B recorded |
| MOTIF | 0.4361 / 0.3491 | done |
| TRIX | 0.4562 / 0.3679 | done; also relation task (0.7564/0.8415 UNFILTERED) and transductive 54/54 |
| SEMMA | 0.4496 / 0.3520 | done |
| KG-ICL | 0.4240 / 0.3722 | done (matched convention; its published edge was mostly preprocessing) |
| FLOCK | 40/41 graphs | HM:indigo done at divisor 8; FBIngram:25 retrying (int32 bug, patches/flock/0005) |
| KGPFN | 10/41 ok | suite running; 13 transient failures (rsync collision, retrying); Metafam 0.4727; first-ever cost numbers |

Published targets live in `shared/published.json` (data, never constants).
Group means regenerate via `scripts/make_summary.py`.

## INCITE: what we tried and what each verdict was

All detail on the `incite` branch under `results/incite/*.md` and
`config_diff.md` (the deviation ledger — read it before changing any
hyperparameter).

1. **Floor (phase 1)**: TRIX's layer algebra (proven equal at 1e-5),
   recomposed; trained 20k steps vs TRIX's ~100k. **Ties TRIX** (0.4553 /
   0.3740). PHASE1_RESULT.md. Honest framing: a re-architected TRIX.
2. **Support lever**: retrieval + hard negatives readout. **Dead** —
   identical to floor to 1e-4 with live stores; labels consumed but
   metric-invariant. Second dead in-context readout after CREST →
   replicated negative result. PHASE22_RESULT.md.
3. **Walks lever**: dead unsupervised (PETALS 47%, truth-uncorrelated),
   **revived by ~5% synthetic automorphic training** (82%/94%). The
   causal finding of the project: capability needs its own supervision.
   PHASE21_RESULT.md / PHASE21B_RESULT.md.
4. **Joint relation head**: one checkpoint, both tasks; entity −0.004,
   relation within 0.03 of the TRIX specialist. PHASE23_RESULT.md.
5. **4-graph mix** (+NELL995): biggest single gain, ind_er 0.3791
   (diet-caveated — baselines are 3-graph). SCALE4G_RESULT.md.
6. **v1 composite** (4g + joint + walks/synth, support dropped): entity
   0.4500/0.3659 (multi-objective tax vs 4g), **relation ind_er 0.8484
   BEATS the TRIX specialist**, PETALS 94.6%/98.6%. Seed 1337 training
   now; seed 7 queued.

## Running and queued right now (all self-driving)

* `scripts/baseline_orchestrator.sh` — currently R3 (composite seed
  1337), then R4 (seed 7). T1 transductive stages for ultra/motif/semma
  + TI failed against mid-prefetch data roots: clear their .failed
  markers and rerun the script after R4.
* `scripts/retry_watcher.sh` — reruns the KGPFN suite (13 transient
  failures) and FLOCK's FBIngram:25 when the GPU path frees.
* `scripts/queued_research.sh` — fires at ranks/kgpfn == 41:
  complementarity report (oracle gap, fusion, where-in-context-wins map)
  then checkpoint soup (floor-family average) + its eval.
* `scripts/research_chain2.sh` — after everything: **TRIX@20k A/B**
  (decides whether "parity at 20% budget" is real) then the
  **synthetic-prior fraction sweep** (25/75/100% rules-prior training —
  the TabPFN-for-KGs experiment; generator in incite/synth.py, design in
  RULES_PRIOR.md).

## Open decisions (agreed with the user)

* **MoE / mixture-of-checkpoints**: decide ONLY after the complementarity
  report — oracle gap small → dead; large and structured → one-day
  stats-router experiment.
* SOTA claims: currently "co-SOTA, tied with TRIX"; every margin is
  single-seed. Never claim "best" without the seed spread (R3/R4) and
  never compare 4g-diet rows against 3-graph baselines without saying so.
* KGPFN's published +0.044: the open question — two dead readouts here
  suggest its edge is the imported TabPFN prior; ablation idea recorded.

## Traps (each cost us real time — do not relearn)

* docker_run.sh forwards a FIXED env list; unforwarded vars are silently
  unset in containers. Grep the list every time you add a knob (5 hits).
* Runners must execute the prepared work tree, never the baked image
  copy (sys.path[0] beats PYTHONPATH; 4 hits).
* Never edit a bash script while an instance runs (incremental reads).
* Never write into a data root a live runner is reading (the KGPFN
  rsync collision).
* `--resume` must continue the step counter (fixed; first real resume
  trained 20k extra steps).
* Read paper AND code for every recipe number (lr, budget: both were
  caught by the user asking questions, not by review).
* Host crashes were hardware (GPU Xid 79, fixed by the user); every
  long run must remain checkpoint-resumable anyway.
* Exit codes lie (FLOCK loop exits 0 on per-graph failures); TIMINGS
  status lines are the truth.
* Upstream drifts: dataset URLs 404 (NELL995 moved, RED-GNN restructured),
  Wikidata labels change. Pre-seed raw files; pinned repos stay pristine
  via patches/ only.

## Conventions

Ranks: 1-based, pessimistic ties, strict filtering (TRIX/INCITE relation
task: unfiltered by design — state it on tables). One processed root and
one claims dir per model; never mix devices in a rank dir. Seed 1024.
Checkpoint selection: zero-shot DEV10, one mean per group, never
pretraining-mix validation. Every deviation goes in the ledger the day it
happens. Stop rules are honored: dead levers get a result file, not
tuning.
