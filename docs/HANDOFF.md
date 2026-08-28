# Handoff — evening of 2026-08-27

Written at end of day two. FLOCK is grinding overnight; everything else is
committed on `claude/gpu-multi-model-baseline` (not pushed).

## The one-table state

| model | entity ranks | relation ranks | criterion A | vs published |
| --- | --- | --- | --- | --- |
| ULTRA | 41/41 | — | 123/123 | −0.004/−0.002, open (see report_notes) |
| MOTIF | 41/41 | — | 123/123 | ≤0.0004 everywhere, PASS |
| TRIX | 41/41 | **41/41** | 123/123 both tasks | PASS; relation matches FLOCK-paper per graph to 3rd decimal |
| SEMMA | 41/41 | — | 123/123 | +0.002..0.003 (their table is 5-run avg) |
| KG-ICL | 41/41 (+13 native-convention) | — | exact on own CSV | reproduces its paper under its own convention |
| FLOCK | **16/41, running overnight** | — | pending | pending |
| KGPFN | not started | — | — | — |
| CREST | moved to the `crest` branch | — | inherits TRIX | **phase 2 STOPPED** — see below |

Headline numbers live in `baseline_report.md` (regenerate:
`python3 scripts/make_summary.py`). Key findings of record:
`docs/report_notes.md`. Per-model detail: `reports/`.

## What happened today (day two)

1. **KG-ICL completed**, including the convention A/B that showed its published
   Fully-Inductive advantage is mostly preprocessing (+0.079 MRR from validation
   edges in the test message graph; collapses to +0.020 under matched data).
2. **CREST was built end to end** — NOTE: all CREST code, plans, ranks, and
   results now live on the `crest` branch, not here. This branch keeps only
   the shared-harness improvements and the TRIX relation baseline.
   Built by two Fable-5 subagents against
   `docs/CREST_PLAN.md` (revision 3: CREST owns its encoder — the TRIX port +
   bitwise-then-modernise verification plan is written but NOT yet executed).
   - Phase 0 PASSED: zero-residual CREST is bit-for-bit TRIX, 185,870 ranks.
   - Phase 1 PASSED: TRIX relation baseline now exists (`ranks-relation/trix/`,
     ind_e 0.7564 / ind_er 0.8415 MRR, unfiltered protocol — state it on any
     table). Recorded in `shared/published.json` under `trix.relation`.
   - Phase 2 **STOPPED** per plan rule 5: `results/STOP.md` (crest branch).
     Three checkpoints
     (stage A frozen; stage B half-lr; stage B full-rate 5k steps) all transfer
     zero. The decisive one: end-to-end training drove the readout to silence
     (0.2% of ranks differ from TRIX). The optimiser discarded the mechanism.
   - Also learned: pretraining-mix validation does NOT predict transfer
     (rose +0.017..0.021 every time, transferred nothing). Checkpoint selection
     must use zero-shot DEV10 (defined in `shared/suite.py` on the crest
     branch; removed here with the rest of CREST).
3. **FLOCK crashed with the host mid-day** (16/41 survived, nothing corrupted),
   resumed this evening. `ranks/flock/RESUME.md` has the resume command.

## FLOCK overnight: what to check tomorrow morning

```bash
ls ranks/flock/*.parquet | wc -l     # want 41
grep -c '!!! FAILED' /tmp/claude-1000/-home-lukef-Dokumente-GitHub-sotaKGFMs/*/tasks/bqwiopvhp.output
```

* If 41/41: run `scripts/collect_flock_results.sh` (workdir
  `output/flock-run`), then criterion A via
  `python3 scripts/make_report.py --ranks ranks/flock --out reports/flock.md`,
  then `python3 scripts/make_summary.py`. Note FLOCK's CSVs are
  `results_*.csv` under `output/flock-run/src_entity/results/` — check
  `analyse.CSV_PATTERNS` has a `flock` entry before criterion A (it may not).
* If it died again: `rm -rf ranks/.claims-flock`, relaunch the exact command in
  `ranks/flock/RESUME.md` (divisor 4 + expandable_segments — never change
  mid-run). Completed graphs skip automatically.
* Remaining cost from 16/41 was ~19 h; the two ~5 h graphs are ILPC2022:large
  and HM:indigo.
* Seed note for the report: FLOCK is stochastic (walk sampling); runs use seed
  1024 with numpy seeded by patches/flock/0004. A few repeats with
  FLOCK_UNSEEDED_WALKS=1 are still wanted to quantify sampling spread.

## Open work, in the order I would do it

1. **Finish FLOCK** (overnight) → criterion B vs its paper (block already in
   `shared/published.json`? check — if absent, add from arXiv 2510.01510).
2. **KGPFN** — last of the seven. Python 3.12, torch 2.5.1, flash-attn
   REQUIRED (prebuilt wheel; the SEMMA wheel-hunt pattern works). Checkpoint
   via `python script/download.py --kgpfn`. Nothing built yet.
3. **Transductive sweep** (13 graphs × models) — deferred all along. KG-ICL
   note: its transductive test sets deduplicate; ConceptNet/AristoV4 differ.
4. **ultra_torchdrug** as an eighth pinned repo — closes ULTRA's criterion B
   gap, still the one open reproduction question (four published sources give
   four different ULTRA ind_e numbers; ours is a fifth).
5. **MOTIF timing re-run** (its first suite predates TIMINGS.jsonl).
6. **CREST tracks A/B if desired** — on the `crest` branch. Untested,
   independent of the readout, start from TRIX directly. The TRIX-port plan
   (CREST_PLAN revision 3 §P, crest branch) is also unexecuted if
   self-containment still matters.

## Traps that bit us (do not re-learn these)

* `docker_run.sh` forwards a fixed env list — an unforwarded variable is
  SILENTLY unset in the container. Bit us three times (FLOCK_BATCH_DIVISOR,
  KGICL naming, CREST_TRAIN_*). Check the list before adding knobs.
* The container images bake `crest/`+repos at build time; `sys.path[0]` (cwd)
  beats PYTHONPATH. Both stage-A OOM and an eval bug came from running baked
  code instead of the prepared work tree. run_crest.sh/train_crest.sh now cd
  into `$WORKDIR` — keep that pattern for new runners.
* `git diff` emits LF; KG-ICL sources are CRLF; patches must normalise both
  sides (gen script + Dockerfile both do).
* Wikidata labels drift (SEMMA patches 0004-0006); entity labels were 225k
  fetched-and-discarded lookups.
* Plan-vs-config: diff hyperparameters line by line
  (`results/crest/config_diff.md` on the crest branch is the standing table).
  The stage-B encoder lr ran at half spec until the user caught it by asking.
* `compute_ranking_relation`'s missing `+1` is CORRECT (unfiltered counts the
  target itself). Do not "fix" it — see report_notes "The unfiltered rank
  offset is correct, and looks like a bug".
* Read `BUILD_EXIT=`/exit lines explicitly; piped tails mask docker failures.

## Standing conventions

Ranks: 1-based, pessimistic ties, strict filtering (except TRIX relation task:
unfiltered by upstream design). One processed root and one TORCH_EXTENSIONS_DIR
per repo. Never mix devices in one rank dir. Every upstream change is a
Reason-headed diff in `patches/<repo>/`; `repos/` stays pristine at
`repos/PINS.json` SHAs. Published targets are data (`shared/published.json`),
never constants. Seed 1024 everywhere. All 21+ commits today are local:
**nothing is pushed**.
