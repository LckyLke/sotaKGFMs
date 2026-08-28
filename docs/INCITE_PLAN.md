# INCITE, adjusted to this harness

Status: plan only. Nothing is implemented. Branch: `incite`, cut from the
baseline branch at fc225e5. INCITE code, ranks, and results stay on this
branch. The baseline branch stays clean, as with CREST.

This document is the original INCITE design plus the changes that fit it to
this benchmark framework. The original design text is kept where it survives
review. Each change states its reason. The CREST post-mortem
(`results/STOP.md`, crest branch) is the main input.

## Verdict on the original plan

The design is sound and the review confirmed its central technical claim.
Three parts do not survive contact with this harness unchanged:

1. The evidence for the support set is one unreproduced paper table.
2. The support pathway repeats the CREST mechanism that training silenced.
3. The targets and the cost claim rest on graph sets and protocols that are
   not ours.

The adjusted plan below keeps the architecture and repairs these three parts.

## Verified claims (checked in this repository, 2026-08-28)

* **The factorization in section A is exact for TRIX as shipped.** The pinned
  TRIX repo uses `message_func: distmult` and `aggregate_func: sum` in every
  layer block of every shipped config. The relation-graph message is
  `z_{r_i} * f(x_v)` summed over co-incidence, so the double sum factorizes.
  Caution: the identity is exact in real arithmetic only. A reordered float
  sum differs in the last bits. The gate below therefore uses `allclose`
  at 1e-5, not the bit-for-bit rule CREST phase 0 used.
* **TRIX cost vs ULTRA, measured here:** suite wall clock 15 min vs 8 min,
  about 1.9x, not the 10x the paper discussion suggests. Per-graph ratios
  are in `ranks/*/TIMINGS.jsonl`. Cost claims in this project cite these
  files, not paper prose.
* **Relation baseline, measured here (unfiltered protocol):** TRIX relation
  MRR ind_e 0.7564, ind_er 0.8415 (`shared/published.json`,
  `trix.relation`). Any INCITE relation number states the same protocol.

Not checked: the 2026 WDsinger analysis, the KMAS study, and the KGPFN paper
numbers. The first two are inputs I could not verify. The third becomes
Phase 0.

## Lessons from CREST that bind this plan

1. **A readout over same-encoder support features died under end-to-end
   training.** The optimizer drove CREST's readout to silence: 0.2% of ranks
   differed from TRIX after full-rate training. INCITE's support set differs
   in four ways: trained from scratch, jointly, with retrieval, and with hard
   negatives in the rows. Whether these differences revive the mechanism is
   the experiment. The plan therefore carries a support-usage probe as a
   stop rule (Phase 2).
2. **Pretraining-mix validation does not predict zero-shot transfer.** It
   rose +0.017 to +0.021 in three CREST regimes and transferred nothing.
   Checkpoint selection uses zero-shot DEV10 only, reported as one mean per
   suite group, never as one number. Copy the DEV10 definition from
   `shared/suite.py` on the crest branch.
3. **Full backprop through support passes does not fit this GPU.** Batch-32
   TRIX backprop alone holds 13.8 GiB here. K=16 support passes per query
   with gradients is impossible. Support rows are computed under `no_grad`,
   detached, and refreshed on an interval (CREST's `BankRefresher` pattern,
   crest branch). Record this as a known deviation and as a suspect if the
   support pathway underperforms.
4. **Config-vs-plan diffs are mandatory.** Every hyperparameter deviation
   goes into `results/incite/config_diff.md` the day it happens. The CREST
   stage-B lr ran at half spec until a human asked.

## Phases

### Phase 0 — evidence gate (no INCITE code)

The support-set bet rests on KGPFN Table 1: +0.044 MRR over zero-shot TRIX.
That table is unreproduced, and this harness showed with KG-ICL that a
published cross-model gap can be mostly data convention (+0.079 of +0.090).

1. Run KGPFN in this harness (open-work item 2 on the baseline branch,
   needed anyway). Measure the KGPFN-minus-TRIX gap on our 41 graphs under
   our rank definition and matched pretraining graphs.
2. Run FLOCK's relation task in this harness (entity is running; relation is
   not). This sets the relation baseline for the target below.

Gate: if the measured KGPFN gap is below +0.02 MRR, stop and redesign the
support component before any INCITE code exists. If the gap holds, its
measured value replaces the paper value in the targets.

### Phase 1 — incidence-graph core, gated

Build the factorized network (section A of the original design). Two gates:

1. **Layer gate.** The factorized relation update equals TRIX's materialized
   relation layer on three real graphs, `allclose` at 1e-5, random and
   trained weights both. The pinned TRIX repo is the reference.
2. **Reduction gate.** INCITE with walks off and support off, trained with
   the TRIX recipe on FB15k-237 + WN18RR + CoDEx-M, lands within 0.01 MRR of
   TRIX per DEV10 group. This is the floor every lever must beat, and it
   validates the claimed cost at the same time: the timed runner must show
   at most 3x ULTRA per `TIMINGS.jsonl`.

### Phase 2 — levers, one at a time, each with a kill switch

Add in this order. Evaluate each on zero-shot DEV10 before the next.

1. **Walks (section C).** Kill switch: PETALS accuracy. Build PETALS from
   the FLOCK paper's construction under `diagnostics/`, outside the 41-graph
   suite accounting. Pass: above 90% where deterministic models sit at 50%,
   with DEV10 within noise of the Phase 1 floor.
2. **Support set (section B).** Kill switch: the usage probe. At every
   evaluation, score DEV10 once normally and once with permuted support
   labels. If the rank sets are near-identical (the CREST failure signature),
   the mechanism is dead: stop, write `results/STOP.md`, and do not tune
   around it.
3. **Joint relation task (section D).** Kill switch: entity DEV10 must not
   drop more than 0.005 MRR when the relation loss is added.

### Phase 3 — scale and sweep

Only after Phase 2 survives: 8-graph pretraining, the full 41-graph sweep,
the ablation table (section 5.5 of the original, run on DEV10, not on 41
graphs), and the transductive graphs when the baseline branch adds them.
Three pretraining seeds first. Five seeds only for the final configuration.

## Targets, restated against this harness

Paper-set numbers (57 graphs, 54 graphs) are not comparable to our suite.
Targets bind to measured values in this repository:

| quantity | baseline measured here | INCITE target |
| --- | --- | --- |
| entity MRR, ind_e (18 graphs) | TRIX 0.4562 | TRIX + measured KGPFN gap (Phase 0) |
| entity MRR, ind_er (23 graphs) | TRIX 0.3679 | TRIX + measured KGPFN gap (Phase 0) |
| relation MRR, unfiltered | TRIX 0.7564 / 0.8415 | at or above FLOCK measured here (Phase 0.2) |
| cost, suite wall clock | ULTRA 8 min, TRIX 15 min | at most 3x ULTRA |
| PETALS | deterministic models 50% | above 90% |

The original entity target (0.45 on 57 graphs) assumed TRIX-over-ULTRA adds
to KGPFN's number. That additivity stays a hypothesis and is not a target.

## Amendments to the original design

* **Test-time walk averaging is cut from the primary protocol.** Four passes
  multiply inference to about 12x ULTRA and break the cost target. Primary
  protocol: one pass, seeded (seed 1024, the FLOCK convention,
  patches/flock/0004 precedent). Sampling spread is measured separately with
  unseeded repeats on DEV10 and reported next to the headline number.
* **Support precompute is timed.** The once-per-graph entity encoding and
  the capped 64-per-relation support passes run inside the timed runner, so
  `TIMINGS.jsonl` carries them. A cost claim that excludes precompute is the
  kind this harness exists to catch.
* **Half-link scenario reporting is post-hoc.** Seen/unseen status of head,
  tail, and relation is computable from the rank parquets plus the graph
  files. Write `scripts/halflink_report.py`. No parquet schema change.
* **Gradient flow through support rows is specified** (detached + refresh
  interval), not left open. See lesson 3 above.

## Harness mechanics (all mandatory)

* Self-contained `incite/` package, own container, runners dump per-query
  parquets in the shared schema. Port the skeleton from `crest/run.py`
  (crest branch): it already handles claims, provenance, timings, and the
  rank definition (1-based, pessimistic ties, strict filtering).
* Relation dumps go to `ranks-relation/incite/` with the protocol stated.
* `repos/PINS.json` needs an `incite` entry. INCITE wraps no upstream repo,
  so pin the content: the entry records the incite-branch commit hash, and
  the image tag derives from it as for every other model. (The crest-era
  TRIX alias in `docker_run.sh` was removed with the split — do not
  reintroduce an alias, give INCITE its own entry.)
* Every new env knob goes into the `docker_run.sh` forwarding list the day
  it is born. An unforwarded variable is silently unset in the container.
  This bit three times.
* Runners `cd` into `$WORKDIR` before python starts. `sys.path[0]` beats
  PYTHONPATH, and the baked image copy shadows the prepared tree otherwise.
  This caused both a 14.85 GiB OOM and an eval bug in CREST.
* Never mix CPU and GPU ranks in one directory. Seed 1024 everywhere.

## What stays from the original design, unchanged

Sections A (incidence graph, both channels), B's retrieval-and-hard-negative
construction with PU down-weighting, C's walk protocol and GRU encoder, D's
task flag, prototype head, and joint loss, and E's self-adversarial
negatives and 3-then-8 graph staging. The reasoning in those sections is
good and is not repeated here.

## Decision log

* 2026-08-28 (user): implement and pretrain INCITE FIRST. KGPFN runs only
  after the INCITE pretrain and eval. This inverts Phase 0: the KGPFN
  evidence gate becomes a post-hoc check, and the support-set component is
  built on the paper's number alone. The usage probe (Phase 2.2) therefore
  carries more weight: it is now the only in-harness check on the support
  mechanism until KGPFN lands.
* 2026-08-28: FLOCK stands at 17/41 on the baseline branch, deprioritized
  behind INCITE (user decision). HM:indigo is a suspected host-RAM killer;
  see ranks/flock/RESUME.md on the baseline branch.
