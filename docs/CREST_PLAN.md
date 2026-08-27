# CREST implementation plan, revision 2

CREST is a knowledge graph foundation model. The encoder is TRIX. The readout is
an in-context module that reads a bank of example triples per relation. One
optional random channel breaks relation symmetries.

This revision changes three things about how the work is run. Nothing about the
model changes.

1. **CREST is the eighth model in this harness**, not a separate project. It
   dumps per-query ranks into `ranks/crest/` under the same schema as the other
   seven, and every number it reports is computed by `shared/metrics.py` outside
   the container.
2. **Every gate compares against numbers measured here**, never against another
   paper's table. The reason is in `docs/report_notes.md`: four published sources
   give four different values for ULTRA's ind_e MRR, and one preprocessing
   choice in KG-ICL was worth 0.079 MRR. A gate of plus or minus 0.01 against a
   third party's transcription measures the harness, not the model.
3. **One environment.** CREST runs on the TRIX stack: Python 3.9, torch 2.1.0,
   CUDA 11.8, PyG 2.4.0. FLOCK is not a runtime dependency. See section 6.

## 1. The contract CREST must satisfy

CREST is our code, so `repos/` discipline does not apply to it: it lives at
`crest/` in the work tree and is versioned like `shared/`. Everything else is
the same contract every model here follows.

| requirement | where |
| --- | --- |
| one container, built from a Dockerfile in the repository | `containers/crest/Dockerfile` |
| one runner that loops the suite and dumps ranks | `scripts/run_crest.sh` |
| ranks in the shared schema, one parquet per graph | `ranks/crest/<graph>.parquet` |
| the model's own metric values, for criterion A | `results/crest/CREST_results.csv` |
| device, seed and checkpoint recorded | `ranks/crest/PROVENANCE.json` |
| wall clock per graph | `ranks/crest/TIMINGS.jsonl` |
| published targets as data | a `crest` block in `shared/published.json` |
| dataset list | `shared/suite.py`, never a private yaml |

TRIX itself stays pristine at its pin. CREST imports it from the patched tree
that `scripts/prepare_trix_workdir.sh` produces, exactly as the TRIX runner
does. Any change to TRIX is a new diff in `patches/trix/` with a stated Reason.

Seed is **1024** for every run until phase 4, matching every other model here.

## 2. Phase 0. Integration, and an exact identity gate

1. Create the package layout of section 7.
2. `containers/crest/Dockerfile`: start from the TRIX stack. Copy `shared/`,
   `repos/trix/`, `patches/trix/`, and `crest/`. Apply the TRIX patches. Compile
   `rspmm` at build time, not at run time.
3. `scripts/run_crest.sh`: model the file on `scripts/run_trix.sh`. Keep the
   provenance guard that refuses to mix CPU and GPU ranks in one directory, the
   atomic claim per graph, and the per-graph timing record.
4. Add `crest` to `analyse.CSV_PATTERNS` and to `MODELS` in
   `scripts/make_summary.py`.
5. Define `DEV10` as a named group in `shared/suite.py`. It holds FBIngram:25,
   WKIngram:25, NLIngram:0, WikiTopicsMT1:tax, Metafam, FBNELL,
   FB15k237Inductive:v1, WN18RRInductive:v1, NELLInductive:v1 and CoDExSmall.
   **Report it as three group means, never as one.** It mixes 6 ind_er, 3 ind_e
   and 1 transductive graph, and a single mean over those is not a quantity.
6. Build `CRESTEntity` with the readout present but the last linear layer of its
   MLP at zero, so `s = s_v0 + 0`.

### Stop rule for phase 0

Run CREST with the zeroed residual over all 41 inductive graphs and compare
`ranks/crest/` against `ranks/trix/` **row by row**.

* **Pass:** every graph has the same row count, and the `rank` column is
  identical on every row.
* **Stop:** any rank differs. A zeroed residual is arithmetically TRIX, so a
  difference is a defect in the wrapper, the data root or the dump, and no later
  number can be trusted until it is found.

This replaces the old "mean entity MRR within 0.01 of 0.475". It is a stronger
test, it is free, and it is only possible because TRIX's ranks are already
dumped here. A tolerance gate cannot distinguish a correct wrapper from two
errors that cancel.

## 3. Phase 1. Build the relation-prediction baseline

This harness has never run relation prediction. There is nothing to compare
against, so the baseline must be produced before CREST's relation model can be
gated. This phase is new and is not optional.

1. **Do not patch `compute_ranking_relation`.** An earlier revision of this
   plan claimed its unfiltered branch was 0-based and had to be fixed. That was
   wrong, and the arithmetic is in `docs/report_notes.md` under "The unfiltered
   rank offset is correct, and looks like a bug". Without a mask the target
   counts itself, so the sum is never below 1 and the branch is already 1-based;
   adding `+ 1` would inflate every relation rank by one. Record instead that
   **TRIX evaluates relation prediction unfiltered**, so other true relations
   between the same pair are not removed from the candidate set. State that on
   every relation table. It is a modelling choice, not a defect.
2. Extend the rank schema for the second task. Relation dumps go to
   `ranks-relation/<model>/<graph>.parquet`, same columns, with
   `direction = "relation"` and `n_candidates` counting relations rather than
   entities. `shared/metrics.py` gains a `task` argument so its direction
   validation accepts it.
3. Run TRIX relation prediction on all 41 graphs with
   `relation_prediction.pth`. Never `entity_prediction.pth`: they are separate
   models with separate checkpoints.
4. Record the group means in `shared/published.json` under `trix`, as a
   `relation` block, sourced to this project rather than to a paper.

### Stop rule for phase 1

* **Pass:** criterion A passes, meaning every integer hit count from
  `shared/metrics.py` matches what TRIX itself computed over the same ranks.
  That is the same gate the other four models passed at 123 of 123.
* **Stop:** any count differs. A count disagreement is a disagreement about the
  rank definition, not a rounding difference.

Cross-check, not a gate: compare the result against TRIX's own paper. A
disagreement is a finding to record, not a failure.

## 4. Phase 2. CREST v1

Unchanged from revision 1 in substance. Sections 3.1, 3.2 and 3.3 of the
original plan stand, with the corrections below.

### 4.1 Specification corrections

* **`Ans(u, r)` is computed over the inference graph only**, never over a graph
  containing test edges. State it in `bank.py` and assert it in the leakage
  test. The original plan left the source graph unspecified, which is a leak
  waiting to happen in negative sampling.
* **Bank rows come from the inference graph only.** This was already stated in
  the original section 6 and is repeated here because it is the property the
  whole design rests on.
* **Chunk size 4096 is a memory parameter, not a model parameter.** Record it in
  `results/config_diff.md` and in `PROVENANCE.json`. This GPU has 15.6 GB, not
  the 24 GB the original plan assumed.

### 4.2 Cost control, which the original plan did not budget

Measured here: the 41 inductive graphs carry 3241 base relations, so 6482
relation ids. At 20 sampled edges per id and two tasks that is **259,280 encoder
forward passes** to build every bank once. For scale, TRIX evaluates all 41
graphs in 16.7 minutes.

Building a bank once per graph for inference is acceptable. **Stage B is not**,
as originally written: rebuilding the whole bank of FB15k-237 every 500 steps
costs about 18,960 forwards per rebuild, and no stop rule covered it.

Three changes:

1. **Cache banks on disk**, keyed by graph id, checkpoint hash and seed, under
   `data/roots/crest/banks/`. A rebuild that would produce an identical bank
   must not run.
2. **Refresh only the relation ids touched since the last refresh**, not the
   whole graph.
3. **Gate it.** Bank build time per 500 steps must stay under 20 percent of
   training step time over the same window. Log both to
   `results/phase2_cost.json`. If the gate fails, raise the refresh interval
   before anything else.

### 4.3 Tests, `crest/tests/`

Keep all five from the original plan, with two corrections.

* `test_residual_zero.py` is now the phase 0 gate as well, run over real graphs
  rather than only a toy one.
* `test_equivariance.py` applies to the deterministic model only. Track B exists
  to break relation symmetry, so the test must be skipped, not failed, when a
  random channel is active. The original plan did not scope it.

Add one test:

* `test_schema.py`: a CREST dump satisfies `shared/suite.RANK_COLUMNS`, ranks
  are 1-based, and `rank <= n_candidates + 1` on every row. This is what
  `scripts/verify_rank_dump.py` already checks for the other models.

### 4.4 Stop rule for phase 2

Measured baseline, from `ranks/trix/` in this repository:

| group | TRIX MRR | TRIX Hits@10 |
| --- | --- | --- |
| ind_e, 18 graphs | 0.4562 | 0.5931 |
| ind_er, 23 graphs | 0.3679 | 0.5409 |

Evaluate the best stage-B checkpoint zero-shot on DEV10 first, then on all 41.

* **Pass:** on all 41, ind_e MRR at least 0.4712 and ind_er MRR at least 0.3829,
  both plus 0.015 over TRIX, with neither group regressing on Hits@10.
* **Grid:** if either group lands between plus 0.005 and plus 0.015, run the
  grid once, on DEV10 only, and bound it to six cells rather than twelve:
  positives in {5, 20} crossed with negatives in {uniform, type-compatible,
  type-compatible plus self-adversarial}, at stage A only. Carry the best cell
  into stage B. The original twelve-cell grid did not fit its own budget.
* **Stop:** below plus 0.005 on both groups after the grid.
* **Cost:** inference time per graph after the bank build must stay within 3x
  TRIX, measured from `ranks/*/TIMINGS.jsonl`. Report bank build time
  separately.

Why plus 0.015: the whole measured ULTRA to TRIX gap here is plus 0.040 on ind_e
and plus 0.026 on ind_er. A readout module delivering 0.015 is roughly 40
percent of a full model generation. That is the bet this plan makes, and phase 2
is where it becomes visible.

## 5. Phase 3. Three tracks

Order A, B, C. Each starts from the phase-2 checkpoint. Each reports through the
same harness.

### Track A. Order-sensitive message

Unchanged. Replace the DistMult message `x * z_r` with `m = U (z_r * (V x))`,
one 32x32 pair per layer, initialised to the identity so the model starts at
DistMult. Keep it if mean relation MRR rises by at least 0.02 over the phase 1
baseline. Report entity prediction with the same message.

### Track B. Random channel

**FLOCK is not imported.** Its walk sampler is expensive because it samples per
*candidate entity*, which is why it costs 24x to 412x ULTRA here and saturates
16 CPU cores. Track B-iii samples 8 walks from `h` only, per query. That is a
few hundred steps per query and needs no C++ extension. Implement the sampler in
`crest/randchan.py` in plain torch, on the TRIX stack.

PETALS is a data dependency, not a code dependency. Generate the 220 instances
**once** with the FLOCK container already built here, write them to
`data/raw/petals/` with a manifest recording the generator seed, and never
import FLOCK again. Note that FLOCK ships `np.random.seed` commented out in
`set_seed`, so the generation script must seed numpy explicitly or the instances
cannot be regenerated.

Variants B1-i, B1-ii and B1-iii as originally specified. Average 8 passes at
test time.

* **Stop rule B1:** a variant proceeds only at 95 percent or better on PETALS.
* **Stop rule B2, revised:** keep the channel if mean entity MRR over all 41
  stays within 0.005 of the phase-2 result **and** ind_er MRR rises by at least
  0.01, or if mean relation MRR rises by at least 0.01.

The original gate keyed on Metafam entity MRR rising 0.05. Metafam is the single
most idiosyncratic graph in the suite: it is the only one of 41 with exactly one
answer per query, and measured model scores on it span 0.2556 to 0.4213. One
graph that noisy cannot carry a gate. The ind_er group is where relation
symmetry actually matters and it has 23 graphs.

### Track C. One network for both tasks

Unchanged. Share the encoder, keep both readouts and both banks, alternate
batches one to one, train stage A then stage B. Keep the joint model if each
task stays within 0.005 MRR of its separate model on all 41.

## 6. Why one environment is enough

The original plan cloned FLOCK for the PETALS generator, the walk sampler and
the recording protocol. That cannot share an environment with TRIX: FLOCK
specifies `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel`, which is Python 3.11
and CUDA 12.6, against TRIX's Python 3.9 and CUDA 11.8.

It does not need to. The dependency splits cleanly:

| what | when | how |
| --- | --- | --- |
| PETALS instances | once, offline | the FLOCK container already in this repository, output to `data/raw/petals/` |
| walk sampler | every forward pass | reimplemented in torch in `crest/randchan.py` |
| recording protocol | every forward pass | anonymised ids, direction bits and query flags are a specification, not a library |

CREST therefore builds on one image, on the TRIX stack, with no cross-stack
imports.

## 7. Package layout

```
crest/
  bank.py        ContextBank, build_bank_entity, build_bank_relation
  pfn.py         RowEncoder, ContextTransformer, QueryReader
  model.py       CRESTEntity, CRESTRelation, CRESTJoint
  messages.py    BilinearMessage
  randchan.py    NoiseChannel, WalkChannel, walk sampler
  train.py       stage_a, stage_b, joint
  tests/
containers/crest/Dockerfile
scripts/run_crest.sh, scripts/prepare_crest_workdir.sh, scripts/collect_crest_results.sh
configs/crest_v1.yaml, track_a.yaml, track_b.yaml, track_c.yaml
```

No `crest/eval.py`. Evaluation is `shared/metrics.py`, `shared/analyse.py` and
`scripts/make_report.py`, which every other model already uses. A private
evaluation path is the thing this project exists to avoid.

`results/<phase>_<name>.json` keeps per-dataset MRR and Hits@10 for entity
prediction, MRR and Hits@1 for relation prediction, time per batch, and peak GPU
memory. Every changed hyperparameter goes in `results/config_diff.md`. A failed
stop rule writes `results/STOP.md` with the numbers and the most likely cause.

## 8. Phase 4. Full evaluation, in this harness

1. All 41 inductive graphs, both tasks, 3 pretraining seeds. Report mean and
   standard deviation through `scripts/make_summary.py`, so CREST appears in the
   same table as the other seven models.
2. The 13 transductive graphs, same treatment. Note that KG-ICL deduplicates
   test triples and two transductive graphs contain duplicates, so query counts
   differ there. This is recorded in `docs/report_notes.md`.
3. Finetuned row, using the TRIX finetune recipe. Rebuild the bank afterwards.
4. Cost on the 5 largest graphs: time per batch and peak memory for TRIX and
   CREST, from `TIMINGS.jsonl`. Plot peak memory against `|E|`.
5. If track B is on, plot MRR against walk count in {2, 4, 8, 16, 32}. A flat
   curve is the expected result.
6. State the protocol on every table: the bank uses edges of the inference graph
   only, ranks are 1-based with pessimistic ties under strict filtering, and the
   test message graph is the inference graph alone.

### Targets, restated against measurements from this repository

| quantity | TRIX here | CREST target |
| --- | --- | --- |
| ind_e MRR | 0.4562 | 0.4712 |
| ind_e Hits@10 | 0.5931 | no regression |
| ind_er MRR | 0.3679 | 0.3829 |
| ind_er Hits@10 | 0.5409 | no regression |
| relation MRR | phase 1 establishes it | phase 1 plus 0.02 |

The original targets of 0.42 entity MRR and 0.80 relation MRR came from a
54-graph mean in another paper. They are not comparable with anything this
project measures and are dropped. If a 54-graph number is wanted for
comparison with the literature, compute it, report it beside the group means,
and label the difference.

## 9. Time budget

Recalibrated against measurements here. TRIX evaluates 41 graphs in 16.7
minutes; CREST at 3x is under an hour per full sweep, so evaluation is not the
cost. Bank building and training are.

| phase | budget | dominant cost |
| --- | --- | --- |
| 0, integration and identity gate | 2 days | none, this is wiring |
| 1, relation baseline | 2 days | one TRIX sweep plus the ranking patch |
| 2, CREST v1 | 8 days | stage B, plus bank refresh |
| 3, three tracks | 8 days | track B pretraining on PETALS |
| 4, full evaluation | 10 days | 3 seeds x 2 tasks x 54 graphs, plus finetune |

If a phase exceeds twice its budget, write `results/STOP.md` and stop.

## 10. Concerns from the review, and their disposition

| concern | status |
| --- | --- |
| gates calibrated on another paper's table | **fixed**, every gate now uses `ranks/*` measured here |
| `compute_ranking_relation` is 0-based unfiltered | **still live**, phase 1 task 1 |
| FLOCK and TRIX stacks cannot share an environment | **fixed**, FLOCK is offline-only, section 6 |
| FLOCK's numpy seed is commented out | **narrowed**, affects PETALS generation only |
| 24 GB assumed, 15.6 GB available | **addressed**, chunk size recorded, cost gate added |
| DEV10 mixes three regimes in one mean | **fixed**, reported per group |
| bank cost unbudgeted | **fixed**, caching, partial refresh, explicit gate |
| no check that our metrics match the model's own | **fixed**, criterion A is the phase 1 gate |
| equivariance test contradicts track B | **fixed**, scoped to the deterministic model |
| Metafam carries a track B gate | **fixed**, replaced by the ind_er group |
| twelve-cell grid does not fit its budget | **fixed**, six cells, stage A only |
| seed 0 conflicts with the project | **fixed**, 1024 throughout |
| `Ans(u, r)` source graph unspecified | **fixed**, inference graph, asserted in the test |
| "wrap, do not edit" meets a repo that resists it | **accepted**, TRIX changes are diffs in `patches/trix/` |

---

# Revision 3. CREST owns its encoder

This section is additive. It revises sections 1, 2 and 7 above, and changes
nothing about the model, the readout, the bank or any stop rule after phase 1.
It will be folded into the body once the current implementation pass lands.

Decision: **CREST reimplements the TRIX encoder inside `crest/` and loads TRIX's
released weights.** It does not import `repos/trix` at run time.

Two things follow. CREST becomes a self-contained implementation that can be
built and run without cloning another repository. And the stack stops being
pinned to torch 2.1.0, CUDA 11.8 and Python 3.9 by somebody else's requirements.

## P.1 What is ported, and what is not

Measured surface of TRIX at pin 7596e14e:

| file | lines | ported |
| --- | --- | --- |
| `models_entity.py` | 243 | yes |
| `models_relation.py` | 256 | yes |
| `layers.py` | 235 | yes |
| `tasks.py` | 277 | ranking and relation-graph construction only |
| `util.py` | 165 | partly |
| `datasets.py` | 1237 | **no** |

`datasets.py` is not ported. `scripts/build_kgicl_datasets.py` already
demonstrates that all 54 graphs are readable with one generic loader plus a
per-family table of file names, in roughly 200 lines. Reuse that mapping. It is
the same raw data every other model here reads.

## P.2 The checkpoint pins the architecture

CREST loads `entity_prediction.pth` and `relation_prediction.pth`: 154 tensors
and 87138 parameters for the entity model. Keys look like
`relation_model.layers_hh.0.relation_projection.0.weight`.

This is a re-expression with identical parameter semantics, **not** a redesign.
Either match the module structure so the state dict loads directly, or write an
explicit key-remapping loader and test it. State which was chosen.

For scale, and it is worth knowing when reading phase 2: the readout specified
in section 3.3 is 427009 parameters, **4.9 times the encoder**. TRIX is a very
small model. If CREST gains, most of the capacity doing the gaining is new.

The checkpoints are committed in TRIX's own repository, are 1.2 MB and 2.1 MB,
and are therefore declared as a hashed data dependency rather than reached for
inside a clone:

```
data/raw/trix-checkpoints/entity_prediction.pth     sha256 8f6e7266093c2d15...
data/raw/trix-checkpoints/relation_prediction.pth   sha256 5d15d0b50c9df4e4...
```

Record them in `data/raw/MANIFEST-trix-checkpoints.json` in the same shape as
the other manifests, with the source URL and the pin they came from.

## P.3 Verification: two tiers, in this order

Bitwise verification and newer packages are **mutually exclusive**. `rspmm`
fuses its reduction; any `scatter_reduce` equivalent sums in a different order,
so float32 results differ and ranks flip on ties. Changing torch versions does
the same. This is measured, not theoretical: moving ULTRA from CPU to GPU
changed ranks on 1 of 37 graphs here, worth 0.0145 MRR on NELLInductive:v1.

So the two variables are separated, and the order is mandatory.

### Tier 1. Prove the port

Stack unchanged: torch 2.1.0, CUDA 11.8, Python 3.9, `rspmm` kept as it is.
Only the code changes.

* **Gate:** run the port over all 41 inductive graphs and compare against
  `ranks/trix/*.parquet` row by row. Every rank must be identical. Roughly
  700000 per-query ranks, from a model we did not write.
* **Stop:** any rank differs. Find it before continuing. There is no tolerance
  at this tier, because there does not need to be one.

### Tier 2. Move the stack

Now, and only now, change one thing at a time: torch 2.5 or newer, `rspmm`
replaced by `scatter_reduce`. KG-ICL's `EntityEncoder` carries a working
pure-torch fallback of the same shape (`message = hs + hr`, then `scatter_add`)
and is the reference for what the kernel computes.

* **Gate:** compare against tier 1's own dump. Report the percentage of ranks
  that are identical and the change in each group mean. This is a **measured
  and published property of the port**, not a pass or fail.
* Record it in `docs/report_notes.md` beside the CPU-versus-GPU note, which is
  the same class of effect.

If tier 1 and tier 2 are done together and a rank differs, nothing can be
concluded about which change caused it. That is the entire reason for the split.

## P.4 What the container no longer needs

`containers/crest/Dockerfile` copies `shared/` and `crest/`. It does not copy
`repos/trix`, does not apply `patches/trix/`, and does not compile `rspmm` from
another repository's source tree at tier 2.

`patches/trix/` still exists and is still applied when running TRIX itself,
including the `compute_ranking_relation` fix from phase 1. CREST's own port must
carry that fix natively: 1-based ranks in both branches, filtered and
unfiltered.

## P.5 Effect on the phases above

* **Phase 0** is unchanged in intent and stronger in effect. The identity gate
  now proves two things at once: that the readout is correctly zeroed, and that
  the ported encoder reproduces TRIX exactly.
* **Phase 1** is unchanged. The relation baseline still comes from running TRIX
  itself, from `repos/trix`, so the port is checked against an independent
  implementation rather than against itself.
* **Phases 2, 3 and 4** are unchanged.
* Budget: add **4 days** for the port and its two verification tiers, between
  phase 1 and phase 2.
