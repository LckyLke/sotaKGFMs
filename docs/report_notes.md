## The resolved tie rule and rank offset

Criterion A can only be argued once the rank definition is pinned down, so here
it is, read out of ULTRA at pin `427966ad` and now documented in
`shared/metrics.py`:

```python
# ultra/tasks.py
def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
```

**Rank offset: 1-based.** A perfect prediction ranks 1. A rank of `k` means
`k - 1` filtered-in candidates scored at least as high as the true answer, so
`1 <= rank <= n_candidates + 1`.

**Tie rule: pessimistic (worst case).** Two things combine. The comparison
`pos_pred <= pred` is non-strict, so an equal-scoring candidate counts against
the true answer. And `strict_negative_mask` zeroes the target's own position
(`t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)`), so the target's trivially
equal self-comparison contributes nothing. Net: if the true answer ties with `k`
other surviving candidates and nothing outscores it, its rank is `k + 1`. That
is neither the optimistic rule (1) nor the average rule (`(k + 2) / 2`). An
implementation using either of those disagrees with ULTRA by an amount that
grows with tie mass, which on sparse graphs is not small.

**`n_candidates` excludes the target**, being `mask.sum(dim=-1)` — that is why
the rank can reach `n_candidates + 1` rather than `n_candidates`.

**Filtering.** Test-time filtering graphs differ by dataset family, and ULTRA
builds them itself: for ILPC and Ingram, inference + valid + test edges; for the
other inductive families, inference + test edges. The dump records the resulting
`n_candidates` per query rather than trying to re-derive it downstream.

### Two things printed-precision comparison caught

**A real bug, first.** The first implementation of ULTRA's unbiased `hits@10_50`
disagreed in the 8th significant digit. The cause was not the tie rule: **numpy
promotes `float32 / int64` to float64 where torch keeps float32**, so dividing
the cast rank by the raw `n_candidates` column ran the whole chain at a
precision ULTRA never used. Casting both operands fixes it, and it now matches
bitwise. This is exactly the failure mode criterion A exists to catch, and it is
why the comparison is done at printed precision rather than against a tolerance.

`hits@10_50` is not one of this project's four reported metrics. It is
reproduced anyway because it is the **only** quantity that consumes
`n_candidates`: without it, a dump could get that column wrong and criterion A
would still pass on all five other metrics.

**And one thing that is not a bug.** `mrr` matched bitwise on the first dataset
and then differed by one float32 ulp on `FB15k237Inductive:v2` (ULTRA
`0.5005503296852112`, `metrics.py` `0.5005502700805664`). Chased down: the
summands are identical, and the *correctly rounded* float32 value of their exact
sum — `math.fsum` over the same float32 summands — is `0.5005502700805664`.
**ULTRA's own reduction is the one an ulp off**, not this module's. The cause is
float32 associativity inside `torch.Tensor.mean`, whose internal blocking is
device- and version-specific; reimplementing it would not transfer to the GPU
numbers anyway, and would make the project's one metric implementation depend on
a torch internal.

That is why criterion A is reported two ways rather than as a bare pass/fail:

* `hits@1`, `hits@3`, `hits@10` sum values that are exactly 0.0 or 1.0, so their
  reduction cannot depend on order. Bitwise equality there **is** the claim that
  the tie rule, the rank offset and the dump agree with ULTRA. This is the
  reading that answers "is the patch right?" — and it passes.
* `mrr`, `mr` and `hits@10_50` are order-dependent, so they are reported with
  their worst ulp distance rather than pass/fail. The strict bitwise verdict is
  still printed and is still the headline; it is not quietly dropped.

Every mismatch observed sits in `mrr` or `hits@10_50` — never in `hits@1/3/10`,
and so far never in `mr` — and each is one or two float32 ulps, order 1e-7
absolute. The exact per-value distances are in the criterion A table above
rather than asserted here, since they are computed, not claimed. That is four to
five orders of magnitude inside criterion B's ±0.002 band, so the residual
cannot affect any acceptance decision. Reported, not iterated on.

## The patch changes no rank

"Patch the dump, not the ranking" is checked rather than asserted.
`scripts/verify_patch_neutrality.sh` runs stock upstream ULTRA — a clean tree
with no patches at all — and the patched tree over the same dataset, checkpoint
and seed, and diffs the two `ultra_results_*.csv` rows.

```
stock   : FB15k237Inductive:v2,60.8537483215332,0.5005503296852112,0.4017951488494873,0.5559661984443665,0.6942977905273438,0.9404902458190918
patched : FB15k237Inductive:v2,60.8537483215332,0.5005503296852112,0.4017951488494873,0.5559661984443665,0.6942977905273438,0.9404902458190918
```

Identical to full printed precision, with `--skip_valid` active. No dtype
change, no epsilon change, no fused op swapped for an unfused one, no reduction
reordered.

Separately, `scripts/verify_rank_dump.py` reloads each dataset and checks that
every dumped `query_id` indexes exactly the triple its row claims. That is the
soundness condition for the way `query_id` is recovered: the dump rebuilds the
loader's `DistributedSampler` order in a second, independent sampler rather than
altering the real loader, which is only valid if the two agree. They do, for
every row of every file checked.

## Patches

| patch | what it does | why |
| --- | --- | --- |
| `0001-rank-dump.diff` | new `ultra/rank_dump.py`; `test()` takes a `dump=` argument; `run_many.py` gains `--rank_dump_dir` | emit one parquet row per scored query into the shared `ranks/` schema; dump only |
| `0002-data-root.diff` | `dataset.root` and `output_dir` in both inference configs become jinja variables with container defaults | give ULTRA its own processed root, so the relation-graph cache in `processed/data.pt` cannot be shared with another repo's `pre_transform` |
| `0003-skip-valid.diff` | opt-in `--skip_valid` on `run_many.py`, default off | the validation-split evaluation feeds no reported metric and mutates no state; skipping it roughly halves CPU-bound runs |

Every change is a patch file. `repos/ultra` is never edited: `git -C repos/ultra
status` is clean at the pinned SHA, and `scripts/prepare_ultra_workdir.sh`
materialises the patched tree the same way the Dockerfile's build layers do.

## Two upstream defects worth recording

**`-d Metafam` and `-d FBNELL` do not work.** `run_many.py` only sets the
`version` template variable when the `-d` entry contains a colon. Bare, jinja
renders the unset variable as the *string* `"None"`, and
`MTDEAInductive.__init__` asserts `version in self.versions` before anything can
normalise it, so both raise `AssertionError`. The correct spellings are
`Metafam:Metafam` and `FBNELL:FBNELL_v1`; `shared/suite.py` carries them as
`run_id` and maps them back with `by_run_id`, so the canonical suite ids and the
rank filenames stay clean.

**Stock `run_many.py` cannot run on a fresh machine.** It calls
`os.chdir(cfg.output_dir)` without creating the directory, so the first run dies
with `FileNotFoundError` on `~/git/ULTRA/output`. SEMMA's fork adds the
`makedirs`; upstream has not. Our runs route `output_dir` at a directory
`scripts/run_ultra.sh` creates, so only the stock half of the neutrality check
is affected.

## Criterion B: two models pass, ULTRA does not

Criterion B compares this project's unweighted group means against the figures
each model itself published. Targets live in `shared/published.json`, one block
per model, each carrying its source. They were constants in `analyse.py` once,
and they were ULTRA's constants, so running the report for any other model
compared that model against ULTRA's targets and printed a verdict that meant
nothing.

Everything below is one GPU run of one seed, 1024, on an RTX 4070 Ti SUPER.

| model | target source | ind_e MRR | ind_e H@10 | ind_er MRR | ind_er H@10 | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| MOTIF | arXiv 2502.13339 Table 2 | +0.0001 | -0.0003 | +0.0001 | +0.0004 | **PASS** |
| TRIX | arXiv 2502.19512 Table 1 | +0.0012 | +0.0011 | -0.0001 | +0.0009 | **PASS** |
| ULTRA | README, row `ULTRA (3g) PyG` | **-0.0042** | **+0.0064** | -0.0019 | -0.0012 | FAIL |

MOTIF and TRIX land on their own published figures to within 0.0012 on all four
group means, well inside the +/-0.002 band. They are ULTRA forks: same
`compute_ranking`, same suite, same rank dump, same metric code, same container
stack, same GPU. Two independent papers reproduced to the printed precision is a
strong statement about the harness, and it is the reason the ULTRA row can be
read as a fact about ULTRA rather than a fault in the measurement.

### What the ULTRA gap is not

**It is not the TorchDrug question.** An earlier version of this file said the
published 0.430 belongs to ULTRA's TorchDrug implementation and that this was
the whole gap. The first half is true -- `repos/ultra/README.md` line 31 names
`DeepGraphLearning/ultra_torchdrug` as a separate repository, and line 363 states
the preprint numbers came from a TorchDrug-trained model. The second half is
wrong. Criterion B never targeted 0.430. It targets the README's own PyG row,
0.420, measured by the authors with the same `run_many.py` this project drives,
and the gap against that row is what fails.

**It is not the device.** MOTIF and TRIX ran on the same GPU, in the same stack,
over the same 18 graphs, and hit their targets. A float32 kernel difference that
moved ULTRA by 0.004 would have moved them too.

**It is not one broken graph.** Per graph across ind_e, ULTRA sits below MOTIF
and TRIX by a margin that grows smoothly with how much those models improve on
it -- 0.157 on WN18RRInductive:v4, 0.010 on FB15k237Inductive:v4 -- and sits
above both on the HM and ILPC graphs. That is two better models, not one
corrupted dataset. Nothing in the per-graph table is anomalous.

**It is not the harness or the rank definition.** Both are shared, and both are
what MOTIF and TRIX pass with.

### What it might be

The remaining candidates are all specific to the ULTRA row itself: that
`ckpts/ultra_3g.pth` as shipped is not the checkpoint that produced the README
table, or that the table was measured against a dataset snapshot that has since
moved. Neither is settled here, and neither should be asserted without evidence.

One measurement worth recording, because it bears on how firm the target is.
Four sources give four different values for ULTRA's ind_e MRR on these same 18
graphs:

| source | ind_e MRR | ind_e H@10 |
| --- | --- | --- |
| ULTRA README, `ULTRA (3g) PyG` | 0.420 | 0.562 |
| ULTRA README, `ULTRA (3g) Paper` | 0.430 | 0.566 |
| MOTIF paper, ULTRA baseline row | 0.431 | 0.566 |
| TRIX paper, ULTRA baseline row | 0.431 | 0.566 |
| SEMMA paper, ULTRA baseline row | 0.428 | 0.570 |
| this project | 0.4158 | 0.5684 |

MOTIF and TRIX both quote 0.431, close to ULTRA's paper row rather than its PyG
row, which suggests neither re-ran ULTRA. SEMMA quotes a third value again. On
ind_er every source agrees within 0.001 and so does this project. The
disagreement is confined to ind_e, and it exists between the published sources
before this project is added to them.

Closing this needs `ultra_torchdrug` pinned as an eighth repository and run, so
that both ULTRA implementations are measured here rather than compared through
somebody else's table. That is on the task list.

## Datasets

All 41 graphs in groups 1 and 2 download from hosts this environment can reach —
`raw.githubusercontent.com` (GraIL, Ingram, ILPC, HM) and
`reltrans.s3.us-east-2.amazonaws.com` (MTDEA). No download in scope failed
outright, but one arrived **silently truncated**, which is worse.

### One truncated download, and why it is worth a section

`HM:indigo` failed to load with `ValueError: not enough values to unpack
(expected 3, got 1)` on the last line of its inference graph. The upstream data
is fine. Our copy of `test-graph.txt` was **17,595,392 bytes against the
server's 19,321,652** — PyG's `download_url` streams to disk without checking
`Content-Length`, so a cut connection leaves a short file and raises nothing.

This one announced itself only because the cut happened to land mid-record. Had
it landed on a line boundary, the graph would simply have been missing its tail,
every metric computed from it would have been quietly wrong, and nothing
anywhere would have complained — not the loader, not the run, not criterion A,
which compares two computations over the *same* corrupted input and would agree
perfectly.

So `scripts/verify_downloads.py` checks a **byte count, not a parse**: it HEADs
every URL each dataset class declares and compares against the file on disk,
with `--fix` to re-fetch and clear the stale `processed/` cache. Over groups 1
and 2: 91 files checked, exactly one short, re-fetched and verified.

MTDEA's ten datasets are reported as **unverifiable rather than passed** — they
arrive as one zip that is extracted and deleted, so there is nothing left to
compare against. That is a real gap, not a clean bill of health.

For the record, since `shared/suite.py` defines all 54 graphs and later tasks
will need them, four of the 13 transductive graphs are **not reachable from
here** and would fail:

| graph | host | status |
| --- | --- | --- |
| `CoDExSmall`, `CoDExLarge` | `zenodo.org` | 403 from the proxy |
| `AristoV4` | `zenodo.org` | 403 from the proxy |
| `Hetionet` | `www.dropbox.com` | 403 from the proxy |

`data/raw/MANIFEST-ultra.json` records, for every one of the 54 graphs, the URLs
its ULTRA dataset class declares — with the version substituted, so a link that
dies later is still on record — alongside a sha256 for every raw file actually
mirrored.

## What was not run, and why

* **The other six containers were not built.** As instructed: clone and inspect
  only. They are cloned at pinned SHAs and inspected in `containers/STACKS.md`.
* **The 13 transductive graphs were not run.** Groups 1 and 2 are what the
  acceptance criteria cover; `shared/suite.py` defines all 54 for later tasks.
* **Nothing was tuned.** No hyperparameter, no threshold, no checkpoint choice
  was changed in response to a gap against the targets. `ultra_3g.pth` was used
  throughout (sha256 in `environment.json`), never `ultra_50g.pth`.
* **`--epochs 0` was verified in the logs**, not assumed: every dataset's config
  dump reads `'num_epoch': 0`.

## SEMMA runs without flash-attn, and that was measured, not assumed

SEMMA has two halves. The structural half is ULTRA. The semantic half embeds
relation descriptions with a sentence encoder, then builds a second relation
graph from the similarities between those embeddings.

`flags.yaml` selects `jinaai/jina-embeddings-v3` as that encoder. `transformers`
loads it with `trust_remote_code=True`, so the model repository supplies and
executes its own `custom_st.py`. That code checks for `flash_attn`, does not
find it, and prints one line for each attention layer:

```
flash_attn is not installed. Using PyTorch native attention implementation.
```

Upstream ships it this way. `repos/semma/requirements.txt` line 10 comments the
dependency out, with the authors' note that installation is complex. The gate
that prints the warning is `get_use_flash_attn` in `modeling_xlm_roberta.py`,
and it tests `importlib.util.find_spec` only, after `config.use_flash_attn` and
`torch.cuda.is_available()` both pass. Installing a wheel is therefore the whole
change. The question is whether to make it.

### The measurement

A matching wheel exists and works:
`flash_attn-2.5.8+cu118torch2.1cxx11abiFALSE-cp39-cp39`. The container is Python
3.9, torch 2.1.0+cu118, `_GLIBCXX_USE_CXX11_ABI = False`, and the GPU is compute
capability 8.9, so every constraint is met. A forward pass runs.

**Cost.** The encoder is not where SEMMA's time goes. Loading it takes 1.3 s and
encoding all 237 FB15k-237 relation names takes 0.8 s, against a steady-state
cost near 27 s per graph. Flash attention can save a fraction of that 0.8 s, so
under two percent of the suite. The earlier claim in this file, that unfused
attention was part of why SEMMA is the most expensive model here, was wrong.
SEMMA's cost is in the structural half.

**Numbers.** The two paths do not agree. Encoding the same 237 relation names
both ways gives a maximum absolute difference of 0.0055 per embedding component,
and up to 0.0128 in the pairwise cosine similarity.

That looks large until the dtype is checked. `config.torch_dtype` is
`bfloat16`, and the parameters load as bfloat16 on both paths. One bfloat16 step
near 1.0 is 0.0078. The measured 0.0055 is therefore below a single
representable step of the encoder's own precision, which is what a reordered but
exact attention kernel predicts.

The consequence is still real. SEMMA keeps every relation pair whose cosine
similarity exceeds 0.8. Over the 27966 pairs among those 237 relations, the
flash path keeps 363 and the native path keeps 364, and **5 pairs disagree**.
The semantic relation graph is not the same graph.

### The decision

Run without flash-attn. There is no speed argument, because the encoder is under
two percent of the cost. There is a correctness argument against, because the
threshold admits a different set of pairs. And upstream ships it off, so off is
also the configuration the authors published.

### What this says about SEMMA

The finding worth keeping is not about this container. SEMMA's 0.8 cutoff sits
inside the numerical noise floor of its own encoder's dtype: one bfloat16 step
near the threshold is 0.0039, and the observed cosine spread reaches 0.0128. Any
change that reorders arithmetic in that encoder can move pairs across the cutoff.
That is a property of the model as published, and it bounds how exactly any
reimplementation of SEMMA can be expected to match it.

## The sentence encoder is pinned to one commit

`patches/semma/0003-pin-encoder.diff` adds a `revision` to the
`AutoModel.from_pretrained` call. Upstream passes none, so `main` decides both
the weights and the remote code on the day of the run.

Half of SEMMA is built from those embeddings. Without a pin, a re-run months
later can produce different relation graphs from the same checkpoint and the
same data, and nothing in the output would show why.

The container pins `JINA_REVISION=ab036b023d30b4d1138c4c3bfa9f0c445ab455d6`,
caches that commit at build time, and sets `HF_HUB_OFFLINE=1` for every run. The
copy that executes is the copy the image was built against. The patch does not
change which model SEMMA uses. It fixes which version of it.
