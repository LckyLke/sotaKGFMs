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

### One trap, since it produced a wrong answer before it produced a right one

The first implementation of ULTRA's unbiased `hits@10_50` disagreed in the 8th
significant digit. The cause was not the tie rule: **numpy promotes
`float32 / int64` to float64 where torch keeps float32**, so dividing the cast
rank by the raw `n_candidates` column ran the whole chain at a precision ULTRA
never used. Casting both operands fixes it. This is exactly the failure mode
criterion A exists to catch, and it is why the comparison is done at printed
precision rather than with a tolerance.

`hits@10_50` is not one of this project's four reported metrics. It is
reproduced anyway because it is the **only** quantity that consumes
`n_candidates`: without it, a dump could get that column wrong and criterion A
would still pass on all five other metrics.

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

## Deviations from the specified procedure

The environment this ran in has **no GPU** — no `nvidia-smi`, no `nvcc`,
`torch.cuda.is_available()` is False, 4 CPU cores — and its egress policy blocks
several hosts the specified procedure requires. Each item below is a fact
checked in this environment, not an assumption.

| step as specified | status | what actually happened |
| --- | --- | --- |
| Build `containers/ultra/` | **not done** | Blocked three independent ways. No Docker daemon is running (`/var/run/docker.sock` absent; only the client is installed). The registry blob CDN `production.cloudfront.docker.com` answers 403, so the CUDA devel base cannot be pulled. And both wheel indexes below are blocked, so the pip layers could not complete even with a daemon. The Dockerfile is written and is the deliverable; it has not been built. |
| `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118` | **blocked** | `download.pytorch.org` — proxy answers 403 to CONNECT (organization egress policy). |
| `pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/...` | **blocked** | `data.pyg.org` — 403. |
| Run on a single GPU with `--gpus [0]` | **substituted** | Run on CPU with `--gpus null`, which ULTRA documents and supports; `rspmm` has a CPU code path and was compiled here from source. |
| CUDA 11.8 devel base, Python 3.9, torch 2.1.0, PyG 2.4.0 | **partly met** | Python 3.9.25, torch 2.1.0, torch-geometric 2.4.0, torch-scatter 2.1.2 — ULTRA's pins exactly. Only the CUDA half is absent. torch came from PyPI (the same 2.1.0, CUDA-12 build, used on CPU) because the pinned index is blocked; torch-scatter 2.1.2 was compiled from its PyPI sdist. **No pin was relaxed to make an install succeed.** One build-tool pin was added: `setuptools==69.5.1`, because torch 2.1.0's `cpp_extension` imports `pkg_resources.packaging`, which setuptools removed in 70. |

Consequences to keep in mind when reading the numbers:

* **Criterion A is unaffected by any of this.** It compares this project's metric
  code against ULTRA's own metric code over the same ranks from the same
  process. Whether that process ran on a GPU or a CPU is irrelevant to whether
  the two agree.
* **Criterion B is affected in principle.** ULTRA's published figures were
  produced on an RTX 3090. CPU and CUDA float32 kernels differ in the low-order
  bits, which can flip a near-tie and move a rank. The effect is small, but it
  is not nothing, and a CPU-derived group mean is not strictly the same
  measurement as the published one.

## Datasets

All 41 graphs in groups 1 and 2 download from hosts this environment can reach —
`raw.githubusercontent.com` (GraIL, Ingram, ILPC, HM) and
`reltrans.s3.us-east-2.amazonaws.com` (MTDEA). No download in scope failed.

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
