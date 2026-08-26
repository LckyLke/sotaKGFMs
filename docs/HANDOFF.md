# Handoff to a GPU machine

Written at the end of a CPU-only session. Everything below is either done and
reusable, or explicitly still owed.

## Do this first

```bash
git checkout claude/workspace-ultra-baseline-m1muyd
scripts/clone_repos.sh            # repos/ is gitignored; restores all seven at their pinned SHAs

rm -rf ranks/ultra ranks/.claims  # <- REQUIRED. See "Why the ranks must be discarded".
```

`scripts/run_ultra.sh` will refuse to start if `ranks/ultra/PROVENANCE.json`
records a different device from the one it is running on, so forgetting this
step produces a clear abort rather than a silently mixed rank directory. It is
still better to clear it deliberately.

## Why the ranks must be discarded

`ranks/ultra/` currently holds **37 of 41** rank dumps produced **on CPU**,
marked as such in `ranks/ultra/PROVENANCE.json`. They are real, verified output
and they are what criterion A was established against — but they are not GPU
ranks. CPU and CUDA float32 kernels differ in low-order bits, which can flip a
near-tie and move a rank. A directory holding some of each yields a group mean
corresponding to no single measurement, and no check downstream would catch it.

So: regenerate all 41 on the GPU. Criterion A will re-establish itself from the
new run in minutes; nothing about it depends on the old files.

## What is done and does not need redoing

| deliverable | state |
| --- | --- |
| `repos/PINS.json` | done — seven SHAs, restore with `scripts/clone_repos.sh` |
| `containers/STACKS.md` | done — stacks, extensions, entry points, fork status for all seven |
| `containers/ultra/Dockerfile` | written, **never built** (see below) |
| `shared/suite.py` | done — the 54 graphs, frozen |
| `shared/metrics.py` | done — and validated bitwise against ULTRA itself |
| `shared/analyse.py`, `scripts/make_report.py` | done — regenerate the report in one command |
| `patches/ultra/*.diff` | done — three patches, one reason each, all verified ranking-neutral |
| `environment.json` | regenerate on the GPU box (`scripts/environment.py`) |
| `ranks/ultra/*.parquet` | **redo on GPU** |
| `results/ultra_results_*.csv` | **redo on GPU** (current ones are the CPU run's) |
| `baseline_report.md` | regenerate after the GPU run |

## Criterion A is already answered, and it is device-independent

Across the 37 graphs run here, **every one of the 111 order-independent
comparisons** (`hits@1`, `hits@3`, `hits@10`) reproduced ULTRA's own
`ultra_results_*.csv` **bitwise, to all 17 printed digits**. That is the
comparison that actually tests the tie rule, the rank offset and the dump.

The resolved rule, now documented in `shared/metrics.py`:

* **rank is 1-based** — `1 <= rank <= n_candidates + 1`
* **ties are pessimistic** — the comparison `pos_pred <= pred` is non-strict and
  `strict_negative_mask` zeroes the target's own slot, so every tied candidate
  counts against the true answer
* **`n_candidates` excludes the target**

`mrr`, `mr` and `hits@10_50` are order-dependent and disagreed on some graphs by
1–3 float32 ulps. That is float32 associativity inside `torch.Tensor.mean`, not
a metric disagreement: `math.fsum` over the identical summands returns the value
`metrics.py` gives, so ULTRA's own reduction is the one an ulp off. Expect the
same shape of residual on GPU, with a different set of graphs landing exactly,
since CUDA's block-tree reduction differs again from torch's CPU cascade.

## What is still owed

1. **Run all 41 graphs on the GPU** and regenerate the report. Four never
   completed here: `ILPC2022:large`, `HM:indigo`, `WKIngram:50`, `WKIngram:100`.
2. **Criterion B is unanswered.** Group means need all 18 and all 23. For
   reference only — CPU, incomplete, *not* a result:

   | group | coverage | MRR | target | Hits@10 | target |
   | --- | --- | --- | --- | --- | --- |
   | inductive (e) | 16/18 | 0.4228 | 0.420 | 0.5729 | 0.562 |
   | inductive (e,r) | 21/23 | 0.3587 | 0.344 | 0.5305 | 0.511 |

   Both missing `ind_e` graphs are large and both missing `ind_er` graphs are
   WKIngram, so these will move. Do not read them as near-misses.
3. **Build the container.** It could not be built here on three independent
   counts: no Docker daemon, the registry blob CDN 403s, and both
   `download.pytorch.org` and `data.pyg.org` are blocked by egress policy. On a
   machine with a working daemon and open egress this should be the first thing
   attempted, because it also validates that ULTRA's pinned wheels still exist.

## Running it

```bash
scripts/prepare_ultra_workdir.sh                       # or use the container
ULTRA_EXTRA_ARGS=--skip_valid scripts/run_ultra.sh ind_e   '[0]'
ULTRA_EXTRA_ARGS=--skip_valid scripts/run_ultra.sh ind_er  '[0]'
scripts/collect_results.sh
scripts/environment.py --ckpt repos/ultra/ckpts/ultra_3g.pth
python scripts/make_report.py --csv-glob 'results/ultra_results_*.csv'
```

Several workers can be given the same list; they claim graphs atomically
(`ranks/.claims`, `mkdir`) and divide the suite between them. On a GPU, one
worker per device is the sane default — the claim mechanism exists for the
CPU case and for resuming, not for oversubscribing a GPU.

`--skip_valid` is optional and off by default. It skips the validation-split
evaluation, which feeds no reported metric; it was worth roughly a 2x speedup on
CPU and matters much less on a GPU. It is proven not to change any test rank
(`scripts/verify_patch_neutrality.sh`), but if you would rather run stock
behaviour, just drop it.

## Checks worth re-running on the new machine

```bash
scripts/verify_downloads.py --ultra /path/to/patched/ultra --root data/roots/ultra
scripts/verify_patch_neutrality.sh FB15k237Inductive:v1
scripts/verify_rank_dump.py --ultra /path/to/patched/ultra --root data/roots/ultra
```

The first one is not optional. PyG's `download_url` does not verify
`Content-Length`, and in this session INDIGO-BM's inference graph arrived
1.7 MB short. It surfaced only because the truncation landed mid-record; on a
line boundary it would have silently removed part of the graph, and criterion A
would still have passed, because it compares two computations over the *same*
corrupted input.

## Known upstream defects, already worked around

* `-d Metafam` and `-d FBNELL` raise `AssertionError`. `run_many.py` leaves the
  `version` template variable unset, jinja renders it as the string `"None"`,
  and `MTDEAInductive.__init__` asserts before normalising. Use
  `Metafam:Metafam` and `FBNELL:FBNELL_v1`; `shared/suite.py` already does.
* Stock `run_many.py` `os.chdir`s into `cfg.output_dir` without creating it, so
  a first run on a clean machine dies with `FileNotFoundError`. Patch 0002
  routes `output_dir` somewhere the runner creates.

## Note on the local environment left behind

The CPU environment used here is at `/home/user/kgfm-cpu` (Python 3.9.25, torch
2.1.0, PyG 2.4.0, torch-scatter 2.1.2 built from sdist) and the patched tree at
`/home/user/ultra-run`. Both are outside the repo and disappear with the
container. `scripts/prepare_ultra_workdir.sh` rebuilds the tree; the environment
is only reproducible from `environment.json`'s `pip_freeze`, and on a GPU box
you should be installing ULTRA's actual CUDA pins instead.
