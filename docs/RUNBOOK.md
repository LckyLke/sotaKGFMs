# Runbook — ULTRA baseline

Everything below assumes the workspace root as the working directory.

## 0. Clone and pin

```bash
scripts/clone_repos.sh
```

Reads `repos/PINS.json` and checks each clone out **detached** at its pinned
SHA. No branch is ever tracked. Re-running is idempotent.

## 1. Build the container

```bash
docker build -f containers/ultra/Dockerfile -t kgfm/ultra:427966ad .
```

Build context is the workspace root, not `containers/ultra/`. The image copies
`repos/ultra` and `patches/ultra`, applies every patch with `patch -p1`, and
bakes `shared/` onto `PYTHONPATH` so the rank dump can import `suite`.

Base is `nvidia/cuda:11.8.0-devel-ubuntu22.04`. **Devel, not runtime**: `rspmm`
is JIT-built by `torch.utils.cpp_extension` at first use and needs `nvcc`.

## 2. Warm the cache — one dataset, alone

```bash
docker run --rm --gpus '"device=0"' \
  -v "$PWD/data:/kgfm/data" -v "$PWD/ranks:/kgfm/ranks" \
  kgfm/ultra:427966ad \
  python3.9 script/run_many.py \
    -c /kgfm/repos/ultra/config/inductive/inference.yaml \
    --gpus '[0]' --ckpt /kgfm/repos/ultra/ckpts/ultra_3g.pth \
    --rank_dump_dir /kgfm/ranks/ultra \
    -d FB15k237Inductive:v1
```

This compiles `rspmm` (a few minutes, cached afterwards) and downloads and
processes that one dataset. **Do this before launching anything in parallel.**
Parallel first launches race twice over: on the extension build directory, and
on the dataset download and `processed/data.pt` write.

## 3. Run the groups

```bash
scripts/run_ultra.sh ind_e          # 18 graphs
scripts/run_ultra.sh ind_er         # 23 graphs
```

Both resolve their dataset list from `shared/suite.py` — the list is not
repeated in the script.

Three things that bite:

* **Use full paths for `-c` and `--ckpt`.** `run_many.py` calls
  `create_working_directory()`, which `os.chdir`s into a fresh timestamped
  directory for every dataset. Anything relative resolves against a different
  directory from the second dataset onward. `--rank_dump_dir` has the same
  requirement. `scripts/run_ultra.sh` passes absolutes throughout.
* **`ultra_3g.pth`, never `ultra_50g.pth`.** The 50g checkpoint is trained on 50
  graphs, most of this suite among them, so nothing measured with it is
  zero-shot.
* **Confirm `--epochs 0` in the logs.** With neither `-ft` nor `-tr`,
  `run_many.py` sets `epochs, batch_per_epoch = 0, 'null'`; the config dump at
  the top of each dataset's log must read `'num_epoch': 0`. Grep for it:
  `grep -c "'num_epoch': 0" logs/ind_e.log` should equal the dataset count.

`Metafam` and `FBNELL` must be spelled `Metafam:Metafam` and
`FBNELL:FBNELL_v1`. Bare, `run_many.py` never sets the `version` template
variable, jinja renders it as the *string* `"None"`, and
`MTDEAInductive.__init__` asserts `version in self.versions` before anything
normalises it. `suite.run_id` carries the correct spelling; `suite.by_run_id`
maps it back.

## 4. Metrics

```bash
python shared/analyse.py --ranks ranks/ultra --ultra-csv <ultra_results_*.csv>
```

Criterion A compares `shared/metrics.py` over the dumped ranks against ULTRA's
own CSV, per dataset, at printed precision. If they disagree, either the dump is
wrong or the tie rule differs — fix `metrics.py`, never ULTRA. Criterion B
compares unweighted group means against the ULTRA repository's PyG figures.

## 5. Record the environment

```bash
scripts/environment.py --ckpt repos/ultra/ckpts/ultra_3g.pth
```

## Data

```bash
scripts/mirror_data.py --ultra /path/to/patched/ultra
```

Copies every raw download into `data/raw/<repo>/` read-only with a sha256 each,
and records the URL each suite graph declares — so a link that dies later is
still on record. `data/roots/<repo>/` stays per-repo: PyG keys the cached
relation graph in `processed/data.pt` by directory, not by which `pre_transform`
built it.
