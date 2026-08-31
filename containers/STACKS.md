# Per-repo stacks, extensions and runners

One container per repo. Each repo runs on the stack its authors pinned; nothing
is shared between containers except the four cross-repo artifacts
(`data/raw/`, `shared/suite.py`, `ranks/`, `shared/metrics.py`).

All facts below were read out of the pinned clones in `repos/` at the SHAs in
`repos/PINS.json`, not from memory: `requirements.txt`, the README install
block, and the runner scripts themselves.

---

## Summary

| Repo | Python | torch | CUDA | PyG | Compiled extension | Needs `nvcc` / devel base |
| --- | --- | --- | --- | --- | --- | --- |
| `ultra` | 3.9 | 2.1.0 | 11.8 | 2.4.0 | `rspmm` (JIT, `cpp_extension.load`) | **yes** |
| `motif` | 3.9 (implied) | 2.1.0 | 11.8 | 2.4.0 | `rspmm` + **Triton** kernel | **yes** |
| `trix` | unpinned | >=2.1.0 | unpinned | >=2.4.0 | `rspmm` (JIT) | **yes** |
| `flock` | 3.11 (base image) | 2.8.0 | 12.6 | 2.4.0 | `graph-walker` (pybind11, ahead-of-time) + `rspmm` in `src_synthetic` | **yes** |
| `semma` | 3.9 | 2.1.0 | 11.8 | >=2.4.0 | `rspmm` (JIT) | **yes** |
| `kg-icl` | 3.9 | 2.2.0 | 11.8 | 2.4.0 | `rspmm` (JIT, **optional** — `use_rspmm` defaults to `False`) | no (optional) |
| `kgpfn` | 3.12 | 2.5.1 | 12.1 | 2.7.0 | `rspmm` x2 (JIT) + `flash-attn` (prebuilt wheel) | **yes** |

The three flagged in the task brief are confirmed: ULTRA `rspmm`, MOTIF's Triton
kernel, FLOCK's `graph-walker`. Two more turned up that also need a devel base:
TRIX and SEMMA both carry ULTRA's `rspmm` verbatim, and KGPFN carries two copies
of it plus a `flash-attn` wheel pinned to a specific CUDA/torch/ABI triple.

---

## Runner entry points and ULTRA-fork status

| Repo | Entry point(s) | ULTRA fork? |
| --- | --- | --- |
| `ultra` | `script/run_many.py`, `script/run.py` | reference |
| `motif` | `script/run_many.py`, `script/run.py` | **yes**, direct |
| `semma` | `script/run_many.py`, `script/run.py` | **yes**, direct |
| `trix` | `src/run_entity.py`, `src/run_relation.py` | **yes**, restructured |
| `flock` | `scripts/{entity,relation}_{zeroshot,finetune}.sh` driving `src_entity/run_many.py` and `src_relation/run.py` | derived, restructured |
| `kg-icl` | `src/evaluation.py` via `shell/test.sh` | **no — unverified** |
| `kgpfn` | `script/test_kgpfn.py` | **no — unverified** |

"Unverified" means: no shared lineage with ULTRA's evaluation loop, so the
rank definition, the filtering graph and the tie rule must be established from
that repo's own code before any of its numbers may be compared to ULTRA's.
Evidence for the classification is in the per-repo sections below.

---

## `ultra` — DeepGraphLearning/ULTRA @ `427966ad`

Reference implementation. Everything else is validated against this.

* **Python 3.9**, torch **2.1.0** + **cu118**, torch-scatter **2.1.2**,
  torch-sparse **0.6.18**, torch-geometric **2.4.0**, plus `ninja easydict pyyaml`.
  `requirements.txt` only states floors (`torch>=2.1.0`); the exact versions come
  from the README install block, which is what the container pins.
* `export CUDA_HOME=/usr/local/cuda-11.8/`.
* **Compiled extension:** `ultra/rspmm/` — relational sparse matrix multiply,
  JIT-built at first use through `torch.utils.cpp_extension.load` from
  `source/rspmm.cpp` and `source/rspmm.cu`. Compiled once, then cached. The
  README is explicit that a **devel** base image is required, since the runtime
  images have no `nvcc`. There is a working CPU code path (`rspmm_*_cpu`),
  which is what makes CPU-only inference possible at all.
* **Entry points:** `script/run.py` (one dataset), `script/run_many.py` (a list,
  appends to `ultra_results_<timestamp>.csv`), `script/pretrain.py`.
  `run_many.py` imports `train_and_validate` and `test` from `script/run.py`, so
  a patch to `run.py::test` covers both runners.
* **Config:** `config/inductive/inference.yaml`, `config/transductive/inference.yaml`.
  Both are jinja templates over `dataset, version, gpus, epochs, bpe, ckpt`.

## `motif` — HxyScotthuang/MOTIF @ `aab68802`

Direct ULTRA fork: `script/run_many.py` differs from ULTRA's by ~30 lines
(package renamed `ultra` -> `motif`, `MOTIF` model added alongside `Ultra`,
per-dataset epoch/batch table retuned, ILPC handling changed).

* Same install block as ULTRA — torch **2.1.0** + **cu118**, torch-scatter 2.1.2,
  torch-sparse 0.6.18, PyG 2.4.0 — **plus `triton-nightly`**. (The README line
  reads `pip triton-nightly`, missing `install`; that typo needs a patch.)
* **Compiled extensions, two of them:**
  * `motif/rspmm/` — ULTRA's kernel but *modified* (`rspmm.cpp` differs from
    ULTRA's by content hash), so it must be built from MOTIF's own sources.
  * `motif/rspmm/triton_rspmm.py` — a Triton kernel, imported lazily from
    `motif/layers.py:196` (`RelConvSumAggr`) and `:300` (`HyperRelConvSumAggr`,
    `HyperRelConvMeanAggr`). This is the higher-order motif path.
* Needs `nvcc` and a devel base for the same reason as ULTRA; Triton additionally
  compiles at runtime and wants a matching CUDA toolchain.

## `trix` — yuchengz99/TRIX @ `7596e14e`

ULTRA fork, restructured. `src/trix/rspmm/source/rspmm.cpp` is **byte-identical**
to ULTRA's, so the lineage is unambiguous, but the runners were split.

* `requirements.txt`: `torch>=2.1.0`, `torch-scatter>=2.1.2`,
  `torch-geometric>=2.4.0`, `ninja easydict pyyaml`, **`google-generativeai`**.
  No Python or CUDA pin anywhere; README is just `pip install -r requirements.txt`.
  The container has to pick a version — pin it to ULTRA's (3.9 / 2.1.0 / cu118),
  which satisfies every floor, and record that as our choice, not theirs.
* **Compiled extension:** `src/trix/rspmm/` (JIT, needs `nvcc`).
* **Entry points:** `src/run_entity.py` and `src/run_relation.py` — two separate
  tasks (entity prediction, relation prediction) with separate checkpoints
  (`entity_prediction.pth`, `relation_prediction.pth`, both committed in-tree)
  and separate configs (`config/run_entity_{transductive,inductive}.yaml`).
  Only `run_entity.py` is comparable to ULTRA's entity-prediction numbers.

## `flock` — jw9730/flock @ `f35103d2`

Derived from the ULTRA/TRIX/MOTIF line — `src_synthetic/` vendors all three
(`src_synthetic/ultra/rspmm/source/rspmm.cpp` is byte-identical to ULTRA's) —
but the evaluation path is its own.

* **Base image the authors name:** `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel`
  (so torch **2.8.0**, **CUDA 12.6**, Python 3.11 from that image).
* `pip3 install torch_geometric==2.4.0 easydict pybind11 pyyaml jinja2`, then
  `torch-scatter` from `data.pyg.org/whl/torch-2.8.0+cu126.html`, then
  `bash install_walker.sh`.
* **Compiled extension:** `graph-walker/` — a pybind11 C++ extension (`_walker`,
  built ahead of time from `graph-walker/src/*.cpp` with `-O3 -std=c++11`,
  installed editable via `pip3 install -e .`). Not JIT: it must be built at
  image build time. Also depends on `networkx numpy scipy pybind11 scikit-learn`.
* **Entry points:** shell scripts, not a Python list argument.
  `scripts/entity_zeroshot.sh` loops `src_entity/run_many.py` one dataset per
  line, with a *different config per dataset* selecting the walk count and
  ensemble size (`n16_ensemble16` ... `n512_ensemble2`). `CKPT` is read from the
  environment. `scripts/relation_zeroshot.sh` drives `src_relation/run.py`
  instead, with explicit `--dataset`/`--version`/`--epochs 0`/`--bpe null`.
  The per-dataset config choice is part of the published protocol and cannot be
  collapsed into a single command line.

## `semma` — arvindh75/semma @ `ed525ac1`

Direct ULTRA fork. `ultra/rspmm/source/rspmm.cpp` byte-identical to ULTRA's; the
package is still called `ultra`. `script/run_many.py` differs from ULTRA's mainly
by routing config loading through a new `ultra/parse.py` and reading a
repo-level `flags.yaml`, plus absolute-path handling for the working directory.

* **Python 3.9**, torch **2.1.0**, PyG **>=2.4.0** (README: "PyTorch 2.1 and
  PyTorch-Geometric 2.4 ... CUDA 11.8 or later").
* `requirements.txt` adds the semantic half of the model: `transformers`,
  `sentence-transformers`, `einops`, `nltk`, `python-dotenv`, `httpx`, `aiohttp`,
  and **`numpy==1.24.1`** (a hard pin — the only one in the file).
  `flash-attn` is commented out with a note that it is awkward to install.
* `setup.sh` additionally `wget`s `fb_mid2name.tsv` from a **Google Drive** link
  and warns that it may arrive as a zip. That download is fragile and is a
  likely failure point; it also reaches a host outside the dataset mirror.
* **Compiled extension:** `ultra/rspmm/` (JIT, needs `nvcc`).
  Note the clone ships a stale `__pycache__/rspmm.cpython-39.pyc` — confirmation
  that upstream ran this on Python 3.9, and a file to delete at image build.
* **Entry points:** `script/run_many.py`, `script/run.py` — same CLI shape as ULTRA.

## `kg-icl` — nju-websoft/KG-ICL @ `6a3166e3`

**Not an ULTRA fork.** Its `src/rspmm/source/rspmm.cpp` differs from ULTRA's by
content hash, there is no `ultra`-shaped package, no `run_many.py`, and the
evaluation loop is its own. Treat every number from it as unverified until its
rank definition has been read directly.

* **Python 3.9** ("Please use python 3.9 to run the code"), torch **2.2.0** +
  **cu118**, torch-scatter 2.1.2, torch-sparse 0.6.18, PyG 2.4.0,
  `ninja easydict pyyaml tqdm`, and **`numpy==1.24.0`** if the default numpy
  conflicts. No `requirements.txt` — the README install block is the only spec.
* **Compiled extension:** `src/rspmm/` — **optional**. Upstream changed
  `use_rspmm` to default `False` in `evaluation.py` precisely because people
  could not build it, so this container can be built on a runtime base. Whether
  the kernel is on or off is a numerical choice and must be recorded per run.
* **Entry points:** `src/evaluation.py` (`--checkpoint_path --test_dataset_list
  --gpu --n_layer --hidden_dim --MSG --attn_dim --shot --act --note`), driven by
  `shell/test.sh` over a hardcoded 43-dataset list with its own dataset naming
  (`fb237_v1_ind`, `FB-25`, `WD-singer`, ...) that does **not** match ULTRA's
  class names. A name mapping into `shared/suite.py` ids will be required.
* **Data:** ships `datasets.zip` in-tree plus a `datasets/process.sh` step; it
  does not use ULTRA's downloaders, so `data/raw/` mirroring does not apply
  directly.

## `kgpfn` — HKUST-KnowComp/KGPFN @ `af415c33`

**Not an ULTRA fork.** ULTRA appears only as a vendored baseline under
`model/ultra/`; the model itself is `pfn/` + `model/kgpfnsem.py` and a
tabular prior-fitted network (`tabicl` / `tabpfn` / `limix`) as feature
transformer. Verified 2026-08-31 while building the container:
`pfn/tasks.py::compute_ranking` and `::strict_negative_mask` are
byte-identical to ULTRA's (1-based, pessimistic ties, strict filter, target
excluded), so the tie rule and offset carry over. Its filter GRAPH omits the
validation targets ULTRA's protocol filters on ILPC/Ingram, and it asks the
head question as the inverse-relation tail question -- both handled by the
dual-column rank dump; see patches/kgpfn/0001. Its five tail-only names in
`pfn/tasks.py::TAIL_ONLY_DATASETS` match `shared/suite.py`'s five.

* **Python 3.12** (`conda create -n kgpfn python=3.12`), torch **2.5.1+cu121**
  (with matching torchvision/torchaudio), PyG **2.7.0**, torch-scatter
  `2.1.2+pt25cu121`, torch-sparse `0.6.18+pt25cu121`, torch-cluster
  `1.6.3+pt25cu121`, `pyg-lib 0.4.0+pt25cu121`.
  This is the only repo with a fully pinned `requirements.txt` — every line is
  `==`, including `transformers==5.4.0`, `numpy==2.4.3`, `pandas==3.0.2`,
  `scipy==1.17.1`, `scikit-learn==1.8.0`.
* **`flash-attn` is required** (for TabICL/LimiX) and is installed from a
  **prebuilt wheel URL** matched to CUDA/torch/cxx11-ABI/cpython version. The
  README's example wheel (`cu12torch2.7...cp312`) does **not** match the pinned
  torch 2.5.1 — the correct `cu12torch2.5` wheel has to be selected at build
  time. Resolved: the container installs
  `flash_attn-2.8.0.post2+cu12torch2.5cxx11abiFALSE-cp312` (same release
  family as the README's example, torch2.5 build, old-C++-ABI to match the
  PyPI/download.pytorch.org torch 2.5.1 binaries); a copy plus sha256 sits in
  `data/raw/kgpfn/` + `data/raw/MANIFEST-kgpfn.json`.
* **Compiled extensions:** two copies of `rspmm` (`model/ultra/rspmm/`,
  `pfn/rspmm/`), both byte-identical to ULTRA's, both JIT — so `nvcc` and a devel
  base are needed despite the PFN framing.
* **Entry points:** `script/test_kgpfn.py -c config/script/test.yaml --gpus [0]`
  for evaluation; `script/pretrain_pfn.py` under `accelerate launch` for
  training; `script/download.py` fetches checkpoints into `./cache/`.
  Note the pin's commit subject is "Support model configs and tail-only
  evaluation" — tail-only handling exists here and must be cross-checked against
  `shared/suite.py`'s five tail-only graphs rather than assumed compatible.

---

## Egress notes that affect these builds

Verified against this session's egress policy, not assumed:

* `download.pytorch.org` — **blocked** (proxy answers 403 to CONNECT).
* `data.pyg.org` — **blocked** (403).

Every repo above installs torch and/or torch-scatter/-sparse from one of those
two hosts, so no container in this project can be built until they are allowed.
See `baseline_report.md` for what that blocked and what was done instead.
