# Handoff — 2026-08-27

Written when the SEMMA run was stopped for the night. The GPU handoff that the
previous version of this file described is complete. Nothing in it is still owed.

## State

| model | ranks | notes |
| --- | --- | --- |
| ULTRA | 41/41 GPU | criterion A passes; criterion B explained (see below) |
| MOTIF | 41/41 GPU | reproduces its own paper to mean abs delta 0.0007 MRR |
| TRIX | 41/41 GPU | reproduces its own paper to mean abs delta 0.0014 MRR |
| SEMMA | 12/41, quarantined | `ranks/semma-noflash/`, see `QUARANTINE.md` there |
| FLOCK | none | repo restored, surveyed, no patches yet |
| KG-ICL | none | repo restored, surveyed, no patches yet |
| KGPFN | none | repo restored, surveyed, no patches yet |

Group means over all 41, GPU, zero-shot, entity prediction:

| model | ind_e MRR | ind_e H@10 | ind_er MRR | ind_er H@10 |
| --- | --- | --- | --- | --- |
| ULTRA | 0.4158 | 0.5684 | 0.3421 | 0.5098 |
| MOTIF | 0.4361 | 0.5767 | 0.3491 | 0.5254 |
| TRIX | 0.4562 | 0.5931 | 0.3679 | 0.5409 |

ULTRA's published 0.430 is its **TorchDrug** implementation, not the PyG rewrite
in `repos/ultra`. That is the whole criterion B gap. `ultra_torchdrug` is a
separate repository, and pinning it as an eighth repo is on the list below.

## Task 1 — flash-attn for SEMMA

The night's run was stopped on purpose to try this. SEMMA embeds relation
descriptions with `jina-embeddings-v3`, which prints one warning per attention
layer and falls back to PyTorch native attention. Full background is in
`docs/report_notes.md`, section "SEMMA runs without flash-attn".

The hardware supports it. The GPU is an RTX 4070 Ti SUPER, compute capability
8.9, and flash-attn 2.x requires 8.0 or newer.

Steps:

1. Find a prebuilt wheel matching the container exactly: Python 3.9, torch
   2.1.0, CUDA 11.8, and the ABI that PyPI torch uses (`cxx11abiFALSE`).
   Releases are at `Dao-AILab/flash-attention`.
2. If no such wheel exists, stop and report. Do not build from source first.
   A source build of flash-attn takes hours and needs large amounts of RAM.
3. If a wheel exists, add it to `containers/semma/Dockerfile` after the torch
   layer, and extend the existing build-time stack guard to import `flash_attn`.
4. Rebuild the image. Read the explicit `BUILD_EXIT=` line, never a piped tail.
5. Run one graph and confirm that the warning is gone.

Two outcomes:

* **It works.** Re-run all 41 into a clean `ranks/semma/`. Keep
  `ranks/semma-noflash/` until the new run is complete. The pair is then a free
  sensitivity measurement of the 0.8 similarity threshold, worth one table.
* **It does not work.** Move `ranks/semma-noflash/` back to `ranks/semma/`,
  recreate `ranks/.claims-semma` from the 12 graph ids there, and resume.
  Running without flash-attn matches upstream's own `requirements.txt`, so this
  outcome costs nothing except time.

Measured cost without flash-attn: about 6 times TRIX over the same graphs, which
projects to roughly 95 minutes for all 41.

## Task 2 — FLOCK

Stack: torch 2.8.0, CUDA 12.6, Python 3.11, `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel`.
`install_walker.sh` builds `graph-walker` with pybind11. Build it ahead of time
in the image, not at run time.

Checkpoint: `checkpoints/flock_entity.pth`. Never `flock_relation.pth`.

`scripts/entity_zeroshot.sh` covers all 41 graphs, but with **four different
configs**, not one. The map is by graph size:

| config | walk_num | batch_size | test_samples |
| --- | --- | --- | --- |
| `n16_ensemble16.yaml` | 16 | 32 | 16 |
| `n32_ensemble16.yaml` | 32 | 16 | 16 |
| `n64_ensemble16.yaml` | 64 | 8 | 16 |
| `n128_ensemble16.yaml` | 128 | 4 | 16 |

Read the graph-to-config assignment out of `scripts/entity_zeroshot.sh`. Do not
retype it. A runner that uses one config for all 41 does not measure FLOCK.

Every FLOCK number is a 16-sample ensemble over random walks. `run_many.py`
seeds from the fixed list `[1024, 42, 1337, 512, 256]` indexed by repeat, so one
repeat is deterministic. Record the seed in `PROVENANCE.json`.

## Task 3 — KG-ICL

Checkpoint: `KG-ICL-6L`.

The bundled `datasets.zip` holds 55 graphs and covers only **27 of our 41**.
Absent: all four `HM`, all eight `WikiTopics`, `Metafam`, `FBNELL`.

Drop `FB15k237Inductive:v1`, `NELLInductive:v1` and `CoDExSmall` as KG-ICL's own
pretraining data. That leaves **25 comparable graphs** unless the 14 absent ones
are converted into KG-ICL's format.

Before anything else, prove that KG-ICL's bundled copy of a shared graph equals
ULTRA's copy. Compare triple counts and the entity and relation vocabularies.
Its names differ: `fb237_v1_ind` is `FB15k237Inductive:v1`.

The rank patch is decided and approved: patch KG-ICL to also compute ranks under
the shared definition, 1-based with pessimistic ties.

## Task 4 — KGPFN

Stack: Python 3.12, torch 2.5.1. flash-attn is **required**, not optional, for
the TabICL backbone. Use a prebuilt wheel.

Checkpoint: `python script/download.py --kgpfn`, the default TabICL variant.
Files land in `./cache/`.

`config/script/train_all.yaml` holds `dataset.root`, so the data-root patch is
one line.

## Task 5 — at the very end, in this order

1. Transductive sweep, 13 graphs, all seven models. `NELL995`, `CoDExSmall` and
   `CoDExLarge` answer 404 from SEMMA's URLs. Fix that first.
2. MOTIF timing re-run. The first MOTIF suite predates `TIMINGS.jsonl`.
3. `ultra_torchdrug` as an eighth pinned repo, to close criterion B.

## Rules that still hold

* `repos/` stays pristine. Every change is a `.diff` in `patches/<repo>/` with a
  stated Reason. `git -C repos/<name> status` must report clean.
* One processed root per repo, `data/roots/<repo>/`. PyG caches `pre_transform`
  output by directory, so a shared root hands one model another model's graph
  with no error and no warning.
* One `TORCH_EXTENSIONS_DIR` per repo. Every repo names its extension `rspmm`,
  so a shared directory makes concurrent runs block on one `FileBaton` lock.
* Never mix devices in one rank directory. The runners refuse this already.
* Prefetch raw files before GPU time: `scripts/prefetch_raw.py`.
* Read the explicit exit-code line from a build. A piped `tail` or `grep` hides
  docker's exit status, and a failed build then reads as a successful one.

## Loose ends worth ten minutes

* ULTRA's 41 result CSVs sit loose in `results/`, while every other model has
  `results/<model>/`. Moving them means updating the `--csv-glob` in the
  `make_report.py` call.
* `baseline_report.md` and `docs/report_notes.md` are still written around
  ULTRA alone. With four models and more coming, the reporting needs a shape
  that holds all of them. Decide that before adding a fifth.
* **Nothing is committed.** 144 files, including all 123 GPU rank dumps.
