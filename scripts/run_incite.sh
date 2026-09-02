#!/usr/bin/env bash
# Zero-shot INCITE over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_incite.sh <ind_e|ind_er|transductive> [gpus] [python]
#
# Modeled on scripts/run_crest.sh: same provenance guard against mixing CPU
# and GPU ranks in one directory, same atomic claim per graph, same per-graph
# timing record. The driver is incite/run.py; support precompute runs inside
# the timed python invocation, so TIMINGS.jsonl carries it (PLAN amendment).
#
# Knobs (forwarded into the container by scripts/docker_run.sh -- every new
# one MUST be added to the forwarding list there the day it is born):
#   INCITE_CKPT       REQUIRED. Checkpoint path, or the literal "none" for a
#                     random-weight smoke of the dump path. Random-weight
#                     ranks must never remain under ranks/incite/.
#   INCITE_CONFIG     config yaml; default configs/incite_v1.yaml
#   INCITE_TASK       entity (default) | relation; relation dumps to
#                     ranks-relation/incite with the unfiltered protocol
#   INCITE_SUPPORT    build (default) | skip
#   INCITE_DATA       processed dataset root; default data/roots/trix
#                     (processed roots are shared across models)
#   INCITE_DATASETS   comma list of run_ids, defaults to the whole group
#   INCITE_SHARD      shard tag for output/claims bookkeeping
#   INCITE_REDO       rerun graphs that already have a parquet
#   INCITE_EXTRA_ARGS appended verbatim to incite/run.py
#   INCITE_WORKDIR    prepared work tree (prepare_incite_workdir.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_incite.sh <ind_e|ind_er|transductive> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python}"

# Default to a freshly prepared work tree, never the baked image copy: the
# bake matches the branch only at build time, and a stale bake ran once
# unnoticed (the old /kgfm default). Preparation costs seconds.
WORKDIR="${INCITE_WORKDIR:-$ROOT/output/incite-run}"
if [ "$WORKDIR" = "$ROOT/output/incite-run" ]; then
  "$ROOT/scripts/prepare_incite_workdir.sh" "$WORKDIR"
fi
if [ -d "$WORKDIR/repos/trix" ]; then
  TRIXDIR="$WORKDIR/repos/trix"
else
  TRIXDIR="$WORKDIR/trix"
fi
CKPT="${INCITE_CKPT:?set INCITE_CKPT to a checkpoint path, or to 'none' for a random-weight smoke (never kept)}"
CONFIG="${INCITE_CONFIG:-$ROOT/configs/incite_v1.yaml}"
TASK="${INCITE_TASK:-entity}"
SUPPORT="${INCITE_SUPPORT:-build}"
DATA_ROOT="${INCITE_DATA:-$ROOT/data/roots/trix}"
SUPPORT_ROOT="$ROOT/data/roots/incite/support"
if [ "$TASK" = "relation" ]; then
  RANKS="$ROOT/ranks-relation/incite"
  CLAIMS="$ROOT/ranks-relation/.claims-incite"
else
  RANKS="$ROOT/ranks/incite"
  CLAIMS="$ROOT/ranks/.claims-incite"
fi
# Lever evals (phase 2) dump beside the phase-1 ranks, never over them:
# one directory per model variant, like one directory per model.
if [ -n "${INCITE_RANKS:-}" ]; then
  RANKS="$INCITE_RANKS"
  CLAIMS="$(dirname "$RANKS")/.claims-$(basename "$RANKS")"
fi
OUT="$ROOT/output/incite/${INCITE_SHARD:-all}"

case "$GROUP" in
  transductive) TASK_NAME="TransductiveInference" ;;
  ind_e|ind_er) TASK_NAME="InductiveInference" ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

DATASETS="${INCITE_DATASETS:-$($PY "$ROOT/shared/suite.py" "$GROUP")}"
SHARD="${INCITE_SHARD:-}"

mkdir -p "$DATA_ROOT" "$SUPPORT_ROOT" "$RANKS" "$OUT"

echo "group     : $GROUP"
echo "task      : $TASK"
echo "config    : $CONFIG"
echo "ckpt      : $CKPT"
echo "support   : $SUPPORT"
echo "data root : $DATA_ROOT"
echo "ranks     : $RANKS"
echo "datasets  : $(tr ',' '\n' <<<"$DATASETS" | wc -l)${SHARD:+ (shard $SHARD)}"

# Ranks from a CPU run and ranks from a GPU run must never end up in one
# directory: the float32 kernels differ in low-order bits, which can flip a
# near-tie and move a rank, so a mixed directory yields a group mean that
# corresponds to no single measurement and nothing downstream would notice.
DEVICE=cpu; [ "$GPUS" != "null" ] && DEVICE=gpu
PROV="$RANKS/PROVENANCE.json"
if [ -f "$PROV" ]; then
  had=$($PY -c "import json;print(json.load(open('$PROV'))['device'])" 2>/dev/null || echo unknown)
  if [ "$had" != "$DEVICE" ] && [ -z "${INCITE_REDO:-}" ]; then
    echo "REFUSING TO RUN: $RANKS holds $had ranks, this run is $DEVICE." >&2
    echo "Clear it first:  rm -rf $RANKS $CLAIMS" >&2
    exit 3
  fi
fi
$PY - "$PROV" "$DEVICE" "$CKPT" "$CONFIG" "$TASK" "$SUPPORT" <<'PROVPY'
import json, os, platform, subprocess, sys
path, device, ckpt, config, task, support = sys.argv[1:7]
prov = {"device": device, "host": platform.platform(), "cpu_count": os.cpu_count(),
        "seed": 1024, "checkpoint": os.path.basename(ckpt),
        "random_weights": ckpt.lower() == "none",
        "config": os.path.basename(config), "task": task, "support": support,
        # the relation task ranks with the unfiltered protocol (run.py docstring)
        "protocol": "unfiltered" if task == "relation" else "filtered"}
# which stack produced this (2026-09-02: a CUDA 12.8 image exists beside
# the cu118 one; dumps from the two must never be confused)
prov["image"] = os.environ.get("KGFM_IMAGE")
try:
    import torch
    prov["torch"] = torch.__version__
    prov["cuda"] = torch.version.cuda
except Exception:
    prov["torch"] = prov["cuda"] = None
try:
    prov["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    prov["gpu"] = None
json.dump(prov, open(path, "w"), indent=2, sort_keys=True)
PROVPY

mkdir -p "$CLAIMS"
# Reclaim stale claims (2026-09-01): a claim without a parquet belongs to a
# worker that died or was stopped mid-graph (the plan v3 takeover stopped
# one, and E2 then skipped that graph as "taken"). Only one worker runs
# per ranks dir in this project, so a parquet-less claim is always stale.
for c in "$CLAIMS"/*/; do
  [ -d "$c" ] || continue
  id="$(basename "$c")"
  [ -s "$RANKS/$id.parquet" ] || { rmdir "$c" && echo "reclaimed stale claim: $id"; }
done
export TRIX_ROOT="$TRIXDIR"
export PYTHONPATH="$WORKDIR:$ROOT/shared${PYTHONPATH:+:$PYTHONPATH}"
# python -m puts the *current directory* ahead of PYTHONPATH, and the image's
# working directory carries the incite/ baked in at build time. Run from the
# work tree so the intended copy is the one that imports -- sys.path[0] beats
# PYTHONPATH; this exact shadowing caused two CREST bugs (a 14.85 GiB OOM and
# an eval bug).
cd "$WORKDIR"

ran=0; skipped=0; failed=0
for d in ${DATASETS//,/ }; do
  id="$($PY -c "import sys;sys.path.insert(0,'$ROOT/shared');import suite;print(suite.by_run_id('$d').id.replace(':','_'))")"
  if [ -z "${INCITE_REDO:-}" ] && [ -s "$RANKS/$id.parquet" ]; then
    skipped=$((skipped+1)); continue
  fi
  if ! mkdir "$CLAIMS/$id" 2>/dev/null; then
    skipped=$((skipped+1)); continue
  fi

  # run_id carries the spelling upstream needs, which is not always the suite
  # id: Metafam is Metafam:Metafam and FBNELL is FBNELL:FBNELL_v1.
  DS="${d%%:*}"
  VERSION="${d#*:}"
  [ "$VERSION" = "$d" ] && VERSION=""

  echo ">>> $d"
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  index=$((ran + failed + 1))
  t0=$(date +%s.%N)
  if INCITE_RANK_DUMP_DIR="$RANKS" $PY -m incite.run \
      -c "$CONFIG" \
      --dataset "$DS" \
      --version "$VERSION" \
      --gpus "$GPUS" \
      --task_name "$TASK_NAME" \
      --task "$TASK" \
      --ckpt "$CKPT" \
      --support "$SUPPORT" \
      --support_root "$SUPPORT_ROOT" \
      --output_dir "$OUT" \
      --data_root "$DATA_ROOT" \
      --seed 1024 \
      ${INCITE_EXTRA_ARGS:-}; then
    status=ok; ran=$((ran+1))
  else
    status=failed
    failed=$((failed+1)); rmdir "$CLAIMS/$id" 2>/dev/null || true
    echo "!!! FAILED: $d"
  fi
  t1=$(date +%s.%N)
  $PY - "$RANKS/TIMINGS.jsonl" "$id" "$d" "$DEVICE" "$status" "$started" "$t0" "$t1" "$index" <<'TIMEPY'
import json, sys
path, gid, run_id, device, status, started, t0, t1, index = sys.argv[1:10]
with open(path, "a") as handle:
    handle.write(json.dumps({
        "dataset": gid, "run_id": run_id, "model": "incite", "device": device,
        "status": status, "started": started, "seconds": round(float(t1) - float(t0), 3),
        "index": int(index),
    }, sort_keys=True) + "\n")
TIMEPY
done
echo "worker ${SHARD:-$GROUP} done: $ran ran, $skipped skipped, $failed failed"
