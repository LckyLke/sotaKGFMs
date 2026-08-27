#!/usr/bin/env bash
# Pretrain CREST's readout (stage A) or the full model (stage B).
#
#   usage: scripts/train_crest.sh <a|b> [gpus] [python]
#
# Modeled on scripts/run_crest.sh: same work-tree resolution, same provenance
# record, seed 1024. No suite loop -- training runs once over the graph mix
# (TRIX's own pretraining mix by default) via crest/pretrain.py, which owns
# alternation, validation, and checkpointing. Checkpoints and logs land under
# output/crest-pretrain/stage<a|b>/; the best checkpoint is what
# CREST_READOUT_CKPT then feeds to scripts/run_crest.sh.
#
# Knobs (forwarded into the container by scripts/docker_run.sh):
#   CREST_WORKDIR          prepared work tree (prepare_crest_workdir.sh)
#   CREST_TRAIN_GRAPHS     comma list; pretrain names or Dataset:version specs
#   CREST_TRAIN_STEPS      override configs/crest_v1.yaml train.steps
#   CREST_VAL_INTERVAL     override train.val_interval
#   CREST_VAL_SAMPLES      override train.val_samples
#   CREST_READOUT_CKPT     readout init (stage B starts from stage A's best)
#   CREST_TRAIN_EXTRA_ARGS appended verbatim to crest/pretrain.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:?usage: train_crest.sh <a|b> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python}"

case "$STAGE" in
  a|b) ;;
  *) echo "unknown stage: $STAGE (want a or b)" >&2; exit 2 ;;
esac

WORKDIR="${CREST_WORKDIR:-/kgfm}"
if [ -d "$WORKDIR/repos/trix" ]; then
  TRIXDIR="$WORKDIR/repos/trix"
else
  TRIXDIR="$WORKDIR/trix"
fi
CKPT="$TRIXDIR/entity_prediction.pth"
CONFIG="$ROOT/configs/crest_v1.yaml"
DATA_ROOT="$ROOT/data/roots/crest"
RAW_ROOT="$ROOT/data/raw/ultra-pretrain"
OUT="$ROOT/output/crest-pretrain/stage$STAGE"

mkdir -p "$OUT"

DEVICE=cpu; [ "$GPUS" != "null" ] && DEVICE=gpu
echo "stage     : $STAGE"
echo "config    : $CONFIG"
echo "trix ckpt : $CKPT"
echo "readout   : ${CREST_READOUT_CKPT:-<fresh, zero residual>}"
echo "graphs    : ${CREST_TRAIN_GRAPHS:-<config mix>}"
echo "output    : $OUT"

# The same record run_crest.sh keeps: which device, seed and weights produced
# what is in this directory, so a checkpoint is never orphaned from its setup.
$PY - "$OUT/PROVENANCE.json" "$DEVICE" "$CKPT" "${CREST_READOUT_CKPT:-}" \
    "${CREST_TRAIN_GRAPHS:-}" "$STAGE" <<'PROVPY'
import json, os, platform, subprocess, sys
path, device, ckpt, readout, graphs, stage = sys.argv[1:7]
prov = {"device": device, "host": platform.platform(), "cpu_count": os.cpu_count(),
        "seed": 1024, "stage": stage, "checkpoint": os.path.basename(ckpt),
        "readout_checkpoint": os.path.basename(readout) if readout else None,
        "graphs": graphs or "config mix"}
try:
    prov["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    prov["gpu"] = None
json.dump(prov, open(path, "w"), indent=2, sort_keys=True)
PROVPY

export TRIX_ROOT="$TRIXDIR"
export PYTHONPATH="$WORKDIR:$ROOT/shared${PYTHONPATH:+:$PYTHONPATH}"
# python -m puts the *current directory* ahead of PYTHONPATH, and the image's
# working directory carries the crest/ baked in at build time. Run from the
# prepared work tree so the freshly copied package is the one that imports.
cd "$WORKDIR"

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
t0=$(date +%s.%N)
$PY -m crest.pretrain \
    -c "$CONFIG" \
    --stage "$STAGE" \
    --gpus "$GPUS" \
    --ckpt "$CKPT" \
    ${CREST_READOUT_CKPT:+--readout_ckpt "$CREST_READOUT_CKPT"} \
    ${CREST_TRAIN_GRAPHS:+--graphs "$CREST_TRAIN_GRAPHS"} \
    ${CREST_TRAIN_STEPS:+--steps "$CREST_TRAIN_STEPS"} \
    ${CREST_VAL_INTERVAL:+--val_interval "$CREST_VAL_INTERVAL"} \
    ${CREST_VAL_SAMPLES:+--val_samples "$CREST_VAL_SAMPLES"} \
    --data_root "$DATA_ROOT" \
    --raw_root "$RAW_ROOT" \
    --output_dir "$OUT" \
    --seed 1024 \
    ${CREST_TRAIN_EXTRA_ARGS:-} 2>&1 | tee -a "$OUT/train.log"
status=${PIPESTATUS[0]}
t1=$(date +%s.%N)
$PY - "$OUT/TIMINGS.jsonl" "$STAGE" "$DEVICE" "$status" "$started" "$t0" "$t1" <<'TIMEPY'
import json, sys
path, stage, device, status, started, t0, t1 = sys.argv[1:8]
with open(path, "a") as handle:
    handle.write(json.dumps({
        "stage": stage, "model": "crest", "device": device,
        "status": "ok" if status == "0" else "failed",
        "started": started, "seconds": round(float(t1) - float(t0), 3),
    }, sort_keys=True) + "\n")
TIMEPY
exit "$status"
