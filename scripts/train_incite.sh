#!/usr/bin/env bash
# Pretrain INCITE on the TRIX mix with zero-shot DEV10 checkpoint selection.
#
#   usage: scripts/train_incite.sh [gpus] [python]
#
# Modeled on scripts/train_crest.sh: same work-tree resolution, same
# provenance record, seed 1024. incite/pretrain.py owns alternation,
# validation (zero-shot DEV10, one mean per suite group -- PLAN lesson 2),
# and checkpointing. Checkpoints and logs land under
# output/incite-pretrain/; the best checkpoint is what INCITE_CKPT then
# feeds to scripts/run_incite.sh.
#
# Knobs (forwarded into the container by scripts/docker_run.sh):
#   INCITE_CONFIG           config yaml; default configs/incite_phase1.yaml
#                           (phase order: the reduction gate trains first)
#   INCITE_WORKDIR          prepared work tree (prepare_incite_workdir.sh)
#   INCITE_TRAIN_GRAPHS     comma list of pretrain mix names
#   INCITE_TRAIN_STEPS      override train.steps
#   INCITE_VAL_INTERVAL     override train.val_interval
#   INCITE_VAL_SAMPLES      override train.val_samples
#   INCITE_DEV_GRAPHS       override the DEV10 list (suite ids)
#   INCITE_RESUME           checkpoint to resume from
#   INCITE_INIT_FROM        weights-only warm start (phase-2 levers)
#   INCITE_TRAIN_EXTRA_ARGS appended verbatim to incite/pretrain.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${1:-[0]}"
PY="${2:-python}"

# Default to a freshly prepared work tree, never the baked image copy: the
# bake matches the branch only at build time, and a stale bake ran once
# unnoticed (the old /kgfm default). Preparation costs seconds against
# hours of training.
WORKDIR="${INCITE_WORKDIR:-$ROOT/output/incite-run}"
if [ "$WORKDIR" = "$ROOT/output/incite-run" ]; then
  "$ROOT/scripts/prepare_incite_workdir.sh" "$WORKDIR"
fi
if [ -d "$WORKDIR/repos/trix" ]; then
  TRIXDIR="$WORKDIR/repos/trix"
else
  TRIXDIR="$WORKDIR/trix"
fi
CONFIG="${INCITE_CONFIG:-$ROOT/configs/incite_phase1.yaml}"
DATA_ROOT="$ROOT/data/roots/incite"
RAW_ROOT="$ROOT/data/raw/ultra-pretrain"
DEV_ROOT="$ROOT/data/roots/trix"
OUT="$ROOT/output/incite-pretrain"

mkdir -p "$OUT" "$DATA_ROOT"

# Seed the pretrain root from what already exists rather than re-downloading:
# data/roots/crest/pretrain holds the same three graphs in the exact loader
# layout, processed included (the FB15k237/WN18RR loaders rerun a slow
# build_relation_graph otherwise). Copies, not symlinks/hardlinks -- a write
# through a shared inode would corrupt the other model's root.
if [ ! -d "$DATA_ROOT/pretrain" ] && [ -d "$ROOT/data/roots/crest/pretrain" ]; then
  echo "seeding $DATA_ROOT/pretrain from data/roots/crest/pretrain"
  mkdir -p "$DATA_ROOT/pretrain.tmp"
  for d in fb15k237 wn18rr codex-m; do
    [ -d "$ROOT/data/roots/crest/pretrain/$d" ] && \
      cp -r "$ROOT/data/roots/crest/pretrain/$d" "$DATA_ROOT/pretrain.tmp/$d"
  done
  # the collated mix cache is identical content; only the name is ours
  if [ -f "$ROOT/data/roots/crest/pretrain/crest_mix_FB15k237-WN18RR-CoDExMedium.pt" ]; then
    cp "$ROOT/data/roots/crest/pretrain/crest_mix_FB15k237-WN18RR-CoDExMedium.pt" \
       "$DATA_ROOT/pretrain.tmp/incite_mix_FB15k237-WN18RR-CoDExMedium.pt"
  fi
  mv "$DATA_ROOT/pretrain.tmp" "$DATA_ROOT/pretrain"
fi

DEVICE=cpu; [ "$GPUS" != "null" ] && DEVICE=gpu
echo "config    : $CONFIG"
echo "graphs    : ${INCITE_TRAIN_GRAPHS:-<config mix>}"
echo "dev       : ${INCITE_DEV_GRAPHS:-<DEV10>}"
echo "resume    : ${INCITE_RESUME:-<fresh>}"
echo "init_from : ${INCITE_INIT_FROM:-<none>}"
echo "output    : $OUT"

# The same record run_incite.sh keeps: which device, seed and config produced
# what is in this directory, so a checkpoint is never orphaned from its setup.
$PY - "$OUT/PROVENANCE.json" "$DEVICE" "$CONFIG" "${INCITE_TRAIN_GRAPHS:-}" \
    "${INCITE_RESUME:-}" <<'PROVPY'
import json, os, platform, subprocess, sys
path, device, config, graphs, resume = sys.argv[1:6]
prov = {"device": device, "host": platform.platform(), "cpu_count": os.cpu_count(),
        "seed": 1024, "config": os.path.basename(config),
        "graphs": graphs or "config mix",
        "resume": os.path.basename(resume) if resume else None,
        "checkpoint_selection": "zero-shot DEV10, one mean per suite group"}
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
# sys.path[0] beats PYTHONPATH; run from the work tree so the intended incite/
# copy imports, not the one baked into the image (two CREST bugs).
cd "$WORKDIR"

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
t0=$(date +%s.%N)
$PY -m incite.pretrain \
    -c "$CONFIG" \
    --gpus "$GPUS" \
    ${INCITE_TRAIN_GRAPHS:+--graphs "$INCITE_TRAIN_GRAPHS"} \
    ${INCITE_TRAIN_STEPS:+--steps "$INCITE_TRAIN_STEPS"} \
    ${INCITE_VAL_INTERVAL:+--val_interval "$INCITE_VAL_INTERVAL"} \
    ${INCITE_VAL_SAMPLES:+--val_samples "$INCITE_VAL_SAMPLES"} \
    ${INCITE_DEV_GRAPHS:+--dev_graphs "$INCITE_DEV_GRAPHS"} \
    ${INCITE_RESUME:+--resume "$INCITE_RESUME"} \
    ${INCITE_INIT_FROM:+--init_from "$INCITE_INIT_FROM"} \
    --data_root "$DATA_ROOT" \
    --raw_root "$RAW_ROOT" \
    --dev_root "$DEV_ROOT" \
    --output_dir "$OUT" \
    --seed 1024 \
    ${INCITE_TRAIN_EXTRA_ARGS:-} 2>&1 | tee -a "$OUT/train.log"
status=${PIPESTATUS[0]}
t1=$(date +%s.%N)
$PY - "$OUT/TIMINGS.jsonl" "$DEVICE" "$status" "$started" "$t0" "$t1" <<'TIMEPY'
import json, sys
path, device, status, started, t0, t1 = sys.argv[1:7]
with open(path, "a") as handle:
    handle.write(json.dumps({
        "model": "incite", "device": device,
        "status": "ok" if status == "0" else "failed",
        "started": started, "seconds": round(float(t1) - float(t0), 3),
    }, sort_keys=True) + "\n")
TIMEPY
exit "$status"
