#!/usr/bin/env bash
# Zero-shot TRIX over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_trix.sh <ind_e|ind_er|transductive> [gpus] [python]
#
# TRIX has no run_many.py. src/run_entity.py takes exactly one dataset per
# invocation, rendered into the config as jinja variables, so the loop over the
# suite lives here rather than upstream. Every template variable the config
# declares must be supplied: util.parse_args builds the parser from the config
# and marks all of them required, so omitting one is an argparse error rather
# than a default.
#
# entity_prediction.pth, never relation_prediction.pth -- TRIX splits entity and
# relation prediction across separate runners and separate checkpoints, and only
# entity prediction is what this suite measures.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_trix.sh <ind_e|ind_er|transductive> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python}"

WORKDIR="${TRIX_WORKDIR:-/kgfm/repos/trix}"
CKPT="${TRIX_CKPT:-$WORKDIR/entity_prediction.pth}"
DATA_ROOT="$ROOT/data/roots/trix"
RANKS="$ROOT/ranks/trix"
OUT="$ROOT/output/trix/${TRIX_SHARD:-all}"

case "$GROUP" in
  transductive) CONFIG="$WORKDIR/config/run_entity_transductive.yaml" ;;
  ind_e|ind_er) CONFIG="$WORKDIR/config/run_entity_inductive.yaml" ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

DATASETS="${TRIX_DATASETS:-$($PY "$ROOT/shared/suite.py" "$GROUP")}"
SHARD="${TRIX_SHARD:-}"

mkdir -p "$DATA_ROOT" "$RANKS" "$OUT"

echo "group     : $GROUP"
echo "config    : $CONFIG"
echo "ckpt      : $CKPT"
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
  if [ "$had" != "$DEVICE" ] && [ -z "${TRIX_REDO:-}" ]; then
    echo "REFUSING TO RUN: $RANKS holds $had ranks, this run is $DEVICE." >&2
    echo "Clear it first:  rm -rf $RANKS $ROOT/ranks/.claims-trix" >&2
    exit 3
  fi
fi
$PY - "$PROV" "$DEVICE" <<'PROVPY'
import json, os, platform, subprocess, sys
path, device = sys.argv[1], sys.argv[2]
prov = {"device": device, "host": platform.platform(), "cpu_count": os.cpu_count()}
try:
    prov["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    prov["gpu"] = None
json.dump(prov, open(path, "w"), indent=2, sort_keys=True)
PROVPY

CLAIMS="$ROOT/ranks/.claims-trix"
mkdir -p "$CLAIMS"
cd "$WORKDIR"

ran=0; skipped=0; failed=0
for d in ${DATASETS//,/ }; do
  id="$($PY -c "import sys;sys.path.insert(0,'$ROOT/shared');import suite;print(suite.by_run_id('$d').id.replace(':','_'))")"
  if [ -z "${TRIX_REDO:-}" ] && [ -s "$RANKS/$id.parquet" ]; then
    skipped=$((skipped+1)); continue
  fi
  if ! mkdir "$CLAIMS/$id" 2>/dev/null; then
    skipped=$((skipped+1)); continue
  fi

  # run_id carries the spelling upstream needs, which is not always the suite id:
  # Metafam is Metafam:Metafam and FBNELL is FBNELL:FBNELL_v1.
  DS="${d%%:*}"
  VERSION="${d#*:}"
  [ "$VERSION" = "$d" ] && VERSION=""

  echo ">>> $d"
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  index=$((ran + failed + 1))
  t0=$(date +%s.%N)
  # --epochs 0 is what makes this zero-shot: train_and_validate returns -1
  # immediately at num_epoch == 0, which loads cfg.checkpoint and skips the
  # validation pass entirely. There is no --skip_valid to add here.
  if TRIX_RANK_DUMP_DIR="$RANKS" $PY "$WORKDIR/src/run_entity.py" \
      -c "$CONFIG" \
      --dataset "$DS" \
      --version "$VERSION" \
      --gpus "$GPUS" \
      --epochs 0 \
      --bpe null \
      --ckpt "$CKPT" \
      --output_dir "$OUT" \
      --data_root "$DATA_ROOT" \
      ${TRIX_EXTRA_ARGS:-}; then
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
        "dataset": gid, "run_id": run_id, "model": "trix", "device": device,
        "status": status, "started": started, "seconds": round(float(t1) - float(t0), 3),
        "index": int(index),
    }, sort_keys=True) + "\n")
TIMEPY
done
echo "worker ${SHARD:-$GROUP} done: $ran ran, $skipped skipped, $failed failed"
