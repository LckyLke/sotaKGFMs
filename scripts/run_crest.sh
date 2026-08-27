#!/usr/bin/env bash
# Zero-shot CREST over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_crest.sh <ind_e|ind_er|transductive> [gpus] [python]
#
# Modeled on scripts/run_trix.sh: same provenance guard against mixing CPU and
# GPU ranks in one directory, same atomic claim per graph, same per-graph
# timing record. The driver is crest/run.py, which wraps the patched TRIX tree
# (TRIX_ROOT) and adds the in-context residual; with no CREST_READOUT_CKPT the
# readout's last layer is zero and every rank must equal ranks/trix/ row by
# row -- phase 0's gate, checked by scripts/verify_crest_identity.py.
#
# Knobs (forwarded into the container by scripts/docker_run.sh):
#   CREST_DATASETS    comma list of run_ids, defaults to the whole group
#   CREST_SHARD       shard tag for output/claims bookkeeping
#   CREST_REDO        rerun graphs that already have a parquet
#   CREST_EXTRA_ARGS  appended verbatim to crest/run.py (e.g. --bank skip)
#   CREST_WORKDIR     prepared work tree (prepare_crest_workdir.sh)
#   CREST_READOUT_CKPT  trained readout state; unset means zero residual
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_crest.sh <ind_e|ind_er|transductive> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python}"

# Container layout: crest/ and repos/trix live under /kgfm (Dockerfile).
# Host layout: prepare_crest_workdir.sh materialises <workdir>/{crest,trix}.
WORKDIR="${CREST_WORKDIR:-/kgfm}"
if [ -d "$WORKDIR/repos/trix" ]; then
  TRIXDIR="$WORKDIR/repos/trix"
else
  TRIXDIR="$WORKDIR/trix"
fi
CKPT="$TRIXDIR/entity_prediction.pth"
CONFIG="$ROOT/configs/crest_v1.yaml"
DATA_ROOT="$ROOT/data/roots/crest"
BANK_ROOT="$ROOT/data/roots/crest/banks"
RANKS="$ROOT/ranks/crest"
OUT="$ROOT/output/crest/${CREST_SHARD:-all}"

case "$GROUP" in
  transductive) TASK_NAME="TransductiveInference" ;;
  ind_e|ind_er) TASK_NAME="InductiveInference" ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

DATASETS="${CREST_DATASETS:-$($PY "$ROOT/shared/suite.py" "$GROUP")}"
SHARD="${CREST_SHARD:-}"

mkdir -p "$DATA_ROOT" "$BANK_ROOT" "$RANKS" "$OUT"

echo "group     : $GROUP"
echo "config    : $CONFIG"
echo "trix ckpt : $CKPT"
echo "readout   : ${CREST_READOUT_CKPT:-<none: zero residual, phase 0>}"
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
  if [ "$had" != "$DEVICE" ] && [ -z "${CREST_REDO:-}" ]; then
    echo "REFUSING TO RUN: $RANKS holds $had ranks, this run is $DEVICE." >&2
    echo "Clear it first:  rm -rf $RANKS $ROOT/ranks/.claims-crest" >&2
    exit 3
  fi
fi
# chunk_size is a memory parameter and the plan requires it in PROVENANCE.json
CHUNK=$($PY -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['chunk_size'])")
$PY - "$PROV" "$DEVICE" "$CKPT" "${CREST_READOUT_CKPT:-}" "$CHUNK" <<'PROVPY'
import json, os, platform, subprocess, sys
path, device, ckpt, readout, chunk = sys.argv[1:6]
prov = {"device": device, "host": platform.platform(), "cpu_count": os.cpu_count(),
        "seed": 1024, "checkpoint": os.path.basename(ckpt),
        "readout_checkpoint": os.path.basename(readout) if readout else None,
        "chunk_size": int(chunk)}
try:
    prov["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    prov["gpu"] = None
json.dump(prov, open(path, "w"), indent=2, sort_keys=True)
PROVPY

CLAIMS="$ROOT/ranks/.claims-crest"
mkdir -p "$CLAIMS"
export TRIX_ROOT="$TRIXDIR"
export PYTHONPATH="$WORKDIR:$ROOT/shared${PYTHONPATH:+:$PYTHONPATH}"

ran=0; skipped=0; failed=0
for d in ${DATASETS//,/ }; do
  id="$($PY -c "import sys;sys.path.insert(0,'$ROOT/shared');import suite;print(suite.by_run_id('$d').id.replace(':','_'))")"
  if [ -z "${CREST_REDO:-}" ] && [ -s "$RANKS/$id.parquet" ]; then
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
  if CREST_RANK_DUMP_DIR="$RANKS" $PY -m crest.run \
      -c "$CONFIG" \
      --dataset "$DS" \
      --version "$VERSION" \
      --gpus "$GPUS" \
      --task_name "$TASK_NAME" \
      --ckpt "$CKPT" \
      ${CREST_READOUT_CKPT:+--readout_ckpt "$CREST_READOUT_CKPT"} \
      --output_dir "$OUT" \
      --data_root "$DATA_ROOT" \
      --bank_root "$BANK_ROOT" \
      --seed 1024 \
      ${CREST_EXTRA_ARGS:-}; then
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
        "dataset": gid, "run_id": run_id, "model": "crest", "device": device,
        "status": status, "started": started, "seconds": round(float(t1) - float(t0), 3),
        "index": int(index),
    }, sort_keys=True) + "\n")
TIMEPY
done
echo "worker ${SHARD:-$GROUP} done: $ran ran, $skipped skipped, $failed failed"
