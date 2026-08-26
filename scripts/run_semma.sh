#!/usr/bin/env bash
# Zero-shot SEMMA over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_semma.sh <ind_e|ind_er|transductive> [gpus] [python]
#
# Every path handed to run_many.py is absolute on purpose: run_many.py chdir's
# into a fresh timestamped working directory for each dataset, so anything
# relative resolves against a different directory on the second dataset onward.
#
# semma.pth -- plain SEMMA. The repo also ships ULTRA's four checkpoints as its
# baseline; running those would measure ULTRA, not SEMMA. SEMMA-Hybrid is named
# in the paper but not released, so it is out of scope here.
#
# ultra/datasets.py, layers.py, models.py and tasks.py all capture
# mydir = os.getcwd() at import time, and resolve flags.yaml, fb_mid2name.tsv and
# the LLM relation descriptions against it. The cd into WORKDIR below is load
# bearing: run from anywhere else and SEMMA reads none of them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_semma.sh <ind_e|ind_er|transductive> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python}"

WORKDIR="${SEMMA_WORKDIR:-/kgfm/repos/semma}"
CKPT="$WORKDIR/ckpts/semma.pth"
DATA_ROOT="$ROOT/data/roots/semma"
RANKS="$ROOT/ranks/semma"
# Per-shard output_dir: create_working_directory() writes a working_dir.tmp in
# cfg.output_dir and deletes it again, so two processes sharing one output_dir
# race on that file.
OUT="$ROOT/output/semma/${SEMMA_SHARD:-all}"

case "$GROUP" in
  transductive) CONFIG="$WORKDIR/config/transductive/inference.yaml" ;;
  ind_e|ind_er) CONFIG="$WORKDIR/config/inductive/inference.yaml" ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

# SEMMA_DATASETS lets one group be split across processes; when unset the whole
# group runs, resolved from shared/suite.py and never retyped here.
DATASETS="${SEMMA_DATASETS:-$($PY "$ROOT/shared/suite.py" "$GROUP")}"
SHARD="${SEMMA_SHARD:-}"

mkdir -p "$DATA_ROOT" "$RANKS" "$OUT"

echo "group     : $GROUP"
echo "config    : $CONFIG"
echo "ckpt      : $CKPT"
echo "data root : $DATA_ROOT"
echo "ranks     : $RANKS"
echo "datasets  : $(tr ',' '\n' <<<"$DATASETS" | wc -l)${SHARD:+ (shard $SHARD)}"
echo "extra args: ${SEMMA_EXTRA_ARGS:-none}"

# One invocation per graph, each preceded by an atomic claim. Several workers
# can then be handed the same list and will divide it between them: whoever
# claims a graph runs it, the others move on. That self-balances a suite whose
# per-graph cost spans two orders of magnitude (411 to 21k queries, 4k to 105k
# nodes) far better than any static split guessed up front.
#
# mkdir is the claim: it is atomic on POSIX, succeeds for exactly one caller,
# and needs no lock daemon. A claim is released again if the run fails, so a
# later pass retries rather than silently skipping.
# Ranks from a CPU run and ranks from a GPU run must never end up in one
# directory. The float32 kernels differ in low-order bits, which can flip a
# near-tie and move a rank, so a mixed directory yields a group mean that
# corresponds to no single measurement -- and nothing downstream would notice.
DEVICE=cpu; [ "$GPUS" != "null" ] && DEVICE=gpu
PROV="$RANKS/PROVENANCE.json"
if [ -f "$PROV" ]; then
  had=$($PY -c "import json;print(json.load(open('$PROV'))['device'])" 2>/dev/null || echo unknown)
  if [ "$had" != "$DEVICE" ] && [ -z "${SEMMA_REDO:-}" ]; then
    echo "REFUSING TO RUN: $RANKS holds $had ranks, this run is $DEVICE." >&2
    echo "Mixing devices in one rank directory silently corrupts the group means." >&2
    echo "Clear it first:  rm -rf $RANKS $ROOT/ranks/.claims" >&2
    exit 3
  fi
fi
mkdir -p "$RANKS"
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

CLAIMS="$ROOT/ranks/.claims-semma"
mkdir -p "$CLAIMS"
cd "$WORKDIR"

ran=0; skipped=0; failed=0
for d in ${DATASETS//,/ }; do
  id="$($PY -c "import sys;sys.path.insert(0,'$ROOT/shared');import suite;print(suite.by_run_id('$d').id.replace(':','_'))")"
  if [ -z "${SEMMA_REDO:-}" ] && [ -s "$RANKS/$id.parquet" ]; then
    skipped=$((skipped+1)); continue
  fi
  if ! mkdir "$CLAIMS/$id" 2>/dev/null; then
    skipped=$((skipped+1)); continue
  fi
  echo ">>> $d"
  # Wall clock around the whole invocation, not around the evaluation loop.
  # Cost per graph is a reported quantity here, not a diagnostic: seven models
  # are being compared on one suite, and how long a model takes to answer is
  # part of what distinguishes them. Timing only the scoring loop would hide
  # dataset loading and relation-graph construction, which is exactly where the
  # models differ -- MOTIF builds a higher-order relation graph that ULTRA does
  # not build at all.
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  index=$((ran + failed + 1))
  t0=$(date +%s.%N)
  if $PY "$WORKDIR/script/run_many.py" \
      -c "$CONFIG" \
      --gpus "$GPUS" \
      --ckpt "$CKPT" \
      --data_root "$DATA_ROOT" \
      --output_dir "$OUT" \
      --rank_dump_dir "$RANKS" \
      ${SEMMA_EXTRA_ARGS:-} \
      -d "$d"; then
    status=ok; ran=$((ran+1))
  else
    status=failed
    failed=$((failed+1)); rmdir "$CLAIMS/$id" 2>/dev/null || true
    echo "!!! FAILED: $d"
  fi
  t1=$(date +%s.%N)
  # One JSON object per line: appendable from several workers without a lock,
  # and readable even if a run is interrupted halfway through the suite.
  #
  # `index` is this graph's position in the worker's sequence, and it is recorded
  # because position 1 is not comparable with the rest: it absorbs the one-time
  # JIT compile of rspmm, which is minutes on a cold extension cache. Every graph
  # additionally parses its own raw files and builds its own relation graph the
  # first time it is seen, so a run against a fresh data root is cold throughout
  # and a re-run against a warm one is not. Read timings within a run.
  $PY - "$RANKS/TIMINGS.jsonl" "$id" "$d" "$DEVICE" "$status" "$started" "$t0" "$t1" "$index" <<'TIMEPY'
import json, sys
path, gid, run_id, device, status, started, t0, t1, index = sys.argv[1:10]
with open(path, "a") as handle:
    handle.write(json.dumps({
        "dataset": gid, "run_id": run_id, "model": "semma", "device": device,
        "status": status, "started": started, "seconds": round(float(t1) - float(t0), 3),
        "index": int(index),
    }, sort_keys=True) + "\n")
TIMEPY
done
echo "worker ${SHARD:-$GROUP} done: $ran ran, $skipped skipped, $failed failed"
