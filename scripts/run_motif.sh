#!/usr/bin/env bash
# Zero-shot MOTIF over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_motif.sh <ind_e|ind_er|transductive> [gpus] [python]
#
# Every path handed to run_many.py is absolute on purpose: run_many.py chdir's
# into a fresh timestamped working directory for each dataset, so anything
# relative resolves against a different directory on the second dataset onward.
#
# motif_3g.pth, the 3-graph mixture (FB15k237, WN18RR, CoDExMedium), matching the
# checkpoint ULTRA is measured with here. MOTIF also ships ultra_3g.pth; that one
# is its ULTRA baseline, not MOTIF, and running it would measure the wrong model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_motif.sh <ind_e|ind_er|transductive> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python}"

WORKDIR="${MOTIF_WORKDIR:-/kgfm/repos/motif}"
CKPT="$WORKDIR/ckpts/motif_3g.pth"
DATA_ROOT="$ROOT/data/roots/motif"
RANKS="$ROOT/ranks/motif"
# Per-shard output_dir: create_working_directory() writes a working_dir.tmp in
# cfg.output_dir and deletes it again, so two processes sharing one output_dir
# race on that file.
OUT="$ROOT/output/motif/${MOTIF_SHARD:-all}"

case "$GROUP" in
  transductive) CONFIG="$WORKDIR/config/transductive/MOTIF_inference.yaml" ;;
  ind_e|ind_er) CONFIG="$WORKDIR/config/inductive/MOTIF_inference.yaml" ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

# MOTIF_DATASETS lets one group be split across processes; when unset the whole
# group runs, resolved from shared/suite.py and never retyped here.
DATASETS="${MOTIF_DATASETS:-$($PY "$ROOT/shared/suite.py" "$GROUP")}"
SHARD="${MOTIF_SHARD:-}"

mkdir -p "$DATA_ROOT" "$RANKS" "$OUT"

echo "group     : $GROUP"
echo "config    : $CONFIG"
echo "ckpt      : $CKPT"
echo "data root : $DATA_ROOT"
echo "ranks     : $RANKS"
echo "datasets  : $(tr ',' '\n' <<<"$DATASETS" | wc -l)${SHARD:+ (shard $SHARD)}"
echo "extra args: ${MOTIF_EXTRA_ARGS:-none}"

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
  if [ "$had" != "$DEVICE" ] && [ -z "${MOTIF_REDO:-}" ]; then
    echo "REFUSING TO RUN: $RANKS holds $had ranks, this run is $DEVICE." >&2
    echo "Mixing devices in one rank directory silently corrupts the group means." >&2
    echo "Clear it first:  rm -rf $RANKS $ROOT/ranks/.claims-motif" >&2
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

CLAIMS="$ROOT/ranks/.claims-motif"
mkdir -p "$CLAIMS"
cd "$WORKDIR"

ran=0; skipped=0; failed=0
for d in ${DATASETS//,/ }; do
  id="$($PY -c "import sys;sys.path.insert(0,'$ROOT/shared');import suite;print(suite.by_run_id('$d').id.replace(':','_'))")"
  if [ -z "${MOTIF_REDO:-}" ] && [ -s "$RANKS/$id.parquet" ]; then
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
  # dataset loading and relation-graph construction, which for MOTIF is where
  # much of the cost sits -- it builds a higher-order relation graph that ULTRA
  # does not build at all, and it runs several times slower for that reason.
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
      ${MOTIF_EXTRA_ARGS:-} \
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
  # JIT compile of rspmm, and for MOTIF the Triton kernel as well. Every graph
  # additionally parses its own raw files and builds its own relation graph the
  # first time it is seen. Read timings within a run.
  $PY - "$RANKS/TIMINGS.jsonl" "$id" "$d" "$DEVICE" "$status" "$started" "$t0" "$t1" "$index" <<'TIMEPY'
import json, sys
path, gid, run_id, device, status, started, t0, t1, index = sys.argv[1:10]
with open(path, "a") as handle:
    handle.write(json.dumps({
        "dataset": gid, "run_id": run_id, "model": "motif", "device": device,
        "status": status, "started": started, "seconds": round(float(t1) - float(t0), 3),
        "index": int(index),
    }, sort_keys=True) + "\n")
TIMEPY
done
echo "worker ${SHARD:-$GROUP} done: $ran ran, $skipped skipped, $failed failed"
