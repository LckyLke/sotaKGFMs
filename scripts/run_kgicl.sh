#!/usr/bin/env bash
# Zero-shot KG-ICL over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_kgicl.sh <ind_e|ind_er|transductive> [gpu] [python]
#
# KG-ICL-6L, never 4L or 5L: the 6-layer checkpoint is the one its README calls
# the full model, and the other two exist for smaller memory budgets.
#
# The datasets are the ones scripts/build_kgicl_datasets.py produced, NOT the
# ones in the repository's datasets.zip. That archive covers 27 of the 41 and
# prepares them with a different test graph and a different filter from every
# other model here; see the header of that script. KGICL_DATA points at the
# rebuilt copies by default.
#
# Two rank columns land in the dump. `rank` is the shared definition every
# cross-model table reads. `rank_native` is what KG-ICL's own cal_ranks
# returned, which breaks ties by entity id, and is what its results CSV
# describes. Criterion A must be run against rank_native, never against rank.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_kgicl.sh <ind_e|ind_er|transductive> [gpu] [python]}"
GPU="${2:-0}"
PY="${3:-python3.9}"

WORKDIR="${KGICL_WORKDIR:-/kgfm/repos/kg-icl}"
CKPT="$WORKDIR/checkpoint/KG-ICL-6L"
DATA="${KGICL_DATA:-$ROOT/output/kgicl-data}"
RANKS="${KGICL_RANKS:-$ROOT/ranks/kg-icl}"
RESULTS="${KGICL_RESULTS:-$ROOT/results/kg-icl/KGICL_results.csv}"

DATASETS="${KGICL_DATASETS:-$($PY "$ROOT/shared/suite.py" "$GROUP")}"

mkdir -p "$RANKS" "$(dirname "$RESULTS")"

echo "group     : $GROUP"
echo "ckpt      : $CKPT"
echo "data      : $DATA"
echo "ranks     : $RANKS"

DEVICE=cpu; [ "$GPU" != "null" ] && DEVICE=gpu
PROV="$RANKS/PROVENANCE.json"
if [ -f "$PROV" ]; then
  had=$($PY -c "import json;print(json.load(open('$PROV'))['device'])" 2>/dev/null || echo unknown)
  if [ "$had" != "$DEVICE" ] && [ -z "${KGICL_REDO:-}" ]; then
    echo "REFUSING TO RUN: $RANKS holds $had ranks, this run is $DEVICE." >&2
    exit 3
  fi
fi
$PY - "$PROV" "$DEVICE" <<'PROVPY'
import json, os, platform, subprocess, sys
path, device = sys.argv[1], sys.argv[2]
prov = {"device": device, "host": platform.platform(), "cpu_count": os.cpu_count(),
        "checkpoint": "KG-ICL-6L", "seed": 1234,
        "datasets": "rebuilt by scripts/build_kgicl_datasets.py under ULTRA's "
                    "test-graph and filter conventions, not repos/kg-icl/datasets.zip",
        "rank_columns": {"rank": "shared definition: 1-based, pessimistic ties, strict filtering",
                         "rank_native": "KG-ICL's own cal_ranks, ordinal ties by entity id"}}
try:
    prov["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    prov["gpu"] = None
json.dump(prov, open(path, "w"), indent=2, sort_keys=True)
PROVPY

# Claims follow the rank directory, not the model name. An alternate build
# writing elsewhere would otherwise see the main run's claims and skip
# every graph, reporting success while doing nothing.
CLAIMS="$(dirname "$RANKS")/.claims-$(basename "$RANKS")"
mkdir -p "$CLAIMS"
cd "$WORKDIR/src"

ran=0; skipped=0; failed=0
for d in ${DATASETS//,/ }; do
  id="$($PY -c "import sys;sys.path.insert(0,'$ROOT/shared');import suite;print(suite.by_run_id('$d').id.replace(':','_'))")"
  if [ -z "${KGICL_REDO:-}" ] && [ -s "$RANKS/$id.parquet" ]; then
    skipped=$((skipped+1)); continue
  fi
  if ! mkdir "$CLAIMS/$id" 2>/dev/null; then
    skipped=$((skipped+1)); continue
  fi
  echo ">>> $d  ($id)"
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  index=$((ran + failed + 1))
  t0=$(date +%s.%N)
  # No training happens: evaluation.py loads the checkpoint and evaluates. The
  # hyper-parameters below are copied from shell/test.sh, which is how the
  # authors evaluate KG-ICL-6L.
  if $PY "$WORKDIR/src/evaluation.py" \
      --checkpoint_path "$CKPT" \
      --data_path "$DATA/" \
      --test_dataset_list "$id" \
      --gpu "$GPU" \
      --n_layer 6 \
      --hidden_dim 32 \
      --MSG concat \
      --attn_dim 5 \
      --shot 5 \
      --act idd \
      --rank_dump_dir "$RANKS" \
      --results_csv "$RESULTS" \
      --note "" \
      ${KGICL_EXTRA_ARGS:-}; then
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
        "dataset": gid, "run_id": run_id, "model": "kg-icl", "device": device,
        "status": status, "started": started, "seconds": round(float(t1) - float(t0), 3),
        "index": int(index),
    }, sort_keys=True) + "\n")
TIMEPY
done
echo "worker $GROUP done: $ran ran, $skipped skipped, $failed failed"
