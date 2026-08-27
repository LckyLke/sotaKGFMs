#!/usr/bin/env bash
# Zero-shot FLOCK over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_flock.sh <ind_e|ind_er|transductive> [gpus] [python]
#
# FLOCK does not evaluate the suite with one config. Walk count rises and batch
# size falls as graphs grow, and its own scripts/entity_zeroshot.sh carries the
# assignment. scripts/flock_config_map.py parses that file, so the map is read
# from upstream rather than copied here; a runner that used one config for all
# 41 graphs would not be measuring FLOCK.
#
# flock_entity.pth, never flock_relation.pth -- FLOCK splits entity and relation
# prediction across separate source trees and separate checkpoints, and only
# entity prediction is what this suite measures.
#
# FLOCK is the one stochastic model here. It scores by sampling random walks and
# averaging test_samples of them, so a rank is reproducible only together with
# its seed. run_many.py takes seeds from the fixed list [1024, 42, 1337, 512,
# 256] indexed by repeat, and --repeats 1 therefore pins seed 1024 -- the same
# seed every other model in this project runs under.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_flock.sh <ind_e|ind_er|transductive> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python3}"

WORKDIR="${FLOCK_WORKDIR:-/kgfm/repos/flock}"
CKPT="$WORKDIR/checkpoints/flock_entity.pth"
DATA_ROOT="$ROOT/data/roots/flock"
RANKS="$ROOT/ranks/flock"
OUT="$ROOT/output/flock/${FLOCK_SHARD:-all}"

mkdir -p "$DATA_ROOT" "$RANKS" "$OUT"

echo "group     : $GROUP"
echo "ckpt      : $CKPT"
echo "data root : $DATA_ROOT"
echo "ranks     : $RANKS"

# Ranks from a CPU run and ranks from a GPU run must never end up in one
# directory: the float32 kernels differ in low-order bits, which can flip a
# near-tie and move a rank, so a mixed directory yields a group mean that
# corresponds to no single measurement and nothing downstream would notice.
DEVICE=cpu; [ "$GPUS" != "null" ] && DEVICE=gpu
PROV="$RANKS/PROVENANCE.json"
if [ -f "$PROV" ]; then
  had=$($PY -c "import json;print(json.load(open('$PROV'))['device'])" 2>/dev/null || echo unknown)
  if [ "$had" != "$DEVICE" ] && [ -z "${FLOCK_REDO:-}" ]; then
    echo "REFUSING TO RUN: $RANKS holds $had ranks, this run is $DEVICE." >&2
    echo "Clear it first:  rm -rf $RANKS $ROOT/ranks/.claims-flock" >&2
    exit 3
  fi
fi
$PY - "$PROV" "$DEVICE" <<'PROVPY'
import json, os, platform, subprocess, sys
path, device = sys.argv[1], sys.argv[2]
# seed is recorded here and not only per row: FLOCK's inference is stochastic,
# so the seed is part of what the numbers mean, not run bookkeeping.
prov = {"device": device, "host": platform.platform(), "cpu_count": os.cpu_count(),
        "seed": 1024, "repeats": 1,
        "note": "FLOCK scores by sampling random walks; results are an ensemble "
                "of test_samples walks and depend on the seed."}
try:
    prov["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    prov["gpu"] = None
json.dump(prov, open(path, "w"), indent=2, sort_keys=True)
PROVPY

CLAIMS="$ROOT/ranks/.claims-flock"
mkdir -p "$CLAIMS"
cd "$WORKDIR"

MAP="$($PY "$ROOT/scripts/flock_config_map.py" --group "$GROUP")"
# FLOCK_DATASETS narrows the group to a comma-separated list of suite ids, for
# a single-graph check or to split a group across workers. The config for each
# still comes from the map, never from a flag.
if [ -n "${FLOCK_DATASETS:-}" ]; then
  MAP="$(awk -F'\t' -v want=",${FLOCK_DATASETS}," 'index(want, ","$1",")' <<<"$MAP")"
  [ -n "$MAP" ] || { echo "no suite id in FLOCK_DATASETS matched group $GROUP" >&2; exit 2; }
fi
echo "datasets  : $(wc -l <<<"$MAP")"

ran=0; skipped=0; failed=0
while IFS=$'\t' read -r gid spelling config; do
  [ -n "$gid" ] || continue
  id="${gid//:/_}"
  if [ -z "${FLOCK_REDO:-}" ] && [ -s "$RANKS/$id.parquet" ]; then
    skipped=$((skipped+1)); continue
  fi
  if ! mkdir "$CLAIMS/$id" 2>/dev/null; then
    skipped=$((skipped+1)); continue
  fi

  # Evaluation batch size, chosen for the GPU rather than for the model.
  # Every config pairs batch_size with walk_num so the product is 512, which is
  # what an H100 holds and a 16 GB card does not: the first graph asks for 18.8
  # GiB. Dividing the batch by FLOCK_BATCH_DIVISOR keeps that product uniform
  # across configs, just smaller. It changes nothing a result depends on --
  # walk_num, walk_len, test_samples and every model hyper-parameter are
  # untouched -- and run.py already honours cfg.train.test_batch_size.
  DIV="${FLOCK_BATCH_DIVISOR:-4}"
  BS="$(grep -oP '^  batch_size: \K\d+' "$WORKDIR/$config" | head -1)"
  TEST_BS=$(( BS / DIV )); [ "$TEST_BS" -ge 1 ] || TEST_BS=1

  echo ">>> $gid  ($(basename "$config"), test_batch_size $TEST_BS of $BS)"
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  index=$((ran + failed + 1))
  t0=$(date +%s.%N)
  # --epochs 0 is what makes this zero-shot: train_and_validate returns
  # immediately at num_epoch == 0, which skips the validation pass entirely.
  if $PY "$WORKDIR/src_entity/run_many.py" \
      --config "$WORKDIR/$config" \
      --gpus "$GPUS" \
      --epochs 0 \
      --bpe null \
      --repeats 1 \
      --ckpt "$CKPT" \
      --output_dir "$OUT" \
      --data_root "$DATA_ROOT" \
      --test_batch_size "$TEST_BS" \
      --rank_dump_dir "$RANKS" \
      ${FLOCK_EXTRA_ARGS:-} \
      -d "$spelling"; then
    status=ok; ran=$((ran+1))
  else
    status=failed
    failed=$((failed+1)); rmdir "$CLAIMS/$id" 2>/dev/null || true
    echo "!!! FAILED: $gid"
  fi
  t1=$(date +%s.%N)
  $PY - "$RANKS/TIMINGS.jsonl" "$id" "$spelling" "$DEVICE" "$status" "$started" "$t0" "$t1" "$index" "$config" "$TEST_BS" <<'TIMEPY'
import json, os, sys
path, gid, run_id, device, status, started, t0, t1, index, config, test_bs = sys.argv[1:12]
with open(path, "a") as handle:
    handle.write(json.dumps({
        "dataset": gid, "run_id": run_id, "model": "flock", "device": device,
        "status": status, "started": started, "seconds": round(float(t1) - float(t0), 3),
        "index": int(index), "config": os.path.basename(config),
        "test_batch_size": int(test_bs),
    }, sort_keys=True) + "\n")
TIMEPY
done <<<"$MAP"
echo "worker ${FLOCK_SHARD:-$GROUP} done: $ran ran, $skipped skipped, $failed failed"
