#!/usr/bin/env bash
# Zero-shot ULTRA over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_ultra.sh <ind_e|ind_er|transductive> [gpus] [python]
#
# Every path handed to run_many.py is absolute on purpose: run_many.py chdir's
# into a fresh timestamped working directory for each dataset, so anything
# relative resolves against a different directory on the second dataset onward.
#
# ultra_3g.pth, never ultra_50g.pth -- 50g is trained on 50 graphs including
# most of this suite, so it is not zero-shot here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_ultra.sh <ind_e|ind_er|transductive> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python}"

WORKDIR="${ULTRA_WORKDIR:-/home/user/ultra-run}"
CKPT="$WORKDIR/ckpts/ultra_3g.pth"
DATA_ROOT="$ROOT/data/roots/ultra"
RANKS="$ROOT/ranks/ultra"
OUT="$ROOT/output/ultra"

case "$GROUP" in
  transductive) CONFIG="$WORKDIR/config/transductive/inference.yaml" ;;
  ind_e|ind_er) CONFIG="$WORKDIR/config/inductive/inference.yaml" ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

DATASETS="$($PY "$ROOT/shared/suite.py" "$GROUP")"
mkdir -p "$DATA_ROOT" "$RANKS" "$OUT"

echo "group     : $GROUP"
echo "config    : $CONFIG"
echo "ckpt      : $CKPT"
echo "data root : $DATA_ROOT"
echo "ranks     : $RANKS"
echo "datasets  : $(tr ',' '\n' <<<"$DATASETS" | wc -l)"

cd "$WORKDIR"
exec $PY "$WORKDIR/script/run_many.py" \
  -c "$CONFIG" \
  --gpus "$GPUS" \
  --ckpt "$CKPT" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT" \
  --rank_dump_dir "$RANKS" \
  -d "$DATASETS"
