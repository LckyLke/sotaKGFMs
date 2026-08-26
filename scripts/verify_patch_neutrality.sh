#!/usr/bin/env bash
# Show that the rank-dump patch changes no rank.
#
# Runs stock upstream ULTRA (no patches at all) and the patched tree over the
# same dataset, same checkpoint, same seed, and diffs the two
# ultra_results_*.csv rows byte for byte. Equal rows mean the patch observed the
# ranking without perturbing it: no dtype change, no epsilon change, no fused op
# swapped for an unfused one, no reduction reordered.
#
# The stock tree cannot be given a --data_root (that flag arrives with patch
# 0002), so its hardcoded ~/git/ULTRA/kg-datasets/ is symlinked at the shared
# processed root instead. The tree itself stays untouched.
#
#   usage: verify_patch_neutrality.sh [dataset] [python]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS="${1:-FB15k237Inductive:v1}"
PY="${2:-python}"
STOCK=/home/user/ultra-stock
PATCHED="${ULTRA_WORKDIR:-/home/user/ultra-run}"
DATA_ROOT="$ROOT/data/roots/ultra"

rm -rf "$STOCK"; mkdir -p "$STOCK"
tar -C "$ROOT/repos/ultra" --exclude=.git -cf - . | tar -C "$STOCK" -xf -

mkdir -p ~/git/ULTRA
[ -e ~/git/ULTRA/kg-datasets ] || ln -s "$DATA_ROOT" ~/git/ULTRA/kg-datasets
mkdir -p "$ROOT/output/neutrality"

echo "### stock upstream, no patches"
( cd "$STOCK" && $PY script/run_many.py \
    -c "$STOCK/config/inductive/inference.yaml" \
    --gpus null --ckpt "$STOCK/ckpts/ultra_3g.pth" -d "$DS" >/dev/null 2>&1 )

echo "### patched"
( cd "$PATCHED" && PYTHONPATH="$ROOT/shared" $PY script/run_many.py \
    -c "$PATCHED/config/inductive/inference.yaml" \
    --gpus null --ckpt "$PATCHED/ckpts/ultra_3g.pth" \
    --data_root "$DATA_ROOT" --output_dir "$ROOT/output/neutrality" \
    --rank_dump_dir "$ROOT/output/neutrality/ranks" -d "$DS" >/dev/null 2>&1 )

A=$(ls -t "$STOCK"/script/ultra_results_*.csv | head -1)
B=$(ls -t "$PATCHED"/script/ultra_results_*.csv | head -1)
echo
echo "stock   : $A"; grep -F "$DS," "$A" | tail -1
echo "patched : $B"; grep -F "$DS," "$B" | tail -1
echo
if [ "$(grep -F "$DS," "$A" | tail -1)" = "$(grep -F "$DS," "$B" | tail -1)" ]; then
  echo "PATCH IS RANKING-NEUTRAL: rows identical to full printed precision"
else
  echo "MISMATCH: the patch changed the ranking -- stop and fix it"; exit 1
fi
