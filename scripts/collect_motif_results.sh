#!/usr/bin/env bash
# Copy MOTIF's raw MOTIF_results_*.csv out of the work tree into results/motif/.
#
# run_many.py writes them next to itself inside the tree, which
# prepare_motif_workdir.sh wipes on rebuild. They are a deliverable and the
# thing criterion A is measured against, so they are copied out unmodified --
# never edited, never merged, never reformatted.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${MOTIF_WORKDIR:-/kgfm-src/output/motif-run}"
mkdir -p "$ROOT/results/motif"
n=0
for f in "$WORKDIR"/script/MOTIF_results/MOTIF_results_*.csv; do
  [ -e "$f" ] || continue
  cp -p "$f" "$ROOT/results/motif/"
  n=$((n+1))
done
echo "collected $n csv(s) into results/motif/"
ls -1 "$ROOT/results/motif" 2>/dev/null | sed 's/^/  /'
