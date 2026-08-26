#!/usr/bin/env bash
# Copy SEMMA's raw ultra_results_*.csv out of the work tree into results/semma/.
#
# run_many.py writes them next to itself inside the tree, which
# prepare_semma_workdir.sh wipes on rebuild. They are a deliverable and the
# thing criterion A is measured against, so they are copied out unmodified --
# never edited, never merged, never reformatted.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${SEMMA_WORKDIR:-/kgfm-src/output/semma-run}"
mkdir -p "$ROOT/results/semma"
n=0
for f in "$WORKDIR"/script/ultra_results_*.csv; do
  [ -e "$f" ] || continue
  cp -p "$f" "$ROOT/results/semma/"
  n=$((n+1))
done
echo "collected $n csv(s) into results/semma/"
ls -1 "$ROOT/results/semma" 2>/dev/null | sed 's/^/  /'
