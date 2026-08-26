#!/usr/bin/env bash
# Copy TRIX's TRIX_results_*.csv out of the work tree into results/trix/.
#
# run_entity.py writes them next to itself inside the tree, which
# prepare_trix_workdir.sh wipes on rebuild. They are a deliverable and the
# thing criterion A is measured against, so they are copied out unmodified --
# never edited, never merged, never reformatted.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${TRIX_WORKDIR:-/kgfm-src/output/trix-run}"
mkdir -p "$ROOT/results/trix"
n=0
for f in "$WORKDIR"/src/TRIX_results/TRIX_results_*.csv; do
  [ -e "$f" ] || continue
  cp -p "$f" "$ROOT/results/trix/"
  n=$((n+1))
done
echo "collected $n csv(s) into results/trix/"
ls -1 "$ROOT/results/trix" 2>/dev/null | sed 's/^/  /'
