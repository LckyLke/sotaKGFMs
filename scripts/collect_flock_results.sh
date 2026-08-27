#!/usr/bin/env bash
# Copy FLOCK's raw results_*.csv out of the work tree into results/flock/.
#
# run_many.py writes them to src_entity/results/ inside the tree, which
# prepare_flock_workdir.sh wipes on rebuild. They are a deliverable and the
# thing criterion A is measured against, so they are copied out unmodified.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${FLOCK_WORKDIR:-/kgfm-src/output/flock-run}"
mkdir -p "$ROOT/results/flock"
n=0
for f in "$WORKDIR"/src_entity/results/results_*.csv; do
  [ -e "$f" ] || continue
  cp -p "$f" "$ROOT/results/flock/"
  n=$((n+1))
done
echo "collected $n csv(s) into results/flock/"
