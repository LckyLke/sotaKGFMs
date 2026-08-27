#!/usr/bin/env bash
# Copy CREST's CREST_results_*.csv into results/crest/.
#
# crest/run.py writes one timestamped CSV per graph under
# output/crest/<shard>/CREST_results/. They are the model's own metric values
# -- the thing criterion A is measured against -- so they are copied out
# unmodified: never edited, never merged, never reformatted.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${CREST_OUTPUT:-$ROOT/output/crest}"
mkdir -p "$ROOT/results/crest"
n=0
for f in "$OUT"/*/CREST_results/CREST_results_*.csv; do
  [ -e "$f" ] || continue
  cp -p "$f" "$ROOT/results/crest/"
  n=$((n+1))
done
echo "collected $n csv(s) into results/crest/"
ls -1 "$ROOT/results/crest" 2>/dev/null | sed 's/^/  /'
