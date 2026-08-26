#!/usr/bin/env bash
# Materialise a patched TRIX tree without touching repos/trix.
#
# This is exactly what containers/trix/Dockerfile does in its COPY + patch
# layers, factored out so the same patch set can be exercised outside a
# container.  repos/trix stays pristine: `git -C repos/trix status` must
# report a clean tree at the pinned SHA at all times.
#
#   usage: scripts/prepare_trix_workdir.sh [destination]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/kgfm-src/output/trix-run}"

if [ -n "$(git -C "$ROOT/repos/trix" status --porcelain)" ]; then
  echo "refusing to build: repos/trix is dirty; patches belong in patches/" >&2
  exit 1
fi

# run_entity.py writes TRIX_results/*.csv under the work tree.
# Rescue any before wiping, or a rebuild silently destroys the raw results that
# criterion A is measured against.
if compgen -G "$DEST/src/TRIX_results/TRIX_results_*.csv" > /dev/null; then
  mkdir -p "$ROOT/results/trix"
  cp -n "$DEST"/src/TRIX_results/TRIX_results_*.csv "$ROOT/results/trix/"
  echo "rescued $(ls "$DEST"/src/TRIX_results/TRIX_results_*.csv | wc -l) result csv(s) into results/trix/"
fi

rm -rf "$DEST"
mkdir -p "$DEST"
tar -C "$ROOT/repos/trix" --exclude=.git -cf - . | tar -C "$DEST" -xf -

for p in "$ROOT"/patches/trix/*.diff; do
  echo "applying $(basename "$p")"
  patch -d "$DEST" -p1 --batch --forward < "$p"
done
echo "patched TRIX tree at $DEST"
