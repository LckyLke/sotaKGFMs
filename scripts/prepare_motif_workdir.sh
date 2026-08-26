#!/usr/bin/env bash
# Materialise a patched MOTIF tree without touching repos/motif.
#
# This is exactly what containers/motif/Dockerfile does in its COPY + patch
# layers, factored out so the same patch set can be exercised outside a
# container.  repos/motif stays pristine: `git -C repos/motif status` must
# report a clean tree at the pinned SHA at all times.
#
#   usage: scripts/prepare_motif_workdir.sh [destination]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/kgfm-src/output/motif-run}"

if [ -n "$(git -C "$ROOT/repos/motif" status --porcelain)" ]; then
  echo "refusing to build: repos/motif is dirty; patches belong in patches/" >&2
  exit 1
fi

# run_many.py writes MOTIF_results/MOTIF_results_*.csv under the work tree.
# Rescue any before wiping, or a rebuild silently destroys the raw results that
# criterion A is measured against.
if compgen -G "$DEST/script/MOTIF_results/MOTIF_results_*.csv" > /dev/null; then
  mkdir -p "$ROOT/results/motif"
  cp -n "$DEST"/script/MOTIF_results/MOTIF_results_*.csv "$ROOT/results/motif/"
  echo "rescued $(ls "$DEST"/script/MOTIF_results/MOTIF_results_*.csv | wc -l) result csv(s) into results/motif/"
fi

rm -rf "$DEST"
mkdir -p "$DEST"
tar -C "$ROOT/repos/motif" --exclude=.git -cf - . | tar -C "$DEST" -xf -

for p in "$ROOT"/patches/motif/*.diff; do
  echo "applying $(basename "$p")"
  patch -d "$DEST" -p1 --batch --forward < "$p"
done
echo "patched MOTIF tree at $DEST"
