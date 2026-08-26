#!/usr/bin/env bash
# Materialise a patched ULTRA tree without touching repos/ultra.
#
# This is exactly what containers/ultra/Dockerfile does in its COPY + patch
# layers, factored out so the same patch set can be exercised outside a
# container.  repos/ultra stays pristine: `git -C repos/ultra status` must
# report a clean tree at the pinned SHA at all times.
#
#   usage: scripts/prepare_ultra_workdir.sh [destination]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/home/user/ultra-run}"

if [ -n "$(git -C "$ROOT/repos/ultra" status --porcelain)" ]; then
  echo "refusing to build: repos/ultra is dirty; patches belong in patches/" >&2
  exit 1
fi

# run_many.py writes ultra_results_*.csv next to itself, inside the work tree.
# Rescue any before wiping, or a rebuild silently destroys the raw results that
# criterion A is measured against.
if compgen -G "$DEST/script/ultra_results_*.csv" > /dev/null; then
  mkdir -p "$ROOT/results"
  cp -n "$DEST"/script/ultra_results_*.csv "$ROOT/results/"
  echo "rescued $(ls "$DEST"/script/ultra_results_*.csv | wc -l) result csv(s) into results/"
fi

rm -rf "$DEST"
mkdir -p "$DEST"
tar -C "$ROOT/repos/ultra" --exclude=.git -cf - . | tar -C "$DEST" -xf -

for p in "$ROOT"/patches/ultra/*.diff; do
  echo "applying $(basename "$p")"
  patch -d "$DEST" -p1 --batch --forward < "$p"
done
echo "patched ULTRA tree at $DEST"
