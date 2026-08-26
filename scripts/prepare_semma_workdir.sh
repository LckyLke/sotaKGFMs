#!/usr/bin/env bash
# Materialise a patched SEMMA tree without touching repos/semma.
#
# This is exactly what containers/semma/Dockerfile does in its COPY + patch
# layers, factored out so the same patch set can be exercised outside a
# container.  repos/semma stays pristine: `git -C repos/semma status` must
# report a clean tree at the pinned SHA at all times.
#
#   usage: scripts/prepare_semma_workdir.sh [destination]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/kgfm-src/output/semma-run}"

if [ -n "$(git -C "$ROOT/repos/semma" status --porcelain)" ]; then
  echo "refusing to build: repos/semma is dirty; patches belong in patches/" >&2
  exit 1
fi

# run_many.py writes ultra_results_*.csv next to itself, inside the work tree.
# Rescue any before wiping, or a rebuild silently destroys the raw results that
# criterion A is measured against.
if compgen -G "$DEST/script/ultra_results_*.csv" > /dev/null; then
  mkdir -p "$ROOT/results/semma"
  cp -n "$DEST"/script/ultra_results_*.csv "$ROOT/results/semma/"
  echo "rescued $(ls "$DEST"/script/ultra_results_*.csv | wc -l) result csv(s) into results/semma/"
fi

rm -rf "$DEST"
mkdir -p "$DEST"
tar -C "$ROOT/repos/semma" --exclude=.git -cf - . | tar -C "$DEST" -xf -

for p in "$ROOT"/patches/semma/*.diff; do
  echo "applying $(basename "$p")"
  patch -d "$DEST" -p1 --batch --forward < "$p"
done
# ultra/datasets.py resolves fb_mid2name.tsv against os.getcwd(), captured at
# import time, so it has to sit at the root of the work tree. It is 230 MB and
# not part of the repository, so it is copied from the mirror rather than baked
# into the image or committed. Without it FB15k237Inductive, FBIngram and HM --
# 12 of the 41 graphs -- raise FileNotFoundError.
MID2NAME="$ROOT/data/raw/semma/fb_mid2name.tsv"
if [ -f "$MID2NAME" ]; then
  cp "$MID2NAME" "$DEST/fb_mid2name.tsv"
  echo "placed fb_mid2name.tsv ($(du -h "$MID2NAME" | cut -f1))"
else
  echo "WARNING: $MID2NAME missing; run scripts/fetch_semma_mid2name.py first." >&2
  echo "         FB15k237Inductive, FBIngram and HM will fail without it." >&2
fi

# SEMMA reads flags.yaml from os.getcwd() too; it ships one and it is copied
# with the tree, so nothing to do here beyond noting why it must not move.

echo "patched SEMMA tree at $DEST"
