#!/usr/bin/env bash
# Materialise a patched FLOCK tree without touching repos/flock.
#
# Same reason as the other prepare_* scripts: run_many.py writes its results
# CSV inside the source tree, and a tree that lives only inside the image dies
# with the container. This puts the patched tree on the bind mount instead.
#
# graph-walker is NOT rebuilt here. The image installs it editable against
# /kgfm/repos/flock/graph-walker, so `import walker` resolves to the copy the
# image compiled whatever directory the run starts from. The walker is
# unpatched, so the two copies are the same code.
#
#   usage: scripts/prepare_flock_workdir.sh [destination]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/kgfm-src/output/flock-run}"

if [ -n "$(git -C "$ROOT/repos/flock" status --porcelain)" ]; then
  echo "refusing to build: repos/flock is dirty; patches belong in patches/" >&2
  exit 1
fi

# Rescue results before wiping, or a rebuild destroys the raw numbers criterion
# A is measured against.
if compgen -G "$DEST/src_entity/results/results_*.csv" > /dev/null; then
  mkdir -p "$ROOT/results/flock"
  cp -n "$DEST"/src_entity/results/results_*.csv "$ROOT/results/flock/"
  echo "rescued $(ls "$DEST"/src_entity/results/results_*.csv | wc -l) result csv(s)"
fi

rm -rf "$DEST"
mkdir -p "$DEST"
# checkpoints/ is 20 MB of .pth and is copied with the tree: run_flock.sh reads
# flock_entity.pth from the work tree, not from the image.
tar -C "$ROOT/repos/flock" --exclude=.git -cf - . | tar -C "$DEST" -xf -

for p in "$ROOT"/patches/flock/*.diff; do
  echo "applying $(basename "$p")"
  patch -d "$DEST" -p1 --batch --forward < "$p"
done
echo "patched FLOCK tree at $DEST"
