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

rm -rf "$DEST"
mkdir -p "$DEST"
tar -C "$ROOT/repos/ultra" --exclude=.git -cf - . | tar -C "$DEST" -xf -

for p in "$ROOT"/patches/ultra/*.diff; do
  echo "applying $(basename "$p")"
  patch -d "$DEST" -p1 --batch --forward < "$p"
done
echo "patched ULTRA tree at $DEST"
