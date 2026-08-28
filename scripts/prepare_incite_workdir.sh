#!/usr/bin/env bash
# Materialise an INCITE work tree: a patched TRIX plus the incite package.
#
# Mirrors what containers/incite/Dockerfile does in its COPY + patch layers,
# factored out so the same tree can be exercised outside a container (crest
# precedent). repos/trix stays pristine at its pin -- the TRIX half is
# delegated to prepare_trix_workdir.sh, which enforces that and applies
# patches/trix/*.diff. incite/ and configs/ are our own code and are copied
# verbatim; they are never patched.
#
#   usage: scripts/prepare_incite_workdir.sh [destination]
#
# Layout produced (what run_incite.sh's INCITE_WORKDIR expects):
#   <dest>/trix      patched TRIX tree     -> TRIX_ROOT
#   <dest>/incite    the incite package    -> on PYTHONPATH via <dest>
#   <dest>/configs   the config yamls
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/kgfm-src/output/incite-run}"

# INCITE results are written under output/incite*, not inside this tree, so a
# rebuild destroys no deliverable.
"$ROOT/scripts/prepare_trix_workdir.sh" "$DEST/trix"

rm -rf "$DEST/incite" "$DEST/configs"
mkdir -p "$DEST/incite" "$DEST/configs"
tar -C "$ROOT/incite" --exclude=__pycache__ --exclude=tests -cf - . | tar -C "$DEST/incite" -xf -
cp "$ROOT"/configs/incite_*.yaml "$DEST/configs/"
echo "incite package at $DEST/incite"
