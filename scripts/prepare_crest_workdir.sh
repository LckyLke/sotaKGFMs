#!/usr/bin/env bash
# Materialise a CREST work tree: a patched TRIX plus the crest package.
#
# Mirrors what containers/crest/Dockerfile does in its COPY + patch layers,
# factored out so the same tree can be exercised outside a container.
# repos/trix stays pristine at its pin -- the TRIX half is delegated to
# prepare_trix_workdir.sh, which enforces that and applies patches/trix/*.diff
# (including 0004, the relation-ranking offset fix). crest/ is our own code
# and is copied verbatim; it is never patched.
#
#   usage: scripts/prepare_crest_workdir.sh [destination]
#
# Layout produced (what run_crest.sh's CREST_WORKDIR expects):
#   <dest>/trix    patched TRIX tree        -> TRIX_ROOT
#   <dest>/crest   the crest package        -> on PYTHONPATH via <dest>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-/kgfm-src/output/crest-run}"

# CREST results are written under output/crest/, not inside this tree, so a
# rebuild destroys no deliverable the way a TRIX rebuild would.
"$ROOT/scripts/prepare_trix_workdir.sh" "$DEST/trix"

rm -rf "$DEST/crest"
mkdir -p "$DEST/crest"
tar -C "$ROOT/crest" --exclude=__pycache__ --exclude=tests -cf - . | tar -C "$DEST/crest" -xf -
echo "crest package at $DEST/crest"

# entity_prediction.pth is fetched separately (it is a release artifact, not
# repository content); run_crest.sh expects it at <dest>/trix/.
if [ ! -f "$DEST/trix/entity_prediction.pth" ]; then
  echo "note: $DEST/trix/entity_prediction.pth is not present yet;"
  echo "      copy the TRIX entity checkpoint there before running."
fi
