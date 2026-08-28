#!/usr/bin/env bash
# Run the INCITE test suite inside the incite container.
#
#   usage: scripts/test_incite.sh [pytest args...]
#
# The host has no torch; tests run only in the container. The cd into
# /kgfm-src puts the mounted work tree at sys.path[0], ahead of the copy
# baked into the image (sys.path[0] beats PYTHONPATH -- the CREST shadowing
# trap), so the tests exercise the checked-out code, not the baked snapshot.
# TRIX_ROOT stays the image's patched tree (/kgfm/repos/trix), which is what
# the layer gate must compare against.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/docker_run.sh" incite \
  bash -c "cd /kgfm-src && python -m pytest incite/tests -q ${*:-}"
