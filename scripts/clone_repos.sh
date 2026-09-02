#!/usr/bin/env bash
# Clone the seven upstream repositories into repos/ at the SHAs recorded in
# repos/PINS.json. Re-running is idempotent: an existing clone is fetched and
# hard-reset to its pinned SHA. Never tracks a branch.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINS="$ROOT/repos/PINS.json"
[ -f "$PINS" ] || { echo "missing $PINS" >&2; exit 1; }

python3 - "$PINS" <<'PY' | while IFS=$'\t' read -r dir url sha; do
import json, sys
pins = json.load(open(sys.argv[1]))
for d, e in pins["repos"].items():
    if not e.get("url") or not e.get("sha"):
        continue  # our own models (incite) wrap no upstream: nothing to clone
    print(f'{d}\t{e["url"]}\t{e["sha"]}')
PY
  dest="$ROOT/repos/$dir"
  if [ -d "$dest/.git" ]; then
    echo "== $dir: fetching $sha"
    git -C "$dest" fetch --quiet origin "$sha" 2>/dev/null || git -C "$dest" fetch --quiet origin
  else
    echo "== $dir: cloning $url"
    git clone --quiet "$url" "$dest" || { echo "!! clone failed: $url"; continue; }
  fi
  git -C "$dest" checkout --quiet --detach "$sha" || echo "!! checkout failed: $dir@$sha"
  echo "   $dir -> $(git -C "$dest" rev-parse HEAD)"
done
