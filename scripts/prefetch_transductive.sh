#!/usr/bin/env bash
# Pre-download the 13 transductive datasets into every model's processed
# root while the internet lasts (2026-08-31: imminent outage). Download is
# the only internet-bound step; processing is CPU and reruns offline.
# Idempotent: loaders skip datasets whose processed files exist.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/output/prefetch-transductive.log"
cd "$ROOT"
echo "=== prefetch start $(date) ===" >> "$LOG"
for m in trix ultra motif semma; do
  echo "--- $m ---" >> "$LOG"
  scripts/docker_run.sh "$m" bash -c '
    cd /kgfm-src
    PYROOT=/kgfm-src/data/roots/'"$m"'
    python - "$PYROOT" << "EOF"
import sys, traceback
sys.path.insert(0, "/kgfm-src/shared")
root = sys.argv[1]
import suite
# every pinned repo in this family ships ULTRA-lineage loaders under a
# different import name; try the known layouts
for mod, sub in (("trix", "src"), ("ultra", ""), ("motif", ""), ("semma", "")):
    for p in ("/kgfm/repos/%s/%s" % (mod, sub), "/kgfm/%s" % mod, "/kgfm"):
        sys.path.insert(0, p)
try:
    from trix import util
except Exception:
    try:
        from ultra import util  # type: ignore
    except Exception:
        import util  # type: ignore
from easydict import EasyDict
ok = fail = 0
for gid in suite.ids("transductive"):
    g = suite.by_id(gid)
    cfg = {"class": g.dataset, "root": root}
    if g.version is not None:
        cfg["version"] = util.literal_eval(g.version)
    try:
        util.build_dataset(EasyDict({"dataset": cfg}))
        ok += 1
        print("ok", gid, flush=True)
    except Exception as e:
        fail += 1
        print("FAIL", gid, repr(e)[:200], flush=True)
print("done: %d ok, %d fail" % (ok, fail), flush=True)
EOF' >> "$LOG" 2>&1
done
echo "=== prefetch end $(date) ===" >> "$LOG"
tail -6 "$LOG"
