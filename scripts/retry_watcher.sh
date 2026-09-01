#!/usr/bin/env bash
# Retry watcher (2026-09-01): after the current KGPFN pass exits, rerun the
# suite for the 13 rsync-collision failures (parquets skip the 10 done),
# then retry FLOCK's FBIngram:25 with patch 0005 applied to the workdir.
# When ranks/kgpfn reaches 41, research chain 1 (complementarity + soup)
# fires by itself on its parquet-count trigger.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/output/retry-watcher.log"
cd "$ROOT"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== retry watcher start ==="
# wait for the current kgpfn suite container to be gone
while docker ps --format '{{.Image}}' | grep -q '^kgfm/kgpfn:'; do sleep 300; done
say "kgpfn pass ended; retrying failures"
rm -rf "$ROOT/ranks/.claims-kgpfn"
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  scripts/docker_run.sh kgpfn bash -c \
  '/kgfm-src/scripts/run_kgpfn.sh ind_e "[0]"; /kgfm-src/scripts/run_kgpfn.sh ind_er "[0]"' \
  >> "$LOG" 2>&1
n="$(ls "$ROOT/ranks/kgpfn"/*.parquet 2>/dev/null | wc -l)"
say "kgpfn after retry: $n/41"
[ "$n" -ge 41 ] && touch "$ROOT/output/baseline-orchestrator/K1.done"

say "retrying FLOCK FBIngram:25 (patch 0005 in workdir)"
rm -rf "$ROOT/ranks/.claims-flock"
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLOCK_BATCH_DIVISOR=4 \
  FLOCK_DATASETS="FBIngram:25" FLOCK_WORKDIR=/kgfm-src/output/flock-run \
  scripts/docker_run.sh flock /kgfm-src/scripts/run_flock.sh ind_er "[0]" \
  >> "$LOG" 2>&1
[ -f "$ROOT/ranks/flock/FBIngram_25.parquet" ] \
  && say "FLOCK 41/41 COMPLETE" || say "FBIngram:25 failed again -- see log"
say "=== retry watcher done ==="
