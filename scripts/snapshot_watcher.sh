#!/usr/bin/env bash
# Snapshot-soup watcher (2026-09-01): as each decay run of plan v3 finishes
# (its .done marker appears), average its last 5 kept snapshots and
# evaluate the average beside the running plan (eval only, small memory,
# its own work tree). Markers in output/research-plan/snapshot-<stage>.done.
#   nohup scripts/snapshot_watcher.sh >> output/snapshot-watcher-nohup.log 2>&1 & disown
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORC="$ROOT/output/research-plan"
LOG="$ROOT/output/snapshot-watcher.log"
cd "$ROOT"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
# stage -> run suffix : config
JOBS="L1:4g-decay:/kgfm-src/configs/incite_phase1.yaml
M1:4g-mask:/kgfm-src/configs/incite_phase1.yaml
G1:4g-unary:/kgfm-src/configs/incite_phase1_4g_unary.yaml
MG1:4g-maskunary:/kgfm-src/configs/incite_phase1_4g_unary.yaml
L2:decay:/kgfm-src/configs/incite_phase1.yaml"
say "=== snapshot watcher start (pid $$) ==="
while :; do
  pending=0
  while IFS=: read -r st suffix cfg; do
    [ -n "$st" ] || continue
    [ -e "$ORC/snapshot-$st.done" ] && continue
    if [ -e "$ORC/$st.done" ]; then
      # never start an eval beside a training run: wait for GPU memory
      # below 8 GB (training holds about 13 GB; evals 1 to 3 GB)
      while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 8000 ]; do sleep 120; done
      say "$st finished; building the last-5 snapshot soup of $suffix"
      if scripts/snapshot_soup.sh "$suffix" "$cfg" 5 >> "$LOG" 2>&1; then
        touch "$ORC/snapshot-$st.done"; say "snapshot-$st DONE"
      else
        touch "$ORC/snapshot-$st.done"; say "snapshot-$st FAILED (see log)"
      fi
    elif [ -e "$ORC/$st.failed" ]; then
      touch "$ORC/snapshot-$st.done"; say "$st failed upstream; skipping its soup"
    else
      pending=$((pending + 1))
    fi
  done <<< "$JOBS"
  [ "$pending" -eq 0 ] && break
  sleep 300
done
say "=== snapshot watcher done ==="
