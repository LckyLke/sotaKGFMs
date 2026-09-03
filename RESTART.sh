#!/usr/bin/env bash
# ONE command brings the program back after a pause or reboot (rewritten
# 2026-09-01). The research plan is marker-based and self-ordering: it
# resumes pretrain stages from their last checkpoint (optimizer state and
# lr schedule included) and skips finished stages. The KGPFN small retry
# waits for any running KGPFN suite container, then reruns the small failed
# graphs beside the plan.
#
# research_plan.sh (v1) is superseded by v8 (same markers).
# The old queues (baseline_orchestrator.sh R3/R4, retry_watcher.sh,
# queued_research.sh, research_chain2.sh) are superseded: see
# scripts/research_plan_v9.sh's header for why. Do not relaunch them.
#
#   cd ~/Dokumente/GitHub/sotaKGFMs && ./RESTART.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p output/research-plan
if pgrep -f '^bash scripts/research_plan_v9.sh' > /dev/null; then
  echo "research_plan_v9.sh already running"
else
  nohup scripts/research_plan_v9.sh >> output/research-plan/nohup.log 2>&1 & disown
  echo "research_plan_v9.sh launched"
fi
if pgrep -f '^bash scripts/kgpfn_small_retry.sh' > /dev/null; then
  echo "kgpfn_small_retry.sh already running"
else
  nohup scripts/kgpfn_small_retry.sh >> output/kgpfn-small-retry-nohup.log 2>&1 & disown
  echo "kgpfn_small_retry.sh launched"
fi
if pgrep -f '^bash scripts/snapshot_watcher.sh' > /dev/null; then
  echo "snapshot_watcher.sh already running"
else
  nohup scripts/snapshot_watcher.sh >> output/snapshot-watcher-nohup.log 2>&1 & disown
  echo "snapshot_watcher.sh launched"
fi
echo "watch output/research-plan/log.txt"
