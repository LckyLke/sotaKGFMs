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
# scripts/research_plan_v14.sh's header for why. Do not relaunch them.
#
#   cd ~/Dokumente/GitHub/sotaKGFMs && ./RESTART.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p output/research-plan
if pgrep -f '^bash scripts/research_plan_v14.sh' > /dev/null; then
  echo "research_plan_v14.sh already running"
else
  nohup scripts/research_plan_v14.sh >> output/research-plan/nohup.log 2>&1 & disown
  echo "research_plan_v14.sh launched"
fi
# The KGPFN small-graph retry and the snapshot watcher are NOT relaunched
# (2026-09-03, the verifier's finding): the retry shares the GPU with the
# plan's containers and once died of it (3.7 GB beside a training ramp);
# snapshot soups add nothing (finding 5). Nothing is gated on either.
echo "watch output/research-plan/log.txt"
