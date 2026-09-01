#!/usr/bin/env bash
# ONE command brings the whole program back after a pause/reboot
# (2026-09-01). Every queue is marker-based and self-ordering: the
# baseline orchestrator resumes the composite seed runs from their last
# checkpoint, the retry watcher re-runs the KGPFN suite (finished graphs
# skip via parquets) and FLOCK's last graph, research chain 1 fires when
# ranks/kgpfn hits 41, research chain 2 waits for the GPU to free.
#
#   cd ~/Dokumente/GitHub/sotaKGFMs && ./RESTART.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p output/baseline-orchestrator output/research-chain output/research-chain2
nohup scripts/baseline_orchestrator.sh >> output/baseline-orchestrator/nohup.log 2>&1 & disown
nohup scripts/retry_watcher.sh        >> output/retry-watcher-nohup.log        2>&1 & disown
nohup scripts/queued_research.sh      >> output/research-chain-nohup.log       2>&1 & disown
nohup scripts/research_chain2.sh      >> output/research-chain2-nohup.log      2>&1 & disown
echo "all queues relaunched; watch output/*/log.txt"
