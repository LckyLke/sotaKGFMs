#!/usr/bin/env bash
# KGPFN background retry (2026-09-01, user decision: "background only").
# Waits for the running KGPFN suite container to exit, then reruns only the
# SMALL graphs that failed in the rsync collision. The large failed graphs
# (ILPC2022:large, which died after 19.5 h; HM:indigo; FBIngram 25-100) are
# deliberately left out: they need days each on a shared GPU and nothing in
# the research plan depends on them. Runs beside the research plan (KGPFN
# eval holds ~1.7 GB).
#   nohup scripts/kgpfn_small_retry.sh >> output/kgpfn-small-retry-nohup.log 2>&1 & disown
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/output/kgpfn-small-retry.log"
cd "$ROOT"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== kgpfn small retry: waiting for the running suite container ==="
while docker ps --format '{{.Image}}' | grep -q '^kgfm/kgpfn:'; do sleep 600; done
say "suite container gone; ranks/kgpfn has $(ls "$ROOT"/ranks/kgpfn/*.parquet 2>/dev/null | wc -l) parquets"
rm -rf "$ROOT/ranks/.claims-kgpfn"
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  KGPFN_DATASETS="FB15k237Inductive:v1,FB15k237Inductive:v2,FB15k237Inductive:v3,FB15k237Inductive:v4,HM:1k,HM:3k,HM:5k" \
  scripts/docker_run.sh kgpfn /kgfm-src/scripts/run_kgpfn.sh ind_e "[0]" >> "$LOG" 2>&1
say "done; ranks/kgpfn now has $(ls "$ROOT"/ranks/kgpfn/*.parquet 2>/dev/null | wc -l) parquets"
