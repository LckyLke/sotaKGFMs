#!/usr/bin/env bash
# Research chain behind the KGPFN suite (2026-08-31): checkpoint soup +
# complementarity analysis. Waits for ranks/kgpfn to reach 41 parquets,
# then: (1) complementarity report (CPU), (2) build the floor-descendant
# soup, (3) eval it on the 41 graphs. Markers in output/research-chain/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INC="$ROOT/../sotaKGFMs-incite"
ORC="$ROOT/output/research-chain"
LOG="$ORC/log.txt"
mkdir -p "$ORC"
cd "$ROOT"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
skip() { [ -e "$ORC/$1.done" ] || [ -e "$ORC/$1.failed" ]; }

say "=== research chain start (pid $$) ==="

# C0: wait for the KGPFN suite (parquet count is the truth)
until [ "$(ls "$ROOT/ranks/kgpfn"/*.parquet 2>/dev/null | wc -l)" -ge 41 ]; do
  sleep 300
done
say "C0: KGPFN suite complete"

# C1: complementarity (CPU)
if ! skip C1; then
  if python3 scripts/complementarity.py >> "$LOG" 2>&1 \
     && [ -f "$ROOT/results/complementarity.md" ]; then
    touch "$ORC/C1.done"; say "C1 DONE (results/complementarity.md)"
  else
    touch "$ORC/C1.failed"; say "C1 FAILED"
  fi
fi

# C2: build the soup (container CPU; donors = the four floor-descendants)
if ! skip C2; then
  ( cd "$INC" && scripts/docker_run.sh incite bash -c '
      export CUDA_VISIBLE_DEVICES=""
      cd /kgfm-src && python scripts/make_soup.py \
        output/soup_floor_family.pth \
        output/incite-pretrain-phase1/incite_best.pth \
        output/incite-pretrain-phase21b/incite_best.pth \
        output/incite-pretrain-phase22/incite_best.pth \
        output/incite-pretrain-phase23/incite_best.pth' ) >> "$LOG" 2>&1
  if [ -f "$INC/output/soup_floor_family.pth" ]; then
    touch "$ORC/C2.done"; say "C2 DONE"
  else
    touch "$ORC/C2.failed"; say "C2 FAILED"
  fi
fi

# C3: eval the soup (GPU; coexists like the kgpfn suite did)
if ! skip C3; then
  if [ -e "$ORC/C2.done" ]; then
    ( cd "$INC" && env INCITE_CKPT=/kgfm-src/output/soup_floor_family.pth \
        INCITE_CONFIG=/kgfm-src/configs/incite_phase1.yaml \
        INCITE_RANKS=/kgfm-src/ranks/incite-soup \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        scripts/docker_run.sh incite bash -c \
        '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' ) \
        >> "$LOG" 2>&1
    if [ "$(ls "$INC/ranks/incite-soup"/*.parquet 2>/dev/null | wc -l)" -ge 41 ]; then
      touch "$ORC/C3.done"; say "C3 DONE (ranks/incite-soup)"
    else
      touch "$ORC/C3.failed"; say "C3 FAILED"
    fi
  else
    touch "$ORC/C3.failed"; say "C3 FAILED (no soup)"
  fi
fi

say "=== research chain finished ==="
