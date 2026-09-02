#!/usr/bin/env bash
# SUPERSEDED 2026-09-01 by scripts/research_plan.sh (see its header). Do not relaunch.
# Baseline-completion queue (2026-08-31, user order: "complete everything,
# also FLOCK, establish the baseline, then push the research").
#
# Stages (markers in output/baseline-orchestrator/; rerun to resume):
#   K1  KGPFN full 41-graph suite            -> ranks/kgpfn
#   F1  FLOCK ind_er group (23 graphs)       -> ranks/flock
#   F2  FLOCK HM:indigo alone, divisor 8     -> ranks/flock
#   F3  collect FLOCK CSVs                   -> results/flock
#   T1  transductive sweep: ultra motif trix semma kgpfn (13 graphs each)
#   TI  INCITE floor transductive (incite worktree)
#   R1  INCITE v1 composite pretrain (incite worktree, from scratch)
#   R2  composite evals: entity + relation + PETALS
#   R3  composite seed 1337 pretrain + entity eval
#   R4  composite seed 7 pretrain + entity eval
#
# K1 requires the sentinel KGPFN_VERIFIED in the marker dir (the main
# session creates it after verifying the integration agent's work).
# Failure policy: pretrain stages get 3 crash-resumes; failed stages are
# marked and the queue continues. Restart after a crash:
#   nohup scripts/baseline_orchestrator.sh >> output/baseline-orchestrator/nohup.log 2>&1 & disown
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INC="$ROOT/../sotaKGFMs-incite"
ORC="$ROOT/output/baseline-orchestrator"
LOG="$ORC/log.txt"
mkdir -p "$ORC"
cd "$ROOT"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
done_mark() { touch "$ORC/$1.done"; say "$1 DONE"; }
fail_mark() { touch "$ORC/$1.failed"; say "$1 FAILED (queue continues)"; }
skip() { [ -e "$ORC/$1.done" ] || [ -e "$ORC/$1.failed" ]; }
nparquet() { ls "$1"/*.parquet 2>/dev/null | wc -l; }

incite_container_running() { docker ps --format '{{.Image}}' | grep -q '^kgfm/incite:'; }

# INCITE pretrain with crash-resumes, from the incite worktree.
# $1 stage  $2 config  $3 seed  $4 archive-suffix
incite_pretrain() {
  local stage="$1" cfg="$2" seed="$3" suffix="$4" attempt=0
  local logf="$INC/output/incite-pretrain/train.log"
  while :; do
    if [ -f "$logf" ] && grep -q 'checkpoint reload OK' "$logf"; then
      mv "$INC/output/incite-pretrain" "$INC/output/incite-pretrain-$suffix"
      return 0
    fi
    if incite_container_running; then sleep 300; continue; fi
    attempt=$((attempt + 1)); [ "$attempt" -gt 4 ] && return 1
    local resume=""
    [ -f "$INC/output/incite-pretrain/incite_last.pth" ] && \
      resume="/kgfm-src/output/incite-pretrain/incite_last.pth"
    say "$stage: attempt $attempt (resume: ${resume:-no})"
    ( cd "$INC" && env INCITE_CONFIG="$cfg" INCITE_SEED="$seed" \
        ${resume:+INCITE_RESUME="$resume"} \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        scripts/docker_run.sh incite /kgfm-src/scripts/train_incite.sh "[0]" ) \
        >> "$LOG" 2>&1
    sleep 10
  done
}

say "=== baseline orchestrator start (pid $$) ==="

# ---- K1: KGPFN full suite -------------------------------------------------
if ! skip K1; then
  if [ -e "$ORC/KGPFN_VERIFIED" ]; then
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      scripts/docker_run.sh kgpfn bash -c \
      '/kgfm-src/scripts/run_kgpfn.sh ind_e "[0]"; /kgfm-src/scripts/run_kgpfn.sh ind_er "[0]"' \
      >> "$LOG" 2>&1
    [ "$(nparquet "$ROOT/ranks/kgpfn")" -ge 41 ] && done_mark K1 || fail_mark K1
  else
    say "K1: KGPFN_VERIFIED sentinel missing"; fail_mark K1
  fi
fi

# ---- F1: FLOCK ind_er -----------------------------------------------------
if ! skip F1; then
  rm -rf "$ROOT/ranks/.claims-flock"
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLOCK_BATCH_DIVISOR=4 \
    FLOCK_WORKDIR=/kgfm-src/output/flock-run \
    scripts/docker_run.sh flock /kgfm-src/scripts/run_flock.sh ind_er "[0]" \
    >> "$LOG" 2>&1
  [ "$(nparquet "$ROOT/ranks/flock")" -ge 40 ] && done_mark F1 || fail_mark F1
fi

# ---- F2: HM:indigo alone, divisor 8 (the prescribed retry) ---------------
if ! skip F2; then
  rm -rf "$ROOT/ranks/.claims-flock"
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FLOCK_BATCH_DIVISOR=8 \
    FLOCK_DATASETS="HM:indigo" FLOCK_WORKDIR=/kgfm-src/output/flock-run \
    scripts/docker_run.sh flock /kgfm-src/scripts/run_flock.sh ind_e "[0]" \
    >> "$LOG" 2>&1
  [ -f "$ROOT/ranks/flock/HM_indigo.parquet" ] && done_mark F2 || fail_mark F2
fi

# ---- F3: collect FLOCK CSVs ----------------------------------------------
if ! skip F3; then
  scripts/docker_run.sh flock /kgfm-src/scripts/collect_flock_results.sh \
    >> "$LOG" 2>&1
  [ "$(ls "$ROOT/results/flock"/results_*.csv 2>/dev/null | wc -l)" -ge 30 ] \
    && done_mark F3 || fail_mark F3
fi

# ---- T1: transductive sweep ----------------------------------------------
for m in ultra motif trix semma kgpfn; do
  st="T1-$m"
  if ! skip "$st"; then
    if [ "$m" = kgpfn ] && [ ! -e "$ORC/K1.done" ]; then
      fail_mark "$st"; continue
    fi
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      scripts/docker_run.sh "$m" "/kgfm-src/scripts/run_$m.sh" transductive "[0]" \
      >> "$LOG" 2>&1
    [ "$(nparquet "$ROOT/ranks/$m")" -ge 54 ] && done_mark "$st" || fail_mark "$st"
  fi
done

# ---- TI: INCITE floor transductive ---------------------------------------
if ! skip TI; then
  ( cd "$INC" && env INCITE_CKPT=/kgfm-src/output/incite-pretrain-phase1/incite_best.pth \
      INCITE_CONFIG=/kgfm-src/configs/incite_phase1.yaml \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      scripts/docker_run.sh incite /kgfm-src/scripts/run_incite.sh transductive "[0]" ) \
      >> "$LOG" 2>&1
  [ "$(nparquet "$INC/ranks/incite")" -ge 54 ] && done_mark TI || fail_mark TI
fi

# ---- R1/R2: the v1 composite ----------------------------------------------
if ! skip R1; then
  if incite_pretrain R1 /kgfm-src/configs/incite_v1_full.yaml 1024 v1; then
    done_mark R1
  else
    fail_mark R1
  fi
fi
if ! skip R2; then
  if [ -e "$ORC/R1.done" ]; then
    ok=1
    ( cd "$INC" && env INCITE_CKPT=/kgfm-src/output/incite-pretrain-v1/incite_best.pth \
        INCITE_CONFIG=/kgfm-src/configs/incite_v1_full.yaml \
        INCITE_RANKS=/kgfm-src/ranks/incite-v1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        scripts/docker_run.sh incite bash -c \
        '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' ) \
        >> "$LOG" 2>&1
    [ "$(nparquet "$INC/ranks/incite-v1")" -ge 41 ] || ok=0
    ( cd "$INC" && env INCITE_CKPT=/kgfm-src/output/incite-pretrain-v1/incite_best.pth \
        INCITE_CONFIG=/kgfm-src/configs/incite_v1_full.yaml \
        INCITE_TASK=relation INCITE_RANKS=/kgfm-src/ranks-relation/incite-v1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        scripts/docker_run.sh incite bash -c \
        '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' ) \
        >> "$LOG" 2>&1
    ( cd "$INC" && scripts/docker_run.sh incite bash -c \
        'cd /kgfm-src/output/incite-run && PYTHONPATH=/kgfm-src/output/incite-run:/kgfm-src/shared TRIX_ROOT=/kgfm-src/output/incite-run/trix python /kgfm-src/diagnostics/petals_eval.py --ckpt /kgfm-src/output/incite-pretrain-v1/incite_best.pth --config /kgfm-src/configs/incite_v1_full.yaml --instances /kgfm-src/diagnostics/petals --out /kgfm-src/results/incite/petals_v1.json' ) \
        >> "$LOG" 2>&1
    [ "$ok" -eq 1 ] && done_mark R2 || fail_mark R2
  else
    fail_mark R2
  fi
fi

# ---- R3/R4: seed repeats --------------------------------------------------
for pair in "R3 1337" "R4 7"; do
  set -- $pair; st="$1"; seed="$2"
  if ! skip "$st"; then
    if incite_pretrain "$st" /kgfm-src/configs/incite_v1_full.yaml "$seed" "v1-seed$seed"; then
      ( cd "$INC" && env INCITE_CKPT="/kgfm-src/output/incite-pretrain-v1-seed$seed/incite_best.pth" \
          INCITE_CONFIG=/kgfm-src/configs/incite_v1_full.yaml \
          INCITE_RANKS="/kgfm-src/ranks/incite-v1-seed$seed" \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          scripts/docker_run.sh incite bash -c \
          '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' ) \
          >> "$LOG" 2>&1
      done_mark "$st"
    else
      fail_mark "$st"
    fi
  fi
done

say "=== baseline orchestrator finished ==="
