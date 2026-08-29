#!/usr/bin/env bash
# Autonomous queue for the INCITE phase-2 program (2026-08-29, user order:
# "schedule it so everything runs by itself for at least 24 h").
#
# Stages (markers under output/orchestrator/, rerunning this script resumes):
#   S0  wait for the LIVE phase-2.2 support run (never touch a healthy run;
#       resume it per RESUME policy if it died mid-flight)
#   S1  archive phase-2.2 outputs
#   S2  41-graph entity eval of the phase-2.2 best   -> ranks/incite-support
#   S3  support usage probe (the 2.2 kill switch)    -> results/incite/support_probe.json
#   S4  phase-2.1b pretrain: walks + synthetic automorphic mix (revival)
#   S5  2.1b evals: PETALS + 41 graphs               -> ranks/incite-walksynth
#   S6  phase-2.3 pretrain: joint relation loss from the floor
#   S7  2.3 evals: entity + relation task            -> ranks/incite-joint, ranks-relation/incite-joint
#   S8  4-graph-mix floor pretrain (design E scale-up, ultra_4g precedent)
#   S9  4g eval                                      -> ranks/incite-4g
#
# Failure policy: a pretrain stage gets up to 3 crash-resumes; a stage that
# still fails is marked .failed and the queue continues with stages that do
# not depend on it (S6/S8 are independent of S4). Everything logs to
# output/orchestrator/log.txt.
#
# Restart after a host crash:  nohup scripts/orchestrator.sh & disown
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORC="$ROOT/output/orchestrator"
LOG="$ORC/log.txt"
FLOOR="/kgfm-src/output/incite-pretrain-phase1/incite_best.pth"
mkdir -p "$ORC"
cd "$ROOT"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
done_mark() { touch "$ORC/$1.done"; say "$1 DONE"; }
fail_mark() { touch "$ORC/$1.failed"; say "$1 FAILED (queue continues)"; }
skip() { [ -e "$ORC/$1.done" ] || [ -e "$ORC/$1.failed" ]; }

container_running() { docker ps --format '{{.Image}}' | grep -q '^kgfm/incite:'; }

train_finished() {  # $1 = train.log path
  [ -f "$1" ] && grep -q 'checkpoint reload OK' "$1"
}

# Launch one pretrain to completion with crash-resumes.
# $1 stage  $2 config(container path)  $3 init_from(container path or "")  $4 extra env "K=V ..."
run_pretrain() {
  local stage="$1" cfg="$2" init="$3" extra="$4" attempt=0
  local logf="$ROOT/output/incite-pretrain/train.log"
  while :; do
    if train_finished "$logf"; then return 0; fi
    if container_running; then sleep 300; continue; fi
    attempt=$((attempt + 1))
    [ "$attempt" -gt 4 ] && return 1
    local resume=""
    if [ -f "$ROOT/output/incite-pretrain/incite_last.pth" ]; then
      resume="/kgfm-src/output/incite-pretrain/incite_last.pth"
      say "$stage: resuming (attempt $attempt)"
    else
      say "$stage: launching (attempt $attempt)"
    fi
    env INCITE_CONFIG="$cfg" \
        ${init:+INCITE_INIT_FROM="$init"} \
        ${resume:+INCITE_RESUME="$resume"} \
        ${resume:+INCITE_INIT_FROM=} \
        $extra \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        scripts/docker_run.sh incite /kgfm-src/scripts/train_incite.sh "[0]" \
        >> "$LOG" 2>&1
    sleep 10
  done
}

eval_entity() {  # $1 ckpt  $2 config  $3 ranks dir (container paths for 1,2; host rel for 3)
  # stale claims from a crashed eval attempt block the retry; parquets, not
  # claims, are the completion record
  rm -rf "$ROOT/$(dirname "$3")/.claims-$(basename "$3")"
  env INCITE_CKPT="$1" INCITE_CONFIG="$2" INCITE_RANKS="/kgfm-src/$3" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      scripts/docker_run.sh incite bash -c \
      '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' \
      >> "$LOG" 2>&1
  [ "$(ls "$ROOT/$3"/*.parquet 2>/dev/null | wc -l)" -eq 41 ]
}

say "=== orchestrator start (pid $$) ==="

# ---- S0: the live phase-2.2 run ------------------------------------------
if ! skip S0; then
  say "S0: waiting for the live phase-2.2 run"
  if run_pretrain S0 /kgfm-src/configs/incite_phase22_support.yaml "$FLOOR" ""; then
    done_mark S0
  else
    fail_mark S0
  fi
fi

# ---- S1: archive ----------------------------------------------------------
if ! skip S1; then
  if [ -d "$ROOT/output/incite-pretrain" ] && [ ! -d "$ROOT/output/incite-pretrain-phase22" ]; then
    mv "$ROOT/output/incite-pretrain" "$ROOT/output/incite-pretrain-phase22"
  fi
  [ -d "$ROOT/output/incite-pretrain-phase22" ] && done_mark S1 || fail_mark S1
fi

# ---- S2: eval phase-2.2 ---------------------------------------------------
if ! skip S2; then
  if [ -e "$ORC/S1.done" ] && eval_entity \
       /kgfm-src/output/incite-pretrain-phase22/incite_best.pth \
       /kgfm-src/configs/incite_phase22_support.yaml ranks/incite-support; then
    done_mark S2
  else
    fail_mark S2
  fi
fi

# ---- S3: usage probe ------------------------------------------------------
if ! skip S3; then
  if [ -e "$ORC/S1.done" ] && \
     scripts/docker_run.sh incite bash -c \
       'cd /kgfm-src/output/incite-run && PYTHONPATH=/kgfm-src/output/incite-run:/kgfm-src/shared TRIX_ROOT=/kgfm-src/output/incite-run/trix python /kgfm-src/diagnostics/support_probe.py --ckpt /kgfm-src/output/incite-pretrain-phase22/incite_best.pth --config /kgfm-src/configs/incite_phase22_support.yaml --out /kgfm-src/results/incite/support_probe.json' \
       >> "$LOG" 2>&1 && [ -f "$ROOT/results/incite/support_probe.json" ]; then
    done_mark S3
  else
    fail_mark S3
  fi
fi

# ---- S4: walks revival pretrain ------------------------------------------
if ! skip S4; then
  if [ -f "$ROOT/configs/incite_phase21b_walksynth.yaml" ]; then
    if run_pretrain S4 /kgfm-src/configs/incite_phase21b_walksynth.yaml "$FLOOR" ""; then
      mv "$ROOT/output/incite-pretrain" "$ROOT/output/incite-pretrain-phase21b"
      done_mark S4
    else
      fail_mark S4
    fi
  else
    say "S4: config missing (synth work not landed)"; fail_mark S4
  fi
fi

# ---- S5: 2.1b evals -------------------------------------------------------
if ! skip S5; then
  if [ -e "$ORC/S4.done" ]; then
    scripts/docker_run.sh incite bash -c \
      'cd /kgfm-src/output/incite-run && PYTHONPATH=/kgfm-src/output/incite-run:/kgfm-src/shared TRIX_ROOT=/kgfm-src/output/incite-run/trix python /kgfm-src/diagnostics/petals_eval.py --ckpt /kgfm-src/output/incite-pretrain-phase21b/incite_best.pth --config /kgfm-src/configs/incite_phase21b_walksynth.yaml --instances /kgfm-src/diagnostics/petals --out /kgfm-src/results/incite/petals_walksynth.json' \
      >> "$LOG" 2>&1
    if eval_entity /kgfm-src/output/incite-pretrain-phase21b/incite_best.pth \
         /kgfm-src/configs/incite_phase21b_walksynth.yaml ranks/incite-walksynth \
       && [ -f "$ROOT/results/incite/petals_walksynth.json" ]; then
      done_mark S5
    else
      fail_mark S5
    fi
  else
    fail_mark S5
  fi
fi

# ---- S6: joint relation pretrain -----------------------------------------
if ! skip S6; then
  if run_pretrain S6 /kgfm-src/configs/incite_phase23_joint.yaml "$FLOOR" ""; then
    mv "$ROOT/output/incite-pretrain" "$ROOT/output/incite-pretrain-phase23"
    done_mark S6
  else
    fail_mark S6
  fi
fi

# ---- S7: 2.3 evals --------------------------------------------------------
if ! skip S7; then
  if [ -e "$ORC/S6.done" ]; then
    ok=0
    eval_entity /kgfm-src/output/incite-pretrain-phase23/incite_best.pth \
      /kgfm-src/configs/incite_phase23_joint.yaml ranks/incite-joint && ok=1
    env INCITE_CKPT=/kgfm-src/output/incite-pretrain-phase23/incite_best.pth \
        INCITE_CONFIG=/kgfm-src/configs/incite_phase23_joint.yaml \
        INCITE_TASK=relation INCITE_RANKS=/kgfm-src/ranks-relation/incite-joint \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        scripts/docker_run.sh incite bash -c \
        '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' \
        >> "$LOG" 2>&1
    [ "$ok" -eq 1 ] && done_mark S7 || fail_mark S7
  else
    fail_mark S7
  fi
fi

# ---- S8: 4-graph mix floor pretrain (design E, ultra_4g precedent) -------
if ! skip S8; then
  if run_pretrain S8 /kgfm-src/configs/incite_phase1.yaml "" \
       "INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995"; then
    mv "$ROOT/output/incite-pretrain" "$ROOT/output/incite-pretrain-4g"
    done_mark S8
  else
    fail_mark S8
  fi
fi

# ---- S9: 4g eval ----------------------------------------------------------
if ! skip S9; then
  if [ -e "$ORC/S8.done" ] && eval_entity \
       /kgfm-src/output/incite-pretrain-4g/incite_best.pth \
       /kgfm-src/configs/incite_phase1.yaml ranks/incite-4g; then
    done_mark S9
  else
    fail_mark S9
  fi
fi

say "=== orchestrator finished; stage markers in $ORC ==="
