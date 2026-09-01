#!/usr/bin/env bash
# Research plan (2026-09-01, approved by the user): test-time levers first,
# then learning-rate-decay continuations, the fixed TRIX matched-budget A/B,
# and one synthetic-prior pilot. Supersedes baseline_orchestrator.sh's R3/R4
# (they trained at seed 1024 under a "seed 1337" label -- the seed knob was
# never read), queued_research.sh (gated on a KGPFN suite that needs days)
# and research_chain2.sh (its TRIX stage wrote checkpoints to an unmounted
# path). Markers in output/research-plan/; rerun the script to resume.
#
#   E1  4-graph LAST checkpoint eval           -> ranks/incite-4g-last
#   E2  floor-family weight soup + eval        -> ranks/incite-soup
#   E3  DEV10 re-ranking sweep (4g best)       -> results/incite/rerank_dev.json
#   E4  41-graph re-ranking eval, 4g best      -> ranks/incite-4g-rerank
#   E5  41-graph re-ranking eval, floor best   -> ranks/incite-rerank
#   E6  score ensemble of four trunks          -> ranks/incite-ens4
#   F0  FLOCK FBIngram:25 retry (patch 0005)   -> ranks/flock 41/41
#   L1  4g continuation 20k->30k, linear decay -> ranks/incite-4g-decay(-last)
#   L2  floor continuation 20k->30k            -> ranks/incite-decay(-last)
#   X1  TRIX@20k from scratch, their code      -> output/trix-20k
#   X2  eval TRIX@20k best epoch and last      -> ranks/trix-20k-best / -last
#   P1  synthetic-prior 100% pilot, 10k steps  -> ranks/incite-synth100-pilot
#
# E4/E5 carry a stop rule: if E3 finds no k that lifts the DEV10 selection
# scalar by >= 0.002 over k=0, the re-ranking evals are skipped and the
# lever is recorded dead. GPU sharing: the KGPFN suite (1.7 GB) may run
# alongside; nothing else may. Restart after a crash or reboot:
#   nohup scripts/research_plan.sh >> output/research-plan/nohup.log 2>&1 & disown
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INC="$ROOT/../sotaKGFMs-incite"
ORC="$ROOT/output/research-plan"
LOG="$ORC/log.txt"
mkdir -p "$ORC"
cd "$ROOT"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
done_mark() { touch "$ORC/$1.done"; say "$1 DONE"; }
fail_mark() { touch "$ORC/$1.failed"; say "$1 FAILED (plan continues)"; }
skip() { [ -e "$ORC/$1.done" ] || [ -e "$ORC/$1.failed" ]; }
nparquet() { ls "$1"/*.parquet 2>/dev/null | wc -l; }
incite_container_running() { docker ps --format '{{.Image}}' | grep -q '^kgfm/incite:'; }
ALLOC=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 41-graph entity eval of one INCITE checkpoint (or a comma list = ensemble).
# $1 ckpt(s, container paths)  $2 config (container path)  $3 ranks dir name
# $4 extra args for incite/run.py (optional)
incite_eval() {
  local ckpt="$1" cfg="$2" name="$3" extra="${4:-}"
  ( cd "$INC" && env INCITE_CKPT="$ckpt" INCITE_CONFIG="$cfg" \
      INCITE_RANKS="/kgfm-src/ranks/$name" INCITE_SUPPORT=skip \
      ${extra:+INCITE_EXTRA_ARGS="$extra"} $ALLOC \
      scripts/docker_run.sh incite bash -c \
      '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' ) \
      >> "$LOG" 2>&1
  [ "$(nparquet "$INC/ranks/$name")" -ge 41 ]
}

# INCITE pretrain stage with crash-resumes; output/incite-pretrain is moved
# to output/incite-pretrain-<suffix> on completion. Extra env comes in $5+.
# $1 stage  $2 config  $3 suffix  $4 total steps  $5.. env assignments
incite_pretrain() {
  local stage="$1" cfg="$2" suffix="$3" total="$4"; shift 4
  local logf="$INC/output/incite-pretrain/train.log" attempt=0
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
    say "$stage: attempt $attempt (resume: ${resume:-initial})"
    ( cd "$INC" && env INCITE_CONFIG="$cfg" INCITE_TRAIN_STEPS="$total" \
        ${resume:+INCITE_RESUME="$resume"} $ALLOC "$@" \
        scripts/docker_run.sh incite /kgfm-src/scripts/train_incite.sh "[0]" ) \
        >> "$LOG" 2>&1
    sleep 10
  done
}

say "=== research plan start (pid $$) ==="

# The killed seed-1024-under-a-1337-label run left output/incite-pretrain
# behind; keep it for the record, out of the way.
if [ -d "$INC/output/incite-pretrain" ] && [ ! -e "$INC/output/incite-pretrain-v1-seed1024-dup-killed" ]; then
  if grep -q '"step": 100,' "$INC/output/incite-pretrain/train.log" 2>/dev/null; then
    mv "$INC/output/incite-pretrain" "$INC/output/incite-pretrain-v1-seed1024-dup-killed"
    say "archived the killed duplicate run"
  fi
fi

# ---- E1: the 4g LAST checkpoint (selection-protocol check) ----------------
if ! skip E1; then
  if incite_eval /kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       /kgfm-src/configs/incite_phase1.yaml incite-4g-last; then done_mark E1; else fail_mark E1; fi
fi

# ---- E2: weight soup of the floor family -----------------------------------
if ! skip E2; then
  ( cd "$INC" && scripts/docker_run.sh incite bash -c '
      export CUDA_VISIBLE_DEVICES=""
      cd /kgfm-src && python scripts/make_soup.py \
        output/soup_floor_family.pth \
        output/incite-pretrain-phase1/incite_best.pth \
        output/incite-pretrain-phase21b/incite_best.pth \
        output/incite-pretrain-phase22/incite_best.pth \
        output/incite-pretrain-phase23/incite_best.pth' ) >> "$LOG" 2>&1
  if [ -f "$INC/output/soup_floor_family.pth" ] && \
     incite_eval /kgfm-src/output/soup_floor_family.pth \
       /kgfm-src/configs/incite_phase1.yaml incite-soup; then done_mark E2; else fail_mark E2; fi
fi

# ---- E3: DEV10 re-ranking sweep (valid splits, never test) ----------------
if ! skip E3; then
  ( cd "$INC" && scripts/docker_run.sh incite bash -c '
      cd /kgfm-src/output/incite-run && \
      PYTHONPATH=/kgfm-src/output/incite-run:/kgfm-src/shared \
      TRIX_ROOT=/kgfm-src/output/incite-run/trix \
      python /kgfm-src/diagnostics/rerank_dev.py \
        --config /kgfm-src/configs/incite_phase1.yaml \
        --ckpt /kgfm-src/output/incite-pretrain-4g/incite_best.pth \
        --ks 0,4,8,16 --weights 0.5,1.0 --val_samples 500 \
        --out /kgfm-src/results/incite/rerank_dev.json' ) >> "$LOG" 2>&1
  if [ -f "$INC/results/incite/rerank_dev.json" ]; then done_mark E3; else fail_mark E3; fi
fi

# the stop rule: best (k, weight) by the selection scalar, must beat k=0
pick_rerank() {
  python3 - "$INC/results/incite/rerank_dev.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
def sel(c): g = c["dev10_groups"]; return sum(g.values()) / len(g)
base = [c for c in r["cells"] if c["k"] == 0][0]
best = max((c for c in r["cells"] if c["k"] > 0), key=sel)
if sel(best) - sel(base) >= 0.002:
    print("%d %g" % (best["k"], best["weight"]))
else:
    print("dead")
PY
}

if [ -e "$ORC/E3.done" ]; then
  PICK="$(pick_rerank)"
  say "re-ranking pick: $PICK"
else
  PICK="dead"
fi

# ---- E4/E5: 41-graph re-ranking evals ------------------------------------
for pair in "E4 /kgfm-src/output/incite-pretrain-4g/incite_best.pth incite-4g-rerank" \
            "E5 /kgfm-src/output/incite-pretrain-phase1/incite_best.pth incite-rerank"; do
  set -- $pair; st="$1"; ckpt="$2"; name="$3"
  if skip "$st"; then continue; fi
  if [ "$PICK" = dead ]; then
    say "$st: re-ranking lever dead on DEV10, skipped"; fail_mark "$st"; continue
  fi
  set -- $PICK; K="$1"; W="$2"
  if incite_eval "$ckpt" /kgfm-src/configs/incite_phase1.yaml "$name" \
       "--rerank_k $K --rerank_weight $W"; then done_mark "$st"; else fail_mark "$st"; fi
done

# ---- E6: score ensemble of four trunks ------------------------------------
if ! skip E6; then
  if incite_eval "/kgfm-src/output/incite-pretrain-phase1/incite_best.pth,/kgfm-src/output/incite-pretrain-4g/incite_best.pth,/kgfm-src/output/incite-pretrain-phase23/incite_best.pth,/kgfm-src/output/incite-pretrain-phase22/incite_best.pth" \
       /kgfm-src/configs/incite_phase1.yaml incite-ens4; then done_mark E6; else fail_mark E6; fi
fi

# ---- F0: FLOCK's last graph ----------------------------------------------
if ! skip F0; then
  rm -rf "$ROOT/ranks/.claims-flock"
  env $ALLOC FLOCK_BATCH_DIVISOR=4 FLOCK_DATASETS="FBIngram:25" \
    FLOCK_WORKDIR=/kgfm-src/output/flock-run \
    scripts/docker_run.sh flock /kgfm-src/scripts/run_flock.sh ind_er "[0]" >> "$LOG" 2>&1
  if [ -f "$ROOT/ranks/flock/FBIngram_25.parquet" ]; then done_mark F0; else fail_mark F0; fi
fi

# ---- L1/L2: learning-rate-decay continuations -----------------------------
DECAY="--lr_schedule linear --lr_final 0 --warmup_steps 500 --keep_every 1000 --schedule_start 20001"
if ! skip L1; then
  rm -rf "$INC/output/incite-pretrain"
  mkdir -p "$INC/output/incite-pretrain"
  cp "$INC/output/incite-pretrain-4g/incite_last.pth" "$INC/output/incite-pretrain/incite_last.pth"
  if incite_pretrain L1 /kgfm-src/configs/incite_phase1_4g.yaml 4g-decay 30000 \
       INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
       INCITE_TRAIN_EXTRA_ARGS="$DECAY"; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-decay/incite_best.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-decay || ok=0
    incite_eval /kgfm-src/output/incite-pretrain-4g-decay/incite_last.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-decay-last || ok=0
    [ "$ok" -eq 1 ] && done_mark L1 || fail_mark L1
  else
    fail_mark L1
  fi
fi
if ! skip L2; then
  rm -rf "$INC/output/incite-pretrain"
  mkdir -p "$INC/output/incite-pretrain"
  cp "$INC/output/incite-pretrain-phase1/incite_last.pth" "$INC/output/incite-pretrain/incite_last.pth"
  if incite_pretrain L2 /kgfm-src/configs/incite_phase1.yaml decay 30000 \
       INCITE_TRAIN_EXTRA_ARGS="$DECAY"; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-decay/incite_best.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-decay || ok=0
    incite_eval /kgfm-src/output/incite-pretrain-decay/incite_last.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-decay-last || ok=0
    [ "$ok" -eq 1 ] && done_mark L2 || fail_mark L2
  else
    fail_mark L2
  fi
fi

# ---- X1: TRIX@20k matched-budget pretrain (output_dir FIXED) --------------
if ! skip X1; then
  mkdir -p "$ROOT/output/trix-20k" "$ROOT/data/roots/trix-20k"
  # its own processed root, copied from INCITE's pretrain root (the same
  # TRIX loaders built it): never write into a root a live runner reads
  cp -rn "$INC/data/roots/incite/pretrain/." "$ROOT/data/roots/trix-20k/" 2>/dev/null
  env $ALLOC scripts/docker_run.sh trix bash -c '
    cd /kgfm-src/output/trix-20k
    cp /kgfm/repos/trix/config/pretrain_entity.yaml cfg.yaml
    sed -i "s/batch_per_epoch: 80000/batch_per_epoch: 2000/; s|root: /kg-datasets/|root: /kgfm-src/data/roots/trix-20k/|; s|output_dir: /output|output_dir: /kgfm-src/output/trix-20k|" cfg.yaml
    grep -E "output_dir|batch_per_epoch|root:" cfg.yaml
    python /kgfm/repos/trix/src/pretrain_entity.py -c cfg.yaml --gpus "[0]"' \
    >> "$LOG" 2>&1
  if ls "$ROOT"/output/trix-20k/TRIX/*/*/model_epoch_*.pth >/dev/null 2>&1; then
    done_mark X1
  else
    fail_mark X1
  fi
fi

# ---- X2: eval best epoch (from TRIX's own log) and the last epoch ---------
if ! skip X2; then
  if [ -e "$ORC/X1.done" ]; then
    wd="$(dirname "$(ls -t "$ROOT"/output/trix-20k/TRIX/*/*/model_epoch_*.pth | head -1)")"
    last="$(ls "$wd"/model_epoch_*.pth | sort -t_ -k3 -n | tail -1)"
    bestep="$(grep -o 'Load checkpoint from model_epoch_[0-9]*.pth' "$wd/log.txt" 2>/dev/null | tail -1 | grep -o '[0-9]*' | tail -1)"
    best="$wd/model_epoch_${bestep:-10}.pth"
    say "X2: last=$last best=$best"
    ok=1
    for pair in "$best trix-20k-best" "$last trix-20k-last"; do
      set -- $pair; ck="$1"; name="$2"
      env TRIX_CKPT="${ck/$ROOT//kgfm-src}" TRIX_RANKS="/kgfm-src/ranks/$name" $ALLOC \
        scripts/docker_run.sh trix bash -c \
        '/kgfm-src/scripts/run_trix.sh ind_e "[0]"; /kgfm-src/scripts/run_trix.sh ind_er "[0]"' \
        >> "$LOG" 2>&1
      [ "$(nparquet "$ROOT/ranks/$name")" -ge 41 ] || ok=0
    done
    [ "$ok" -eq 1 ] && done_mark X2 || fail_mark X2
  else
    fail_mark X2
  fi
fi

# ---- P1: synthetic-prior 100% pilot, 10k steps ----------------------------
if ! skip P1; then
  rm -rf "$INC/output/incite-pretrain"
  if incite_pretrain P1 /kgfm-src/configs/incite_synthsweep_100.yaml synth100-pilot 10000 \
       INCITE_TRAIN_EXTRA_ARGS="--keep_every 1000"; then
    if incite_eval /kgfm-src/output/incite-pretrain-synth100-pilot/incite_best.pth \
         /kgfm-src/configs/incite_synthsweep_100.yaml incite-synth100-pilot; then
      done_mark P1
    else
      fail_mark P1
    fi
  else
    fail_mark P1
  fi
fi

say "=== research plan finished ==="
