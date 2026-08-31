#!/usr/bin/env bash
# Research chain 2 (2026-08-31): the TRIX matched-budget A/B and the
# synthetic-prior fraction sweep. Runs AFTER the baseline orchestrator's
# process exits (its R-stages own the GPU until then) and after research
# chain 1's markers are settled. Markers in output/research-chain2/.
#
#   X1  TRIX@20k from scratch (10 epochs x 2000 batches, their code)
#   X2  eval the TRIX@20k best epoch on the 41 graphs -> ranks/trix-20k
#   P*  synthetic-prior sweep (gated on PRIOR_VERIFIED sentinel):
#       P25/P75/P100 pretrain + 41-graph eval each -> ranks/incite-synth<f>
#
# Restart: nohup scripts/research_chain2.sh >> output/research-chain2/nohup.log 2>&1 & disown
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INC="$ROOT/../sotaKGFMs-incite"
ORC="$ROOT/output/research-chain2"
LOG="$ORC/log.txt"
mkdir -p "$ORC"
cd "$ROOT"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
skip() { [ -e "$ORC/$1.done" ] || [ -e "$ORC/$1.failed" ]; }

say "=== research chain 2 start (pid $$) ==="

# wait for the baseline orchestrator process to exit (it owns the GPU queue)
while pgrep -f 'bash scripts/baseline_orchestrator.sh' > /dev/null; do sleep 600; done
while pgrep -f 'bash scripts/queued_research.sh' > /dev/null; do sleep 300; done
say "upstream queues finished; GPU is mine"

# ---- X1: TRIX@20k matched-budget pretrain --------------------------------
if ! skip X1; then
  mkdir -p "$ROOT/output/trix-20k"
  scripts/docker_run.sh trix bash -c '
    cd /kgfm-src/output/trix-20k
    cp /kgfm/repos/trix/config/pretrain_entity.yaml cfg.yaml
    sed -i "s/num_epoch: 10/num_epoch: 10/; s/batch_per_epoch: 80000/batch_per_epoch: 2000/; s|root: /kg-datasets/|root: /kgfm-src/data/roots/trix/|" cfg.yaml
    python /kgfm/repos/trix/src/pretrain_entity.py -c cfg.yaml --gpus "[0]"' \
    >> "$LOG" 2>&1
  best="$(ls -t "$ROOT"/output/trix-20k/*/model_epoch_*.pth 2>/dev/null | head -1)"
  if [ -n "$best" ]; then
    touch "$ORC/X1.done"; say "X1 DONE (checkpoints under output/trix-20k)"
  else
    touch "$ORC/X1.failed"; say "X1 FAILED"
  fi
fi

# ---- X2: eval the A/B checkpoint -----------------------------------------
if ! skip X2; then
  if [ -e "$ORC/X1.done" ]; then
    # pretrain_entity.py reloads its best epoch at the end; the newest file
    # in the experiment dir after that reload is the one to evaluate
    best="$(ls -t "$ROOT"/output/trix-20k/*/model_epoch_*.pth | head -1)"
    say "X2: evaluating $best"
    env TRIX_CKPT="${best/$ROOT//kgfm-src}" TRIX_RANKS=/kgfm-src/ranks/trix-20k \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      scripts/docker_run.sh trix bash -c \
      '/kgfm-src/scripts/run_trix.sh ind_e "[0]"; /kgfm-src/scripts/run_trix.sh ind_er "[0]"' \
      >> "$LOG" 2>&1
    [ "$(ls "$ROOT/ranks/trix-20k"/*.parquet 2>/dev/null | wc -l)" -ge 41 ] \
      && { touch "$ORC/X2.done"; say "X2 DONE"; } \
      || { touch "$ORC/X2.failed"; say "X2 FAILED"; }
  else
    touch "$ORC/X2.failed"; say "X2 FAILED (no X1)"
  fi
fi

# ---- P-stages: the synthetic-prior sweep ---------------------------------
if [ ! -e "$ORC/PRIOR_VERIFIED" ]; then
  say "P: PRIOR_VERIFIED sentinel missing -- sweep deferred (rerun me later)"
else
  for f in 25 75 100; do
    st="P$f"
    if skip "$st"; then continue; fi
    logf="$INC/output/incite-pretrain/train.log"
    attempt=0; okstage=0
    while :; do
      if [ -f "$logf" ] && grep -q 'checkpoint reload OK' "$logf"; then
        mv "$INC/output/incite-pretrain" "$INC/output/incite-pretrain-synth$f"
        okstage=1; break
      fi
      if docker ps --format '{{.Image}}' | grep -q '^kgfm/incite:'; then sleep 300; continue; fi
      attempt=$((attempt + 1)); [ "$attempt" -gt 4 ] && break
      resume=""
      [ -f "$INC/output/incite-pretrain/incite_last.pth" ] && \
        resume="/kgfm-src/output/incite-pretrain/incite_last.pth"
      say "$st: attempt $attempt (resume: ${resume:-no})"
      ( cd "$INC" && env INCITE_CONFIG="/kgfm-src/configs/incite_synthsweep_$f.yaml" \
          ${resume:+INCITE_RESUME="$resume"} \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          scripts/docker_run.sh incite /kgfm-src/scripts/train_incite.sh "[0]" ) \
          >> "$LOG" 2>&1
      sleep 10
    done
    if [ "$okstage" = 1 ]; then
      ( cd "$INC" && env INCITE_CKPT="/kgfm-src/output/incite-pretrain-synth$f/incite_best.pth" \
          INCITE_CONFIG="/kgfm-src/configs/incite_synthsweep_$f.yaml" \
          INCITE_RANKS="/kgfm-src/ranks/incite-synth$f" \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          scripts/docker_run.sh incite bash -c \
          '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' ) \
          >> "$LOG" 2>&1
      touch "$ORC/$st.done"; say "$st DONE"
    else
      touch "$ORC/$st.failed"; say "$st FAILED"
    fi
  done
fi

say "=== research chain 2 finished ==="
