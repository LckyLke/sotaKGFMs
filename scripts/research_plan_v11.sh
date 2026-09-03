#!/usr/bin/env bash
# Research plan v11 (2026-09-03, 13:40): v10 with two fixes after its
# takeover misfired. (1) The boundary detector counted EVERY marker in
# output/research-plan/, and the snapshot watcher's snapshot-MX1.done
# arrived while MXG1's evals ran: v10 stopped the eval container and
# started MXG1 from scratch (killed after one minute; the finished run
# was intact). Snapshot markers are ignored now. (2) The `trained` guard
# now covers MXG1 and G1 too (v10 had it only on MX2, MX15 and the new
# stages), so a stage whose run directory finished is never retrained.
# Everything else is v10 verbatim (its header follows).
#
# Research plan v10 (2026-09-03, 14:00): "queue everything, then wait".
# After MX2 (v9's addition) come the three generator-side hypotheses, each
# a 10k continuation paired against MX2: PG1 (the proof-guided propagation
# gate, plus PG1P = its pruning curve on DEV10), RR1 (rule recovery from
# the relation states), MXS (the query draw at the benchmark's unseen-answer
# share). Then MX15 and the chores as before. Then R1: the winner among all
# continuation levers, trained FROM SCRATCH for 30k steps (20k constant,
# warmup and linear decay over the last 10k) as the paper recipe, evaluated,
# and R2/R3: the same recipe at seeds 1337 and 7 -- automatic now (the
# user's 2026-09-03 instruction), held back only if
# output/research-plan/SEEDS_HOLD exists. Takes over from the running plan
# (v8 or v9) at its current stage boundary. A `trained` guard skips the
# training of a stage whose run directory already finished, so a takeover
# between a finished training and its marker cannot retrain it.
# Everything else is v9 verbatim (its header follows).
#
# Research plan v9 (2026-09-03, 12:10): adds MX2 -- MX1 plus the
# generator-side fixes of the rules prior (certified negatives, many of
# them and half of them hard; per-instance relation blocks; full-closure
# positives; configs/incite_phase1_4g_synth30_v2.yaml) -- right after
# MXG1 and BEFORE MX15, because the paper recipe depends on it and MX15 is
# a fraction ablation. Takes over from v8 at the MXG1 stage boundary: waits
# for v8's MXG1 marker, stops v8 and the MX15 container it has just
# started (identified by output/incite-pretrain/STAGE = MX15), then runs
# the marker-based stage list, which skips everything already done.
# Everything else is v8 verbatim (its header follows).
#
# Research plan v8 (2026-09-03, 09:40): MX1 is the first lever that moves
# ind_e (results/incite/SYNTH_MIX_RESULT.md). Adds, right after MX1 and
# before the chores: MXG1 (synthetic mix 30% + unary channel, warm start)
# and MX15 (mix at 15%). Takeover also stops kgfm/trix containers.
# Restart: nohup scripts/research_plan_v8.sh >> output/research-plan/nohup.log 2>&1 & disown
#
# ---- v7 header follows ----
# Research plan v7 (2026-09-03, 02:40): adds MX1 right after L2 -- the
# 4-graph continuation with 30 percent synthetic rules-prior steps, paired
# against L1 (results/incite/SYNTH_PILOT_RESULT.md motivates it: the
# synthetic-only model's scenario profile complements real data).
# Restart: nohup scripts/research_plan_v7.sh >> output/research-plan/nohup.log 2>&1 & disown
#
# ---- v6 header follows ----
# Research plan v6 (2026-09-02, 12:50): seed repeats deferred (user decision:
# run them once the paper model is known). Order after L1:
#   G1 M2 P1 L2 X1 X2 E4 E5 E6 F0, then B2 B3 C2 C3 ONLY if the sentinel
#   output/research-plan/SEEDS_GO exists (touch it to release them).
# Restart: nohup scripts/research_plan_v6.sh >> output/research-plan/nohup.log 2>&1 & disown
#
# ---- v5 header follows ----
# Research plan v5 (2026-09-02, 10:40): v4 with the masking dose corrected.
# M1 (p 0.3/0.3, hubs included) was net-negative and inverted
# (results/incite/MASKING_RESULT.md). MG1 (masking + unary at that dose)
# is replaced by M2: answer-only masking at p 0.5 for targets with at most
# 10 incoming query-relation edges (--mask_answer_maxdeg 10). Order after
# L1: G1 M2, then the seed stages pick the winner among L1/M1/G1/M2.
# Restart: nohup scripts/research_plan_v5.sh >> output/research-plan/nohup.log 2>&1 & disown
#
# ---- v4 header follows ----
# Research plan v4 (2026-09-02, 04:45): v3 with IDEMPOTENT stage directories.
# v3 cleared output/incite-pretrain at the top of every pretrain stage,
# before the completion check, so a relaunch of the plan wiped the finished
# M1 run (4.8 GPU hours) and restarted it. Now seed_pretrain_dir keeps a
# directory that belongs to the current stage (tag file STAGE, or an
# untagged run from v3) and clears only foreign ones.
# Restart: nohup scripts/research_plan_v4.sh >> output/research-plan/nohup.log 2>&1 & disown
#
# ---- v3 header follows ----
# Research plan v3 (2026-09-01, 23:15). Same markers as v1/v2. Fastest path
# to a three-seed headline table, in this order after E1-E3:
#   M1 L1 G1 MG1        the lever runs, all paired from the 4g last checkpoint
#   B2 B3               backbone seed repeats (4g, 20k, seeds 1337 and 7):
#                       lever-independent, so they run before the winner is known
#   C2 C3               the winning continuation recipe applied to each seed
#                       (picked from the -last dumps of M1/L1/G1/MG1)
#   E4 E5 E6            test-time levers (re-ranking evals, ensemble)
#   F0 L2 X1 X2 P1      baseline chores
# The takeover from v1 now happens after E3 (the DEV10 sweep), not E6.
# Restart: nohup scripts/research_plan_v3.sh >> output/research-plan/nohup.log 2>&1 & disown
#
# ---- v2 header follows ----
# Research plan v2 (2026-09-01, late evening). Same markers as v1
# (output/research-plan/), same stages, plus the two diagnosis-driven levers
# from results/incite/halflink.json and reachability.json:
#   M1  4g continuation WITH half-link masking (paired against L1)
#   G1  4g + unary channel, warm start, 10k steps with decay (paired vs L1)
# and a takeover: v2 waits for v1 to finish the cheap E-stages, then stops
# v1 and its containers and owns the GPU. Order after the E-stages:
#   L1 M1 G1 F0 L2 X1 X2 P1.
# Restart (old): nohup scripts/research_plan_v2.sh >> output/research-plan/nohup.log 2>&1 & disown
#
# ---- v1 header follows ----
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
# the run directory of a stage suffix finished training (moved into place)
trained() { [ -f "$INC/output/incite-pretrain-$1/train.log" ] && grep -q 'checkpoint reload OK' "$INC/output/incite-pretrain-$1/train.log"; }
nparquet() { ls "$1"/*.parquet 2>/dev/null | wc -l; }
incite_container_running() { docker ps --format '{{.Image}}' | grep -q '^kgfm/incite:'; }
ALLOC=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DECAY="--lr_schedule linear --lr_final 0 --warmup_steps 500 --keep_every 1000 --schedule_start 20001"
UDECAY="--lr_schedule linear --lr_final 0 --warmup_steps 500 --keep_every 1000"

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
    local resume="" init=""
    [ -f "$INC/output/incite-pretrain/incite_last.pth" ] && \
      resume="/kgfm-src/output/incite-pretrain/incite_last.pth"
    # a warm start (INIT_FROM in $PLAN_INIT_FROM) applies to the first
    # attempt only; crash-resumes continue from incite_last.pth instead
    [ -z "$resume" ] && [ -n "${PLAN_INIT_FROM:-}" ] && init="$PLAN_INIT_FROM"
    say "$stage: attempt $attempt (resume: ${resume:-initial}${init:+, init_from $init})"
    ( cd "$INC" && env INCITE_CONFIG="$cfg" INCITE_TRAIN_STEPS="$total" \
        ${resume:+INCITE_RESUME="$resume"} ${init:+INCITE_INIT_FROM="$init"} $ALLOC "$@" \
        scripts/docker_run.sh incite /kgfm-src/scripts/train_incite.sh "[0]" ) \
        >> "$LOG" 2>&1
    sleep 10
  done
}


# Seed output/incite-pretrain for a stage, idempotently. A directory with a
# train.log that carries this stage's tag (or no tag: a v3-era run) is kept
# for incite_pretrain to resume or move; a foreign stage's leftovers are
# cleared. $1 stage  $2 checkpoint to copy in as incite_last.pth (optional)
seed_pretrain_dir() {
  local stage="$1" src="${2:-}" d="$INC/output/incite-pretrain"
  if [ -f "$d/train.log" ]; then
    if [ ! -f "$d/STAGE" ] || [ "$(cat "$d/STAGE")" = "$stage" ]; then
      say "$stage: keeping the existing run directory"
      return 0
    fi
    say "$stage: clearing leftovers of stage $(cat "$d/STAGE")"
  fi
  rm -rf "$d"
  mkdir -p "$d"
  echo "$stage" > "$d/STAGE"
  [ -n "$src" ] && cp "$src" "$d/incite_last.pth"
  return 0
}

say "=== research plan v11 start (pid $$) ==="

# ---- takeover from the running plan (v8 or v9) at its stage boundary ------
# The stage in progress is the STAGE tag of output/incite-pretrain (absent
# between a finished training and its marker, while the evals run). Wait
# for that stage's marker, stop the predecessor, then stop the containers it
# started for its NEXT stage. Containers are stopped only at a boundary: a
# predecessor that died mid-stage leaves a live training container that the
# stage below waits for and resumes.
PRED='^bash scripts/research_plan_v([89]|10)\.sh'
if pgrep -f "$PRED" > /dev/null; then
  cur="$(cat "$INC/output/incite-pretrain/STAGE" 2>/dev/null || true)"
  if [ -z "$cur" ]; then
    say "predecessor between stages; waiting for a STAGE tag or a new marker"
    n0="$(ls "$ORC"/*.done "$ORC"/*.failed 2>/dev/null | grep -vc '/snapshot-')"
    until [ -f "$INC/output/incite-pretrain/STAGE" ] || [ "$(ls "$ORC"/*.done "$ORC"/*.failed 2>/dev/null | grep -vc '/snapshot-')" -gt "$n0" ] || ! pgrep -f "$PRED" > /dev/null; do
      sleep 15
    done
    cur="$(cat "$INC/output/incite-pretrain/STAGE" 2>/dev/null || true)"
  fi
  if [ -n "$cur" ] && ! skip "$cur"; then
    say "predecessor running stage $cur; waiting for its marker"
    until skip "$cur" || ! pgrep -f "$PRED" > /dev/null; do sleep 15; done
  fi
  boundary=0
  if [ -z "$cur" ] || skip "$cur"; then boundary=1; fi
  pkill -f "$PRED" && say "predecessor stopped (boundary=$boundary)"
  sleep 2
  if [ "$boundary" -eq 1 ]; then
    for img in kgfm/incite kgfm/flock kgfm/trix; do
      for c in $(docker ps --format '{{.ID}} {{.Image}}' | grep " $img:" | cut -d' ' -f1); do
        docker stop "$c" > /dev/null && say "stopped container $c ($img, the predecessor's next stage)"
      done
    done
  fi
fi
for img in kgfm/trix; do
  for c in $(docker ps --format '{{.ID}} {{.Image}}' | grep " $img:" | cut -d' ' -f1); do
    docker stop "$c" > /dev/null && say "stopped container $c ($img, left by a predecessor)"
  done
done
while docker ps --format '{{.Image}}' | grep -qE '^kgfm/(incite|flock|trix):'; do sleep 30; done

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
cells = [c for c in r["cells"] if c["k"] > 0]
best = max(cells, key=sel)
if sel(best) - sel(base) >= 0.002:
    # the cheapest cell within 0.001 of the best (eval cost grows with k)
    ok = [c for c in cells if sel(best) - sel(c) <= 0.001]
    pick = min(ok, key=lambda c: (c["k"], -sel(c)))
    print("%d %g" % (pick["k"], pick["weight"]))
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

# ---- M1: half-link masking, paired against L1 (same start, same decay) ---
if ! skip M1; then
  seed_pretrain_dir M1 "$INC/output/incite-pretrain-4g/incite_last.pth"
  if incite_pretrain M1 /kgfm-src/configs/incite_phase1_4g.yaml 4g-mask 30000 \
       INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
       INCITE_TRAIN_EXTRA_ARGS="$DECAY --mask_answer_p 0.3 --mask_query_p 0.3"; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-mask/incite_last.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-mask-last || ok=0
    incite_eval /kgfm-src/output/incite-pretrain-4g-mask/incite_best.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-mask || ok=0
    [ "$ok" -eq 1 ] && done_mark M1 || fail_mark M1
  else
    fail_mark M1
  fi
fi

# ---- L1: 4g decay continuation (the paired baseline) -----------------------------
if ! skip L1; then
  seed_pretrain_dir L1 "$INC/output/incite-pretrain-4g/incite_last.pth"
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
# ---- G1: the unary channel, warm start from the 4g last checkpoint ---------
if ! skip G1; then
  if trained 4g-unary || {
       seed_pretrain_dir G1 &&
       PLAN_INIT_FROM=/kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       incite_pretrain G1 /kgfm-src/configs/incite_phase1_4g_unary.yaml 4g-unary 10000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$UDECAY"; }; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-unary/incite_last.pth \
      /kgfm-src/configs/incite_phase1_4g_unary.yaml incite-4g-unary-last || ok=0
    incite_eval /kgfm-src/output/incite-pretrain-4g-unary/incite_best.pth \
      /kgfm-src/configs/incite_phase1_4g_unary.yaml incite-4g-unary || ok=0
    [ "$ok" -eq 1 ] && done_mark G1 || fail_mark G1
  else
    fail_mark G1
  fi
fi

# ---- M2: answer-only masking with the in-degree cap, paired against L1 ----
if ! skip M2; then
  seed_pretrain_dir M2 "$INC/output/incite-pretrain-4g/incite_last.pth"
  if incite_pretrain M2 /kgfm-src/configs/incite_phase1_4g.yaml 4g-mask2 30000 \
       INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
       INCITE_TRAIN_EXTRA_ARGS="$DECAY --mask_answer_p 0.5 --mask_query_p 0 --mask_answer_maxdeg 10"; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-mask2/incite_last.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-mask2-last || ok=0
    incite_eval /kgfm-src/output/incite-pretrain-4g-mask2/incite_best.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-mask2 || ok=0
    [ "$ok" -eq 1 ] && done_mark M2 || fail_mark M2
  else
    fail_mark M2
  fi
fi

# ---- P1: synthetic-prior 100% pilot, 10k steps ----------------------------
if ! skip P1; then
  seed_pretrain_dir P1
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

# ---- L2: floor continuation -------------------------------------------
if ! skip L2; then
  seed_pretrain_dir L2 "$INC/output/incite-pretrain-phase1/incite_last.pth"
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

# ---- MX1: 4g continuation with 30% synthetic rules-prior steps (vs L1) ---
if ! skip MX1; then
  seed_pretrain_dir MX1 "$INC/output/incite-pretrain-4g/incite_last.pth"
  if incite_pretrain MX1 /kgfm-src/configs/incite_phase1_4g_synth30.yaml 4g-synth30 30000 \
       INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
       INCITE_TRAIN_EXTRA_ARGS="$DECAY"; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-synth30/incite_last.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-synth30-last || ok=0
    incite_eval /kgfm-src/output/incite-pretrain-4g-synth30/incite_best.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-synth30 || ok=0
    [ "$ok" -eq 1 ] && done_mark MX1 || fail_mark MX1
  else
    fail_mark MX1
  fi
fi

# ---- MXG1: synthetic mix 30% + unary channel (warm start, 10k decay) ------
if ! skip MXG1; then
  if trained 4g-unary-synth30 || {
       seed_pretrain_dir MXG1 &&
       PLAN_INIT_FROM=/kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       incite_pretrain MXG1 /kgfm-src/configs/incite_phase1_4g_unary_synth30.yaml 4g-unary-synth30 10000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$UDECAY"; }; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-unary-synth30/incite_last.pth \
      /kgfm-src/configs/incite_phase1_4g_unary_synth30.yaml incite-4g-unary-synth30-last || ok=0
    incite_eval /kgfm-src/output/incite-pretrain-4g-unary-synth30/incite_best.pth \
      /kgfm-src/configs/incite_phase1_4g_unary_synth30.yaml incite-4g-unary-synth30 || ok=0
    [ "$ok" -eq 1 ] && done_mark MXG1 || fail_mark MXG1
  else
    fail_mark MXG1
  fi
fi

# ---- MX2: MX1 plus the generator-side fixes (vs MX1 and L1) ---------------
if ! skip MX2; then
  if trained 4g-synth30v2 || {
       seed_pretrain_dir MX2 "$INC/output/incite-pretrain-4g/incite_last.pth" &&
       incite_pretrain MX2 /kgfm-src/configs/incite_phase1_4g_synth30_v2.yaml 4g-synth30v2 30000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-synth30v2/incite_last.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-synth30v2-last || ok=0
    [ "$ok" -eq 1 ] && done_mark MX2 || fail_mark MX2
  else
    fail_mark MX2
  fi
fi

# ---- PG1: MX2 plus the proof-guided propagation gate (vs MX2) ----
if ! skip PG1; then
  if trained 4g-synth30v2-gate || {
       seed_pretrain_dir PG1 &&
       PLAN_INIT_FROM=/kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       incite_pretrain PG1 /kgfm-src/configs/incite_phase1_4g_synth30_v2_gate.yaml 4g-synth30v2-gate 10000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$UDECAY"; }; then
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth30v2-gate/incite_last.pth \
         /kgfm-src/configs/incite_phase1_4g_synth30_v2_gate.yaml incite-4g-synth30v2-gate-last; then done_mark PG1; else fail_mark PG1; fi
  else
    fail_mark PG1
  fi
fi

# ---- PG1P: the pruning curve of the gate on DEV10 (valid splits) ----------
if ! skip PG1P; then
  if trained 4g-synth30v2-gate; then
    ( cd "$INC" && scripts/docker_run.sh incite bash -c '
        cd /kgfm-src/output/incite-run && \
        PYTHONPATH=/kgfm-src/output/incite-run:/kgfm-src/shared \
        TRIX_ROOT=/kgfm-src/output/incite-run/trix \
        python /kgfm-src/diagnostics/gate_prune_dev.py \
          --config /kgfm-src/configs/incite_phase1_4g_synth30_v2_gate.yaml \
          --ckpt /kgfm-src/output/incite-pretrain-4g-synth30v2-gate/incite_last.pth \
          --fracs 0,0.2,0.4,0.6,0.8,0.9,0.95 --val_samples 500 --batch_size 4 \
          --out /kgfm-src/results/incite/gate_prune.json' ) >> "$LOG" 2>&1
    if [ -f "$INC/results/incite/gate_prune.json" ]; then done_mark PG1P; else fail_mark PG1P; fi
  else
    fail_mark PG1P
  fi
fi

# ---- RR1: MX2 plus rule recovery from the relation states (vs MX2) ----
if ! skip RR1; then
  if trained 4g-synth30v2-rules || {
       seed_pretrain_dir RR1 &&
       PLAN_INIT_FROM=/kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       incite_pretrain RR1 /kgfm-src/configs/incite_phase1_4g_synth30_v2_rules.yaml 4g-synth30v2-rules 10000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$UDECAY"; }; then
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth30v2-rules/incite_last.pth \
         /kgfm-src/configs/incite_phase1_4g_synth30_v2_rules.yaml incite-4g-synth30v2-rules-last; then done_mark RR1; else fail_mark RR1; fi
  else
    fail_mark RR1
  fi
fi

# ---- MXS: MX2 with the query draw at the benchmark's unseen-answer share (vs MX2) ----
if ! skip MXS; then
  if trained 4g-synth30v2-s37 || {
       seed_pretrain_dir MXS &&
       PLAN_INIT_FROM=/kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       incite_pretrain MXS /kgfm-src/configs/incite_phase1_4g_synth30_v2_share37.yaml 4g-synth30v2-s37 10000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$UDECAY"; }; then
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth30v2-s37/incite_last.pth \
         /kgfm-src/configs/incite_phase1_4g_synth30_v2_share37.yaml incite-4g-synth30v2-s37-last; then done_mark MXS; else fail_mark MXS; fi
  else
    fail_mark MXS
  fi
fi

# ---- MX15: synthetic mix at 15% (vs L1 and MX1) ----------------------------
if ! skip MX15; then
  if trained 4g-synth15 || {
       seed_pretrain_dir MX15 "$INC/output/incite-pretrain-4g/incite_last.pth" &&
       incite_pretrain MX15 /kgfm-src/configs/incite_phase1_4g_synth15.yaml 4g-synth15 30000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-synth15/incite_last.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-synth15-last || ok=0
    [ "$ok" -eq 1 ] && done_mark MX15 || fail_mark MX15
  else
    fail_mark MX15
  fi
fi

# ---- R1: the paper recipe from scratch, then its seeds -------------------
# The winner among the continuation levers by mean group MRR of the -last
# dumps is trained FROM SCRATCH for 30k steps: constant lr to 20k, then
# warmup and linear decay over the last 10k (--schedule_start 20001; steps
# before it run at the base lr since 2026-09-03). R2/R3 repeat it at seeds
# 1337 and 7, automatically, unless output/research-plan/SEEDS_HOLD exists.
pick_winner() {
  python3 - "$INC/ranks" <<'PY'
import os, sys
sys.path.insert(0, "/home/lukef/Dokumente/GitHub/sotaKGFMs/shared")
import metrics, suite
cands = {"L1": "incite-4g-decay-last", "G1": "incite-4g-unary-last",
         "MX1": "incite-4g-synth30-last", "MXG1": "incite-4g-unary-synth30-last",
         "MX15": "incite-4g-synth15-last", "MX2": "incite-4g-synth30v2-last",
         "PG1": "incite-4g-synth30v2-gate-last", "RR1": "incite-4g-synth30v2-rules-last",
         "MXS": "incite-4g-synth30v2-s37-last"}
best, best_v = None, -1.0
for name, d in cands.items():
    p = os.path.join(sys.argv[1], d)
    per = metrics.compute_dir(p, dtype="float64") if os.path.isdir(p) else {}
    if len(per) < 41:
        continue
    e = metrics.group_mean({k: v for k, v in per.items() if suite.by_id(k).group == "ind_e"}, "mrr")
    er = metrics.group_mean({k: v for k, v in per.items() if suite.by_id(k).group == "ind_er"}, "mrr")
    v = (e + er) / 2
    if v > best_v:
        best, best_v = name, v
print(best or "none")
PY
}
WIN="$(pick_winner)"
say "winning continuation recipe: $WIN"
case "$WIN" in
  L1)   WCFG=/kgfm-src/configs/incite_phase1_4g.yaml ;;
  G1)   WCFG=/kgfm-src/configs/incite_phase1_4g_unary.yaml ;;
  MX1)  WCFG=/kgfm-src/configs/incite_phase1_4g_synth30.yaml ;;
  MXG1) WCFG=/kgfm-src/configs/incite_phase1_4g_unary_synth30.yaml ;;
  MX15) WCFG=/kgfm-src/configs/incite_phase1_4g_synth15.yaml ;;
  MX2)  WCFG=/kgfm-src/configs/incite_phase1_4g_synth30_v2.yaml ;;
  PG1)  WCFG=/kgfm-src/configs/incite_phase1_4g_synth30_v2_gate.yaml ;;
  RR1)  WCFG=/kgfm-src/configs/incite_phase1_4g_synth30_v2_rules.yaml ;;
  MXS)  WCFG=/kgfm-src/configs/incite_phase1_4g_synth30_v2_share37.yaml ;;
  *)    WCFG="" ;;
esac
# one recipe run: $1 stage  $2 seed. From scratch, 30k steps, evaluated.
recipe_stage() {
  local st="$1" seed="$2" suffix
  if skip "$st"; then return 0; fi
  if [ -z "$WCFG" ]; then say "$st: no recipe (winner $WIN)"; fail_mark "$st"; return 0; fi
  if [ "$st" != R1 ] && [ -e "$ORC/SEEDS_HOLD" ]; then say "$st: held back by SEEDS_HOLD"; return 0; fi
  suffix="recipe-$WIN-seed$seed"
  if trained "$suffix" || {
       seed_pretrain_dir "$st" &&
       incite_pretrain "$st" "$WCFG" "$suffix" 30000 \
         INCITE_SEED="$seed" INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    if incite_eval "/kgfm-src/output/incite-pretrain-$suffix/incite_last.pth" \
         "$WCFG" "incite-$suffix-last"; then done_mark "$st"; else fail_mark "$st"; fi
  else
    fail_mark "$st"
  fi
}
recipe_stage R1 1024

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

# ---- R2/R3: the recipe at seeds 1337 and 7 (SEEDS_HOLD holds them) -------
recipe_stage R2 1337
recipe_stage R3 7

say "=== research plan finished ==="
