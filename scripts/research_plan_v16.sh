#!/usr/bin/env bash
# Research plan v16 (2026-09-03, 23:16, machine clock): v15 plus the stratified
# dev protocol. D0 (22:48) ranked MX1 0.0084 BELOW L1 on the uniform dev
# sample while the benchmark ranks it above: transductive valid queries
# are almost all seen-answer, the cells where the prior costs, and the
# benchmark is 61 / 66 percent such queries. diagnostics/dev_eval.py now
# samples per half-link scenario and weights the cells the benchmark's
# way (protocol "stratified_v2"). New stage D0W, after SC1: the four
# reference dev numbers recomputed under that protocol (the uniform
# files are kept as *.uniform.json). The recipe decision accepts only
# stratified_v2 dev files, so no uniform number is ever compared with a
# stratified one. A failed D0W sets output/research-plan/RECIPE_HOLD and
# the recipe decision waits for a person (delete the file), so R0/R1
# never run on a silent default. Takes over from v15 at the stage
# boundary. Everything else is v15 verbatim (its header follows).
#
# Research plan v15 (2026-09-03, 21:00, machine clock): v14 with the second
# verifier's fixes. (1) dev_eval ran `$ALLOC` as a command (exit 127): now
# `env $ALLOC`. (2) MX15 had no dev-suite number, so it could never be the
# accepted modification. (3) UDECAY stages re-anchored their decay on a
# crash-resume: `--schedule_start 1` pins it. (4) dev_eval at batch 8 and
# 1000 queries per graph; the recipe rule compares a candidate with MX1
# on the graphs BOTH have (at least six). (5) the seed repeats pass
# --synth_seed 2048 + seed - 1024 so their synthetic stream differs too.
# (6) after a takeover, stale extension locks are removed and a finished
# PG2P is not repeated. Everything else is v14 verbatim (its header follows).
#
# Research plan v14 (2026-09-03, 21:40): v13 reordered so that EVERY recipe
# candidate lands before the recipe decision and the paper-model runs come
# after all of them (the user's point: the recipe is fixed once). Order
# after PG2/PG2P and D0: SC1, FMX, MX15, MX45, MXS9, MX2a, RR2, the recipe
# decision (at most ONE accepted modification, the one with the largest
# dev-suite gain; never an untested combination), R0 and R1 at seed 1024,
# X1/X2, MX2H, the seeds, the chores, R1L. Takes over from v12/v13 at the
# stage boundary. Everything else is v13 verbatim (its header follows).
#
# Research plan v13 (2026-09-03, 21:00): the queue after the independent
# review (docs/HANDOFF.md, "the review"). Protocol: every lever now gets a
# DEV-SUITE verdict first (diagnostics/dev_eval.py: valid splits of nine
# transductive graphs outside the diet and outside the 41 test graphs,
# results/incite/dev/<name>.json), then its 41-graph dump. The paper
# recipe is fixed by a rule recorded here, not by a test-set argmax.
# Order after PG2/PG2P: D0 (dev numbers of L1, MX1, G1), SC1 (the
# scenario-conditioned readout), FMX (the 3-graph floor plus the mix: the
# matched-diet test), X1/X2 (TRIX at 20k: the matched-budget test), the
# recipe decision, R0 (from scratch, no mix) and R1 (from scratch, the
# recipe) at seed 1024, MX15 and MX45 (the dose curve), MX2H (the bundle at
# half dose), the seeds 1337 and 7 of R1 and R0, MX2a, RR2 (weight 0.2),
# MXS9 (share 0.9), the chores, and R1L (the recipe at 60k steps, one
# seed). SEEDS_HOLD holds the seed stages. Takes over from v12 at its
# stage boundary. Everything else is v12 verbatim (its header follows).
#
# Research plan v12 (2026-09-03, 19:15): MX2 LOST (0.4512 / 0.3717, below
# MX1 on both groups with intervals excluding zero, SYNTH_V2_RESULT.md),
# and v11 had just started PG1 on MX2's recipe. v12 stops that at once and
# rebases the three hypotheses on MX1's recipe: PG2 (the gate) + PG2P (its
# pruning curve), MX2a (isolated relation blocks alone: bisection step one
# and the base RR2 needs), RR2 (rule recovery on MX2a), MXS2 (the query
# draw at the benchmark's unseen-answer share), MX2b (64 uniform certified
# negatives + 4 positives, no isolation: bisection step two). Then MX15, R1
# (the winner from scratch), the chores, R2/R3. Everything else is v11
# verbatim (its header follows).
#
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
UDECAY="--lr_schedule linear --lr_final 0 --warmup_steps 500 --keep_every 1000 --schedule_start 1"

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

# Dev-suite evaluation (the review's protocol): valid splits of nine
# transductive graphs outside the diet and the 41 test graphs.
# $1 ckpt (container path)  $2 config (container path)  $3 name
dev_eval() {
  local ckpt="$1" cfg="$2" name="$3"
  [ -f "$INC/results/incite/dev/$name.json" ] && return 0
  ( cd "$INC" && env $ALLOC scripts/docker_run.sh incite bash -c "
      cd /kgfm-src/output/incite-run && \
      PYTHONPATH=/kgfm-src/output/incite-run:/kgfm-src/shared \
      TRIX_ROOT=/kgfm-src/output/incite-run/trix \
      python /kgfm-src/diagnostics/dev_eval.py --config $cfg --ckpt $ckpt \
        --val_samples 1000 --batch_size 8 --out /kgfm-src/results/incite/dev/$name.json" ) >> "$LOG" 2>&1
  [ -f "$INC/results/incite/dev/$name.json" ]
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

say "=== research plan v16 start (pid $$) ==="

# ---- takeover from v12/v13/v14/v15 at the stage boundary ------------------
# The stage in progress is the STAGE tag of output/incite-pretrain (absent
# between a finished training and its marker, and during PG2P). Wait for
# that stage's marker (snapshot-* markers do not count), stop v12, then
# stop the containers it started for its NEXT stage. Containers are
# stopped only at a boundary.
PRED='^bash scripts/research_plan_v1[2345]\.sh'
markers() { ls "$ORC"/*.done "$ORC"/*.failed 2>/dev/null | grep -vc '/snapshot-'; }
if pgrep -f "$PRED" > /dev/null; then
  cur="$(cat "$INC/output/incite-pretrain/STAGE" 2>/dev/null || true)"
  if [ -z "$cur" ]; then
    say "predecessor between stages; waiting for a STAGE tag or a new marker"
    n0="$(markers)"
    until [ -f "$INC/output/incite-pretrain/STAGE" ] || [ "$(markers)" -gt "$n0" ] || ! pgrep -f "$PRED" > /dev/null; do
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
    # a container stopped mid-start leaves torch's extension build lock
    # behind (younger than in_container.sh's ten-minute tolerance)
    find "$ROOT/output" -maxdepth 3 -path '*torch_extensions*' -name lock -delete 2>/dev/null
    # a pruning curve the predecessor finished is not repeated
    if [ -f "$INC/results/incite/gate_prune.json" ] && ! skip PG2P; then done_mark PG2P; fi
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

# ---- PG2: MX1 plus the proof-guided propagation gate (vs MX1) ----
if ! skip PG2; then
  if trained 4g-synth30-gate || {
       seed_pretrain_dir PG2 &&
       PLAN_INIT_FROM=/kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       incite_pretrain PG2 /kgfm-src/configs/incite_phase1_4g_synth30_gate.yaml 4g-synth30-gate 10000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$UDECAY"; }; then
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth30-gate/incite_last.pth \
         /kgfm-src/configs/incite_phase1_4g_synth30_gate.yaml incite-4g-synth30-gate-last; then done_mark PG2; else fail_mark PG2; fi
  else
    fail_mark PG2
  fi
fi

# ---- PG2P: the pruning curve of the gate on DEV10 (valid splits) ----------
if ! skip PG2P; then
  if trained 4g-synth30-gate; then
    ( cd "$INC" && scripts/docker_run.sh incite bash -c '
        cd /kgfm-src/output/incite-run && \
        PYTHONPATH=/kgfm-src/output/incite-run:/kgfm-src/shared \
        TRIX_ROOT=/kgfm-src/output/incite-run/trix \
        python /kgfm-src/diagnostics/gate_prune_dev.py \
          --config /kgfm-src/configs/incite_phase1_4g_synth30_gate.yaml \
          --ckpt /kgfm-src/output/incite-pretrain-4g-synth30-gate/incite_last.pth \
          --fracs 0,0.2,0.4,0.6,0.8,0.9,0.95 --val_samples 500 --batch_size 4 \
          --out /kgfm-src/results/incite/gate_prune.json' ) >> "$LOG" 2>&1
    if [ -f "$INC/results/incite/gate_prune.json" ]; then done_mark PG2P; else fail_mark PG2P; fi
  else
    fail_mark PG2P
  fi
fi

# ---- D0: dev-suite numbers of the references ------------------------------
if ! skip D0; then
  ok=1
  dev_eval /kgfm-src/output/incite-pretrain-4g-decay/incite_last.pth /kgfm-src/configs/incite_phase1.yaml L1 || ok=0
  dev_eval /kgfm-src/output/incite-pretrain-4g-synth30/incite_last.pth /kgfm-src/configs/incite_phase1.yaml MX1 || ok=0
  dev_eval /kgfm-src/output/incite-pretrain-4g-unary/incite_last.pth /kgfm-src/configs/incite_phase1_4g_unary.yaml G1 || ok=0
  dev_eval /kgfm-src/output/incite-pretrain-decay/incite_last.pth /kgfm-src/configs/incite_phase1.yaml L2 || ok=0
  [ "$ok" -eq 1 ] && done_mark D0 || fail_mark D0
fi

# ---- SC1: MX1 plus the scenario-conditioned readout (vs MX1) ----
if ! skip SC1; then
  if trained 4g-synth30-scenario || {
       seed_pretrain_dir SC1 &&
       PLAN_INIT_FROM=/kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       incite_pretrain SC1 /kgfm-src/configs/incite_phase1_4g_synth30_scenario.yaml 4g-synth30-scenario 10000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$UDECAY"; }; then
    dev_eval /kgfm-src/output/incite-pretrain-4g-synth30-scenario/incite_last.pth /kgfm-src/configs/incite_phase1_4g_synth30_scenario.yaml SC1 || say "SC1: dev eval failed (continuing)"
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth30-scenario/incite_last.pth \
         /kgfm-src/configs/incite_phase1_4g_synth30_scenario.yaml incite-4g-synth30-scenario-last; then done_mark SC1; else fail_mark SC1; fi
  else
    fail_mark SC1
  fi
fi

# ---- D0W: the references' dev numbers under the stratified protocol -------
# D0 ran the uniform protocol (2026-09-03, 22:48). A uniform valid sample is
# almost all seen-answer queries, so it ranked MX1 0.0084 below L1 while
# the benchmark ranks it above. diagnostics/dev_eval.py now stratifies by
# half-link scenario and weights the cells the benchmark's way; the four
# reference numbers are recomputed once so that every candidate is
# compared like for like. A uniform file is kept beside as *.uniform.json.
if ! skip D0W; then
  for n in L1 MX1 G1 L2; do
    f="$INC/results/incite/dev/$n.json"
    if [ -f "$f" ] && ! grep -q '"protocol": "stratified_v2"' "$f"; then
      mv "$f" "$INC/results/incite/dev/$n.uniform.json"
    fi
  done
  ok=1
  dev_eval /kgfm-src/output/incite-pretrain-4g-decay/incite_last.pth /kgfm-src/configs/incite_phase1.yaml L1 || ok=0
  dev_eval /kgfm-src/output/incite-pretrain-4g-synth30/incite_last.pth /kgfm-src/configs/incite_phase1.yaml MX1 || ok=0
  dev_eval /kgfm-src/output/incite-pretrain-4g-unary/incite_last.pth /kgfm-src/configs/incite_phase1_4g_unary.yaml G1 || ok=0
  dev_eval /kgfm-src/output/incite-pretrain-decay/incite_last.pth /kgfm-src/configs/incite_phase1.yaml L2 || ok=0
  if [ "$ok" -eq 1 ]; then
    done_mark D0W
  else
    fail_mark D0W
    touch "$ORC/RECIPE_HOLD"
    say "D0W failed: RECIPE_HOLD set, the recipe decision waits (fix, then delete the file)"
  fi
fi

# ---- FMX: the 3-graph floor plus the mix, matched diet (vs L2 and TRIX) ----
if ! skip FMX; then
  if trained synth30 || {
       seed_pretrain_dir FMX "$INC/output/incite-pretrain-phase1/incite_last.pth" &&
       incite_pretrain FMX /kgfm-src/configs/incite_phase1_synth30.yaml synth30 30000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    dev_eval /kgfm-src/output/incite-pretrain-synth30/incite_last.pth /kgfm-src/configs/incite_phase1.yaml FMX || say "FMX: dev eval failed (continuing)"
    if incite_eval /kgfm-src/output/incite-pretrain-synth30/incite_last.pth \
         /kgfm-src/configs/incite_phase1.yaml incite-synth30-last; then done_mark FMX; else fail_mark FMX; fi
  else
    fail_mark FMX
  fi
fi

# ---- MX15: synthetic mix at 15% (vs L1 and MX1) ----------------------------
if ! skip MX15; then
  if trained 4g-synth15 || {
       seed_pretrain_dir MX15 "$INC/output/incite-pretrain-4g/incite_last.pth" &&
       incite_pretrain MX15 /kgfm-src/configs/incite_phase1_4g_synth15.yaml 4g-synth15 30000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    dev_eval /kgfm-src/output/incite-pretrain-4g-synth15/incite_last.pth /kgfm-src/configs/incite_phase1.yaml MX15 || say "MX15: dev eval failed (continuing)"
    ok=1
    incite_eval /kgfm-src/output/incite-pretrain-4g-synth15/incite_last.pth \
      /kgfm-src/configs/incite_phase1.yaml incite-4g-synth15-last || ok=0
    [ "$ok" -eq 1 ] && done_mark MX15 || fail_mark MX15
  else
    fail_mark MX15
  fi
fi

# ---- MX45: the mix at 45 percent, the dose curve's high point (vs MX1) ----
if ! skip MX45; then
  if trained 4g-synth45 || {
       seed_pretrain_dir MX45 "$INC/output/incite-pretrain-4g/incite_last.pth" &&
       incite_pretrain MX45 /kgfm-src/configs/incite_phase1_4g_synth45.yaml 4g-synth45 30000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    dev_eval /kgfm-src/output/incite-pretrain-4g-synth45/incite_last.pth /kgfm-src/configs/incite_phase1.yaml MX45 || say "MX45: dev eval failed (continuing)"
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth45/incite_last.pth \
         /kgfm-src/configs/incite_phase1.yaml incite-4g-synth45-last; then done_mark MX45; else fail_mark MX45; fi
  else
    fail_mark MX45
  fi
fi

# ---- MXS9: MX1 with 90 percent unseen-answer synthetic queries (vs MX1) ----
if ! skip MXS9; then
  if trained 4g-synth30-s90 || {
       seed_pretrain_dir MXS9 "$INC/output/incite-pretrain-4g/incite_last.pth" &&
       incite_pretrain MXS9 /kgfm-src/configs/incite_phase1_4g_synth30_share90.yaml 4g-synth30-s90 30000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    dev_eval /kgfm-src/output/incite-pretrain-4g-synth30-s90/incite_last.pth /kgfm-src/configs/incite_phase1.yaml MXS9 || say "MXS9: dev eval failed (continuing)"
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth30-s90/incite_last.pth \
         /kgfm-src/configs/incite_phase1.yaml incite-4g-synth30-s90-last; then done_mark MXS9; else fail_mark MXS9; fi
  else
    fail_mark MXS9
  fi
fi

# ---- MX2a: MX1 plus isolated relation blocks only, bisection step one (vs MX1) ----
if ! skip MX2a; then
  if trained 4g-synth30-iso || {
       seed_pretrain_dir MX2a "$INC/output/incite-pretrain-4g/incite_last.pth" &&
       incite_pretrain MX2a /kgfm-src/configs/incite_phase1_4g_synth30_iso.yaml 4g-synth30-iso 30000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    dev_eval /kgfm-src/output/incite-pretrain-4g-synth30-iso/incite_last.pth /kgfm-src/configs/incite_phase1.yaml MX2a || say "MX2a: dev eval failed (continuing)"
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth30-iso/incite_last.pth \
         /kgfm-src/configs/incite_phase1.yaml incite-4g-synth30-iso-last; then done_mark MX2a; else fail_mark MX2a; fi
  else
    fail_mark MX2a
  fi
fi

# ---- RR2: MX2a plus rule recovery at weight 0.2 (vs MX2a and MX1) ----
if ! skip RR2; then
  if trained 4g-synth30-iso-rules || {
       seed_pretrain_dir RR2 &&
       PLAN_INIT_FROM=/kgfm-src/output/incite-pretrain-4g/incite_last.pth \
       incite_pretrain RR2 /kgfm-src/configs/incite_phase1_4g_synth30_iso_rules.yaml 4g-synth30-iso-rules 10000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$UDECAY"; }; then
    dev_eval /kgfm-src/output/incite-pretrain-4g-synth30-iso-rules/incite_last.pth /kgfm-src/configs/incite_phase1_4g_synth30_iso_rules.yaml RR2 || say "RR2: dev eval failed (continuing)"
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth30-iso-rules/incite_last.pth \
         /kgfm-src/configs/incite_phase1_4g_synth30_iso_rules.yaml incite-4g-synth30-iso-rules-last; then done_mark RR2; else fail_mark RR2; fi
  else
    fail_mark RR2
  fi
fi

# ---- the recipe decision (recorded before any R stage runs) ---------------
# The paper recipe is MX1's (the mix in the decay phase only) plus AT MOST
# ONE modification: among SC1, MX15, MX45, MXS9 and RR2, a candidate is
# accepted if BOTH hold: its dev-suite mean exceeds MX1's by at least
# 0.003, and its paired 41-graph bootstrap against MX1 has an interval
# above zero on at least one group while the other group's point
# estimate is above -0.002; the accepted candidate with the largest
# dev-suite gain wins. Never an untested combination (the MX2 lesson).
recipe_decision() {
  python3 - "$INC" "$ROOT" <<'PY'
import json, os, re, subprocess, sys
inc, root = sys.argv[1], sys.argv[2]
def dev_graphs(name):
    """Per-graph dev numbers of a stratified_v2 file; None for a missing
    file or one written under the uniform protocol (never compared)."""
    p = os.path.join(inc, "results/incite/dev", name + ".json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    if j.get("protocol") != "stratified_v2":
        return None
    g = j.get("graphs", {})
    return {k: v for k, v in g.items() if v is not None}
def dev_gain(cand, ref):
    """Mean over the graphs both have (at least six), else None."""
    a, b = dev_graphs(cand), dev_graphs(ref)
    if a is None or b is None:
        return None
    common = sorted(set(a) & set(b))
    if len(common) < 6:
        return None
    return sum(a[g] - b[g] for g in common) / len(common)
def paired(a, b):
    out = subprocess.run([sys.executable, os.path.join(root, "scripts/paired_bootstrap.py"),
                          os.path.join(inc, "ranks", a), os.path.join(inc, "ranks", b)],
                         capture_output=True, text=True).stdout
    cells = {}
    for m in re.finditer(r"(ind_e|ind_er)\s+n=\d+\s+A [\d.]+\s+B [\d.]+\s+A-B ([+-][\d.]+)\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]", out):
        cells[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return cells if len(cells) == 2 else None
# candidate modifications of MX1's recipe, each paired against MX1
cands = {"mx1sc": ("SC1", "incite-4g-synth30-scenario-last"),
         "mx1f15": ("MX15", "incite-4g-synth15-last"),
         "mx1f45": ("MX45", "incite-4g-synth45-last"),
         "mx1s90": ("MXS9", "incite-4g-synth30-s90-last"),
         "mx1rules": ("RR2", "incite-4g-synth30-iso-rules-last")}
best, best_gain, notes = "mx1", 0.0, []
for tag, (stage, dump) in cands.items():
    gain = dev_gain(stage, "MX1")
    if gain is None:
        notes.append("%s: no dev number (or fewer than six common graphs)" % stage); continue
    note = "%s dev %+.4f" % (stage, gain)
    if gain >= 0.003 and os.path.isdir(os.path.join(inc, "ranks", dump)):
        cells = paired(dump, "incite-4g-synth30-last")
        if cells:
            e, er = cells["ind_e"], cells["ind_er"]
            ok = ((e[1] > 0) or (er[1] > 0)) and (e[0] >= -0.002) and (er[0] >= -0.002)
            note += ", test ind_e %+.4f [%+.4f, %+.4f], ind_er %+.4f [%+.4f, %+.4f] -> %s" % (e + er + ("accepted" if ok else "rejected",))
            if ok and gain > best_gain:
                best, best_gain = tag, gain
        else:
            note += ", no paired bootstrap"
    notes.append(note)
print(best + "|" + "; ".join(notes))
PY
}
# D0W's failure sets RECIPE_HOLD: without MX1's stratified dev number the
# decision would default to mx1 silently and R0/R1 (30k steps each) would
# run on that default. The hold waits for a person (delete the file).
while [ -f "$ORC/RECIPE_HOLD" ]; do sleep 300; done
if [ ! -f "$ORC/RECIPE" ]; then
  recipe_decision > "$ORC/RECIPE.tmp" && mv "$ORC/RECIPE.tmp" "$ORC/RECIPE"
fi
RECIPE="$(cut -d'|' -f1 "$ORC/RECIPE")"
say "recipe: $RECIPE ($(cut -d'|' -f2- "$ORC/RECIPE"))"
R0CFG=/kgfm-src/configs/incite_phase1_4g.yaml
case "$RECIPE" in
  mx1sc)    RCFG=/kgfm-src/configs/incite_recipe_mx1_scenario.yaml
            R0CFG=/kgfm-src/configs/incite_phase1_4g_scenario.yaml ;;
  mx1f15)   RCFG=/kgfm-src/configs/incite_recipe_mx1_f15.yaml ;;
  mx1f45)   RCFG=/kgfm-src/configs/incite_recipe_mx1_f45.yaml ;;
  mx1s90)   RCFG=/kgfm-src/configs/incite_recipe_mx1_s90.yaml ;;
  mx1rules) RCFG=/kgfm-src/configs/incite_recipe_mx1_rules.yaml ;;
  *)        RCFG=/kgfm-src/configs/incite_recipe_mx1.yaml ;;
esac

# one from-scratch run: $1 stage  $2 seed  $3 config  $4 suffix  $5 steps  $6 extra flags
scratch_stage() {
  local st="$1" seed="$2" cfg="$3" suffix="$4" total="$5" flags="$6"
  flags="$flags --synth_seed $((2048 + seed - 1024))"
  if skip "$st"; then return 0; fi
  if [ "$st" != R0 ] && [ "$st" != R1 ] && [ -e "$ORC/SEEDS_HOLD" ]; then say "$st: held back by SEEDS_HOLD"; return 0; fi
  if trained "$suffix" || {
       seed_pretrain_dir "$st" &&
       incite_pretrain "$st" "$cfg" "$suffix" "$total" \
         INCITE_SEED="$seed" INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$flags"; }; then
    dev_eval "/kgfm-src/output/incite-pretrain-$suffix/incite_last.pth" "$cfg" "$st" || say "$st: dev eval failed (continuing)"
    if incite_eval "/kgfm-src/output/incite-pretrain-$suffix/incite_last.pth" \
         "$cfg" "incite-$suffix-last"; then done_mark "$st"; else fail_mark "$st"; fi
  else
    fail_mark "$st"
  fi
}
scratch_stage R0 1024 "$R0CFG" "nomix-$RECIPE-seed1024" 30000 "$DECAY"
scratch_stage R1 1024 "$RCFG" "recipe-$RECIPE-seed1024" 30000 "$DECAY"

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

# ---- MX2H: the MX2 bundle at half dose, with the negative mask (vs MX1 and MX15) ----
if ! skip MX2H; then
  if trained 4g-synth15v2 || {
       seed_pretrain_dir MX2H "$INC/output/incite-pretrain-4g/incite_last.pth" &&
       incite_pretrain MX2H /kgfm-src/configs/incite_phase1_4g_synth15_v2.yaml 4g-synth15v2 30000 \
         INCITE_TRAIN_GRAPHS=FB15k237,WN18RR,CoDExMedium,NELL995 \
         INCITE_TRAIN_EXTRA_ARGS="$DECAY"; }; then
    dev_eval /kgfm-src/output/incite-pretrain-4g-synth15v2/incite_last.pth /kgfm-src/configs/incite_phase1.yaml MX2H || say "MX2H: dev eval failed (continuing)"
    if incite_eval /kgfm-src/output/incite-pretrain-4g-synth15v2/incite_last.pth \
         /kgfm-src/configs/incite_phase1.yaml incite-4g-synth15v2-last; then done_mark MX2H; else fail_mark MX2H; fi
  else
    fail_mark MX2H
  fi
fi

# ---- the seeds of R1 and R0 (SEEDS_HOLD holds them) -----------------------
scratch_stage R1S2 1337 "$RCFG" "recipe-$RECIPE-seed1337" 30000 "$DECAY"
scratch_stage R0S2 1337 "$R0CFG" "nomix-$RECIPE-seed1337" 30000 "$DECAY"
scratch_stage R1S3 7 "$RCFG" "recipe-$RECIPE-seed7" 30000 "$DECAY"
scratch_stage R0S3 7 "$R0CFG" "nomix-$RECIPE-seed7" 30000 "$DECAY"

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

# ---- R1L: the recipe at 60k steps, one seed (40k constant, 20k decay) ----
if ! skip R1L; then
  if [ -e "$ORC/SEEDS_HOLD" ]; then say "R1L: held back by SEEDS_HOLD"; else
    sed 's/start_step: 20001/start_step: 40001/' "$INC/configs/$(basename "$RCFG")" > "$INC/configs/generated_recipe_60k.yaml"
    DECAY60="--lr_schedule linear --lr_final 0 --warmup_steps 500 --keep_every 2000 --schedule_start 40001"
    scratch_stage R1L 1024 /kgfm-src/configs/generated_recipe_60k.yaml "recipe-$RECIPE-60k-seed1024" 60000 "$DECAY60"
  fi
fi

say "=== research plan finished ==="
