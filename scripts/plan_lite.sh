#!/usr/bin/env bash
# Plan-lite queue (2026-09-02, branch claude/plan-lite, second machine).
# Markers in output/plan-lite/<stage>.done|.failed|.blocked; rerun to resume.
# Stages are idempotent: nothing is cleared before its completion check.
#
#   CPU stages (run anywhere with the host venv or the incite image):
#   C_<mode>_w<withhold>_s<seed>   diagnostics/context_necessity.py runs
#                                  -> output/context-necessity/<name>/result.json
#   GPU stages (need docker access and the cu128 image on Blackwell):
#   B0  build kgfm/incite:incite01-cu128 from containers/incite/Dockerfile.cu128
#   B1  the INCITE test suite inside that image (kernel gates included)
#   B2  identity check: the committed 4g-last checkpoint on two DEV10 graphs
#       -> ranks/incite-4g-last-bw, compared against ranks/incite-4g-last
#
# Docker is optional for the C stages: PLAN_LITE_PY selects the python that
# runs them (default: the incite image via docker_run.sh with KGFM_STACK; set
# PLAN_LITE_PY=/path/to/venv/bin/python to run on the host CPU). GPU stages
# mark themselves .blocked, not .failed, when docker is unreachable.
#
#   restart:  nohup scripts/plan_lite.sh >> output/plan-lite/nohup.log 2>&1 & disown
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORC="$ROOT/output/plan-lite"
LOG="$ORC/log.txt"
mkdir -p "$ORC" "$ROOT/output/context-necessity"
cd "$ROOT"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
done_mark() { touch "$ORC/$1.done"; say "$1 DONE"; }
fail_mark() { touch "$ORC/$1.failed"; say "$1 FAILED (queue continues)"; }
block_mark() { touch "$ORC/$1.blocked"; say "$1 BLOCKED: $2"; }
skip() { [ -e "$ORC/$1.done" ] || [ -e "$ORC/$1.failed" ]; }
have_docker() { docker info > /dev/null 2>&1; }

PY="${PLAN_LITE_PY:-}"
PARALLEL="${PLAN_LITE_PARALLEL:-3}"      # CPU runs side by side
THREADS="${PLAN_LITE_THREADS:-8}"        # torch threads per CPU run
STEPS="${PLAN_LITE_STEPS:-2000}"

say "=== plan-lite start (pid $$) python=${PY:-container} parallel=$PARALLEL"

# ---- C: context-necessity diagnostic -------------------------------------
# name  mode  withhold  seed  extra
C_RUNS=(
  "floor_w1_s1024        floor        1.0 1024"
  "ctx_w1_s1024          context_only 1.0 1024"
  "resid_w1_s1024        residual     1.0 1024"
  "floor_w1_s1337        floor        1.0 1337"
  "ctx_w1_s1337          context_only 1.0 1337"
  "resid_w1_s1337        residual     1.0 1337"
  "floor_w1_s7           floor        1.0 7"
  "ctx_w1_s7             context_only 1.0 7"
  "resid_w1_s7           residual     1.0 7"
  "ctx_w1_s1024_detach   context_only 1.0 1024 --detach_rows"
  "resid_w1_s1024_detach residual     1.0 1024 --detach_rows"
  "floor_w0_s1024        floor        0.0 1024"
  "ctx_w0_s1024          context_only 0.0 1024"
)

run_c() {  # $1 name $2 mode $3 withhold $4 seed $5.. extra
  local name="$1" mode="$2" w="$3" seed="$4"; shift 4
  local out="$ROOT/output/context-necessity/$name"
  if [ -f "$out/result.json" ]; then done_mark "C_$name"; return 0; fi
  say "C_$name: start"
  if [ -n "$PY" ]; then
    ( cd "$ROOT" && OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        "$PY" diagnostics/context_necessity.py -c configs/context_necessity.yaml \
        --mode "$mode" --withhold "$w" --seed "$seed" --steps "$STEPS" \
        --out "$out" --gpus null "$@" ) > "$out.log" 2>&1
  else
    ( cd "$ROOT" && KGFM_STACK="${KGFM_STACK:-cu128}" scripts/docker_run.sh incite bash -c \
        "cd /kgfm-src && python diagnostics/context_necessity.py -c configs/context_necessity.yaml \
         --mode $mode --withhold $w --seed $seed --steps $STEPS \
         --out /kgfm-src/output/context-necessity/$name --gpus '[0]' $*" ) > "$out.log" 2>&1
  fi
  if [ -f "$out/result.json" ]; then done_mark "C_$name"; else fail_mark "C_$name"; fi
}

if [ -z "$PY" ] && ! have_docker; then
  say "no docker and no PLAN_LITE_PY: C stages need one of them"
else
  running=0
  for spec in "${C_RUNS[@]}"; do
    set -- $spec
    name="$1"
    if skip "C_$name"; then continue; fi
    mkdir -p "$ROOT/output/context-necessity/$name"
    run_c "$@" &
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
  done
  wait
  say "C stages finished"
  "${PY:-python3}" scripts/context_necessity_report.py > "$ROOT/results/incite/context_necessity_table.md" 2>> "$LOG" \
    && say "table: results/incite/context_necessity_table.md"
fi

# ---- B: the cu128 stack on this GPU ---------------------------------------
if ! have_docker; then
  for st in B0 B1 B2; do skip "$st" || [ -e "$ORC/$st.blocked" ] || block_mark "$st" "docker unreachable (add the user to the docker group)"; done
  say "=== plan-lite finished (GPU stages blocked) ==="
  exit 0
fi
rm -f "$ORC"/B*.blocked
IMG="kgfm/incite:$(python3 -c "import json;print(json.load(open('repos/PINS.json'))['repos']['incite']['commit'][:8])")-cu128"
if ! skip B0; then
  if docker image inspect "$IMG" > /dev/null 2>&1 || \
     docker build -f containers/incite/Dockerfile.cu128 -t "$IMG" . >> "$ORC/B0.build.log" 2>&1; then
    # the architecture check the build cannot do: this GPU must be in the
    # wheel's compiled list, or every kernel launch fails at run time
    if docker run --rm ${KGFM_GPU_ARGS:---gpus device=0} "$IMG" python -c "\
import torch; a = torch.cuda.get_arch_list(); n = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0); \
print('runtime:', n, cap, a); assert 'sm_%d%d' % cap in a, (cap, a)" >> "$ORC/B0.build.log" 2>&1; then
      done_mark B0
    else
      fail_mark B0
    fi
  else
    fail_mark B0
  fi
fi
if ! skip B1; then
  if [ -e "$ORC/B0.done" ] && KGFM_STACK=cu128 scripts/test_incite.sh > "$ORC/B1.tests.log" 2>&1; then
    done_mark B1
  else
    fail_mark B1
  fi
fi
if ! skip B2; then
  ok=0
  if [ -e "$ORC/B0.done" ]; then
    ( cd "$ROOT" && env KGFM_STACK=cu128 INCITE_CKPT=/kgfm-src/checkpoints/incite-4g-last-step20k.pth \
        INCITE_CONFIG=/kgfm-src/configs/incite_phase1.yaml INCITE_RANKS=/kgfm-src/ranks/incite-4g-last-bw \
        INCITE_SUPPORT=skip INCITE_DATASETS="NELLInductive:v1,Metafam:Metafam" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        scripts/docker_run.sh incite bash -c \
        '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' ) >> "$LOG" 2>&1
    if [ -f "$ROOT/ranks/incite-4g-last-bw/NELLInductive_v1.parquet" ] && [ -f "$ROOT/ranks/incite-4g-last-bw/Metafam.parquet" ]; then
      "${PY:-python3}" scripts/compare_dumps.py ranks/incite-4g-last ranks/incite-4g-last-bw \
        > "$ROOT/results/incite/stack_identity_cu128.md" 2>> "$LOG" && ok=1
    fi
  fi
  [ "$ok" -eq 1 ] && done_mark B2 || fail_mark B2
fi
say "=== plan-lite finished ==="
