#!/usr/bin/env bash
# Run a command in one model's container with the work tree bind-mounted.
#
#   usage: scripts/docker_run.sh <model> <command> [args...]
#   e.g.   scripts/docker_run.sh semma /kgfm-src/scripts/run_semma.sh ind_e "[0]"
#
# This replaces the four per-model wrappers that used to live in a scratch
# directory. That directory is session-scoped and gets wiped, which left the
# project unable to start a run at all until the wrappers were retyped. Nothing
# needed to reproduce a result belongs outside the repository.
#
# The image tag is the first 8 characters of the pin in repos/PINS.json, so a
# repin cannot silently keep running the old image.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?usage: docker_run.sh <model> <command> [args...]}"; shift

TAG="$(python3 - "$ROOT" "$MODEL" <<'PY'
import json, sys
root, model = sys.argv[1], sys.argv[2]
pins = json.load(open(root + "/repos/PINS.json"))
pins = pins.get("repos", pins)
info = pins[model]
sha = info if isinstance(info, str) else (info.get("commit") or info.get("sha"))
print(sha[:8])
PY
)"
IMAGE="kgfm/${MODEL}:${TAG}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "no such image: $IMAGE" >&2
  echo "build it:  docker build -f containers/$MODEL/Dockerfile -t $IMAGE ." >&2
  exit 4
}

# Snapshot the runner rather than mounting it live. bash reads a script
# incrementally, so editing the mounted file while a container executes it makes
# the running shell resume at a shifted offset and misparse. That happened here
# once and corrupted the trap that hands files back to the host user. A
# per-invocation copy cannot be moved under a running container.
RUNNER="$(mktemp "${TMPDIR:-/tmp}/kgfm-run.XXXXXX")"
cp "$ROOT/scripts/in_container.sh" "$RUNNER"; chmod 0755 "$RUNNER"
trap 'rm -f "$RUNNER"' EXIT

# Pass through this model's own knobs, upper-cased: SEMMA_DATASETS, TRIX_REDO
# and so on. Each runner documents the ones it reads.
# kg-icl -> KGICL, not KG_ICL: the runner reads KGICL_DATASETS and a name
# that does not match is silently unset inside the container, not an error.
UP="$(echo "$MODEL" | tr -d '-' | tr '[:lower:]' '[:upper:]')"
ENVARGS=()
# Any new per-model knob must be added here. A variable that is not in this
# list is not an error: it is silently absent inside the container, and the
# runner falls back to its default as if it had never been set. That cost a
# wasted test run of FLOCK_BATCH_DIVISOR.
# CKPT/TABICL_CONFIG/BATCH_SIZE/NUM_POS/NUM_NEG were born with run_kgpfn.sh.
for suffix in DATASETS SHARD REDO EXTRA_ARGS WORKDIR RANK_DUMP_DIR \
              BATCH_DIVISOR UNSEEDED_WALKS FETCH_ENTITY_LABELS \
              DATA RANKS RESULTS \
              CKPT TABICL_CONFIG BATCH_SIZE NUM_POS NUM_NEG; do
  name="${UP}_${suffix}"
  # Only forward what is set. `-e VAR=` sets the variable to the empty string
  # rather than leaving it unset, and some consumers reject that outright:
  # libgomp aborts with "Invalid value for environment variable
  # OMP_NUM_THREADS" when it is empty.
  [ -n "${!name:-}" ] && ENVARGS+=(-e "${name}=${!name}")
done
for name in PYTORCH_CUDA_ALLOC_CONF OMP_NUM_THREADS; do
  [ -n "${!name:-}" ] && ENVARGS+=(-e "${name}=${!name}")
done

exec docker run --rm \
  --gpus '"device=0"' \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  "${ENVARGS[@]}" \
  -v "$ROOT:/kgfm-src" \
  -v "$ROOT/output:/kgfm/output" \
  -v "$RUNNER:/usr/local/bin/kgfm-run:ro" \
  --shm-size=8g \
  "$IMAGE" \
  kgfm-run "$@"
