#!/usr/bin/env bash
# Zero-shot KGPFN over one suite group, dumping per-query ranks.
#
#   usage: scripts/run_kgpfn.sh <ind_e|ind_er|transductive> [gpus] [python]
#
# KGPFN evaluates one JointDataset spec per invocation, so the loop over the
# suite lives here. The spec spelling is exactly suite.run_id
# ("FB15k237Inductive:v1", "Metafam:Metafam", "FBNELL:FBNELL_v1") -- KGPFN's
# JointDataset parses "Name:version" natively, no translation needed.
#
# ------------------------------------------------------------------------
# WHAT KGPFN READS FROM THE TARGET GRAPH AT EVAL TIME -- state this next to
# any table built from these ranks. This is its method (in-context
# inference), not a protocol violation, but it is context no other model in
# this suite consumes. From the pinned code, not the paper:
#
#   * Per query (each test triple, and again for its head question asked as
#     the inverse-relation tail question), pfn/tasks.py::
#     build_context_relation_aware samples from the INFERENCE (message)
#     graph only -- never from test or validation target edges:
#       - up to num_pos=20 POSITIVES: real edges of the query relation,
#         drawn uniformly from the whole message graph (for head questions:
#         the inverse-typed copies, i.e. the same relation read backwards).
#         The query triple itself is a test edge, absent from that pool by
#         construction, and is excluded from the context by the forbidden
#         set as well. If the relation has fewer than 20 edges, the
#         shortfall is added to the negatives.
#       - num_neg=80 NEGATIVES: (query head, t', query relation) pairs that
#         are NOT edges, 80% with t' from the head's 2-hop out-neighbourhood,
#         the rest uniform, with a deterministic sweep fallback; a batch
#         whose relation is too dense to yield 80 non-edges is SKIPPED
#         upstream (dump row counts catch this).
#     The 20/80 budget is the pinned test.yaml's; the paper states 20/60
#     (see shared/published.json). script/test_kgpfn.py forces num_neg=40
#     on NELLInductive:v1 -- upstream's own special case, kept.
#   * context_label_correction=True then RELABELS context negatives that the
#     frozen structure encoder's MLP scores above the weakest positive to a
#     soft 0.5 -- still in-context, no gradient anywhere.
#   * The labelled context rows condition the TabICL feature transformer;
#     candidate scores come out of one in-context forward per query row.
#
# Cost accounting is a first-class result (the paper reports none):
# TIMINGS.jsonl carries wall seconds per graph invocation (dataset download/
# processing included on first touch -- rerunning a completed graph is the
# steady-state number), and the model's own CSV carries eval_seconds (the
# per-graph eval loop alone, in-context retrieval included) plus
# peak_gpu_mem_bytes. See patches/kgpfn/0004.
# ------------------------------------------------------------------------
#
# Env knobs (all forwarded by scripts/docker_run.sh -- a knob added here must
# be added to its list THE SAME DAY):
#   KGPFN_DATASETS       comma list of run_ids (default: the whole group)
#   KGPFN_SHARD          label for output/claims bookkeeping
#   KGPFN_REDO           re-run graphs that already have a parquet
#   KGPFN_WORKDIR        patched tree (default: the baked /kgfm/repos/kgpfn)
#   KGPFN_DATA           processed dataset root
#   KGPFN_RESULTS        where the model's own CSVs land
#   KGPFN_CKPT           the pretrained KGPFN checkpoint
#   KGPFN_TABICL_CONFIG  TabICL architecture yaml (extracted from its ckpt)
#   KGPFN_BATCH_SIZE     eval batch size (default 64, upstream's)
#   KGPFN_NUM_POS / KGPFN_NUM_NEG   context budget (default 20/80, upstream's)
#   KGPFN_EXTRA_ARGS     appended last, so flags there win
#   KGPFN_RANKS          rank dump dir (default ranks/kgpfn; K3 runs use their own)
#   KGPFN_LABEL_CORRECTION True|False  upstream's encoder relabeling of context
#                        negatives (default True, upstream's)
#   KGPFN_SHUFFLE_LABELS True|False  permute each query's context labels
#                        (patches/kgpfn/0006; the K3 "shuffled" condition)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_kgpfn.sh <ind_e|ind_er|transductive> [gpus] [python]}"
GPUS="${2:-[0]}"
PY="${3:-python3.12}"

WORKDIR="${KGPFN_WORKDIR:-/kgfm/repos/kgpfn}"
CKPT="${KGPFN_CKPT:-/kgfm-src/data/raw/kgpfn/cache/kgpfn_icl.pth}"
TABICL_CONFIG="${KGPFN_TABICL_CONFIG:-/kgfm-src/data/raw/kgpfn/tabicl.yaml}"
DATA_ROOT="${KGPFN_DATA:-$ROOT/data/roots/kgpfn}"
# KGPFN_RANKS (2026-09-02): the K3 context-ablation runs dump beside the
# reference dump, never over it -- one directory per condition.
RANKS="${KGPFN_RANKS:-$ROOT/ranks/kgpfn}"
RESULTS="${KGPFN_RESULTS:-$ROOT/results/kgpfn}"
OUT="$ROOT/output/kgpfn/${KGPFN_SHARD:-all}"
CONFIG="$WORKDIR/config/script/eval_zero_shot.yaml"

case "$GROUP" in
  ind_e|ind_er|transductive) ;;
  *) echo "unknown group: $GROUP" >&2; exit 2 ;;
esac

[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT (see data/raw/MANIFEST-kgpfn.json)" >&2; exit 4; }
[ -f "$TABICL_CONFIG" ] || { echo "no TabICL config at $TABICL_CONFIG" >&2; exit 4; }

DATASETS="${KGPFN_DATASETS:-$($PY "$ROOT/shared/suite.py" "$GROUP")}"
SHARD="${KGPFN_SHARD:-}"

mkdir -p "$DATA_ROOT" "$RANKS" "$RESULTS" "$OUT"

echo "group     : $GROUP"
echo "config    : $CONFIG"
echo "ckpt      : $CKPT"
echo "data root : $DATA_ROOT"
echo "ranks     : $RANKS"
echo "results   : $RESULTS"
echo "context   : num_pos=${KGPFN_NUM_POS:-20} num_neg=${KGPFN_NUM_NEG:-80} batch=${KGPFN_BATCH_SIZE:-64}"
echo "datasets  : $(tr ',' '\n' <<<"$DATASETS" | wc -l)${SHARD:+ (shard $SHARD)}"

# Ranks from a CPU run and ranks from a GPU run must never end up in one
# directory: the float32 kernels differ in low-order bits, which can flip a
# near-tie and move a rank, so a mixed directory yields a group mean that
# corresponds to no single measurement and nothing downstream would notice.
DEVICE=cpu; [ "$GPUS" != "null" ] && DEVICE=gpu
PROV="$RANKS/PROVENANCE.json"
if [ -f "$PROV" ]; then
  had=$($PY -c "import json;print(json.load(open('$PROV'))['device'])" 2>/dev/null || echo unknown)
  if [ "$had" != "$DEVICE" ] && [ -z "${KGPFN_REDO:-}" ]; then
    echo "REFUSING TO RUN: $RANKS holds $had ranks, this run is $DEVICE." >&2
    echo "Clear it first:  rm -rf $RANKS $ROOT/ranks/.claims-kgpfn" >&2
    exit 3
  fi
fi
$PY - "$PROV" "$DEVICE" "$CKPT" "${KGPFN_NUM_POS:-20}" "${KGPFN_NUM_NEG:-80}" <<'PROVPY'
import hashlib, json, os, platform, subprocess, sys
path, device, ckpt, num_pos, num_neg = sys.argv[1:6]
prov = {"device": device, "host": platform.platform(), "cpu_count": os.cpu_count()}
try:
    prov["gpu"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    prov["gpu"] = None
# The checkpoint IS the model; a rank directory must record which bytes made it.
digest = hashlib.sha256()
with open(ckpt, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
prov["checkpoint"] = os.path.basename(ckpt)
prov["checkpoint_sha256"] = digest.hexdigest()
prov["checkpoint_bytes"] = os.path.getsize(ckpt)
# The in-context budget is part of the measurement, not a nuisance parameter.
prov["context_num_pos"] = int(num_pos)
prov["context_label_correction"] = os.environ.get("KGPFN_LABEL_CORRECTION", "True")
prov["context_shuffle_labels"] = os.environ.get("KGPFN_SHUFFLE_LABELS", "False")
prov["context_num_neg"] = int(num_neg)
json.dump(prov, open(path, "w"), indent=2, sort_keys=True)
print("checkpoint sha256:", prov["checkpoint_sha256"])
PROVPY

# one claims dir per ranks dir: a K3 condition run must not inherit the
# reference run's claims (every graph would count as taken)
CLAIMS="$(dirname "$RANKS")/.claims-$(basename "$RANKS")"
mkdir -p "$CLAIMS"
# sys.path[0] is the script's directory here (script/), but datasets, configs
# and the working-directory bookkeeping resolve against the tree; running from
# anywhere else is the exact trap that bit CREST. cd, always.
cd "$WORKDIR"

ran=0; skipped=0; failed=0
for d in ${DATASETS//,/ }; do
  id="$($PY -c "import sys;sys.path.insert(0,'$ROOT/shared');import suite;print(suite.by_run_id('$d').id.replace(':','_'))")"
  if [ -z "${KGPFN_REDO:-}" ] && [ -s "$RANKS/$id.parquet" ]; then
    skipped=$((skipped+1)); continue
  fi
  if ! mkdir "$CLAIMS/$id" 2>/dev/null; then
    skipped=$((skipped+1)); continue
  fi

  echo ">>> $d"
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  index=$((ran + failed + 1))
  t0=$(date +%s.%N)
  if KGPFN_RANK_DUMP_DIR="$RANKS" KGPFN_RESULTS_DIR="$RESULTS" \
     $PY "$WORKDIR/script/test_kgpfn.py" \
      -c "$CONFIG" \
      -s 1024 \
      --dataset "$d" \
      --gpus "$GPUS" \
      --ckpt "$CKPT" \
      --tabicl_config "$TABICL_CONFIG" \
      --output_dir "$OUT" \
      --data_root "$DATA_ROOT" \
      --batch_size "${KGPFN_BATCH_SIZE:-64}" \
      --num_pos "${KGPFN_NUM_POS:-20}" \
      --num_neg "${KGPFN_NUM_NEG:-80}" \
      --context_label_correction "${KGPFN_LABEL_CORRECTION:-True}" \
      --context_shuffle_labels "${KGPFN_SHUFFLE_LABELS:-False}" \
      ${KGPFN_EXTRA_ARGS:-}; then
    status=ok; ran=$((ran+1))
  else
    status=failed
    failed=$((failed+1)); rmdir "$CLAIMS/$id" 2>/dev/null || true
    echo "!!! FAILED: $d"
  fi
  t1=$(date +%s.%N)
  $PY - "$RANKS/TIMINGS.jsonl" "$id" "$d" "$DEVICE" "$status" "$started" "$t0" "$t1" "$index" <<'TIMEPY'
import json, sys
path, gid, run_id, device, status, started, t0, t1, index = sys.argv[1:10]
with open(path, "a") as handle:
    handle.write(json.dumps({
        "dataset": gid, "run_id": run_id, "model": "kgpfn", "device": device,
        "status": status, "started": started, "seconds": round(float(t1) - float(t0), 3),
        "index": int(index),
    }, sort_keys=True) + "\n")
TIMEPY
done
echo "worker ${SHARD:-$GROUP} done: $ran ran, $skipped skipped, $failed failed"
