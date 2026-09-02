#!/usr/bin/env bash
# Snapshot soup (2026-09-01): average the last K kept checkpoints of one
# INCITE pretrain run (incite_step<N>.pth, written with --keep_every) and
# evaluate the average on the 41 inductive graphs. The floor-family soup
# gained +0.002 / +0.0035 over its best donor (E2), so the kept snapshots
# of every decay run get the same treatment.
#
#   usage: scripts/snapshot_soup.sh <run-suffix> <config-container-path> [K]
#   e.g.   scripts/snapshot_soup.sh 4g-decay /kgfm-src/configs/incite_phase1.yaml 5
#
# Reads  ../sotaKGFMs-incite/output/incite-pretrain-<suffix>/incite_step*.pth
# Writes ../sotaKGFMs-incite/output/soup-<suffix>-last<K>.pth
#        ../sotaKGFMs-incite/ranks/incite-<suffix>-soup<K>/
# Uses its own prepared work tree (output/incite-run-soup) so it never
# re-prepares the tree a running plan stage executes from.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INC="$ROOT/../sotaKGFMs-incite"
SUFFIX="${1:?run suffix}"; CFG="${2:?config}"; K="${3:-5}"
RUN="$INC/output/incite-pretrain-$SUFFIX"
donors=$(ls "$RUN"/incite_step*.pth 2>/dev/null | sed 's/.*incite_step\([0-9]*\).pth/\1 &/' | sort -n | tail -n "$K" | cut -d' ' -f2)
n=$(echo "$donors" | grep -c .)
if [ "$n" -lt 2 ]; then echo "need >= 2 kept snapshots under $RUN, found $n" >&2; exit 2; fi
cdonors=$(echo "$donors" | sed "s|$INC|/kgfm-src|")
out="/kgfm-src/output/soup-$SUFFIX-last$K.pth"
[ -d "$INC/output/incite-run-soup/incite" ] || ( cd "$INC" && scripts/prepare_incite_workdir.sh output/incite-run-soup > /dev/null )
( cd "$INC" && scripts/docker_run.sh incite bash -c "export CUDA_VISIBLE_DEVICES=''; cd /kgfm-src && python scripts/make_soup.py $out $(echo $cdonors | tr '\n' ' ')" )
[ -f "$INC/output/soup-$SUFFIX-last$K.pth" ] || { echo "soup build failed" >&2; exit 1; }
( cd "$INC" && env INCITE_WORKDIR=/kgfm-src/output/incite-run-soup INCITE_CKPT="$out" INCITE_CONFIG="$CFG" \
    INCITE_RANKS="/kgfm-src/ranks/incite-$SUFFIX-soup$K" INCITE_SUPPORT=skip \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    scripts/docker_run.sh incite bash -c \
    '/kgfm-src/scripts/run_incite.sh ind_e "[0]"; /kgfm-src/scripts/run_incite.sh ind_er "[0]"' )
echo "ranks: $(ls "$INC/ranks/incite-$SUFFIX-soup$K"/*.parquet 2>/dev/null | wc -l)/41"
