# GPU validation of the throughput/memory optimizations -- run AFTER the live phase-1 pretrain finishes

Nothing below may run while the 16-hour phase-1 run holds the GPU. All
commands write to `output/incite-bench/` (a new directory) and a bench
workdir -- never to `output/incite-run/` (the live snapshot) or
`output/incite-pretrain/` (the phase-1 checkpoints and logs).

What is being validated (results/incite/config_diff.md, 2026-08-28 rows):
per-round activation checkpointing (`train.checkpoint_activations`, default
off) and the fused FactorizedRelationStep forward (always on; ~12 pair_sum
launches per round -> 2). CPU tests already pin gradient correctness and
state-dict compatibility; the GPU questions are memory, step rate, and the
kernel-path gates.

## 1. Full test suite on the GPU (kernel gates included)

    scripts/test_incite.sh

Must pass entirely: the pair_sum/distmult_sum kernel-vs-fallback gates skip
on CPU and only run here; the new fusion tests then cover the fused shapes
through the rspmm path via the layer gate.

## 2. Bench configs (generated, not committed)

    mkdir -p output/incite-bench
    for BS in 8 16 32; do
      sed -e "s/^  batch_size: 8/  batch_size: $BS/" \
          -e "s/^  accum_steps: 4/  accum_steps: $((32 / BS))/" \
          -e "s/^  # checkpoint_activations: yes.*/  checkpoint_activations: yes/" \
          -e "/^  #  *# checkpoint/d" \
          configs/incite_phase1.yaml > output/incite-bench/phase1_ckpt_bs$BS.yaml
    done

(Each keeps the effective batch at 32: 8x4, 16x2, 32x1. Verify by eye that
`checkpoint_activations: yes` is uncommented in each.)

## 3. Memory at batch 16 and 32 with checkpointing

For BS in 16, then 32 (watch `nvidia-smi --query-gpu=memory.used --format=csv -l 1`
in a second terminal and note the peak; the phase-1 launch OOMed both sizes
at ~15.0 GiB without checkpointing):

    INCITE_WORKDIR=$PWD/output/incite-bench/workdir \
    INCITE_CONFIG=output/incite-bench/phase1_ckpt_bs16.yaml \
    INCITE_TRAIN_STEPS=30 INCITE_DEV_GRAPHS=Metafam \
    INCITE_TRAIN_EXTRA_ARGS="--output_dir /kgfm-src/output/incite-bench/bs16 --val_interval 1000000" \
      scripts/docker_run.sh incite scripts/train_incite.sh "[0]"

(prepare the bench workdir once first: `scripts/prepare_incite_workdir.sh
output/incite-bench/workdir`; repeat the command with bs32. The trailing
`--val_interval` in EXTRA_ARGS wins over the earlier flag -- argparse
last-occurrence -- and the final-step validation runs on Metafam only, so
the 30 steps stay a pure train-loop measurement.)

Record: peak memory at 16 and 32, and whether 32 fits under 15.6 GiB.

## 4. Step rate

Read `it_per_s` from the step-30 log line of each bench run
(`output/incite-bench/bs*/pretrain.jsonl`; use `--log_every 10`) and compare
against the launched run's 0.35 updates/s at 8x4 accumulation. Three numbers:

  * bs8, checkpointing OFF (config with the flag still commented): the fused
    relation step alone vs the live run's baseline;
  * bs8, checkpointing ON: the recompute overhead;
  * bs32 single-piece, checkpointing ON: the target configuration.

updates/s = it_per_s (one optimizer update per step; accum halves/quarters
it_per_s at 16/8, so compare updates/s, not raw step time).

## 5. Ledger

Append the measured numbers to results/incite/config_diff.md (memory row +
step-rate row), then, if batch 32 fits and is faster per update, uncomment
`checkpoint_activations: yes` and set `batch_size: 32` / `accum_steps: 1`
in the config used for the NEXT training launch (phase 2). The phase-1
config stays exactly as launched.

## 6. Checkpoint continuity spot-check

Load the phase-1 best checkpoint through the current tree once on GPU
(strict load already covered by CPU tests; this is the end-to-end proof):

    INCITE_CKPT=output/incite-pretrain/incite_best.pth \
      scripts/docker_run.sh incite scripts/run_incite.sh Metafam "[0]"

(read-only use of output/incite-pretrain; adjust dataset arg to the runner's
usual smoke choice. Scores must reproduce the checkpoint's recorded DEV10
Metafam number under support-off eval.)
