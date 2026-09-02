# INCITE checkpoints (committed 2026-09-02 so another machine can continue)

Each file is `{"model": state_dict, "step", "dev10", "seed", ...}`; load
with `incite.run.load_members(cfg, [path])` or `INCITE_CKPT` in
`scripts/run_incite.sh`. Config = the yaml that builds the matching
architecture (`incite/run.py::build_model`).

| file | config | what | ind_e / ind_er MRR (41 graphs) |
| --- | --- | --- | --- |
| incite-floor-3g-best-step17k.pth | configs/incite_phase1.yaml | 3-graph floor, DEV10 best | 0.4553 / 0.3740 |
| incite-floor-3g-last-step20k.pth | configs/incite_phase1.yaml | 3-graph floor, last | start of the L2 continuation |
| incite-4g-best-step17k.pth | configs/incite_phase1.yaml | 4-graph backbone, DEV10 best | 0.4542 / 0.3791 |
| incite-4g-last-step20k.pth | configs/incite_phase1.yaml | 4-graph backbone, last: THE REFERENCE and the start of every continuation | 0.4534 / 0.3825 |
| incite-4g-mask-dose1-last-step30k.pth | configs/incite_phase1.yaml | masking dose 1 (negative result); carries optimizer state | 0.4420 / 0.3604 |
| incite-joint-3g-best.pth | configs/incite_phase23_joint.yaml | joint entity + relation head | 0.4509 / 0.3709; relation 0.7286 / 0.8222 |
| incite-walksynth-3g-best.pth | configs/incite_phase21b_walksynth.yaml | walks + synthetic supervision (PETALS 82 / 94 percent) | 0.4512 / 0.3724 |
| incite-v1-composite-4g-best.pth | configs/incite_v1_full.yaml | v1 composite (walks, synth, joint) | 0.4500 / 0.3659; relation ind_er 0.8484 |
| incite-floor-family-soup.pth | configs/incite_phase1.yaml | average of four floor descendants | 0.4571 / 0.3775 |

Later checkpoints (decay, unary, mask dose 2, seed repeats) are added as
the plan on the first machine finishes them; see docs/HANDOFF.md on the
baseline branch for the schedule.
| incite-4g-decay-last-step30k.pth | configs/incite_phase1.yaml | 4-graph backbone + 10k-step linear decay continuation (L1), last: THE NEW REFERENCE (2026-09-02) | 0.4560 / 0.3852 |
| incite-4g-unary-last-step10k.pth | configs/incite_phase1_4g_unary.yaml | 4g last + unary channel, warm start, 10k decay (G1), last: best single model so far | 0.4571 / 0.3874 |
