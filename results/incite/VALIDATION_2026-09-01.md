# Validation pass of the inherited results (2026-09-01)

A new session re-derived every headline number from the rank parquets
with `shared/metrics.py` (float64) and read the code, the provenance
records and the training logs behind them. This note records what held,
what did not, and what changed as a result.

## Recomputed group means (41 inductive graphs, test splits, seed 1024)

| model | ind_e MRR | ind_er MRR | ind_e H@10 | ind_er H@10 |
| --- | --- | --- | --- | --- |
| TRIX (released, 3 graphs) | 0.4562 | 0.3679 | 0.5931 | 0.5409 |
| FLOCK (40 of 41 graphs; ind_er over 22) | 0.4558 | 0.3674 | 0.6027 | 0.5502 |
| KG-ICL | 0.4240 | 0.3722 | 0.5684 | 0.5494 |
| INCITE floor, 3 graphs | 0.4553 | 0.3740 | 0.5960 | 0.5475 |
| INCITE 4 graphs | 0.4542 | 0.3791 | 0.5970 | 0.5609 |
| INCITE joint (3g) | 0.4509 | 0.3709 | 0.5912 | 0.5504 |
| INCITE walks+synth (3g) | 0.4512 | 0.3724 | 0.5938 | 0.5502 |
| INCITE support (3g) | 0.4553 | 0.3741 | 0.5952 | 0.5503 |
| INCITE v1 composite (4g) | 0.4500 | 0.3659 | 0.5977 | 0.5539 |

Every number in the phase result files reproduces to four decimals.

## What held

* The evaluation driver uses TRIX's own filter graph, strict masks and
  pessimistic 1-based rank; the model runs in eval mode without edge
  removal, as TRIX does. The loss is TRIX's loss, byte-copied. The
  factorized relation step has a gate test against the pinned TRIX layer
  at 1e-5. The 76-test suite passes in the container.
* The 4-graph run loaded NELL995 (PROVENANCE.json and train.log), although
  `configs/incite_phase1_4g.yaml` lists three graphs: the env override
  `INCITE_TRAIN_GRAPHS` carried the fourth. A trap, not a defect.
* The dead-lever verdicts (support twice, unsupervised walks) stand.

## What did not hold

1. **The "seed 1337" repeat trained at seed 1024.** `train_incite.sh`
   hard-coded the seed; the knob was documented and forwarded but never
   read. The run's graph draw sequence matched the seed-1024 run step for
   step. Killed at step ~2700; ledgered; fixed.
2. **The v1 composite is below TRIX on both entity groups** (0.4500 /
   0.3659 vs 0.4562 / 0.3679). The "co-SOTA" statement rests on the floor
   and 4-graph runs. Seed repeats of v1 measured the wrong model.
3. **Per-graph noise is large.** Same-architecture runs swing by up to
   0.10 MRR on WikiTopicsMT1:tax (0.25 to 0.36), Metafam (0.39 to 0.48)
   and NELLInductive:v1 (0.77 to 0.82). The floor's +0.006 on ind_er sits
   inside that noise; the 4-graph +0.011 is borderline and diet-caveated.
4. **Checkpoint selection uses valid splits of ten benchmark graphs.**
   TRIX selects on its pretraining mix. The gap between best and last
   selection scalars is small, but the protocol difference must be stated
   or removed. The plan evaluates last checkpoints beside best ones.
5. **Research chain 2 would have failed.** Its TRIX@20k stage kept
   `output_dir: /output`, an unmounted path; the checkpoints would have
   vanished with the container after the full training time. The eval
   stage picked the newest file, the last epoch, not TRIX's best epoch.
6. **KGPFN is not the +0.044 lever here.** On the 11 graphs measured it
   sits at TRIX level (mean delta about +0.004), except Metafam (+0.13).
   The suite needs days more on a shared GPU (one graph failed after
   19.5 h). Decision: background only, nothing gated on it.

## Rank-level complementarity (ranks only; no score fusion possible)

TRIX and the INCITE floor disagree at Hits@10 on about 6.6 percent of
queries (3.5 percent only-TRIX, 3.1 percent only-floor). The two-model
oracle (best rank per query) reaches 0.488 / 0.409, an upper bound that
every ensemble overstates. Score-level fusion is the E6 stage of the plan.

## What changed

* Seed knob real; optimizer state and lr in checkpoints; lr schedules;
  per-validation checkpoint retention (incite/pretrain.py).
* Test-time levers: bidirectional re-ranking and score ensembles
  (incite/rerank.py, incite/run.py, diagnostics/rerank_dev.py), with a
  DEV10 stop rule before any 41-graph evaluation.
* The queues were replaced by `scripts/research_plan.sh` on the baseline
  branch (order: test-time levers, decay continuations, fixed TRIX A/B,
  one synthetic-prior pilot), plus `scripts/kgpfn_small_retry.sh`.
