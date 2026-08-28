# Phase 1 result: the reduction gate PASSES

Date: 2026-08-29. Checkpoint: incite_best.pth (step 17,000 of 20,000,
selected by zero-shot DEV10). Eval: all 41 inductive graphs, test splits,
shared rank definition, ranks in ranks/incite/ (row counts identical to
ranks/trix/ on every graph).

| quantity | INCITE @ 20k steps | TRIX @ ~100k steps | verdict |
| --- | --- | --- | --- |
| ind_e MRR (18 graphs) | 0.4553 | 0.4562 | parity (-0.0009, within 0.01) |
| ind_er MRR (23 graphs) | **0.3740** | 0.3679 | **exceeds** (+0.0061) |
| ind_e Hits@10 | 0.5960 | 0.5931 | no regression |
| ind_er Hits@10 | 0.5475 | 0.5409 | no regression |
| suite wall clock | 1085 s | 1000 s | 2.08x ULTRA (521 s); target <= 3x |

Context that must accompany these numbers:

* Budget: INCITE trained 20,000 steps; TRIX's released checkpoint trained
  ~100,000 (paper: 10 x 10,000, batch 32, same GPU class). Parity at 20%
  of the budget. The 100k extension is untested headroom, not a claim.
* ind_er 0.3740 is the best measured ind_er in this harness so far
  (ULTRA 0.3421, MOTIF 0.3491, SEMMA 0.3520, TRIX 0.3679; FLOCK and
  KGPFN not yet measured).
* The DEV10-valid "flatline" at ~0.33-0.35 during training was a
  valid-split artifact: the same checkpoint scores 0.3843 on the DEV10
  ind_er TEST splits (TRIX: 0.3765).
* Cost: the backbone costs ~1.09x TRIX in this harness. The design's
  1.4-3x-cheaper-than-TRIX claim assumed the paper's 10x TRIX-vs-ULTRA
  ratio; measured TRIX is 1.9x ULTRA, so there was less to save. The
  <= 3x ULTRA target passes; the "cheaper than TRIX" hope does not.
* Active design parts: A only (walks off, support off, entity loss only).
  Phase 2 levers start from this floor.
