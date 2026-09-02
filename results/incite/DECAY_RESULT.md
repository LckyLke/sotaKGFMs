# Decay continuation (L1): the new reference, and a warning about continuations

Date: 2026-09-02. Setup: the 4-graph run's last checkpoint (step 20,000)
continued for 10,000 steps, linear lr decay 5e-4 to 0, warmup 500 (fresh
AdamW), no other change. Checkpoints kept every 1,000 steps.

## Benchmark (41 graphs, test splits; ranks/incite-4g-decay-last)

| model | ind_e MRR | ind_er MRR | ind_e H@10 | ind_er H@10 |
| --- | --- | --- | --- | --- |
| 4g last, step 20k | 0.4534 | 0.3825 | 0.5954 | 0.5591 |
| 4g decay, last, step 30k | 0.4560 | 0.3852 | 0.5973 | 0.5634 |
| 4g decay, DEV10 best (29k) | 0.4568 | 0.3854 | | |
| TRIX | 0.4562 | 0.3679 | 0.5931 | 0.5409 |

Decay minus start: +0.0025 [−0.002, +0.007] / +0.0028 [−0.002, +0.007],
11 of 18 and 17 of 23 graphs (paired graph bootstrap). Decay minus TRIX:
ind_e −0.0002 (tie), ind_er +0.0174 [+0.005, +0.036], 18 of 23 graphs;
Hits@10 ind_er +0.022 [+0.012, +0.037]. DEV10 scalar 0.4233 -> 0.4323.

## Per-scenario (scripts/halflink_report.py), ind_er

| | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| 4g last | 0.3936 | 0.1825 | 0.5795 | 0.2468 |
| 4g decay last | 0.4079 | 0.1724 | 0.5896 | 0.2032 |

The continuation alone moves scores toward seen-answer candidates:
SQSA/UQSA up 0.010 to 0.016, SQUA/UQUA down 0.010 to 0.044 (ind_e the
same pattern). Masking dose 1 (MASKING_RESULT.md) amplified this trend
three to five fold; its inverted result is partly the continuation's
own drift plus the hub-stripping popularity prior. Any lever tested as a
continuation must be read against THIS row, not against the 20k start.

## Standing

The reference model for every comparison from now on. Single seed.
Reference checkpoint: checkpoints/incite-4g-decay-last-step30k.pth.
