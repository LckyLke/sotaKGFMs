# Half-link masking, first dose (M1): NEGATIVE, and inverted

Date: 2026-09-02. Setup: the 4-graph run's last checkpoint (step 20,000)
continued for 10,000 steps with a linear lr decay to 0 (warmup 500, fresh
AdamW) and half-link masking at p_answer 0.3, p_query 0.3
(incite/model.py::mask_halflinks). Paired baseline L1 (same continuation
without masking) is training.

## Benchmark (41 graphs, test splits; ranks/incite-4g-mask-last)

| checkpoint | ind_e MRR | ind_er MRR | ind_e H@10 | ind_er H@10 |
| --- | --- | --- | --- | --- |
| 4g last, step 20k (start) | 0.4534 | 0.3825 | 0.5954 | 0.5591 |
| + masked continuation, last (30k) | 0.4420 | 0.3604 | 0.5809 | 0.5407 |
| + masked continuation, DEV10 best (26k) | 0.4494 | 0.3706 | | |

Masked minus start: ind_e −0.0114 [−0.021, −0.002], ind_er −0.0220
[−0.037, −0.010] (paired graph bootstrap). DEV10 fell from the first
masked validation on (0.4233 at 20k to 0.4112 at 21k, 0.4045 at 30k).

## Per-scenario (scripts/halflink_report.py)

| ind_er | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| 4g last | 0.3936 | 0.1825 | 0.5795 | 0.2468 |
| masked | 0.4112 | 0.1223 | 0.5736 | 0.1591 |

| ind_e | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| 4g last | 0.4765 | 0.2920 | 0.6070 | 0.3615 |
| masked | 0.4949 | 0.2535 | 0.6224 | 0.2505 |

The intervention moved every cell the WRONG way: seen-answer scenarios
up, unseen-answer scenarios down by 0.04 to 0.11.

## Verified: the masking does what it says

On FB15k237's training graph (output/check_mask.py): every masked target
ends with zero incoming query-relation edges and zero inverse copies,
other edges untouched, head-query rows handled in tail form. Natural
shares among training positives: unseen answer 10.4 percent, unseen
query 22.2 percent; the dose put the training mix near the test mix
(about 37 percent unseen answers), as intended.

## Reading

The same check shows the mechanism: one masked target lost 484 edges of
one relation. Hubs stripped of a relation are not realistic unseen-answer
cases; real unseen-answer targets are rare entities. A model trained on
stripped hubs learns "a popular entity without the query relation is the
answer", a popularity prior, which lifts the popular-answer scenarios
(SQSA, UQSA) and sinks the rare-answer ones (SQUA, UQUA). Exactly the
table above. The batch-shared message graph adds a second cost: every
other query in the micro-batch sees the same holes.

Open confound: the continuation restarts AdamW (the 4-graph checkpoint
carries no optimizer state). L1 will show whether the decay-only
continuation also drops at 21k; if it does, both runs share that cost.

## Standing

Dead at this dose. Next dose (M2), if L1 clears the confound: mask the
answer half only, and only for targets whose query-relation in-degree is
at most 5 (`--mask_answer_maxdeg`), at p 0.5 among eligible rows, so the
synthetic unseen-answer rows look like the real ones and the graph loses
at most a handful of edges per row. The unary channel (G1) is independent
of this result.
