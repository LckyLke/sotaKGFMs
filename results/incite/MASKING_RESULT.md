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

# Dose 2 (M2): answer only, in-degree cap 10, p 0.5 -- NEGATIVE again

Date: 2026-09-03. Same continuation as L1 (10k steps, linear decay),
masking only targets with at most 10 incoming query-relation edges, no
query masking. Paired against the decay-only reference (L1).

| checkpoint | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| L1 decay last (reference) | 0.4560 | 0.3852 |
| M2 last (30k) | 0.4384 | 0.3629 |
| M2 DEV10 best (21k, after 1,000 masked steps) | 0.4481 | 0.3811 |

M2 last minus reference: −0.0175 [−0.026, −0.009] / −0.0223 [−0.036,
−0.011]; 4 of 18 and 2 of 23 graphs. The degree cap did not change the
outcome, so hub stripping was not the whole story.

## The trajectory says what the lever is

| ind_er | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| reference | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| M2 after 1,000 masked steps (21k) | 0.3720 | 0.2061 | 0.5757 | 0.2428 |
| M2 after 10,000 masked steps (30k) | 0.4175 | 0.1045 | 0.5863 | 0.1508 |

After 1,000 masked steps the model moved exactly where the design
wanted: unseen-answer cells +0.034 / +0.040, at a cost of −0.036 on SQSA
(ind_e the same: SQUA +0.022, UQUA +0.038, SQSA −0.035). Net MRR still
below the reference (−0.008 / −0.004). With more masked steps the model
inverted to the M1 pattern. Two readings, both bad for the lever:

* Seen-answer reliance and unseen-answer competence trade against each
  other in this trunk; masking moves the operating point along that
  trade-off, and the test mix (60 to 70 percent seen-answer queries)
  rewards the reliance. The shortcut is rational for the metric.
* Continued masked training at high lr drifts to a worse optimum
  (popularity-like), which the cap did not prevent.

## Verdict

DEAD as a net lever, at both doses. Recorded, not tuned around. What
survives is a diagnostic: half-link masking is a scenario knob that
exposes the trade-off, worth a figure, not a method. Unseen-answer
capability has to come from elsewhere: the unary channel moved those
cells at no SQSA cost (UNARY_RESULT.md), and scenario-targeted synthetic
supervision remains untested.
