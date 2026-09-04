# The dose curve of the synthetic mix: 0, 15, 30, 45 percent (2026-09-04)

Four continuations of the 4-graph last checkpoint, 10k decay steps each,
resume-style (the same step numbering, the coin nested: a step that is
synthetic at 15 percent is synthetic at 30 and at 45), seed 1024, LAST
checkpoints. L1 = 0 percent (`ranks/incite-4g-decay-last`), MX15
(`incite-4g-synth15-last`, results/incite/DOSE15_RESULT.md), MX1
(`incite-4g-synth30-last`), MX45 (`incite-4g-synth45-last`, stage MX45 of
plan v16, 11:43, 4 Sep).

## The curve

| dose | ind_e MRR | ind_er MRR | dev, stratified | dev, graphs' own mix |
| --- | --- | --- | --- | --- |
| 0 (L1) | 0.4560 | 0.3852 | 0.3082 | 0.3378 |
| 15 (MX15) | 0.4621 | 0.3893 | 0.3050 | 0.3360 |
| 30 (MX1) | 0.4606 | 0.3851 | 0.3014 | 0.3290 |
| 45 (MX45) | 0.4585 | 0.3820 | 0.3003 | 0.3300 |

Paired graph bootstrap of MX45:

| pair | ind_e | ind_er |
| --- | --- | --- |
| MX45 − MX1 | −0.0021 [−0.0034, −0.0007], 3 of 18 | −0.0030 [−0.0060, −0.0006], 9 of 23 |
| MX45 − L1 | +0.0026 [−0.0016, +0.0068], 10 of 18 | −0.0032 [−0.0108, +0.0029], 10 of 23 |
| MX45 − MX15 | −0.0036 [−0.0063, −0.0011], 7 of 18 | −0.0073 [−0.0131, −0.0028], 3 of 23 |

Per-scenario MRR:

| group | dose | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | 0 | 0.4929 | 0.2856 | 0.6221 | 0.3187 |
| ind_e | 15 | 0.4856 | 0.3020 | 0.6132 | 0.3756 |
| ind_e | 30 | 0.4805 | 0.3051 | 0.6048 | 0.3864 |
| ind_e | 45 | 0.4803 | 0.3018 | 0.6014 | 0.3851 |
| ind_er | 0 | 0.4079 | 0.1724 | 0.5896 | 0.2032 |
| ind_er | 15 | 0.4013 | 0.1892 | 0.5827 | 0.2518 |
| ind_er | 30 | 0.3942 | 0.1876 | 0.5762 | 0.2684 |
| ind_er | 45 | 0.3911 | 0.1827 | 0.5734 | 0.2696 |

## Reading

1. The unseen-answer gain saturates by 30 percent (UQUA 0.319 → 0.376 →
   0.386 → 0.385 on ind_e), while the seen-answer cost keeps growing with
   the dose (SQSA 0.493 → 0.486 → 0.481 → 0.480; UQSA 0.622 → 0.613 →
   0.605 → 0.601). The net peaks at 15 percent on both groups and on the
   dev suite; 45 percent is below MX1 with both intervals below zero and
   below MX15 by 0.004 / 0.007.
2. On the held-out graphs the cost is monotone in the dose (0.3082 →
   0.3050 → 0.3014 → 0.3003): the prior has no dose at which it helps
   there, and the smallest dose costs least.
3. A dose below 15 percent is untested; the three points 0 / 15 / 30 put
   the peak near 15. MX15 is the leading recipe candidate; nothing in the
   dose direction is queued.
