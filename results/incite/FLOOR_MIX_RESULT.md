# FMX: the 3-graph floor plus the mix, the matched-diet test (2026-09-04)

Stage FMX of plan v16 (`configs/incite_phase1_synth30.yaml`): the 3-graph
floor's last checkpoint (FB15k237, WN18RR, CoDExMedium: TRIX's diet)
continued for 10k decay steps with 30 percent synthetic rules-prior
steps, resume-style (steps 20001..30000, the same coin and instances as
MX1). Paired against L2 (the same continuation without the mix) and
against the released TRIX (the same diet). LAST checkpoint, 41 test
graphs, seed 1024. Not a recipe candidate (the recipe is 4-graph).

## Numbers

| model | diet | ind_e MRR | ind_er MRR |
| --- | --- | --- | --- |
| TRIX (released) | 3g | 0.4562 | 0.3679 |
| L2, floor + decay (`ranks/incite-decay-last`) | 3g | 0.4510 | 0.3745 |
| FMX, floor + decay + mix (`ranks/incite-synth30-last`) | 3g | 0.4563 | 0.3720 |
| L1, 4g + decay | 4g | 0.4560 | 0.3852 |
| MX1, 4g + decay + mix | 4g | 0.4606 | 0.3851 |

Paired graph bootstrap (20,000 resamples):

| pair | ind_e | ind_er |
| --- | --- | --- |
| FMX − L2 (the mix at matched diet) | +0.0053 [−0.0013, +0.0125], 8 of 18 | −0.0025 [−0.0108, +0.0073], 9 of 23 |
| FMX − TRIX (matched diet) | +0.0001 [−0.0061, +0.0061], 9 of 18 | +0.0041 [−0.0057, +0.0165], 16 of 23 |
| L2 − TRIX (matched diet, no mix) | −0.0051 [−0.0106, +0.0006], 5 of 18 | +0.0067 [−0.0056, +0.0236], 15 of 23 |
| FMX − MX1 (the fourth graph) | −0.0043 [−0.0088, −0.0000], 5 of 18 | −0.0131 [−0.0191, −0.0078], 2 of 23 |

Per-scenario MRR:

| group | model | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- | --- |
| ind_e | L2 | 0.4816 | 0.2750 | 0.6132 | 0.3457 |
| ind_e | FMX | 0.4728 | 0.3052 | 0.5930 | 0.3944 |
| ind_e | TRIX | 0.4842 | 0.2932 | 0.6092 | 0.3428 |
| ind_er | L2 | 0.3874 | 0.1651 | 0.5828 | 0.2203 |
| ind_er | FMX | 0.3768 | 0.1758 | 0.5594 | 0.2785 |
| ind_er | TRIX | 0.3709 | 0.1696 | 0.5736 | 0.2361 |

## Reading

1. **At matched diet the mix brings INCITE exactly to TRIX on ind_e**
   (+0.0001) and to +0.004 on ind_er, inside noise. Without the mix the
   3-graph continuation is 0.005 below TRIX on ind_e. So the honest
   matched-diet sentence is: "INCITE with the prior ties the released
   TRIX at 30 percent of its steps; without the prior it is 0.005 below
   on ind_e." No SOTA at matched diet.
2. **The mix's effect at 3 graphs is MX1's effect at 4:** +0.005 on
   ind_e (not significant here: 8 of 18 graphs), zero on ind_er, and the
   same scenario trade (SQSA −0.009 / −0.011, UQSA −0.020 / −0.023
   against SQUA +0.030 / +0.011, UQUA +0.049 / +0.058). The trade is a
   property of the mix, not of the fourth graph.
3. **The fourth graph is worth +0.004 / +0.013** (FMX − MX1, both
   intervals below zero), the same as L1 − L2 (+0.005 / +0.011): the
   ind_er margin over TRIX is the diet, as the review said.

## The dev suite (stratified, `results/incite/dev/FMX.json`)

| model | dev (benchmark-weighted) | graphs' own mix |
| --- | --- | --- |
| L2 | 0.3029 | 0.3365 |
| FMX | 0.2997 | 0.3260 |

FMX − L2: −0.0032 (2 of 8 graphs), −0.0105 on the graphs' own mix. The
mix costs on the held-out graphs at 3 graphs as it does at 4
(results/incite/DEV_SUITE.md): the finding of 4 Sep is not a 4-graph
artefact.
