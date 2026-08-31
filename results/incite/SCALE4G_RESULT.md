# 4-graph scale-up result: the largest single gain of the program

Date: 2026-08-31. Setup: the phase-1 backbone (no levers) pretrained
from scratch on FB15k237 + WN18RR + CoDExMedium + NELL995 (ultra_4g
precedent), 20k steps, micro-batch 16 x accum 2 (NELL995 memory,
config_diff.md), DEV10 best 0.4244 -- the highest of any run.

| | 4g | floor (3g) | TRIX (3g) |
| --- | --- | --- | --- |
| ind_er MRR | 0.3791 | 0.3740 | 0.3679 |
| ind_er Hits@10 | 0.5609 | 0.5475 | 0.5409 |
| ind_e MRR | 0.4542 | 0.4553 | 0.4562 |
| ind_e Hits@10 | 0.5970 | 0.5960 | 0.5931 |

Relation-vocabulary diversity moved exactly the axis it should: unseen
relations (+0.005 MRR / +0.013 H@10 over the floor), ind_e unchanged.

## The fairness caveat (state it on every table)

The 4g row trains on MORE data than every baseline row: TRIX, SEMMA,
MOTIF, and our ULTRA checkpoint are 3-graph pretrains. This is exactly
the comparison class our KG-ICL convention finding warns about. Fair
statements: "4g INCITE vs 3g baselines shows the diet effect" or
compare against ULTRA's published 4g checkpoint (exists upstream, not
yet run here). The matched-diet SOTA row remains the floor (3g).

## Standing

Best measured ind_er by +0.011 over TRIX (diet-caveated), best DEV10
selection overall. Single seed. The obvious v1 recipe going forward:
4-graph (or larger) backbone + joint relation head, walks optional for
the capability class, support dropped.
