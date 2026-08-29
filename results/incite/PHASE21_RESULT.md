# Phase 2.1 result: the walks lever FAILS its kill switch

Date: 2026-08-29. Lever: anonymized walks + GRU (design C), warm-started
from the phase-1 floor, 20k steps (DEV10-selected best at step 8000).

## Benchmark (41 graphs, test splits, ranks/incite-walks/)

| group | walks | floor | delta |
| --- | --- | --- | --- |
| ind_e MRR | 0.4527 | 0.4553 | -0.0026 |
| ind_er MRR | 0.3750 | 0.3740 | +0.0011 |

Within noise both ways; the DEV10-valid edge (0.4204 vs 0.4156) did not
transfer. The walks model also carries an 8k-extra-trunk-steps confound,
so even the ind_er tick is unclaimable.

## PETALS (220 instances, diagnostics/petals_eval.py)

| model | 1-pass acc | 8-pass acc | exact ties | mean margin |
| --- | --- | --- | --- | --- |
| floor | 0.4545 | 0.4545 | 130/220 | 0.000000 |
| walks | 0.4727 | 0.4136 | 0/220 | 0.2555 |

Kill switch: above 90% required. The floor's 130 EXACT ties verify the
automorphism argument empirically. The walks break every tie (margin
0.26) but at chance accuracy: capability without supervision. The
pretraining mix holds no automorphic structures to teach the GRU which
way to break; the injected signal is truth-uncorrelated noise, and the
8-pass average even leans anti-correlated.

## Verdict

The lever as implemented is DEAD: no benchmark gain, kill switch failed.
Per plan discipline: recorded, not tuned around. What would revive it
(future work, not tonight): symmetry-breaking supervision in the mix
(synthetic automorphic graphs with labeled answers) or FLOCK-style
walk-native training. The budget-control run (+20k no-walks) is
unnecessary: there is no gain to attribute.

Phase 2.2 (support lever) proceeds from the unchanged phase-1 floor.
