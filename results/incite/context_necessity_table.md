## Context-necessity diagnostic (held-out synthetic instances, final step)

Held-out set: 100 instances per run, seed 4096; mean 238 nodes, 589 edges, 117.7 eval candidates of which 80.6 inside the head's 3-hop ball. Ranks: 1-based, pessimistic ties, over the type-consistent non-derivable pool.

| mode | withhold | variant | seeds | MRR full | MRR shuffled | MRR none | H@1 full | K3 ordered | full − none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_only | 1.0 | - | 7,1024,1337 | 0.6480 ± 0.0076 | 0.6450 ± 0.0091 | 0.6494 ± 0.0052 | 0.5067 ± 0.0125 | no/no/no | -0.0014 ± 0.0027 |
| context_only | 1.0 | 6000 steps | 1024 | 0.6718 | 0.6486 | 0.6752 | 0.5200 | no | -0.0034 |
| context_only | 1.0 | detached rows | 1024 | 0.6446 | 0.6497 | 0.6515 | 0.4900 | no | -0.0069 |
| floor | 1.0 | - | 7,1024,1337 | 0.6456 ± 0.0175 | 0.6456 ± 0.0175 | 0.6456 ± 0.0175 | 0.5067 ± 0.0249 | n/a | n/a |
| residual | 1.0 | - | 7,1024,1337 | 0.6461 ± 0.0056 | 0.6478 ± 0.0041 | 0.6459 ± 0.0052 | 0.4967 ± 0.0094 | no/yes/no | 0.0002 ± 0.0027 |
| residual | 1.0 | detached rows | 1024 | 0.6550 | 0.6457 | 0.6575 | 0.5100 | no | -0.0025 |
| context_only | 0.0 | - | 1024 | 0.8896 | 0.8901 | 0.8779 | 0.8400 | no | 0.0117 |
| floor | 0.0 | - | 1024 | 0.8070 | 0.8070 | 0.8070 | 0.7000 | n/a | n/a |

Runs: 14. Per-run files: output/context-necessity/<name>/result.json (curve, provenance).
