## Leak split (41 inductive graphs, test splits, seed 1024, MRR)

Leaked: the graph's family is in the model's pretraining set. Means are over the graphs the model has.

| model | pretraining families | leaked ind_e | clean ind_e | leaked ind_er | clean ind_er | leaked all | clean all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ULTRA 3g | CoDEx, FB, WN | 0.3873 (12) | 0.4726 (6) | 0.3856 (4) | 0.3329 (19) | 0.3869 (16) | 0.3664 (25) |
| ULTRA 4g | CoDEx, FB, NELL, WN | 0.4642 (16) | 0.2956 (2) | 0.3944 (9) | 0.3150 (14) | 0.4390 (25) | 0.3125 (16) |
| MOTIF | CoDEx, FB, WN | 0.4167 (12) | 0.4749 (6) | 0.3873 (4) | 0.3411 (19) | 0.4094 (16) | 0.3732 (25) |
| TRIX | CoDEx, FB, WN | 0.4272 (12) | 0.5140 (6) | 0.3926 (4) | 0.3627 (19) | 0.4186 (16) | 0.3990 (25) |
| SEMMA | CoDEx, FB, WN | 0.4265 (12) | 0.4957 (6) | 0.3981 (4) | 0.3423 (19) | 0.4194 (16) | 0.3791 (25) |
| FLOCK | CoDEx, FB, WN | 0.4325 (12) | 0.5022 (6) | 0.4077 (3) | 0.3610 (19) | 0.4275 (15) | 0.3949 (25) |
| KGPFN | CoDEx, FB, WN | 0.6072 (4) | 0.5466 (5) | - | 0.3240 (16) | 0.6072 (4) | 0.3770 (21) |
| INCITE floor 3g | CoDEx, FB, WN | 0.4211 (12) | 0.5238 (6) | 0.3990 (4) | 0.3687 (19) | 0.4156 (16) | 0.4059 (25) |
| INCITE 4g last | CoDEx, FB, NELL, WN | 0.4725 (16) | 0.3010 (2) | 0.4032 (9) | 0.3692 (14) | 0.4475 (25) | 0.3607 (16) |

## Paired comparisons on graphs clean for both models

| model | baseline | clean graphs | mean delta | bootstrap 95% | wins/ties/losses | Wilcoxon p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| INCITE 4g last | ULTRA 4g | 16 | +0.0481 | [+0.0161, +0.0961] | 14/2/0 | 0.0000 | 0.0001 |
| INCITE 4g last | TRIX | 16 | +0.0163 | [+0.0008, +0.0399] | 7/6/3 | 0.1928 | 0.3856 |
| INCITE 4g last | FLOCK | 16 | +0.0082 | [-0.0024, +0.0202] | 6/4/6 | 0.2979 | 0.3856 |
| INCITE floor 3g | ULTRA 3g | 25 | +0.0395 | [+0.0260, +0.0541] | 24/1/0 | 0.0000 | 0.0000 |
| INCITE floor 3g | MOTIF | 25 | +0.0327 | [+0.0171, +0.0484] | 20/2/3 | 0.0004 | 0.0013 |
| INCITE floor 3g | SEMMA | 25 | +0.0268 | [+0.0144, +0.0423] | 20/3/2 | 0.0000 | 0.0001 |
| INCITE floor 3g | TRIX | 25 | +0.0069 | [-0.0066, +0.0189] | 16/6/3 | 0.0219 | 0.0438 |
| INCITE floor 3g | FLOCK | 25 | +0.0110 | [-0.0043, +0.0285] | 11/6/8 | 0.4742 | 0.4742 |
| TRIX | ULTRA 3g | 25 | +0.0325 | [+0.0187, +0.0480] | 19/4/2 | 0.0000 | 0.0000 |

Families: FB 12 graphs, WN 4, NELL 9, WK 14, other 2 (Metafam, FBNELL). Per-graph values from shared/metrics.py over the committed parquets; FLOCK lacks FBIngram:25 (leaked for every model here), KGPFN has 25 of 41 graphs.
