# KGFM baselines on one suite

Generated 2026-08-27 by `scripts/make_summary.py`. Every number is computed from the rank dumps in `ranks/`; nothing is transcribed.

All runs are zero-shot entity prediction, one seed (1024), on a single GPU. Ranks are 1-based with pessimistic ties under strict filtering, identically for every model -- see `shared/metrics.py` for the definition and `docs/report_notes.md` for the evidence that each repository computes it the same way.

## Coverage

| model | ind_e | ind_er | suite wall clock |
| --- | --- | --- | --- |
| ultra | 18/18 | 23/23 | 8 min |
| motif | 18/18 | 23/23 | - |
| trix | 18/18 | 23/23 | 15 min |
| semma | 18/18 | 23/23 | 25 min |
| flock | - | - | _not run_ |
| kg-icl | - | - | _not run_ |
| kgpfn | - | - | _not run_ |

## ind_e (18 graphs)

| model | mrr | hits@1 | hits@3 | hits@10 | published MRR | delta |
| --- | --- | --- | --- | --- | --- | --- |
| ultra | 0.4158 | 0.3364 | 0.4574 | 0.5684 | 0.420 (repo) | -0.0042 |
| motif | 0.4361 | 0.3606 | 0.4764 | 0.5767 | 0.436 (paper) | +0.0001 |
| trix | 0.4562 | 0.3818 | 0.4960 | 0.5931 | 0.455 (paper) | +0.0012 |
| semma | 0.4496 | 0.3753 | 0.4885 | 0.5869 | 0.447 (paper) | +0.0026 |

## ind_er (23 graphs)

| model | mrr | hits@1 | hits@3 | hits@10 | published MRR | delta |
| --- | --- | --- | --- | --- | --- | --- |
| ultra | 0.3421 | 0.2588 | 0.3726 | 0.5098 | 0.344 (repo) | -0.0019 |
| motif | 0.3491 | 0.2600 | 0.3847 | 0.5254 | 0.349 (paper) | +0.0001 |
| trix | 0.3679 | 0.2787 | 0.4055 | 0.5409 | 0.368 (paper) | -0.0001 |
| semma | 0.3520 | 0.2688 | 0.3843 | 0.5152 | 0.350 (paper) | +0.0020 |

## Per-model detail

* [ultra](reports/ultra.md)
* [motif](reports/motif.md)
* [trix](reports/trix.md)
* [semma](reports/semma.md)
