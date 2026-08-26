# ULTRA baseline report

ULTRA is the reference this project measures every other repo against, so this
report is about one question before it is about any number: **does
`shared/metrics.py`, reading only the dumped ranks, reproduce what ULTRA itself
reports?** If it does not, no later comparison means anything.


Generated 2026-08-26 by `scripts/make_report.py`.

## Verdict

| criterion | result | coverage |
| --- | --- | --- |
| A (strict) — every value bitwise identical to ULTRA's CSV | **FAIL** | 190/222 exact over 37 graphs |
| A (metric definition) — order-independent metrics bitwise identical | **PASS** | 111 comparisons; order-dependent ones within 3 float32 ulp |
| B — group means within ±0.002 of the repository figures | **FAIL** | inductive (e) 16/18, inductive (e,r) 21/23 |

Both criteria must pass. They do not. What is missing and why is in *Deviations* below, and the criterion A residual is dissected in *The resolved tie rule* — it is float32 associativity inside ULTRA's own reduction, not a disagreement about what a rank is. Nothing was tuned to close any gap.


## Criterion A — metric equivalence

`shared/metrics.py` recomputes each metric from `ranks/ultra/*.parquet` and is compared against the raw `ultra_results_*.csv` ULTRA wrote during the same run, value by value, at the full 17-digit precision `csv.DictWriter` prints. `ulps(f32)` is the distance in float32 representable steps; 0 means the two floats are the same bit pattern.

Source CSVs (unmodified, kept in `results/`):

* `ultra_results_2026-08-26-13-49-41.csv`
* `ultra_results_2026-08-26-13-49-45.csv`
* `ultra_results_2026-08-26-13-49-47.csv`
* `ultra_results_2026-08-26-13-56-56.csv`
* `ultra_results_2026-08-26-13-56-58.csv`
* `ultra_results_2026-08-26-14-19-21.csv`
* `ultra_results_2026-08-26-14-19-25.csv`
* `ultra_results_2026-08-26-14-19-30.csv`
* `ultra_results_2026-08-26-14-22-10.csv`
* `ultra_results_2026-08-26-14-28-04.csv`
* `ultra_results_2026-08-26-14-40-03.csv`
* `ultra_results_2026-08-26-14-45-35.csv`
* `ultra_results_2026-08-26-15-03-25.csv`
* `ultra_results_2026-08-26-15-07-08.csv`
* `ultra_results_2026-08-26-15-09-16.csv`
* `ultra_results_2026-08-26-15-24-24.csv`
* `ultra_results_2026-08-26-15-32-38.csv`
* `ultra_results_2026-08-26-15-42-42.csv`
* `ultra_results_2026-08-26-15-49-13.csv`
* `ultra_results_2026-08-26-15-51-21.csv`
* `ultra_results_2026-08-26-15-52-19.csv`
* `ultra_results_2026-08-26-15-53-03.csv`
* `ultra_results_2026-08-26-15-54-20.csv`
* `ultra_results_2026-08-26-15-55-22.csv`
* `ultra_results_2026-08-26-15-56-28.csv`
* `ultra_results_2026-08-26-16-00-05.csv`
* `ultra_results_2026-08-26-16-02-39.csv`
* `ultra_results_2026-08-26-16-02-53.csv`

**Criterion A (strict, bitwise): FAIL** -- 222 comparisons, 190 exact, 32 mismatched.

Split by what a mismatch would mean:

* **Order-independent metrics** (`hits@1`, `hits@3`, `hits@10`): 111/111 exact -- **PASS**. These cannot be moved by summation order, so bitwise equality here is exactly the claim that the tie rule, the rank offset and the dump agree with ULTRA.
* **Order-dependent metrics** (`mrr`, `mr`, `hits@10_50`): 79/111 exact; worst disagreement 3 float32 ulp (1.79e-07 absolute).

| dataset | metric | metrics.py | ULTRA csv | exact | |diff| | ulps(f32) |
| --- | --- | --- | --- | --- | --- | --- |
| FB15k237Inductive:v1 | mrr | 0.48626023530960083 | 0.48626023530960083 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@1 | 0.38929441571235657 | 0.38929441571235657 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@3 | 0.5510948896408081 | 0.5510948896408081 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@10 | 0.6569343209266663 | 0.6569343209266663 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | mr | 56.024330139160156 | 56.024330139160156 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@10_50 | 0.9207180142402649 | 0.9207180142402649 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | mrr | 0.5005502700805664 | 0.5005503296852112 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v2 | hits@1 | 0.4017951488494873 | 0.4017951488494873 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@3 | 0.5559661984443665 | 0.5559661984443665 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@10 | 0.6942977905273438 | 0.6942977905273438 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | mr | 60.8537483215332 | 60.8537483215332 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@10_50 | 0.9404902458190918 | 0.9404902458190918 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | mrr | 0.48211702704429626 | 0.48211705684661865 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v3 | hits@1 | 0.39081457257270813 | 0.39081457257270813 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | hits@3 | 0.5337954759597778 | 0.5337954759597778 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | hits@10 | 0.6444252133369446 | 0.6444252133369446 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | mr | 86.31889343261719 | 86.31889343261719 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | hits@10_50 | 0.9457852840423584 | 0.9457852840423584 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | mrr | 0.4768696427345276 | 0.4768696427345276 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@1 | 0.37112677097320557 | 0.37112677097320557 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@3 | 0.5390844941139221 | 0.5390844941139221 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@10 | 0.6707746386528015 | 0.6707746386528015 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | mr | 72.82535552978516 | 72.82535552978516 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@10_50 | 0.9676231741905212 | 0.9676231741905212 | yes | 0.000e+00 | 0 |
| FBIngram:100 | mrr | 0.43833377957344055 | 0.4383338391780853 | **no** | 5.960e-08 | 2 |
| FBIngram:100 | hits@1 | 0.3376985788345337 | 0.3376985788345337 | yes | 0.000e+00 | 0 |
| FBIngram:100 | hits@3 | 0.48797768354415894 | 0.48797768354415894 | yes | 0.000e+00 | 0 |
| FBIngram:100 | hits@10 | 0.6313868761062622 | 0.6313868761062622 | yes | 0.000e+00 | 0 |
| FBIngram:100 | mr | 59.030269622802734 | 59.030269622802734 | yes | 0.000e+00 | 0 |
| FBIngram:100 | hits@10_50 | 0.9655815362930298 | 0.9655815362930298 | yes | 0.000e+00 | 0 |
| FBIngram:25 | mrr | 0.3830017149448395 | 0.3830016851425171 | **no** | 2.980e-08 | 1 |
| FBIngram:25 | hits@1 | 0.26189643144607544 | 0.26189643144607544 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@3 | 0.4356193244457245 | 0.4356193244457245 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@10 | 0.6332225203514099 | 0.6332225203514099 | yes | 0.000e+00 | 0 |
| FBIngram:25 | mr | 82.13068389892578 | 82.13068389892578 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@10_50 | 0.9707428216934204 | 0.97074294090271 | **no** | 1.192e-07 | 2 |
| FBIngram:50 | mrr | 0.3303404152393341 | 0.3303404450416565 | **no** | 2.980e-08 | 1 |
| FBIngram:50 | hits@1 | 0.22995617985725403 | 0.22995617985725403 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@3 | 0.3658159375190735 | 0.3658159375190735 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@10 | 0.5357050895690918 | 0.5357050895690918 | yes | 0.000e+00 | 0 |
| FBIngram:50 | mr | 145.52796936035156 | 145.52796936035156 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@10_50 | 0.9466056227684021 | 0.9466056823730469 | **no** | 5.960e-08 | 1 |
| FBIngram:75 | mrr | 0.39052730798721313 | 0.39052730798721313 | yes | 0.000e+00 | 0 |
| FBIngram:75 | hits@1 | 0.28638118505477905 | 0.28638118505477905 | yes | 0.000e+00 | 0 |
| FBIngram:75 | hits@3 | 0.438023179769516 | 0.438023179769516 | yes | 0.000e+00 | 0 |
| FBIngram:75 | hits@10 | 0.5943335294723511 | 0.5943335294723511 | yes | 0.000e+00 | 0 |
| FBIngram:75 | mr | 75.05988311767578 | 75.05988311767578 | yes | 0.000e+00 | 0 |
| FBIngram:75 | hits@10_50 | 0.9617846012115479 | 0.9617845416069031 | **no** | 5.960e-08 | 1 |
| FBNELL | mrr | 0.47340789437294006 | 0.4734078347682953 | **no** | 5.960e-08 | 2 |
| FBNELL | hits@1 | 0.3793969750404358 | 0.3793969750404358 | yes | 0.000e+00 | 0 |
| FBNELL | hits@3 | 0.5284757018089294 | 0.5284757018089294 | yes | 0.000e+00 | 0 |
| FBNELL | hits@10 | 0.6532663106918335 | 0.6532663106918335 | yes | 0.000e+00 | 0 |
| FBNELL | mr | 66.76214599609375 | 66.76214599609375 | yes | 0.000e+00 | 0 |
| FBNELL | hits@10_50 | 0.9882258176803589 | 0.9882258176803589 | yes | 0.000e+00 | 0 |
| HM:1k | mrr | 0.07903667539358139 | 0.07903667539358139 | yes | 0.000e+00 | 0 |
| HM:1k | hits@1 | 0.04621848836541176 | 0.04621848836541176 | yes | 0.000e+00 | 0 |
| HM:1k | hits@3 | 0.07668067514896393 | 0.07668067514896393 | yes | 0.000e+00 | 0 |
| HM:1k | hits@10 | 0.15021008253097534 | 0.15021008253097534 | yes | 0.000e+00 | 0 |
| HM:1k | mr | 1211.5230712890625 | 1211.5230712890625 | yes | 0.000e+00 | 0 |
| HM:1k | hits@10_50 | 0.7911298871040344 | 0.7911298871040344 | yes | 0.000e+00 | 0 |
| HM:3k | mrr | 0.06343384832143784 | 0.06343384832143784 | yes | 0.000e+00 | 0 |
| HM:3k | hits@1 | 0.03817642852663994 | 0.03817642852663994 | yes | 0.000e+00 | 0 |
| HM:3k | hits@3 | 0.06375093013048172 | 0.06375093013048172 | yes | 0.000e+00 | 0 |
| HM:3k | hits@10 | 0.1208302453160286 | 0.1208302453160286 | yes | 0.000e+00 | 0 |
| HM:3k | mr | 2819.997802734375 | 2819.997802734375 | yes | 0.000e+00 | 0 |
| HM:3k | hits@10_50 | 0.7533155083656311 | 0.7533155083656311 | yes | 0.000e+00 | 0 |
| HM:5k | mrr | 0.05519260838627815 | 0.055192600935697556 | **no** | 7.451e-09 | 2 |
| HM:5k | hits@1 | 0.033192090690135956 | 0.033192090690135956 | yes | 0.000e+00 | 0 |
| HM:5k | hits@3 | 0.05602636560797691 | 0.05602636560797691 | yes | 0.000e+00 | 0 |
| HM:5k | hits@10 | 0.10145951062440872 | 0.10145951062440872 | yes | 0.000e+00 | 0 |
| HM:5k | mr | 3713.28515625 | 3713.28515625 | yes | 0.000e+00 | 0 |
| HM:5k | hits@10_50 | 0.7234638333320618 | 0.7234638929367065 | **no** | 5.960e-08 | 1 |
| ILPC2022:small | mrr | 0.2958506643772125 | 0.2958506643772125 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@1 | 0.21829771995544434 | 0.21829771995544434 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@3 | 0.3325292766094208 | 0.3325292766094208 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@10 | 0.4412474036216736 | 0.4412474036216736 | yes | 0.000e+00 | 0 |
| ILPC2022:small | mr | 443.93280029296875 | 443.93280029296875 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@10_50 | 0.8803183436393738 | 0.8803183436393738 | yes | 0.000e+00 | 0 |
| Metafam | mrr | 0.3297547399997711 | 0.3297547399997711 | yes | 0.000e+00 | 0 |
| Metafam | hits@1 | 0.17119565606117249 | 0.17119565606117249 | yes | 0.000e+00 | 0 |
| Metafam | hits@3 | 0.34239131212234497 | 0.34239131212234497 | yes | 0.000e+00 | 0 |
| Metafam | hits@10 | 0.820652186870575 | 0.820652186870575 | yes | 0.000e+00 | 0 |
| Metafam | mr | 6.6711955070495605 | 6.6711955070495605 | yes | 0.000e+00 | 0 |
| Metafam | hits@10_50 | 0.9999997615814209 | 0.9999997615814209 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | mrr | 0.7310843467712402 | 0.7310842871665955 | **no** | 5.960e-08 | 1 |
| NELLInductive:v1 | hits@1 | 0.6791045069694519 | 0.6791045069694519 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@3 | 0.7512437701225281 | 0.7512437701225281 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@10 | 0.8681591749191284 | 0.8681591749191284 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | mr | 3.9228856563568115 | 3.9228856563568115 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@10_50 | 0.8401319980621338 | 0.8401319980621338 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | mrr | 0.5246646404266357 | 0.5246646404266357 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@1 | 0.4117647111415863 | 0.4117647111415863 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@3 | 0.6026737689971924 | 0.6026737689971924 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@10 | 0.718716561794281 | 0.718716561794281 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | mr | 50.54331588745117 | 50.54331588745117 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@10_50 | 0.9625215530395508 | 0.9625215530395508 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | mrr | 0.5112237334251404 | 0.5112237334251404 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@1 | 0.41419753432273865 | 0.41419753432273865 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@3 | 0.559876561164856 | 0.559876561164856 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@10 | 0.6867284178733826 | 0.6867284178733826 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | mr | 48.098148345947266 | 48.098148345947266 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@10_50 | 0.9846675395965576 | 0.9846675395965576 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | mrr | 0.4903210699558258 | 0.4903210997581482 | **no** | 2.980e-08 | 1 |
| NELLInductive:v4 | hits@1 | 0.37284034490585327 | 0.37284034490585327 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@3 | 0.5656530857086182 | 0.5656530857086182 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@10 | 0.7011057138442993 | 0.7011057138442993 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | mr | 70.29025268554688 | 70.29025268554688 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@10_50 | 0.9645063281059265 | 0.9645061492919922 | **no** | 1.788e-07 | 3 |
| NLIngram:0 | mrr | 0.3471793532371521 | 0.3471793532371521 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@1 | 0.25950196385383606 | 0.25950196385383606 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@3 | 0.37352555990219116 | 0.37352555990219116 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@10 | 0.5144167542457581 | 0.5144167542457581 | yes | 0.000e+00 | 0 |
| NLIngram:0 | mr | 113.7844009399414 | 113.7844009399414 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@10_50 | 0.9074332118034363 | 0.9074332118034363 | yes | 0.000e+00 | 0 |
| NLIngram:100 | mrr | 0.4418483376502991 | 0.4418483078479767 | **no** | 2.980e-08 | 1 |
| NLIngram:100 | hits@1 | 0.341740220785141 | 0.341740220785141 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@3 | 0.48928120732307434 | 0.48928120732307434 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@10 | 0.6305170059204102 | 0.6305170059204102 | yes | 0.000e+00 | 0 |
| NLIngram:100 | mr | 42.890289306640625 | 42.890289306640625 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@10_50 | 0.9670436978340149 | 0.9670436978340149 | yes | 0.000e+00 | 0 |
| NLIngram:25 | mrr | 0.3865111172199249 | 0.3865111172199249 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@1 | 0.2990591526031494 | 0.2990591526031494 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@3 | 0.43212366104125977 | 0.43212366104125977 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@10 | 0.538306474685669 | 0.538306474685669 | yes | 0.000e+00 | 0 |
| NLIngram:25 | mr | 80.88642120361328 | 80.88642120361328 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@10_50 | 0.9463781714439392 | 0.946378231048584 | **no** | 5.960e-08 | 1 |
| NLIngram:50 | mrr | 0.3975643217563629 | 0.3975643515586853 | **no** | 2.980e-08 | 1 |
| NLIngram:50 | hits@1 | 0.3119906783103943 | 0.3119906783103943 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@3 | 0.434225857257843 | 0.434225857257843 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@10 | 0.5488940477371216 | 0.5488940477371216 | yes | 0.000e+00 | 0 |
| NLIngram:50 | mr | 92.65017700195312 | 92.65017700195312 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@10_50 | 0.9446386098861694 | 0.9446386098861694 | yes | 0.000e+00 | 0 |
| NLIngram:75 | mrr | 0.3478427231311798 | 0.3478426933288574 | **no** | 2.980e-08 | 1 |
| NLIngram:75 | hits@1 | 0.2619439959526062 | 0.2619439959526062 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@3 | 0.3649093806743622 | 0.3649093806743622 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@10 | 0.5271828770637512 | 0.5271828770637512 | yes | 0.000e+00 | 0 |
| NLIngram:75 | mr | 51.516475677490234 | 51.516475677490234 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@10_50 | 0.9658229351043701 | 0.9658229351043701 | yes | 0.000e+00 | 0 |
| WKIngram:25 | mrr | 0.3071720004081726 | 0.307172030210495 | **no** | 2.980e-08 | 1 |
| WKIngram:25 | hits@1 | 0.21220159530639648 | 0.21220159530639648 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@3 | 0.33554375171661377 | 0.33554375171661377 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@10 | 0.5070734024047852 | 0.5070734024047852 | yes | 0.000e+00 | 0 |
| WKIngram:25 | mr | 136.36294555664062 | 136.36294555664062 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@10_50 | 0.9306843280792236 | 0.9306842684745789 | **no** | 5.960e-08 | 1 |
| WKIngram:75 | mrr | 0.37314456701278687 | 0.37314459681510925 | **no** | 2.980e-08 | 1 |
| WKIngram:75 | hits@1 | 0.28977271914482117 | 0.28977271914482117 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@3 | 0.4226398468017578 | 0.4226398468017578 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@10 | 0.5187937021255493 | 0.5187937021255493 | yes | 0.000e+00 | 0 |
| WKIngram:75 | mr | 114.0581283569336 | 114.0581283569336 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@10_50 | 0.9438894987106323 | 0.9438896179199219 | **no** | 1.192e-07 | 2 |
| WN18RRInductive:v1 | mrr | 0.5929059982299805 | 0.5929059982299805 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@1 | 0.5013405084609985 | 0.5013405084609985 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@3 | 0.6461126208305359 | 0.6461126208305359 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@10 | 0.7788203954696655 | 0.7788203954696655 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | mr | 53.18632888793945 | 53.18632888793945 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@10_50 | 0.9045056700706482 | 0.9045056700706482 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | mrr | 0.6204630732536316 | 0.6204630732536316 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@1 | 0.5469483733177185 | 0.5469483733177185 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@3 | 0.6678403615951538 | 0.6678403615951538 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@10 | 0.7517605423927307 | 0.7517605423927307 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | mr | 177.44601440429688 | 177.44601440429688 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@10_50 | 0.8935670852661133 | 0.8935671448707581 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v3 | mrr | 0.3712637722492218 | 0.3712637722492218 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@1 | 0.30752405524253845 | 0.30752405524253845 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@3 | 0.40201225876808167 | 0.40201225876808167 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@10 | 0.49387577176094055 | 0.49387577176094055 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | mr | 410.97857666015625 | 410.97857666015625 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@10_50 | 0.8305710554122925 | 0.8305709958076477 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v4 | mrr | 0.4839155077934265 | 0.4839155673980713 | **no** | 5.960e-08 | 2 |
| WN18RRInductive:v4 | hits@1 | 0.3877080976963043 | 0.3877080976963043 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@3 | 0.5332978963851929 | 0.5332978963851929 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@10 | 0.687035083770752 | 0.687035083770752 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | mr | 431.440673828125 | 431.440673828125 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@10_50 | 0.887578547000885 | 0.887578547000885 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | mrr | 0.27942657470703125 | 0.27942660450935364 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT1:health | hits@1 | 0.24457216262817383 | 0.24457216262817383 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@3 | 0.2854406237602234 | 0.2854406237602234 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@10 | 0.3323754668235779 | 0.3323754668235779 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | mr | 398.8834533691406 | 398.8834533691406 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@10_50 | 0.9469472765922546 | 0.9469472765922546 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | mrr | 0.24197177588939667 | 0.24197179079055786 | **no** | 1.490e-08 | 1 |
| WikiTopicsMT1:tax | hits@1 | 0.20474372804164886 | 0.20474372804164886 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@3 | 0.25190839171409607 | 0.25190839171409607 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@10 | 0.30534350872039795 | 0.30534350872039795 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | mr | 590.2685546875 | 590.2685546875 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@10_50 | 0.8914283514022827 | 0.8914283514022827 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | mrr | 0.0831659808754921 | 0.0831659808754921 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@1 | 0.0491601787507534 | 0.0491601787507534 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@3 | 0.08500614762306213 | 0.08500614762306213 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@10 | 0.14543220400810242 | 0.14543220400810242 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | mr | 1063.5966796875 | 1063.5966796875 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@10_50 | 0.7799258232116699 | 0.7799258828163147 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT2:sci | mrr | 0.2576117217540741 | 0.2576117217540741 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@1 | 0.202727273106575 | 0.202727273106575 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@3 | 0.26606062054634094 | 0.26606062054634094 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@10 | 0.348181813955307 | 0.348181813955307 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | mr | 432.4696960449219 | 432.4696960449219 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@10_50 | 0.9145798087120056 | 0.9145798683166504 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:art | mrr | 0.2509790360927582 | 0.2509790360927582 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@1 | 0.16945068538188934 | 0.16945068538188934 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@3 | 0.2793125510215759 | 0.2793125510215759 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@10 | 0.41439124941825867 | 0.41439124941825867 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | mr | 455.25811767578125 | 455.25811767578125 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@10_50 | 0.9193653464317322 | 0.9193654656410217 | **no** | 1.192e-07 | 2 |
| WikiTopicsMT3:infra | mrr | 0.6221163272857666 | 0.6221163272857666 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@1 | 0.5409563183784485 | 0.5409563183784485 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@3 | 0.6625779867172241 | 0.6625779867172241 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@10 | 0.7787941694259644 | 0.7787941694259644 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | mr | 73.00312042236328 | 73.00312042236328 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@10_50 | 0.9895198941230774 | 0.9895198345184326 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT4:health | mrr | 0.5565113425254822 | 0.5565113425254822 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@1 | 0.4770992398262024 | 0.4770992398262024 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@3 | 0.601291835308075 | 0.601291835308075 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@10 | 0.7072812914848328 | 0.7072812914848328 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | mr | 246.49412536621094 | 246.49412536621094 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@10_50 | 0.9557361602783203 | 0.9557361602783203 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | mrr | 0.2934439480304718 | 0.2934439480304718 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@1 | 0.21109509468078613 | 0.21109509468078613 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@3 | 0.32168588042259216 | 0.32168588042259216 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@10 | 0.455331414937973 | 0.455331414937973 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | mr | 340.1603088378906 | 340.1603088378906 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@10_50 | 0.9422851204872131 | 0.9422851800918579 | **no** | 5.960e-08 | 1 |


## Per-dataset results

### ind_e (16 of 18 graphs)

| dataset | family | queries | MRR | Hits@10 |
| --- | --- | --- | --- | --- |
| FB15k237Inductive:v1 | FB | 822 | 0.4863 | 0.6569 |
| FB15k237Inductive:v2 | FB | 1894 | 0.5006 | 0.6943 |
| FB15k237Inductive:v3 | FB | 3462 | 0.4821 | 0.6444 |
| FB15k237Inductive:v4 | FB | 5680 | 0.4769 | 0.6708 |
| WN18RRInductive:v1 | WN | 746 | 0.5929 | 0.7788 |
| WN18RRInductive:v2 | WN | 1704 | 0.6205 | 0.7518 |
| WN18RRInductive:v3 | WN | 2286 | 0.3713 | 0.4939 |
| WN18RRInductive:v4 | WN | 5646 | 0.4839 | 0.6870 |
| NELLInductive:v1 | NELL | 402 | 0.7311 | 0.8682 |
| NELLInductive:v2 | NELL | 1870 | 0.5247 | 0.7187 |
| NELLInductive:v3 | NELL | 3240 | 0.5112 | 0.6867 |
| NELLInductive:v4 | NELL | 2894 | 0.4903 | 0.7011 |
| ILPC2022:small | WK | 5804 | 0.2959 | 0.4412 |
| HM:1k | FB | 952 | 0.0790 | 0.1502 |
| HM:3k | FB | 2698 | 0.0634 | 0.1208 |
| HM:5k | FB | 4248 | 0.0552 | 0.1015 |

### ind_er (21 of 23 graphs)

| dataset | family | queries | MRR | Hits@10 |
| --- | --- | --- | --- | --- |
| FBIngram:25 | FB | 11432 | 0.3830 | 0.6332 |
| FBIngram:50 | FB | 7758 | 0.3303 | 0.5357 |
| FBIngram:75 | FB | 6212 | 0.3905 | 0.5943 |
| FBIngram:100 | FB | 4658 | 0.4383 | 0.6314 |
| WKIngram:25 | WK | 2262 | 0.3072 | 0.5071 |
| WKIngram:75 | WK | 2288 | 0.3731 | 0.5188 |
| NLIngram:0 | NELL | 1526 | 0.3472 | 0.5144 |
| NLIngram:25 | NELL | 1488 | 0.3865 | 0.5383 |
| NLIngram:50 | NELL | 1718 | 0.3976 | 0.5489 |
| NLIngram:75 | NELL | 1214 | 0.3478 | 0.5272 |
| NLIngram:100 | NELL | 1586 | 0.4418 | 0.6305 |
| WikiTopicsMT1:tax | WK | 3668 | 0.2420 | 0.3053 |
| WikiTopicsMT1:health | WK | 3132 | 0.2794 | 0.3324 |
| WikiTopicsMT2:org | WK | 4882 | 0.0832 | 0.1454 |
| WikiTopicsMT2:sci | WK | 3300 | 0.2576 | 0.3482 |
| WikiTopicsMT3:art | WK | 6226 | 0.2510 | 0.4144 |
| WikiTopicsMT3:infra | WK | 4810 | 0.6221 | 0.7788 |
| WikiTopicsMT4:sci | WK | 2776 | 0.2934 | 0.4553 |
| WikiTopicsMT4:health | WK | 3406 | 0.5565 | 0.7073 |
| Metafam | other | 368 | 0.3298 | 0.8207 |
| FBNELL | other | 1194 | 0.4734 | 0.6533 |


## Criterion B — published numbers

Targets are the ULTRA **repository's** PyG figures (README at the pinned SHA), not the paper's. Group means are unweighted over datasets: every graph counts once regardless of how many test queries it has. The last two columns show the distance to the paper numbers as well — landing on those instead of the repository ones would be an anomaly worth reporting.

**Criterion B: FAIL**

| group | metric | datasets | ours | repo target | delta | within +/-0.002 | paper | delta vs paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ind_e | mrr | 16/18 | 0.4228 | 0.420 | +0.0028 | **no** | 0.430 | -0.0072 |
| ind_e | hits@10 | 16/18 | 0.5729 | 0.562 | +0.0109 | **no** | 0.566 | +0.0069 |
| ind_er | mrr | 21/23 | 0.3587 | 0.344 | +0.0147 | **no** | 0.345 | +0.0137 |
| ind_er | hits@10 | 21/23 | 0.5305 | 0.511 | +0.0195 | **no** | 0.512 | +0.0185 |



## The resolved tie rule and rank offset

Criterion A can only be argued once the rank definition is pinned down, so here
it is, read out of ULTRA at pin `427966ad` and now documented in
`shared/metrics.py`:

```python
# ultra/tasks.py
def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
```

**Rank offset: 1-based.** A perfect prediction ranks 1. A rank of `k` means
`k - 1` filtered-in candidates scored at least as high as the true answer, so
`1 <= rank <= n_candidates + 1`.

**Tie rule: pessimistic (worst case).** Two things combine. The comparison
`pos_pred <= pred` is non-strict, so an equal-scoring candidate counts against
the true answer. And `strict_negative_mask` zeroes the target's own position
(`t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)`), so the target's trivially
equal self-comparison contributes nothing. Net: if the true answer ties with `k`
other surviving candidates and nothing outscores it, its rank is `k + 1`. That
is neither the optimistic rule (1) nor the average rule (`(k + 2) / 2`). An
implementation using either of those disagrees with ULTRA by an amount that
grows with tie mass, which on sparse graphs is not small.

**`n_candidates` excludes the target**, being `mask.sum(dim=-1)` — that is why
the rank can reach `n_candidates + 1` rather than `n_candidates`.

**Filtering.** Test-time filtering graphs differ by dataset family, and ULTRA
builds them itself: for ILPC and Ingram, inference + valid + test edges; for the
other inductive families, inference + test edges. The dump records the resulting
`n_candidates` per query rather than trying to re-derive it downstream.

### Two things printed-precision comparison caught

**A real bug, first.** The first implementation of ULTRA's unbiased `hits@10_50`
disagreed in the 8th significant digit. The cause was not the tie rule: **numpy
promotes `float32 / int64` to float64 where torch keeps float32**, so dividing
the cast rank by the raw `n_candidates` column ran the whole chain at a
precision ULTRA never used. Casting both operands fixes it, and it now matches
bitwise. This is exactly the failure mode criterion A exists to catch, and it is
why the comparison is done at printed precision rather than against a tolerance.

`hits@10_50` is not one of this project's four reported metrics. It is
reproduced anyway because it is the **only** quantity that consumes
`n_candidates`: without it, a dump could get that column wrong and criterion A
would still pass on all five other metrics.

**And one thing that is not a bug.** `mrr` matched bitwise on the first dataset
and then differed by one float32 ulp on `FB15k237Inductive:v2` (ULTRA
`0.5005503296852112`, `metrics.py` `0.5005502700805664`). Chased down: the
summands are identical, and the *correctly rounded* float32 value of their exact
sum — `math.fsum` over the same float32 summands — is `0.5005502700805664`.
**ULTRA's own reduction is the one an ulp off**, not this module's. The cause is
float32 associativity inside `torch.Tensor.mean`, whose internal blocking is
device- and version-specific; reimplementing it would not transfer to the GPU
numbers anyway, and would make the project's one metric implementation depend on
a torch internal.

That is why criterion A is reported two ways rather than as a bare pass/fail:

* `hits@1`, `hits@3`, `hits@10` sum values that are exactly 0.0 or 1.0, so their
  reduction cannot depend on order. Bitwise equality there **is** the claim that
  the tie rule, the rank offset and the dump agree with ULTRA. This is the
  reading that answers "is the patch right?" — and it passes.
* `mrr`, `mr` and `hits@10_50` are order-dependent, so they are reported with
  their worst ulp distance rather than pass/fail. The strict bitwise verdict is
  still printed and is still the headline; it is not quietly dropped.

Every mismatch observed sits in `mrr` or `hits@10_50` — never in `hits@1/3/10`,
and so far never in `mr` — and each is one or two float32 ulps, order 1e-7
absolute. The exact per-value distances are in the criterion A table above
rather than asserted here, since they are computed, not claimed. That is four to
five orders of magnitude inside criterion B's ±0.002 band, so the residual
cannot affect any acceptance decision. Reported, not iterated on.

## The patch changes no rank

"Patch the dump, not the ranking" is checked rather than asserted.
`scripts/verify_patch_neutrality.sh` runs stock upstream ULTRA — a clean tree
with no patches at all — and the patched tree over the same dataset, checkpoint
and seed, and diffs the two `ultra_results_*.csv` rows.

```
stock   : FB15k237Inductive:v2,60.8537483215332,0.5005503296852112,0.4017951488494873,0.5559661984443665,0.6942977905273438,0.9404902458190918
patched : FB15k237Inductive:v2,60.8537483215332,0.5005503296852112,0.4017951488494873,0.5559661984443665,0.6942977905273438,0.9404902458190918
```

Identical to full printed precision, with `--skip_valid` active. No dtype
change, no epsilon change, no fused op swapped for an unfused one, no reduction
reordered.

Separately, `scripts/verify_rank_dump.py` reloads each dataset and checks that
every dumped `query_id` indexes exactly the triple its row claims. That is the
soundness condition for the way `query_id` is recovered: the dump rebuilds the
loader's `DistributedSampler` order in a second, independent sampler rather than
altering the real loader, which is only valid if the two agree. They do, for
every row of every file checked.

## Patches

| patch | what it does | why |
| --- | --- | --- |
| `0001-rank-dump.diff` | new `ultra/rank_dump.py`; `test()` takes a `dump=` argument; `run_many.py` gains `--rank_dump_dir` | emit one parquet row per scored query into the shared `ranks/` schema; dump only |
| `0002-data-root.diff` | `dataset.root` and `output_dir` in both inference configs become jinja variables with container defaults | give ULTRA its own processed root, so the relation-graph cache in `processed/data.pt` cannot be shared with another repo's `pre_transform` |
| `0003-skip-valid.diff` | opt-in `--skip_valid` on `run_many.py`, default off | the validation-split evaluation feeds no reported metric and mutates no state; skipping it roughly halves CPU-bound runs |

Every change is a patch file. `repos/ultra` is never edited: `git -C repos/ultra
status` is clean at the pinned SHA, and `scripts/prepare_ultra_workdir.sh`
materialises the patched tree the same way the Dockerfile's build layers do.

## Two upstream defects worth recording

**`-d Metafam` and `-d FBNELL` do not work.** `run_many.py` only sets the
`version` template variable when the `-d` entry contains a colon. Bare, jinja
renders the unset variable as the *string* `"None"`, and
`MTDEAInductive.__init__` asserts `version in self.versions` before anything can
normalise it, so both raise `AssertionError`. The correct spellings are
`Metafam:Metafam` and `FBNELL:FBNELL_v1`; `shared/suite.py` carries them as
`run_id` and maps them back with `by_run_id`, so the canonical suite ids and the
rank filenames stay clean.

**Stock `run_many.py` cannot run on a fresh machine.** It calls
`os.chdir(cfg.output_dir)` without creating the directory, so the first run dies
with `FileNotFoundError` on `~/git/ULTRA/output`. SEMMA's fork adds the
`makedirs`; upstream has not. Our runs route `output_dir` at a directory
`scripts/run_ultra.sh` creates, so only the stock half of the neutrality check
is affected.

## Deviations from the specified procedure

The environment this ran in has **no GPU** — no `nvidia-smi`, no `nvcc`,
`torch.cuda.is_available()` is False, 4 CPU cores — and its egress policy blocks
several hosts the specified procedure requires. Each item below is a fact
checked in this environment, not an assumption.

| step as specified | status | what actually happened |
| --- | --- | --- |
| Build `containers/ultra/` | **not done** | Blocked three independent ways. No Docker daemon is running (`/var/run/docker.sock` absent; only the client is installed). The registry blob CDN `production.cloudfront.docker.com` answers 403, so the CUDA devel base cannot be pulled. And both wheel indexes below are blocked, so the pip layers could not complete even with a daemon. The Dockerfile is written and is the deliverable; it has not been built. |
| `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118` | **blocked** | `download.pytorch.org` — proxy answers 403 to CONNECT (organization egress policy). |
| `pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/...` | **blocked** | `data.pyg.org` — 403. |
| Run on a single GPU with `--gpus [0]` | **substituted** | Run on CPU with `--gpus null`, which ULTRA documents and supports; `rspmm` has a CPU code path and was compiled here from source. |
| CUDA 11.8 devel base, Python 3.9, torch 2.1.0, PyG 2.4.0 | **partly met** | Python 3.9.25, torch 2.1.0, torch-geometric 2.4.0, torch-scatter 2.1.2 — ULTRA's pins exactly. Only the CUDA half is absent. torch came from PyPI (the same 2.1.0, CUDA-12 build, used on CPU) because the pinned index is blocked; torch-scatter 2.1.2 was compiled from its PyPI sdist. **No pin was relaxed to make an install succeed.** One build-tool pin was added: `setuptools==69.5.1`, because torch 2.1.0's `cpp_extension` imports `pkg_resources.packaging`, which setuptools removed in 70. |

Consequences to keep in mind when reading the numbers:

* **Criterion A is essentially unaffected.** It compares this project's metric
  code against ULTRA's own metric code over the same ranks from the same
  process, so whether that process ran on a GPU or a CPU is irrelevant to
  whether the two agree. The one caveat is the ulp residual above: on a GPU the
  reduction is CUDA's block tree rather than torch's CPU cascade, so *which*
  order-dependent values land exactly may differ, while the bound does not.
* **Criterion B is affected in principle.** ULTRA's published figures were
  produced on an RTX 3090. CPU and CUDA float32 kernels differ in the low-order
  bits, which can flip a near-tie and move a rank. The effect is small, but it
  is not nothing, and a CPU-derived group mean is not strictly the same
  measurement as the published one.

## Datasets

All 41 graphs in groups 1 and 2 download from hosts this environment can reach —
`raw.githubusercontent.com` (GraIL, Ingram, ILPC, HM) and
`reltrans.s3.us-east-2.amazonaws.com` (MTDEA). No download in scope failed
outright, but one arrived **silently truncated**, which is worse.

### One truncated download, and why it is worth a section

`HM:indigo` failed to load with `ValueError: not enough values to unpack
(expected 3, got 1)` on the last line of its inference graph. The upstream data
is fine. Our copy of `test-graph.txt` was **17,595,392 bytes against the
server's 19,321,652** — PyG's `download_url` streams to disk without checking
`Content-Length`, so a cut connection leaves a short file and raises nothing.

This one announced itself only because the cut happened to land mid-record. Had
it landed on a line boundary, the graph would simply have been missing its tail,
every metric computed from it would have been quietly wrong, and nothing
anywhere would have complained — not the loader, not the run, not criterion A,
which compares two computations over the *same* corrupted input and would agree
perfectly.

So `scripts/verify_downloads.py` checks a **byte count, not a parse**: it HEADs
every URL each dataset class declares and compares against the file on disk,
with `--fix` to re-fetch and clear the stale `processed/` cache. Over groups 1
and 2: 91 files checked, exactly one short, re-fetched and verified.

MTDEA's ten datasets are reported as **unverifiable rather than passed** — they
arrive as one zip that is extracted and deleted, so there is nothing left to
compare against. That is a real gap, not a clean bill of health.

For the record, since `shared/suite.py` defines all 54 graphs and later tasks
will need them, four of the 13 transductive graphs are **not reachable from
here** and would fail:

| graph | host | status |
| --- | --- | --- |
| `CoDExSmall`, `CoDExLarge` | `zenodo.org` | 403 from the proxy |
| `AristoV4` | `zenodo.org` | 403 from the proxy |
| `Hetionet` | `www.dropbox.com` | 403 from the proxy |

`data/raw/MANIFEST-ultra.json` records, for every one of the 54 graphs, the URLs
its ULTRA dataset class declares — with the version substituted, so a link that
dies later is still on record — alongside a sha256 for every raw file actually
mirrored.

## What was not run, and why

* **The other six containers were not built.** As instructed: clone and inspect
  only. They are cloned at pinned SHAs and inspected in `containers/STACKS.md`.
* **The 13 transductive graphs were not run.** Groups 1 and 2 are what the
  acceptance criteria cover; `shared/suite.py` defines all 54 for later tasks.
* **Nothing was tuned.** No hyperparameter, no threshold, no checkpoint choice
  was changed in response to a gap against the targets. `ultra_3g.pth` was used
  throughout (sha256 in `environment.json`), never `ultra_50g.pth`.
* **`--epochs 0` was verified in the logs**, not assumed: every dataset's config
  dump reads `'num_epoch': 0`.
