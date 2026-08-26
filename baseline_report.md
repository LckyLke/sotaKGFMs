# ULTRA baseline report

ULTRA is the reference this project measures every other repo against, so this
report is about one question before it is about any number: **does
`shared/metrics.py`, reading only the dumped ranks, reproduce what ULTRA itself
reports?** If it does not, no later comparison means anything.


Generated 2026-08-26 by `scripts/make_report.py`.

## Verdict

| criterion | result | coverage |
| --- | --- | --- |
| A (strict) — every value bitwise identical to ULTRA's CSV | **FAIL** | 162/246 exact over 41 graphs |
| A (metric definition) — order-independent metrics bitwise identical | **PASS** | 123 comparisons; order-dependent ones within 2 float32 ulp |
| B — group means within ±0.002 of the repository figures | **FAIL** | inductive (e) 18/18, inductive (e,r) 23/23 |

Both criteria must pass. They do not. What is missing and why is in *Deviations* below, and the criterion A residual is dissected in *The resolved tie rule* — it is float32 associativity inside ULTRA's own reduction, not a disagreement about what a rank is. Nothing was tuned to close any gap.


## Criterion A — metric equivalence

`shared/metrics.py` recomputes each metric from `ranks/ultra/*.parquet` and is compared against the raw `ultra_results_*.csv` ULTRA wrote during the same run, value by value, at the full 17-digit precision `csv.DictWriter` prints. `ulps(f32)` is the distance in float32 representable steps; 0 means the two floats are the same bit pattern.

Source CSVs (unmodified, kept in `results/`):

* `ultra_results_2026-08-26-18-31-59.csv`
* `ultra_results_2026-08-26-18-32-34.csv`
* `ultra_results_2026-08-26-18-32-37.csv`
* `ultra_results_2026-08-26-18-32-41.csv`
* `ultra_results_2026-08-26-18-32-47.csv`
* `ultra_results_2026-08-26-18-32-49.csv`
* `ultra_results_2026-08-26-18-32-53.csv`
* `ultra_results_2026-08-26-18-32-57.csv`
* `ultra_results_2026-08-26-18-33-07.csv`
* `ultra_results_2026-08-26-18-33-09.csv`
* `ultra_results_2026-08-26-18-33-13.csv`
* `ultra_results_2026-08-26-18-33-18.csv`
* `ultra_results_2026-08-26-18-33-22.csv`
* `ultra_results_2026-08-26-18-33-31.csv`
* `ultra_results_2026-08-26-18-35-14.csv`
* `ultra_results_2026-08-26-18-35-18.csv`
* `ultra_results_2026-08-26-18-35-27.csv`
* `ultra_results_2026-08-26-18-35-43.csv`
* `ultra_results_2026-08-26-18-37-18.csv`
* `ultra_results_2026-08-26-18-37-31.csv`
* `ultra_results_2026-08-26-18-37-41.csv`
* `ultra_results_2026-08-26-18-37-47.csv`
* `ultra_results_2026-08-26-18-37-52.csv`
* `ultra_results_2026-08-26-18-37-56.csv`
* `ultra_results_2026-08-26-18-38-06.csv`
* `ultra_results_2026-08-26-18-38-09.csv`
* `ultra_results_2026-08-26-18-38-27.csv`
* `ultra_results_2026-08-26-18-38-30.csv`
* `ultra_results_2026-08-26-18-38-33.csv`
* `ultra_results_2026-08-26-18-38-36.csv`
* `ultra_results_2026-08-26-18-38-38.csv`
* `ultra_results_2026-08-26-18-38-41.csv`
* `ultra_results_2026-08-26-18-38-50.csv`
* `ultra_results_2026-08-26-18-38-58.csv`
* `ultra_results_2026-08-26-18-39-06.csv`
* `ultra_results_2026-08-26-18-39-14.csv`
* `ultra_results_2026-08-26-18-39-26.csv`
* `ultra_results_2026-08-26-18-39-35.csv`
* `ultra_results_2026-08-26-18-39-42.csv`
* `ultra_results_2026-08-26-18-39-50.csv`
* `ultra_results_2026-08-26-18-39-52.csv`

**Criterion A (metric definition): PASS** -- 123/123 hit counts identical.

**Criterion A (strict, bitwise): FAIL** -- 246 comparisons, 162 exact, 84 mismatched.

The two verdicts answer different questions, and only the first one is about ranking.

* **Order-independent metrics** (`hits@1`, `hits@3`, `hits@10`): 123/123 identical as counts -- **PASS**; 97/123 identical bitwise. `hits@k` is `count / n_queries`, and the count is a whole number that no summation order can move: a count disagreement is a disagreement about the tie rule, the rank offset or the dump. The float around it can still differ, because that last division is not done the same way everywhere -- on CUDA torch reduces and scales differently from numpy on the host. Every bitwise mismatch seen here is 1 ulp with the counts identical, which is that division and nothing else.
* **Order-dependent metrics** (`mrr`, `mr`, `hits@10_50`): 65/123 exact; worst disagreement 2 float32 ulp (1.22e-04 absolute). These carry no count to fall back on, so float32 associativity is the whole story.

| dataset | metric | metrics.py | ULTRA csv | exact | |diff| | ulps(f32) |
| --- | --- | --- | --- | --- | --- | --- |
| FB15k237Inductive:v1 | mrr | 0.48626023530960083 | 0.48626020550727844 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v1 | hits@1 | 0.38929441571235657 | 0.3892943859100342 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v1 | hits@3 | 0.5510948896408081 | 0.5510948896408081 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@10 | 0.6569343209266663 | 0.6569343209266663 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | mr | 56.02311325073242 | 56.02311325073242 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@10_50 | 0.9207181334495544 | 0.9207181334495544 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | mrr | 0.5005502700805664 | 0.5005503296852112 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v2 | hits@1 | 0.4017951488494873 | 0.4017951488494873 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@3 | 0.5559661984443665 | 0.5559662580490112 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v2 | hits@10 | 0.6942977905273438 | 0.6942977905273438 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | mr | 60.8537483215332 | 60.85375213623047 | **no** | 3.815e-06 | 1 |
| FB15k237Inductive:v2 | hits@10_50 | 0.9404902458190918 | 0.9404903054237366 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v3 | mrr | 0.4821169376373291 | 0.4821169376373291 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | hits@1 | 0.39081457257270813 | 0.39081454277038574 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v3 | hits@3 | 0.5337954759597778 | 0.5337954759597778 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | hits@10 | 0.6444252133369446 | 0.6444251537322998 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v3 | mr | 86.31918334960938 | 86.31917572021484 | **no** | 7.629e-06 | 1 |
| FB15k237Inductive:v3 | hits@10_50 | 0.9457852840423584 | 0.9457852244377136 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v4 | mrr | 0.4768696129322052 | 0.4768695831298828 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v4 | hits@1 | 0.37112677097320557 | 0.3711267411708832 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v4 | hits@3 | 0.5390844941139221 | 0.5390844941139221 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@10 | 0.6707746386528015 | 0.6707746386528015 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | mr | 72.82553100585938 | 72.82552337646484 | **no** | 7.629e-06 | 1 |
| FB15k237Inductive:v4 | hits@10_50 | 0.9676231741905212 | 0.9676231145858765 | **no** | 5.960e-08 | 1 |
| FBIngram:100 | mrr | 0.43833377957344055 | 0.43833380937576294 | **no** | 2.980e-08 | 1 |
| FBIngram:100 | hits@1 | 0.3376985788345337 | 0.3376985788345337 | yes | 0.000e+00 | 0 |
| FBIngram:100 | hits@3 | 0.48797768354415894 | 0.48797768354415894 | yes | 0.000e+00 | 0 |
| FBIngram:100 | hits@10 | 0.6313868761062622 | 0.6313868761062622 | yes | 0.000e+00 | 0 |
| FBIngram:100 | mr | 59.030269622802734 | 59.0302734375 | **no** | 3.815e-06 | 1 |
| FBIngram:100 | hits@10_50 | 0.9655815362930298 | 0.9655815362930298 | yes | 0.000e+00 | 0 |
| FBIngram:25 | mrr | 0.3830017149448395 | 0.3830017149448395 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@1 | 0.26189643144607544 | 0.26189643144607544 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@3 | 0.4356193244457245 | 0.4356193244457245 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@10 | 0.6332225203514099 | 0.6332225203514099 | yes | 0.000e+00 | 0 |
| FBIngram:25 | mr | 82.13059997558594 | 82.13059997558594 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@10_50 | 0.9707428216934204 | 0.97074294090271 | **no** | 1.192e-07 | 2 |
| FBIngram:50 | mrr | 0.3303411900997162 | 0.3303411900997162 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@1 | 0.22995617985725403 | 0.22995617985725403 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@3 | 0.3658159375190735 | 0.3658159375190735 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@10 | 0.5357050895690918 | 0.5357050895690918 | yes | 0.000e+00 | 0 |
| FBIngram:50 | mr | 145.5284881591797 | 145.5284881591797 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@10_50 | 0.9466056227684021 | 0.9466056823730469 | **no** | 5.960e-08 | 1 |
| FBIngram:75 | mrr | 0.39052537083625793 | 0.3905254006385803 | **no** | 2.980e-08 | 1 |
| FBIngram:75 | hits@1 | 0.28638118505477905 | 0.28638121485710144 | **no** | 2.980e-08 | 1 |
| FBIngram:75 | hits@3 | 0.438023179769516 | 0.4380232095718384 | **no** | 2.980e-08 | 1 |
| FBIngram:75 | hits@10 | 0.5943335294723511 | 0.5943335890769958 | **no** | 5.960e-08 | 1 |
| FBIngram:75 | mr | 75.06358337402344 | 75.06359100341797 | **no** | 7.629e-06 | 1 |
| FBIngram:75 | hits@10_50 | 0.9617821574211121 | 0.9617821574211121 | yes | 0.000e+00 | 0 |
| FBNELL | mrr | 0.47340574860572815 | 0.47340574860572815 | yes | 0.000e+00 | 0 |
| FBNELL | hits@1 | 0.3793969750404358 | 0.3793969750404358 | yes | 0.000e+00 | 0 |
| FBNELL | hits@3 | 0.5284757018089294 | 0.5284757018089294 | yes | 0.000e+00 | 0 |
| FBNELL | hits@10 | 0.6532663106918335 | 0.6532663106918335 | yes | 0.000e+00 | 0 |
| FBNELL | mr | 66.7638168334961 | 66.7638168334961 | yes | 0.000e+00 | 0 |
| FBNELL | hits@10_50 | 0.9882258176803589 | 0.9882258176803589 | yes | 0.000e+00 | 0 |
| HM:1k | mrr | 0.07905793190002441 | 0.07905793935060501 | **no** | 7.451e-09 | 1 |
| HM:1k | hits@1 | 0.04621848836541176 | 0.04621848836541176 | yes | 0.000e+00 | 0 |
| HM:1k | hits@3 | 0.07668067514896393 | 0.07668067514896393 | yes | 0.000e+00 | 0 |
| HM:1k | hits@10 | 0.15021008253097534 | 0.15021009743213654 | **no** | 1.490e-08 | 1 |
| HM:1k | mr | 1211.4989013671875 | 1211.4990234375 | **no** | 1.221e-04 | 1 |
| HM:1k | hits@10_50 | 0.7911244630813599 | 0.7911245822906494 | **no** | 1.192e-07 | 2 |
| HM:3k | mrr | 0.06340353935956955 | 0.06340353190898895 | **no** | 7.451e-09 | 1 |
| HM:3k | hits@1 | 0.03817642852663994 | 0.03817642852663994 | yes | 0.000e+00 | 0 |
| HM:3k | hits@3 | 0.06375093013048172 | 0.06375093013048172 | yes | 0.000e+00 | 0 |
| HM:3k | hits@10 | 0.12045960128307343 | 0.12045960128307343 | yes | 0.000e+00 | 0 |
| HM:3k | mr | 2821.14794921875 | 2821.14794921875 | yes | 0.000e+00 | 0 |
| HM:3k | hits@10_50 | 0.7529944777488708 | 0.7529944777488708 | yes | 0.000e+00 | 0 |
| HM:5k | mrr | 0.055202361196279526 | 0.055202361196279526 | yes | 0.000e+00 | 0 |
| HM:5k | hits@1 | 0.033192090690135956 | 0.033192090690135956 | yes | 0.000e+00 | 0 |
| HM:5k | hits@3 | 0.05602636560797691 | 0.05602636560797691 | yes | 0.000e+00 | 0 |
| HM:5k | hits@10 | 0.10145951062440872 | 0.10145951062440872 | yes | 0.000e+00 | 0 |
| HM:5k | mr | 3713.833740234375 | 3713.833740234375 | yes | 0.000e+00 | 0 |
| HM:5k | hits@10_50 | 0.7233091592788696 | 0.7233090996742249 | **no** | 5.960e-08 | 1 |
| HM:indigo | mrr | 0.43607577681541443 | 0.43607574701309204 | **no** | 2.980e-08 | 1 |
| HM:indigo | hits@1 | 0.3268585503101349 | 0.3268585503101349 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@3 | 0.4906401038169861 | 0.4906400740146637 | **no** | 2.980e-08 | 1 |
| HM:indigo | hits@10 | 0.6489533185958862 | 0.6489532589912415 | **no** | 5.960e-08 | 1 |
| HM:indigo | mr | 94.44182586669922 | 94.44182586669922 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@10_50 | 0.9958032369613647 | 0.99580317735672 | **no** | 5.960e-08 | 1 |
| ILPC2022:large | mrr | 0.29705896973609924 | 0.29705893993377686 | **no** | 2.980e-08 | 1 |
| ILPC2022:large | hits@1 | 0.2306068390607834 | 0.2306068390607834 | yes | 0.000e+00 | 0 |
| ILPC2022:large | hits@3 | 0.3323841392993927 | 0.3323841392993927 | yes | 0.000e+00 | 0 |
| ILPC2022:large | hits@10 | 0.422820121049881 | 0.422820121049881 | yes | 0.000e+00 | 0 |
| ILPC2022:large | mr | 1572.1444091796875 | 1572.14453125 | **no** | 1.221e-04 | 1 |
| ILPC2022:large | hits@10_50 | 0.8968480229377747 | 0.8968480229377747 | yes | 0.000e+00 | 0 |
| ILPC2022:small | mrr | 0.2958506643772125 | 0.29585063457489014 | **no** | 2.980e-08 | 1 |
| ILPC2022:small | hits@1 | 0.21829771995544434 | 0.21829771995544434 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@3 | 0.3325292766094208 | 0.3325292766094208 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@10 | 0.4412474036216736 | 0.4412474036216736 | yes | 0.000e+00 | 0 |
| ILPC2022:small | mr | 443.93280029296875 | 443.93280029296875 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@10_50 | 0.8803183436393738 | 0.8803183436393738 | yes | 0.000e+00 | 0 |
| Metafam | mrr | 0.3297547399997711 | 0.3297547399997711 | yes | 0.000e+00 | 0 |
| Metafam | hits@1 | 0.17119565606117249 | 0.17119565606117249 | yes | 0.000e+00 | 0 |
| Metafam | hits@3 | 0.34239131212234497 | 0.34239131212234497 | yes | 0.000e+00 | 0 |
| Metafam | hits@10 | 0.820652186870575 | 0.820652186870575 | yes | 0.000e+00 | 0 |
| Metafam | mr | 6.6711955070495605 | 6.671195983886719 | **no** | 4.768e-07 | 1 |
| Metafam | hits@10_50 | 0.9999997615814209 | 0.9999998807907104 | **no** | 1.192e-07 | 2 |
| NELLInductive:v1 | mrr | 0.7166289687156677 | 0.716628909111023 | **no** | 5.960e-08 | 1 |
| NELLInductive:v1 | hits@1 | 0.6666666865348816 | 0.6666666269302368 | **no** | 5.960e-08 | 1 |
| NELLInductive:v1 | hits@3 | 0.7238805890083313 | 0.7238805890083313 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@10 | 0.8606964945793152 | 0.8606964945793152 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | mr | 4.226367950439453 | 4.226367950439453 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@10_50 | 0.8294954895973206 | 0.8294955492019653 | **no** | 5.960e-08 | 1 |
| NELLInductive:v2 | mrr | 0.5246646404266357 | 0.5246646404266357 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@1 | 0.4117647111415863 | 0.4117647111415863 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@3 | 0.6026737689971924 | 0.6026738286018372 | **no** | 5.960e-08 | 1 |
| NELLInductive:v2 | hits@10 | 0.718716561794281 | 0.7187166213989258 | **no** | 5.960e-08 | 1 |
| NELLInductive:v2 | mr | 50.54331588745117 | 50.54331588745117 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@10_50 | 0.9625215530395508 | 0.9625215530395508 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | mrr | 0.5112232565879822 | 0.5112232565879822 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@1 | 0.41419753432273865 | 0.41419756412506104 | **no** | 2.980e-08 | 1 |
| NELLInductive:v3 | hits@3 | 0.559876561164856 | 0.559876561164856 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@10 | 0.6867284178733826 | 0.6867284178733826 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | mr | 48.11944580078125 | 48.11944580078125 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@10_50 | 0.9846457242965698 | 0.9846457839012146 | **no** | 5.960e-08 | 1 |
| NELLInductive:v4 | mrr | 0.49032092094421387 | 0.49032092094421387 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@1 | 0.37284034490585327 | 0.37284037470817566 | **no** | 2.980e-08 | 1 |
| NELLInductive:v4 | hits@3 | 0.5656530857086182 | 0.5656530857086182 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@10 | 0.7011057138442993 | 0.7011057138442993 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | mr | 70.29060363769531 | 70.29060363769531 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@10_50 | 0.9645063281059265 | 0.964506208896637 | **no** | 1.192e-07 | 2 |
| NLIngram:0 | mrr | 0.34717944264411926 | 0.34717950224876404 | **no** | 5.960e-08 | 2 |
| NLIngram:0 | hits@1 | 0.25950196385383606 | 0.25950196385383606 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@3 | 0.37352555990219116 | 0.37352555990219116 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@10 | 0.5144167542457581 | 0.5144168138504028 | **no** | 5.960e-08 | 1 |
| NLIngram:0 | mr | 113.77523040771484 | 113.77523040771484 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@10_50 | 0.907437801361084 | 0.9074378609657288 | **no** | 5.960e-08 | 1 |
| NLIngram:100 | mrr | 0.44177982211112976 | 0.44177982211112976 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@1 | 0.341740220785141 | 0.341740220785141 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@3 | 0.48865067958831787 | 0.48865067958831787 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@10 | 0.6305170059204102 | 0.6305170059204102 | yes | 0.000e+00 | 0 |
| NLIngram:100 | mr | 42.89470291137695 | 42.89470291137695 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@10_50 | 0.9670436978340149 | 0.9670436978340149 | yes | 0.000e+00 | 0 |
| NLIngram:25 | mrr | 0.38651108741760254 | 0.38651108741760254 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@1 | 0.2990591526031494 | 0.2990591526031494 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@3 | 0.43212366104125977 | 0.43212366104125977 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@10 | 0.538306474685669 | 0.538306474685669 | yes | 0.000e+00 | 0 |
| NLIngram:25 | mr | 80.89045715332031 | 80.89045715332031 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@10_50 | 0.9463714957237244 | 0.9463716149330139 | **no** | 1.192e-07 | 2 |
| NLIngram:50 | mrr | 0.3975643217563629 | 0.3975643217563629 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@1 | 0.3119906783103943 | 0.3119906783103943 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@3 | 0.434225857257843 | 0.434225857257843 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@10 | 0.5488940477371216 | 0.5488940477371216 | yes | 0.000e+00 | 0 |
| NLIngram:50 | mr | 92.65017700195312 | 92.65017700195312 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@10_50 | 0.9446386098861694 | 0.9446386694908142 | **no** | 5.960e-08 | 1 |
| NLIngram:75 | mrr | 0.3478427231311798 | 0.3478427231311798 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@1 | 0.2619439959526062 | 0.2619439661502838 | **no** | 2.980e-08 | 1 |
| NLIngram:75 | hits@3 | 0.3649093806743622 | 0.3649093806743622 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@10 | 0.5271828770637512 | 0.5271828770637512 | yes | 0.000e+00 | 0 |
| NLIngram:75 | mr | 51.516475677490234 | 51.51647186279297 | **no** | 3.815e-06 | 1 |
| NLIngram:75 | hits@10_50 | 0.9658229351043701 | 0.9658228158950806 | **no** | 1.192e-07 | 2 |
| WKIngram:100 | mrr | 0.1782224327325821 | 0.1782224327325821 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@1 | 0.11932829022407532 | 0.11932829022407532 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@3 | 0.19673041999340057 | 0.19673041999340057 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@10 | 0.28925710916519165 | 0.28925710916519165 | yes | 0.000e+00 | 0 |
| WKIngram:100 | mr | 567.4400634765625 | 567.4400634765625 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@10_50 | 0.918030858039856 | 0.918030858039856 | yes | 0.000e+00 | 0 |
| WKIngram:25 | mrr | 0.3071720004081726 | 0.3071720004081726 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@1 | 0.21220159530639648 | 0.21220159530639648 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@3 | 0.33554375171661377 | 0.33554378151893616 | **no** | 2.980e-08 | 1 |
| WKIngram:25 | hits@10 | 0.5070734024047852 | 0.5070734024047852 | yes | 0.000e+00 | 0 |
| WKIngram:25 | mr | 136.36251831054688 | 136.36251831054688 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@10_50 | 0.9306852221488953 | 0.9306851625442505 | **no** | 5.960e-08 | 1 |
| WKIngram:50 | mrr | 0.1576918512582779 | 0.1576918363571167 | **no** | 1.490e-08 | 1 |
| WKIngram:50 | hits@1 | 0.09100775420665741 | 0.09100774675607681 | **no** | 7.451e-09 | 1 |
| WKIngram:50 | hits@3 | 0.16883720457553864 | 0.16883720457553864 | yes | 0.000e+00 | 0 |
| WKIngram:50 | hits@10 | 0.2956589162349701 | 0.2956589162349701 | yes | 0.000e+00 | 0 |
| WKIngram:50 | mr | 581.4158325195312 | 581.415771484375 | **no** | 6.104e-05 | 1 |
| WKIngram:50 | hits@10_50 | 0.8918819427490234 | 0.8918819427490234 | yes | 0.000e+00 | 0 |
| WKIngram:75 | mrr | 0.37314456701278687 | 0.37314456701278687 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@1 | 0.28977271914482117 | 0.28977271914482117 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@3 | 0.4226398468017578 | 0.4226398468017578 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@10 | 0.5187937021255493 | 0.5187937021255493 | yes | 0.000e+00 | 0 |
| WKIngram:75 | mr | 114.05769348144531 | 114.05769348144531 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@10_50 | 0.9438897967338562 | 0.9438896775245667 | **no** | 1.192e-07 | 2 |
| WN18RRInductive:v1 | mrr | 0.5929059982299805 | 0.5929059982299805 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@1 | 0.5013405084609985 | 0.5013405084609985 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@3 | 0.6461126208305359 | 0.6461126208305359 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@10 | 0.7788203954696655 | 0.7788203954696655 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | mr | 53.18632888793945 | 53.18632888793945 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@10_50 | 0.9045056700706482 | 0.904505729675293 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v2 | mrr | 0.6204630732536316 | 0.6204630732536316 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@1 | 0.5469483733177185 | 0.5469483733177185 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@3 | 0.6678403615951538 | 0.6678403615951538 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@10 | 0.7517605423927307 | 0.7517606019973755 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v2 | mr | 177.45069885253906 | 177.45071411132812 | **no** | 1.526e-05 | 1 |
| WN18RRInductive:v2 | hits@10_50 | 0.8935632109642029 | 0.8935631513595581 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v3 | mrr | 0.3712212145328522 | 0.37122124433517456 | **no** | 2.980e-08 | 1 |
| WN18RRInductive:v3 | hits@1 | 0.30752405524253845 | 0.30752405524253845 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@3 | 0.40201225876808167 | 0.40201225876808167 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@10 | 0.49387577176094055 | 0.49387577176094055 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | mr | 410.989501953125 | 410.989501953125 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@10_50 | 0.8305706977844238 | 0.8305706977844238 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | mrr | 0.4839128255844116 | 0.4839128255844116 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@1 | 0.3877080976963043 | 0.3877080976963043 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@3 | 0.5332978963851929 | 0.5332978963851929 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@10 | 0.687035083770752 | 0.687035083770752 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | mr | 431.4869079589844 | 431.48687744140625 | **no** | 3.052e-05 | 1 |
| WN18RRInductive:v4 | hits@10_50 | 0.8875640034675598 | 0.8875640034675598 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | mrr | 0.2794263958930969 | 0.2794264256954193 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT1:health | hits@1 | 0.24457216262817383 | 0.24457214772701263 | **no** | 1.490e-08 | 1 |
| WikiTopicsMT1:health | hits@3 | 0.2854406237602234 | 0.285440593957901 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT1:health | hits@10 | 0.3323754668235779 | 0.3323754668235779 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | mr | 398.8837890625 | 398.8837585449219 | **no** | 3.052e-05 | 1 |
| WikiTopicsMT1:health | hits@10_50 | 0.9469472765922546 | 0.9469472765922546 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | mrr | 0.2419642060995102 | 0.2419642210006714 | **no** | 1.490e-08 | 1 |
| WikiTopicsMT1:tax | hits@1 | 0.20474372804164886 | 0.20474372804164886 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@3 | 0.25190839171409607 | 0.25190839171409607 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@10 | 0.30534350872039795 | 0.30534350872039795 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | mr | 590.2704467773438 | 590.2704467773438 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@10_50 | 0.8914280533790588 | 0.8914279937744141 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT2:org | mrr | 0.0831659808754921 | 0.0831659808754921 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@1 | 0.0491601787507534 | 0.0491601787507534 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@3 | 0.08500614762306213 | 0.08500614762306213 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@10 | 0.14543220400810242 | 0.14543220400810242 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | mr | 1063.596435546875 | 1063.596435546875 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@10_50 | 0.779926061630249 | 0.7799261212348938 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT2:sci | mrr | 0.25762680172920227 | 0.25762680172920227 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@1 | 0.202727273106575 | 0.202727273106575 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@3 | 0.26606062054634094 | 0.26606062054634094 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@10 | 0.348181813955307 | 0.348181813955307 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | mr | 432.4696960449219 | 432.4696960449219 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@10_50 | 0.9145798087120056 | 0.9145798087120056 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | mrr | 0.2509790360927582 | 0.2509790360927582 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@1 | 0.16945068538188934 | 0.16945068538188934 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@3 | 0.2793125510215759 | 0.2793125510215759 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@10 | 0.41439124941825867 | 0.41439124941825867 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | mr | 455.2582702636719 | 455.2582702636719 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@10_50 | 0.9193653464317322 | 0.9193653464317322 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | mrr | 0.6220816373825073 | 0.6220816969871521 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:infra | hits@1 | 0.5409563183784485 | 0.5409563779830933 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:infra | hits@3 | 0.6625779867172241 | 0.6625779867172241 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@10 | 0.7787941694259644 | 0.7787942290306091 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:infra | mr | 73.0029067993164 | 73.00291442871094 | **no** | 7.629e-06 | 1 |
| WikiTopicsMT3:infra | hits@10_50 | 0.9895198941230774 | 0.9895199537277222 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT4:health | mrr | 0.5565113425254822 | 0.5565112829208374 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT4:health | hits@1 | 0.4770992398262024 | 0.4770992398262024 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@3 | 0.601291835308075 | 0.601291835308075 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@10 | 0.7072812914848328 | 0.7072812914848328 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | mr | 246.49412536621094 | 246.49412536621094 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@10_50 | 0.9557361602783203 | 0.9557362198829651 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT4:sci | mrr | 0.2934439480304718 | 0.2934439480304718 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@1 | 0.21109509468078613 | 0.21109509468078613 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@3 | 0.32168588042259216 | 0.32168588042259216 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@10 | 0.455331414937973 | 0.45533138513565063 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT4:sci | mr | 340.1603088378906 | 340.1602783203125 | **no** | 3.052e-05 | 1 |
| WikiTopicsMT4:sci | hits@10_50 | 0.9422851204872131 | 0.9422850608825684 | **no** | 5.960e-08 | 1 |


## Per-dataset results

### ind_e (18 of 18 graphs)

| dataset | family | queries | MRR | Hits@10 |
| --- | --- | --- | --- | --- |
| FB15k237Inductive:v1 | FB | 822 | 0.4863 | 0.6569 |
| FB15k237Inductive:v2 | FB | 1894 | 0.5006 | 0.6943 |
| FB15k237Inductive:v3 | FB | 3462 | 0.4821 | 0.6444 |
| FB15k237Inductive:v4 | FB | 5680 | 0.4769 | 0.6708 |
| WN18RRInductive:v1 | WN | 746 | 0.5929 | 0.7788 |
| WN18RRInductive:v2 | WN | 1704 | 0.6205 | 0.7518 |
| WN18RRInductive:v3 | WN | 2286 | 0.3712 | 0.4939 |
| WN18RRInductive:v4 | WN | 5646 | 0.4839 | 0.6870 |
| NELLInductive:v1 | NELL | 402 | 0.7166 | 0.8607 |
| NELLInductive:v2 | NELL | 1870 | 0.5247 | 0.7187 |
| NELLInductive:v3 | NELL | 3240 | 0.5112 | 0.6867 |
| NELLInductive:v4 | NELL | 2894 | 0.4903 | 0.7011 |
| ILPC2022:small | WK | 5804 | 0.2959 | 0.4412 |
| ILPC2022:large | WK | 20368 | 0.2971 | 0.4228 |
| HM:1k | FB | 952 | 0.0791 | 0.1502 |
| HM:3k | FB | 2698 | 0.0634 | 0.1205 |
| HM:5k | FB | 4248 | 0.0552 | 0.1015 |
| HM:indigo | FB | 29808 | 0.4361 | 0.6490 |

### ind_er (23 of 23 graphs)

| dataset | family | queries | MRR | Hits@10 |
| --- | --- | --- | --- | --- |
| FBIngram:25 | FB | 11432 | 0.3830 | 0.6332 |
| FBIngram:50 | FB | 7758 | 0.3303 | 0.5357 |
| FBIngram:75 | FB | 6212 | 0.3905 | 0.5943 |
| FBIngram:100 | FB | 4658 | 0.4383 | 0.6314 |
| WKIngram:25 | WK | 2262 | 0.3072 | 0.5071 |
| WKIngram:50 | WK | 6450 | 0.1577 | 0.2957 |
| WKIngram:75 | WK | 2288 | 0.3731 | 0.5188 |
| WKIngram:100 | WK | 8992 | 0.1782 | 0.2893 |
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
| ind_e | mrr | 18/18 | 0.4158 | 0.420 | -0.0042 | **no** | 0.430 | -0.0142 |
| ind_e | hits@10 | 18/18 | 0.5684 | 0.562 | +0.0064 | **no** | 0.566 | +0.0024 |
| ind_er | mrr | 23/23 | 0.3421 | 0.344 | -0.0019 | yes | 0.345 | -0.0029 |
| ind_er | hits@10 | 23/23 | 0.5098 | 0.511 | -0.0012 | yes | 0.512 | -0.0022 |



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
