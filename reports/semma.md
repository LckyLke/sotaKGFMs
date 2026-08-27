# ULTRA baseline report

ULTRA is the reference this project measures every other repo against, so this
report is about one question before it is about any number: **does
`shared/metrics.py`, reading only the dumped ranks, reproduce what ULTRA itself
reports?** If it does not, no later comparison means anything.


Generated 2026-08-27 by `scripts/make_report.py`.

## Verdict

| criterion | result | coverage |
| --- | --- | --- |
| A (strict) — every value bitwise identical to ULTRA's CSV | **FAIL** | 146/246 exact over 41 graphs |
| A (metric definition) — order-independent metrics bitwise identical | **PASS** | 123 comparisons; order-dependent ones within 3 float32 ulp |
| B — group means within ±0.002 of the repository figures | **FAIL** | inductive (e) 18/18, inductive (e,r) 23/23 |

Both criteria must pass. They do not. What is missing and why is in *Deviations* below, and the criterion A residual is dissected in *The resolved tie rule* — it is float32 associativity inside ULTRA's own reduction, not a disagreement about what a rank is. Nothing was tuned to close any gap.


## Criterion A — metric equivalence

`shared/metrics.py` recomputes each metric from `ranks/ultra/*.parquet` and is compared against the raw `ultra_results_*.csv` ULTRA wrote during the same run, value by value, at the full 17-digit precision `csv.DictWriter` prints. `ulps(f32)` is the distance in float32 representable steps; 0 means the two floats are the same bit pattern.

Source CSVs (unmodified, kept in `results/`):

* `ultra_results_2026-08-26-21-36-12.csv`
* `ultra_results_2026-08-26-21-55-15.csv`
* `ultra_results_2026-08-26-21-55-49.csv`
* `ultra_results_2026-08-26-21-56-25.csv`
* `ultra_results_2026-08-26-21-57-09.csv`
* `ultra_results_2026-08-26-21-57-34.csv`
* `ultra_results_2026-08-26-21-57-59.csv`
* `ultra_results_2026-08-26-21-58-26.csv`
* `ultra_results_2026-08-26-21-59-01.csv`
* `ultra_results_2026-08-26-21-59-22.csv`
* `ultra_results_2026-08-26-21-59-47.csv`
* `ultra_results_2026-08-26-22-00-17.csv`
* `ultra_results_2026-08-27-06-02-52.csv`
* `ultra_results_2026-08-27-06-03-24.csv`
* `ultra_results_2026-08-27-06-04-01.csv`
* `ultra_results_2026-08-27-06-04-49.csv`
* `ultra_results_2026-08-27-06-07-58.csv`
* `ultra_results_2026-08-27-06-09-00.csv`
* `ultra_results_2026-08-27-06-09-55.csv`
* `ultra_results_2026-08-27-06-10-46.csv`
* `ultra_results_2026-08-27-06-13-41.csv`
* `ultra_results_2026-08-27-06-14-07.csv`
* `ultra_results_2026-08-27-06-14-34.csv`
* `ultra_results_2026-08-27-06-15-01.csv`
* `ultra_results_2026-08-27-06-15-27.csv`
* `ultra_results_2026-08-27-06-22-04.csv`
* `ultra_results_2026-08-27-06-22-24.csv`
* `ultra_results_2026-08-27-06-49-59.csv`
* `ultra_results_2026-08-27-06-50-41.csv`
* `ultra_results_2026-08-27-06-53-22.csv`
* `ultra_results_2026-08-27-06-53-59.csv`
* `ultra_results_2026-08-27-06-54-25.csv`
* `ultra_results_2026-08-27-06-55-53.csv`
* `ultra_results_2026-08-27-06-56-26.csv`
* `ultra_results_2026-08-27-06-56-57.csv`
* `ultra_results_2026-08-27-06-57-32.csv`
* `ultra_results_2026-08-27-06-58-04.csv`
* `ultra_results_2026-08-27-06-58-42.csv`
* `ultra_results_2026-08-27-06-59-56.csv`
* `ultra_results_2026-08-27-07-00-26.csv`
* `ultra_results_2026-08-27-07-01-11.csv`

**Criterion A (metric definition): PASS** -- 123/123 hit counts identical.

**Criterion A (strict, bitwise): FAIL** -- 246 comparisons, 146 exact, 100 mismatched.

The two verdicts answer different questions, and only the first one is about ranking.

* **Order-independent metrics** (`hits@1`, `hits@3`, `hits@10`): 123/123 identical as counts -- **PASS**; 86/123 identical bitwise. `hits@k` is `count / n_queries`, and the count is a whole number that no summation order can move: a count disagreement is a disagreement about the tie rule, the rank offset or the dump. The float around it can still differ, because that last division is not done the same way everywhere -- on CUDA torch reduces and scales differently from numpy on the host. Every bitwise mismatch seen here is 1 ulp with the counts identical, which is that division and nothing else.
* **Order-dependent metrics** (`mrr`, `mr`, `hits@10_50`): 60/123 exact; worst disagreement 3 float32 ulp (2.44e-04 absolute). These carry no count to fall back on, so float32 associativity is the whole story.

| dataset | metric | metrics.py | ULTRA csv | exact | |diff| | ulps(f32) |
| --- | --- | --- | --- | --- | --- | --- |
| FB15k237Inductive:v1 | mrr | 0.48616304993629456 | 0.48616304993629456 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@1 | 0.38929441571235657 | 0.3892943859100342 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v1 | hits@3 | 0.5377128720283508 | 0.5377128720283508 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@10 | 0.65450119972229 | 0.65450119972229 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | mr | 53.95255661010742 | 53.952552795410156 | **no** | 3.815e-06 | 1 |
| FB15k237Inductive:v1 | hits@10_50 | 0.9281341433525085 | 0.9281342029571533 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v2 | mrr | 0.50285804271698 | 0.50285804271698 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@1 | 0.4017951488494873 | 0.4017951488494873 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@3 | 0.5559661984443665 | 0.5559662580490112 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v2 | hits@10 | 0.6916578412055969 | 0.6916579008102417 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v2 | mr | 56.21277618408203 | 56.2127799987793 | **no** | 3.815e-06 | 1 |
| FB15k237Inductive:v2 | hits@10_50 | 0.9485639929771423 | 0.9485639333724976 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v3 | mrr | 0.4940771460533142 | 0.4940771758556366 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v3 | hits@1 | 0.40179088711738586 | 0.4017908573150635 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v3 | hits@3 | 0.5537261962890625 | 0.5537261366844177 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v3 | hits@10 | 0.6516464352607727 | 0.6516464352607727 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | mr | 86.69093322753906 | 86.69092559814453 | **no** | 7.629e-06 | 1 |
| FB15k237Inductive:v3 | hits@10_50 | 0.9469553232192993 | 0.9469552636146545 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v4 | mrr | 0.4921228587627411 | 0.4921228289604187 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v4 | hits@1 | 0.38961267471313477 | 0.38961267471313477 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@3 | 0.5519366264343262 | 0.5519366264343262 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@10 | 0.6757042407989502 | 0.6757041811943054 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v4 | mr | 68.94630432128906 | 68.94629669189453 | **no** | 7.629e-06 | 1 |
| FB15k237Inductive:v4 | hits@10_50 | 0.9696735143661499 | 0.9696733951568604 | **no** | 1.192e-07 | 2 |
| FBIngram:100 | mrr | 0.4449140429496765 | 0.4449140727519989 | **no** | 2.980e-08 | 1 |
| FBIngram:100 | hits@1 | 0.34478315711021423 | 0.3447831869125366 | **no** | 2.980e-08 | 1 |
| FBIngram:100 | hits@3 | 0.4916273057460785 | 0.4916273355484009 | **no** | 2.980e-08 | 1 |
| FBIngram:100 | hits@10 | 0.6352511644363403 | 0.6352512240409851 | **no** | 5.960e-08 | 1 |
| FBIngram:100 | mr | 57.542720794677734 | 57.542724609375 | **no** | 3.815e-06 | 1 |
| FBIngram:100 | hits@10_50 | 0.9652186036109924 | 0.9652186632156372 | **no** | 5.960e-08 | 1 |
| FBIngram:25 | mrr | 0.39973756670951843 | 0.39973756670951843 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@1 | 0.28210288286209106 | 0.28210288286209106 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@3 | 0.4500524699687958 | 0.45005249977111816 | **no** | 2.980e-08 | 1 |
| FBIngram:25 | hits@10 | 0.6425822377204895 | 0.6425822377204895 | yes | 0.000e+00 | 0 |
| FBIngram:25 | mr | 78.07199096679688 | 78.07199096679688 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@10_50 | 0.9730392694473267 | 0.9730393290519714 | **no** | 5.960e-08 | 1 |
| FBIngram:50 | mrr | 0.3437425196170807 | 0.3437425494194031 | **no** | 2.980e-08 | 1 |
| FBIngram:50 | hits@1 | 0.2446506768465042 | 0.2446506917476654 | **no** | 1.490e-08 | 1 |
| FBIngram:50 | hits@3 | 0.37935033440589905 | 0.37935036420822144 | **no** | 2.980e-08 | 1 |
| FBIngram:50 | hits@10 | 0.5455014109611511 | 0.5455014109611511 | yes | 0.000e+00 | 0 |
| FBIngram:50 | mr | 140.80020141601562 | 140.8002166748047 | **no** | 1.526e-05 | 1 |
| FBIngram:50 | hits@10_50 | 0.9481866955757141 | 0.9481866955757141 | yes | 0.000e+00 | 0 |
| FBIngram:75 | mrr | 0.4039802551269531 | 0.4039802849292755 | **no** | 2.980e-08 | 1 |
| FBIngram:75 | hits@1 | 0.29893752932548523 | 0.2989375591278076 | **no** | 2.980e-08 | 1 |
| FBIngram:75 | hits@3 | 0.4542820453643799 | 0.4542820453643799 | yes | 0.000e+00 | 0 |
| FBIngram:75 | hits@10 | 0.601094663143158 | 0.601094663143158 | yes | 0.000e+00 | 0 |
| FBIngram:75 | mr | 74.13087463378906 | 74.1308822631836 | **no** | 7.629e-06 | 1 |
| FBIngram:75 | hits@10_50 | 0.9622588157653809 | 0.9622587561607361 | **no** | 5.960e-08 | 1 |
| FBNELL | mrr | 0.4811244010925293 | 0.4811244010925293 | yes | 0.000e+00 | 0 |
| FBNELL | hits@1 | 0.38693466782569885 | 0.38693466782569885 | yes | 0.000e+00 | 0 |
| FBNELL | hits@3 | 0.5326633453369141 | 0.5326632857322693 | **no** | 5.960e-08 | 1 |
| FBNELL | hits@10 | 0.6541038751602173 | 0.6541038751602173 | yes | 0.000e+00 | 0 |
| FBNELL | mr | 69.73031616210938 | 69.73031616210938 | yes | 0.000e+00 | 0 |
| FBNELL | hits@10_50 | 0.9860669374465942 | 0.9860668778419495 | **no** | 5.960e-08 | 1 |
| HM:1k | mrr | 0.06198042631149292 | 0.06198043003678322 | **no** | 3.725e-09 | 1 |
| HM:1k | hits@1 | 0.03046218492090702 | 0.03046218678355217 | **no** | 1.863e-09 | 1 |
| HM:1k | hits@3 | 0.06197478994727135 | 0.061974793672561646 | **no** | 3.725e-09 | 1 |
| HM:1k | hits@10 | 0.10924369841814041 | 0.10924370586872101 | **no** | 7.451e-09 | 1 |
| HM:1k | mr | 921.037841796875 | 921.037841796875 | yes | 0.000e+00 | 0 |
| HM:1k | hits@10_50 | 0.8333104848861694 | 0.833310604095459 | **no** | 1.192e-07 | 2 |
| HM:3k | mrr | 0.05592650920152664 | 0.05592651292681694 | **no** | 3.725e-09 | 1 |
| HM:3k | hits@1 | 0.030022239312529564 | 0.030022239312529564 | yes | 0.000e+00 | 0 |
| HM:3k | hits@3 | 0.05856189876794815 | 0.05856189876794815 | yes | 0.000e+00 | 0 |
| HM:3k | hits@10 | 0.10229799896478653 | 0.10229799896478653 | yes | 0.000e+00 | 0 |
| HM:3k | mr | 2194.477294921875 | 2194.4775390625 | **no** | 2.441e-04 | 1 |
| HM:3k | hits@10_50 | 0.7971147298812866 | 0.7971146702766418 | **no** | 5.960e-08 | 1 |
| HM:5k | mrr | 0.054939720779657364 | 0.054939720779657364 | yes | 0.000e+00 | 0 |
| HM:5k | hits@1 | 0.02919020690023899 | 0.02919020690023899 | yes | 0.000e+00 | 0 |
| HM:5k | hits@3 | 0.058851223438978195 | 0.058851223438978195 | yes | 0.000e+00 | 0 |
| HM:5k | hits@10 | 0.10240112990140915 | 0.10240112990140915 | yes | 0.000e+00 | 0 |
| HM:5k | mr | 3156.641357421875 | 3156.641357421875 | yes | 0.000e+00 | 0 |
| HM:5k | hits@10_50 | 0.7714226245880127 | 0.7714225649833679 | **no** | 5.960e-08 | 1 |
| HM:indigo | mrr | 0.4348052442073822 | 0.4348052442073822 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@1 | 0.3264559805393219 | 0.3264559805393219 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@3 | 0.49164652824401855 | 0.49164652824401855 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@10 | 0.6447933316230774 | 0.6447933316230774 | yes | 0.000e+00 | 0 |
| HM:indigo | mr | 86.117919921875 | 86.117919921875 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@10_50 | 0.9968841671943665 | 0.9968841075897217 | **no** | 5.960e-08 | 1 |
| ILPC2022:large | mrr | 0.30719953775405884 | 0.30719953775405884 | yes | 0.000e+00 | 0 |
| ILPC2022:large | hits@1 | 0.2417517602443695 | 0.2417517751455307 | **no** | 1.490e-08 | 1 |
| ILPC2022:large | hits@3 | 0.34107422828674316 | 0.34107422828674316 | yes | 0.000e+00 | 0 |
| ILPC2022:large | hits@10 | 0.42880991101264954 | 0.42880991101264954 | yes | 0.000e+00 | 0 |
| ILPC2022:large | mr | 1469.0836181640625 | 1469.083740234375 | **no** | 1.221e-04 | 1 |
| ILPC2022:large | hits@10_50 | 0.9100488424301147 | 0.9100488424301147 | yes | 0.000e+00 | 0 |
| ILPC2022:small | mrr | 0.2985585927963257 | 0.2985585927963257 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@1 | 0.2217436283826828 | 0.2217436283826828 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@3 | 0.3318400979042053 | 0.3318400979042053 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@10 | 0.44986215233802795 | 0.44986215233802795 | yes | 0.000e+00 | 0 |
| ILPC2022:small | mr | 407.0937194824219 | 407.0937194824219 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@10_50 | 0.8923389315605164 | 0.8923389315605164 | yes | 0.000e+00 | 0 |
| Metafam | mrr | 0.2556276321411133 | 0.2556276321411133 | yes | 0.000e+00 | 0 |
| Metafam | hits@1 | 0.14130434393882751 | 0.14130434393882751 | yes | 0.000e+00 | 0 |
| Metafam | hits@3 | 0.26902174949645996 | 0.26902174949645996 | yes | 0.000e+00 | 0 |
| Metafam | hits@10 | 0.5353260636329651 | 0.5353261232376099 | **no** | 5.960e-08 | 1 |
| Metafam | mr | 31.627717971801758 | 31.627717971801758 | yes | 0.000e+00 | 0 |
| Metafam | hits@10_50 | 0.9281200170516968 | 0.9281198978424072 | **no** | 1.192e-07 | 2 |
| NELLInductive:v1 | mrr | 0.7976762652397156 | 0.7976762056350708 | **no** | 5.960e-08 | 1 |
| NELLInductive:v1 | hits@1 | 0.7363184094429016 | 0.7363184094429016 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@3 | 0.8283582329750061 | 0.8283581733703613 | **no** | 5.960e-08 | 1 |
| NELLInductive:v1 | hits@10 | 0.9353233575820923 | 0.9353233575820923 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | mr | 2.7263681888580322 | 2.7263681888580322 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@10_50 | 0.9840313196182251 | 0.9840313196182251 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | mrr | 0.5440688133239746 | 0.5440688133239746 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@1 | 0.4433155059814453 | 0.4433155059814453 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@3 | 0.6005347371101379 | 0.6005347967147827 | **no** | 5.960e-08 | 1 |
| NELLInductive:v2 | hits@10 | 0.7310160398483276 | 0.7310160398483276 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | mr | 48.83529281616211 | 48.835296630859375 | **no** | 3.815e-06 | 1 |
| NELLInductive:v2 | hits@10_50 | 0.9664080142974854 | 0.9664079546928406 | **no** | 5.960e-08 | 1 |
| NELLInductive:v3 | mrr | 0.5300691723823547 | 0.5300692319869995 | **no** | 5.960e-08 | 1 |
| NELLInductive:v3 | hits@1 | 0.42901235818862915 | 0.42901235818862915 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@3 | 0.5839506387710571 | 0.5839506387710571 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@10 | 0.720370352268219 | 0.7203704118728638 | **no** | 5.960e-08 | 1 |
| NELLInductive:v3 | mr | 45.97993850708008 | 45.97993850708008 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@10_50 | 0.9856904745101929 | 0.9856905341148376 | **no** | 5.960e-08 | 1 |
| NELLInductive:v4 | mrr | 0.4964834153652191 | 0.4964834451675415 | **no** | 2.980e-08 | 1 |
| NELLInductive:v4 | hits@1 | 0.3693849444389343 | 0.3693849444389343 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@3 | 0.5784381628036499 | 0.5784381628036499 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@10 | 0.7290946841239929 | 0.7290946841239929 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | mr | 59.214927673339844 | 59.214927673339844 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@10_50 | 0.9696961641311646 | 0.9696961045265198 | **no** | 5.960e-08 | 1 |
| NLIngram:0 | mrr | 0.3670283257961273 | 0.3670283257961273 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@1 | 0.2726081311702728 | 0.2726081311702728 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@3 | 0.40039318799972534 | 0.40039318799972534 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@10 | 0.5648754835128784 | 0.5648754835128784 | yes | 0.000e+00 | 0 |
| NLIngram:0 | mr | 109.1657943725586 | 109.1657943725586 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@10_50 | 0.9175929427146912 | 0.9175929427146912 | yes | 0.000e+00 | 0 |
| NLIngram:100 | mrr | 0.4627409279346466 | 0.4627408981323242 | **no** | 2.980e-08 | 1 |
| NLIngram:100 | hits@1 | 0.34993696212768555 | 0.34993696212768555 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@3 | 0.5283732414245605 | 0.5283732414245605 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@10 | 0.6752837300300598 | 0.6752837300300598 | yes | 0.000e+00 | 0 |
| NLIngram:100 | mr | 47.68347930908203 | 47.68347930908203 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@10_50 | 0.9653714299201965 | 0.9653714299201965 | yes | 0.000e+00 | 0 |
| NLIngram:25 | mrr | 0.38745248317718506 | 0.38745248317718506 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@1 | 0.2909946143627167 | 0.2909946143627167 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@3 | 0.43615591526031494 | 0.43615591526031494 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@10 | 0.5477150678634644 | 0.5477150678634644 | yes | 0.000e+00 | 0 |
| NLIngram:25 | mr | 102.23722839355469 | 102.23722839355469 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@10_50 | 0.9348455667495728 | 0.9348455667495728 | yes | 0.000e+00 | 0 |
| NLIngram:50 | mrr | 0.4085414409637451 | 0.4085415005683899 | **no** | 5.960e-08 | 2 |
| NLIngram:50 | hits@1 | 0.3236321210861206 | 0.3236321210861206 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@3 | 0.4423748552799225 | 0.4423748552799225 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@10 | 0.5733410716056824 | 0.5733411312103271 | **no** | 5.960e-08 | 1 |
| NLIngram:50 | mr | 110.91036224365234 | 110.91036224365234 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@10_50 | 0.9372111558914185 | 0.9372110962867737 | **no** | 5.960e-08 | 1 |
| NLIngram:75 | mrr | 0.3523981273174286 | 0.3523980677127838 | **no** | 5.960e-08 | 2 |
| NLIngram:75 | hits@1 | 0.2627677023410797 | 0.2627677023410797 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@3 | 0.3632619380950928 | 0.3632619380950928 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@10 | 0.5444810390472412 | 0.5444810390472412 | yes | 0.000e+00 | 0 |
| NLIngram:75 | mr | 66.3797378540039 | 66.3797378540039 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@10_50 | 0.9578821063041687 | 0.9578821063041687 | yes | 0.000e+00 | 0 |
| WKIngram:100 | mrr | 0.17956234514713287 | 0.17956233024597168 | **no** | 1.490e-08 | 1 |
| WKIngram:100 | hits@1 | 0.11499110609292984 | 0.11499109864234924 | **no** | 7.451e-09 | 1 |
| WKIngram:100 | hits@3 | 0.19884341955184937 | 0.19884341955184937 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@10 | 0.30115658044815063 | 0.30115658044815063 | yes | 0.000e+00 | 0 |
| WKIngram:100 | mr | 655.9358520507812 | 655.9358520507812 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@10_50 | 0.9048652648925781 | 0.9048652648925781 | yes | 0.000e+00 | 0 |
| WKIngram:25 | mrr | 0.29886990785598755 | 0.29886990785598755 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@1 | 0.20026525855064392 | 0.20026525855064392 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@3 | 0.336870014667511 | 0.3368700444698334 | **no** | 2.980e-08 | 1 |
| WKIngram:25 | hits@10 | 0.5088417530059814 | 0.5088417530059814 | yes | 0.000e+00 | 0 |
| WKIngram:25 | mr | 168.4739227294922 | 168.4739227294922 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@10_50 | 0.9197267293930054 | 0.9197267889976501 | **no** | 5.960e-08 | 1 |
| WKIngram:50 | mrr | 0.17419195175170898 | 0.1741919368505478 | **no** | 1.490e-08 | 1 |
| WKIngram:50 | hits@1 | 0.10775193572044373 | 0.10775193572044373 | yes | 0.000e+00 | 0 |
| WKIngram:50 | hits@3 | 0.1835658848285675 | 0.1835658848285675 | yes | 0.000e+00 | 0 |
| WKIngram:50 | hits@10 | 0.3176744282245636 | 0.3176743984222412 | **no** | 2.980e-08 | 1 |
| WKIngram:50 | mr | 685.7049560546875 | 685.7049560546875 | yes | 0.000e+00 | 0 |
| WKIngram:50 | hits@10_50 | 0.8608521819114685 | 0.8608521223068237 | **no** | 5.960e-08 | 1 |
| WKIngram:75 | mrr | 0.3873063623905182 | 0.3873063623905182 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@1 | 0.30725523829460144 | 0.30725523829460144 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@3 | 0.4326923191547394 | 0.432692289352417 | **no** | 2.980e-08 | 1 |
| WKIngram:75 | hits@10 | 0.5253496766090393 | 0.5253496170043945 | **no** | 5.960e-08 | 1 |
| WKIngram:75 | mr | 117.12980651855469 | 117.12980651855469 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@10_50 | 0.9390631318092346 | 0.9390631914138794 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v1 | mrr | 0.724313497543335 | 0.7243136167526245 | **no** | 1.192e-07 | 2 |
| WN18RRInductive:v1 | hits@1 | 0.6702412962913513 | 0.6702412962913513 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@3 | 0.7573726773262024 | 0.7573726773262024 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@10 | 0.8163539171218872 | 0.8163539171218872 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | mr | 50.70509338378906 | 50.70509719848633 | **no** | 3.815e-06 | 1 |
| WN18RRInductive:v1 | hits@10_50 | 0.905509889125824 | 0.9055100083351135 | **no** | 1.192e-07 | 2 |
| WN18RRInductive:v2 | mrr | 0.704979419708252 | 0.704979419708252 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@1 | 0.6514084339141846 | 0.6514084339141846 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@3 | 0.7400234937667847 | 0.7400234937667847 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@10 | 0.8034037351608276 | 0.8034037947654724 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v2 | mr | 145.5627899169922 | 145.5627899169922 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@10_50 | 0.9036383032798767 | 0.9036382436752319 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v3 | mrr | 0.44172966480255127 | 0.44172966480255127 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@1 | 0.3722659647464752 | 0.3722659647464752 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@3 | 0.4724409580230713 | 0.4724409580230713 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@10 | 0.5765529274940491 | 0.5765529274940491 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | mr | 242.44662475585938 | 242.44664001464844 | **no** | 1.526e-05 | 1 |
| WN18RRInductive:v3 | hits@10_50 | 0.9164958000183105 | 0.9164958000183105 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | mrr | 0.6644384264945984 | 0.6644384860992432 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v4 | hits@1 | 0.620970606803894 | 0.620970606803894 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@3 | 0.6884520053863525 | 0.6884520053863525 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@10 | 0.7412327527999878 | 0.741232693195343 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v4 | mr | 406.8092346191406 | 406.8092346191406 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@10_50 | 0.8935173153877258 | 0.8935171365737915 | **no** | 1.788e-07 | 3 |
| WikiTopicsMT1:health | mrr | 0.3357005715370178 | 0.33570051193237305 | **no** | 5.960e-08 | 2 |
| WikiTopicsMT1:health | hits@1 | 0.28097063302993774 | 0.28097063302993774 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@3 | 0.3550446927547455 | 0.3550446927547455 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@10 | 0.43486589193344116 | 0.43486589193344116 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | mr | 379.8106689453125 | 379.8106689453125 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@10_50 | 0.941929280757904 | 0.941929280757904 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | mrr | 0.23042136430740356 | 0.23042133450508118 | **no** | 2.980e-08 | 2 |
| WikiTopicsMT1:tax | hits@1 | 0.17884404957294464 | 0.17884404957294464 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@3 | 0.2562704384326935 | 0.2562704384326935 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@10 | 0.3012540936470032 | 0.3012540638446808 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT1:tax | mr | 1174.7462158203125 | 1174.74609375 | **no** | 1.221e-04 | 1 |
| WikiTopicsMT1:tax | hits@10_50 | 0.7871698141098022 | 0.7871697545051575 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT2:org | mrr | 0.09642425179481506 | 0.09642425924539566 | **no** | 7.451e-09 | 1 |
| WikiTopicsMT2:org | hits@1 | 0.06145022436976433 | 0.06145022436976433 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@3 | 0.099754199385643 | 0.099754199385643 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@10 | 0.1573125720024109 | 0.1573125720024109 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | mr | 920.9566040039062 | 920.95654296875 | **no** | 6.104e-05 | 1 |
| WikiTopicsMT2:org | hits@10_50 | 0.8057937622070312 | 0.8057937622070312 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | mrr | 0.25744980573654175 | 0.25744980573654175 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@1 | 0.1942424178123474 | 0.1942424327135086 | **no** | 1.490e-08 | 1 |
| WikiTopicsMT2:sci | hits@3 | 0.27242425084114075 | 0.27242425084114075 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@10 | 0.3909091055393219 | 0.3909091055393219 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | mr | 673.1008911132812 | 673.1008911132812 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@10_50 | 0.8603289127349854 | 0.8603289127349854 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | mrr | 0.2778133153915405 | 0.2778133451938629 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT3:art | hits@1 | 0.20558945834636688 | 0.20558945834636688 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@3 | 0.30485063791275024 | 0.30485060811042786 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT3:art | hits@10 | 0.41663989424705505 | 0.41663989424705505 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | mr | 439.4169616699219 | 439.4169616699219 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@10_50 | 0.9220288991928101 | 0.9220288395881653 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:infra | mrr | 0.6494522094726562 | 0.6494522094726562 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@1 | 0.5804573893547058 | 0.5804573893547058 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@3 | 0.6848232746124268 | 0.6848233342170715 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:infra | hits@10 | 0.7833679914474487 | 0.7833679914474487 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | mr | 83.53056335449219 | 83.53056335449219 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@10_50 | 0.9867120385169983 | 0.9867121577262878 | **no** | 1.192e-07 | 2 |
| WikiTopicsMT4:health | mrr | 0.6146615147590637 | 0.6146615147590637 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@1 | 0.5475631356239319 | 0.5475631356239319 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@3 | 0.6520845293998718 | 0.6520845293998718 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@10 | 0.7386963963508606 | 0.7386963963508606 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | mr | 190.7721710205078 | 190.7721710205078 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@10_50 | 0.9636988639831543 | 0.9636988639831543 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | mrr | 0.287224680185318 | 0.287224680185318 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@1 | 0.20425072312355042 | 0.20425070822238922 | **no** | 1.490e-08 | 1 |
| WikiTopicsMT4:sci | hits@3 | 0.31484150886535645 | 0.31484147906303406 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT4:sci | hits@10 | 0.4542507231235504 | 0.454250693321228 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT4:sci | mr | 370.93084716796875 | 370.9308166503906 | **no** | 3.052e-05 | 1 |
| WikiTopicsMT4:sci | hits@10_50 | 0.9397869110107422 | 0.9397867918014526 | **no** | 1.192e-07 | 2 |


## Per-dataset results

### ind_e (18 of 18 graphs)

| dataset | family | queries | MRR | Hits@10 |
| --- | --- | --- | --- | --- |
| FB15k237Inductive:v1 | FB | 822 | 0.4862 | 0.6545 |
| FB15k237Inductive:v2 | FB | 1894 | 0.5029 | 0.6917 |
| FB15k237Inductive:v3 | FB | 3462 | 0.4941 | 0.6516 |
| FB15k237Inductive:v4 | FB | 5680 | 0.4921 | 0.6757 |
| WN18RRInductive:v1 | WN | 746 | 0.7243 | 0.8164 |
| WN18RRInductive:v2 | WN | 1704 | 0.7050 | 0.8034 |
| WN18RRInductive:v3 | WN | 2286 | 0.4417 | 0.5766 |
| WN18RRInductive:v4 | WN | 5646 | 0.6644 | 0.7412 |
| NELLInductive:v1 | NELL | 402 | 0.7977 | 0.9353 |
| NELLInductive:v2 | NELL | 1870 | 0.5441 | 0.7310 |
| NELLInductive:v3 | NELL | 3240 | 0.5301 | 0.7204 |
| NELLInductive:v4 | NELL | 2894 | 0.4965 | 0.7291 |
| ILPC2022:small | WK | 5804 | 0.2986 | 0.4499 |
| ILPC2022:large | WK | 20368 | 0.3072 | 0.4288 |
| HM:1k | FB | 952 | 0.0620 | 0.1092 |
| HM:3k | FB | 2698 | 0.0559 | 0.1023 |
| HM:5k | FB | 4248 | 0.0549 | 0.1024 |
| HM:indigo | FB | 29808 | 0.4348 | 0.6448 |

### ind_er (23 of 23 graphs)

| dataset | family | queries | MRR | Hits@10 |
| --- | --- | --- | --- | --- |
| FBIngram:25 | FB | 11432 | 0.3997 | 0.6426 |
| FBIngram:50 | FB | 7758 | 0.3437 | 0.5455 |
| FBIngram:75 | FB | 6212 | 0.4040 | 0.6011 |
| FBIngram:100 | FB | 4658 | 0.4449 | 0.6353 |
| WKIngram:25 | WK | 2262 | 0.2989 | 0.5088 |
| WKIngram:50 | WK | 6450 | 0.1742 | 0.3177 |
| WKIngram:75 | WK | 2288 | 0.3873 | 0.5253 |
| WKIngram:100 | WK | 8992 | 0.1796 | 0.3012 |
| NLIngram:0 | NELL | 1526 | 0.3670 | 0.5649 |
| NLIngram:25 | NELL | 1488 | 0.3875 | 0.5477 |
| NLIngram:50 | NELL | 1718 | 0.4085 | 0.5733 |
| NLIngram:75 | NELL | 1214 | 0.3524 | 0.5445 |
| NLIngram:100 | NELL | 1586 | 0.4627 | 0.6753 |
| WikiTopicsMT1:tax | WK | 3668 | 0.2304 | 0.3013 |
| WikiTopicsMT1:health | WK | 3132 | 0.3357 | 0.4349 |
| WikiTopicsMT2:org | WK | 4882 | 0.0964 | 0.1573 |
| WikiTopicsMT2:sci | WK | 3300 | 0.2574 | 0.3909 |
| WikiTopicsMT3:art | WK | 6226 | 0.2778 | 0.4166 |
| WikiTopicsMT3:infra | WK | 4810 | 0.6495 | 0.7834 |
| WikiTopicsMT4:sci | WK | 2776 | 0.2872 | 0.4543 |
| WikiTopicsMT4:health | WK | 3406 | 0.6147 | 0.7387 |
| Metafam | other | 368 | 0.2556 | 0.5353 |
| FBNELL | other | 1194 | 0.4811 | 0.6541 |


## Criterion B — published numbers

Targets are the ULTRA **repository's** PyG figures (README at the pinned SHA), not the paper's. Group means are unweighted over datasets: every graph counts once regardless of how many test queries it has. The last two columns show the distance to the paper numbers as well — landing on those instead of the repository ones would be an anomaly worth reporting.

**Criterion B (semma): FAIL**

Target: `arXiv 2505.20422, Table 1, row 'Semma'`.

AVERAGED OVER 5 RUNS. This project runs one seed (1024), so a difference within run-to-run spread is expected and the tolerance below is not a like-for-like test. SEMMA-Hybrid is named in the paper but not released, so it is out of scope. Note also that this table's own ULTRA baseline row reads 0.428/0.570 for ind_e, which is neither figure ULTRA's README gives.

| group | metric | datasets | ours | paper target | delta | within +/-0.002 |
| --- | --- | --- | --- | --- | --- | --- |
| ind_e | mrr | 18/18 | 0.4496 | 0.447 | +0.0026 | **no** |
| ind_e | hits@10 | 18/18 | 0.5869 | 0.584 | +0.0029 | **no** |
| ind_er | mrr | 23/23 | 0.3520 | 0.350 | +0.0020 | **no** |
| ind_er | hits@10 | 23/23 | 0.5152 | 0.514 | +0.0012 | yes |



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

## Criterion B: two models pass, ULTRA does not

Criterion B compares this project's unweighted group means against the figures
each model itself published. Targets live in `shared/published.json`, one block
per model, each carrying its source. They were constants in `analyse.py` once,
and they were ULTRA's constants, so running the report for any other model
compared that model against ULTRA's targets and printed a verdict that meant
nothing.

Everything below is one GPU run of one seed, 1024, on an RTX 4070 Ti SUPER.

| model | target source | ind_e MRR | ind_e H@10 | ind_er MRR | ind_er H@10 | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| MOTIF | arXiv 2502.13339 Table 2 | +0.0001 | -0.0003 | +0.0001 | +0.0004 | **PASS** |
| TRIX | arXiv 2502.19512 Table 1 | +0.0012 | +0.0011 | -0.0001 | +0.0009 | **PASS** |
| ULTRA | README, row `ULTRA (3g) PyG` | **-0.0042** | **+0.0064** | -0.0019 | -0.0012 | FAIL |

MOTIF and TRIX land on their own published figures to within 0.0012 on all four
group means, well inside the +/-0.002 band. They are ULTRA forks: same
`compute_ranking`, same suite, same rank dump, same metric code, same container
stack, same GPU. Two independent papers reproduced to the printed precision is a
strong statement about the harness, and it is the reason the ULTRA row can be
read as a fact about ULTRA rather than a fault in the measurement.

### What the ULTRA gap is not

**It is not the TorchDrug question.** An earlier version of this file said the
published 0.430 belongs to ULTRA's TorchDrug implementation and that this was
the whole gap. The first half is true -- `repos/ultra/README.md` line 31 names
`DeepGraphLearning/ultra_torchdrug` as a separate repository, and line 363 states
the preprint numbers came from a TorchDrug-trained model. The second half is
wrong. Criterion B never targeted 0.430. It targets the README's own PyG row,
0.420, measured by the authors with the same `run_many.py` this project drives,
and the gap against that row is what fails.

**It is not the device.** MOTIF and TRIX ran on the same GPU, in the same stack,
over the same 18 graphs, and hit their targets. A float32 kernel difference that
moved ULTRA by 0.004 would have moved them too.

**It is not one broken graph.** Per graph across ind_e, ULTRA sits below MOTIF
and TRIX by a margin that grows smoothly with how much those models improve on
it -- 0.157 on WN18RRInductive:v4, 0.010 on FB15k237Inductive:v4 -- and sits
above both on the HM and ILPC graphs. That is two better models, not one
corrupted dataset. Nothing in the per-graph table is anomalous.

**It is not the harness or the rank definition.** Both are shared, and both are
what MOTIF and TRIX pass with.

### What it might be

The remaining candidates are all specific to the ULTRA row itself: that
`ckpts/ultra_3g.pth` as shipped is not the checkpoint that produced the README
table, or that the table was measured against a dataset snapshot that has since
moved. Neither is settled here, and neither should be asserted without evidence.

One measurement worth recording, because it bears on how firm the target is.
Four sources give four different values for ULTRA's ind_e MRR on these same 18
graphs:

| source | ind_e MRR | ind_e H@10 |
| --- | --- | --- |
| ULTRA README, `ULTRA (3g) PyG` | 0.420 | 0.562 |
| ULTRA README, `ULTRA (3g) Paper` | 0.430 | 0.566 |
| MOTIF paper, ULTRA baseline row | 0.431 | 0.566 |
| TRIX paper, ULTRA baseline row | 0.431 | 0.566 |
| SEMMA paper, ULTRA baseline row | 0.428 | 0.570 |
| this project | 0.4158 | 0.5684 |

MOTIF and TRIX both quote 0.431, close to ULTRA's paper row rather than its PyG
row, which suggests neither re-ran ULTRA. SEMMA quotes a third value again. On
ind_er every source agrees within 0.001 and so does this project. The
disagreement is confined to ind_e, and it exists between the published sources
before this project is added to them.

Closing this needs `ultra_torchdrug` pinned as an eighth repository and run, so
that both ULTRA implementations are measured here rather than compared through
somebody else's table. That is on the task list.

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

## SEMMA runs without flash-attn, and that was measured, not assumed

SEMMA has two halves. The structural half is ULTRA. The semantic half embeds
relation descriptions with a sentence encoder, then builds a second relation
graph from the similarities between those embeddings.

`flags.yaml` selects `jinaai/jina-embeddings-v3` as that encoder. `transformers`
loads it with `trust_remote_code=True`, so the model repository supplies and
executes its own `custom_st.py`. That code checks for `flash_attn`, does not
find it, and prints one line for each attention layer:

```
flash_attn is not installed. Using PyTorch native attention implementation.
```

Upstream ships it this way. `repos/semma/requirements.txt` line 10 comments the
dependency out, with the authors' note that installation is complex. The gate
that prints the warning is `get_use_flash_attn` in `modeling_xlm_roberta.py`,
and it tests `importlib.util.find_spec` only, after `config.use_flash_attn` and
`torch.cuda.is_available()` both pass. Installing a wheel is therefore the whole
change. The question is whether to make it.

### The measurement

A matching wheel exists and works:
`flash_attn-2.5.8+cu118torch2.1cxx11abiFALSE-cp39-cp39`. The container is Python
3.9, torch 2.1.0+cu118, `_GLIBCXX_USE_CXX11_ABI = False`, and the GPU is compute
capability 8.9, so every constraint is met. A forward pass runs.

**Cost.** The encoder is not where SEMMA's time goes. Loading it takes 1.3 s and
encoding all 237 FB15k-237 relation names takes 0.8 s, against a steady-state
cost near 27 s per graph. Flash attention can save a fraction of that 0.8 s, so
under two percent of the suite. The earlier claim in this file, that unfused
attention was part of why SEMMA is the most expensive model here, was wrong.
SEMMA's cost is in the structural half.

**Numbers.** The two paths do not agree. Encoding the same 237 relation names
both ways gives a maximum absolute difference of 0.0055 per embedding component,
and up to 0.0128 in the pairwise cosine similarity.

That looks large until the dtype is checked. `config.torch_dtype` is
`bfloat16`, and the parameters load as bfloat16 on both paths. One bfloat16 step
near 1.0 is 0.0078. The measured 0.0055 is therefore below a single
representable step of the encoder's own precision, which is what a reordered but
exact attention kernel predicts.

The consequence is still real. SEMMA keeps every relation pair whose cosine
similarity exceeds 0.8. Over the 27966 pairs among those 237 relations, the
flash path keeps 363 and the native path keeps 364, and **5 pairs disagree**.
The semantic relation graph is not the same graph.

### The decision

Run without flash-attn. There is no speed argument, because the encoder is under
two percent of the cost. There is a correctness argument against, because the
threshold admits a different set of pairs. And upstream ships it off, so off is
also the configuration the authors published.

### What this says about SEMMA

The finding worth keeping is not about this container. SEMMA's 0.8 cutoff sits
inside the numerical noise floor of its own encoder's dtype: one bfloat16 step
near the threshold is 0.0039, and the observed cosine spread reaches 0.0128. Any
change that reorders arithmetic in that encoder can move pairs across the cutoff.
That is a property of the model as published, and it bounds how exactly any
reimplementation of SEMMA can be expected to match it.

## SEMMA could not run 14 of the 41 graphs, for two separate reasons

ILPC2022, WKIngram and WikiTopics -- 2 graphs in ind_e and 12 in ind_er -- do
not use their raw relation identifiers as relation names. The raw files carry
Wikidata property codes such as `P101`, while SEMMA's own LLM relation
descriptions in `openrouter/descriptions/` are keyed by English labels such as
`field of work`. `datasets.py` bridges the two by resolving each code through
the Wikidata API, and builds `edge2id` from the labels it gets back.

Both halves of that were broken, and neither said so.

### The API refuses the default User-Agent

Wikimedia's policy rejects `python-requests`' default string. Every call
answered 403. `fetch_wikidata` caught the failure and returned `None`, the
helpers substituted the string `"Failed to retrieve data"` for every relation,
and `edge2id = {v: k for k, v in id2relation.items()}` then collapsed every
relation onto that single key. The run died about 300 lines later in
`order_embeddings` with `KeyError: 0`, which names a relation id and says
nothing about a network call.

`patches/semma/0004` sends the User-Agent the policy asks for, raises instead
of substituting placeholder text, and caches what it read. The cache is the
reproducibility half: a Wikidata label is editable, and SEMMA builds half its
model from the embedding of that label.

### Seven property labels have been renamed since

With the API answering, the labels still did not all match. Wikidata labels are
editable and seven of these properties have changed:

| property | label now | key in SEMMA's description file |
| --- | --- | --- |
| P112 | founder | founded by |
| P355 | child organization or unit | has subsidiary |
| P410 | military, police or special rank | military or police rank |
| P488 | chairman | chairperson |
| P607 | participated in conflict | conflict |
| P749 | parent organization or unit | parent organization |
| P1029 | crew members | crew member |

The authors saw at least one. `datasets.py` carries the bare comment
`# parent organization/unit -> parent organization` above the ILPC2022 class.

`patches/semma/0005` recovers six of the seven from each property's own
Wikidata aliases. It first discards any alias that is another property's
current label in the same graph, and that guard is the point of the patch
rather than a detail: P112's aliases include `creator`, which is a real key in
these description files -- but it belongs to P170, which appears in the same
graphs. Matching on it would attach one relation's description and its
embedding to a different relation, with no error anywhere. P112 is then the
only entry in `LABEL_OVERRIDES`. Anything unresolved raises and names the
property rather than being guessed at.

On the first affected graph the patch reports what it did:

```
reconciled 3 renamed Wikidata label(s):
  P112: 'founder' -> 'founded by' (override);
  P749: 'parent organization or unit' -> 'parent organization' (alias);
  P488: 'chairman' -> 'chairperson' (alias)
```

### What this means for SEMMA as published

SEMMA reads a live, editable database at dataset-build time and matches what it
finds against a file generated once. That is not a bug in one function, it is
the design of its relation naming, and it means SEMMA's published numbers are
attached to a moment in Wikidata's history that is not recorded anywhere. The
cache this project writes pins the moment for these runs. Nothing can recover
the authors'.

## SEMMA fetched 225816 labels it then deleted

For the same 14 graphs, `datasets.py` also resolved every *entity* id to a
label -- 225816 entities, roughly 4500 API requests -- copied the result onto
`train_`, `valid_` and `test_` prefixed attributes, and then deleted the
originals in `attrs_to_remove` before saving the dataset.

`id2entity` appears nowhere in the repository outside `datasets.py`, and never
on the right-hand side of an expression. Neither do its prefixed copies. The
semantic stream is built from relation descriptions, so no entity label can
reach a number.

`patches/semma/0006` skips the fetch. Measured on ILPC2022:small, which is not
the largest of the 14: 142 s and a failure before, 42.8 s and a rank dump
after. `SEMMA_FETCH_ENTITY_LABELS=1` restores upstream behaviour.

## FLOCK: two things stood between it and a result

### It does not fit on a 16 GB card

FLOCK's README states it was tested on H100s. Every zero-shot config pairs
`batch_size` with `walk_num` so the product is 512 -- 32x16, 16x32, 8x64,
4x128 -- and scoring one batch materialises a GRU input of
`batch x candidates x walk_num x walk_len x hidden`. On the first graph that
asks for 18.8 GiB against 15.6 available.

`run.py` already reads `cfg.train.test_batch_size` and falls back to
`batch_size`, so `patches/flock/0003` only exposes the knob it already honours,
defaulting to each config's own `batch_size`. The runner divides by 4, holding
the product uniform at 128 across all configs. No walk count, walk length,
ensemble size or model hyper-parameter changes.

### It is not reproducible as shipped

FLOCK is the only stochastic model in this suite. It scores by sampling random
walks and averaging an ensemble of `test_samples` of them. `run_many.py`'s
`set_seed` seeds python's `random` and torch, and ships the numpy line
commented out:

```
    # np.random.seed(seed + util.get_rank())
```

Numpy's generator is the one the walk sampler uses, through
`graph_walker.random_walks_fast(seed=None)` and `graph_walker._seed(None)`. So
as shipped, two runs of one graph from one checkpoint return different numbers.

`patches/flock/0004` uncomments it. This does not make the run match the
published one: that run was not reproducible either, and no run can be made to
match it. It makes this run reproducible at all, at seed 1024, the seed every
other model here uses. `FLOCK_UNSEEDED_WALKS=1` restores upstream behaviour,
which is how to measure what share of a FLOCK number is walk-sampling noise.
Until that is measured, FLOCK's figures carry an uncertainty the other six do
not, and a comparison that ignores it is not on equal footing.

## The sentence encoder is pinned to one commit

`patches/semma/0003-pin-encoder.diff` adds a `revision` to the
`AutoModel.from_pretrained` call. Upstream passes none, so `main` decides both
the weights and the remote code on the day of the run.

Half of SEMMA is built from those embeddings. Without a pin, a re-run months
later can produce different relation graphs from the same checkpoint and the
same data, and nothing in the output would show why.

The container pins `JINA_REVISION=ab036b023d30b4d1138c4c3bfa9f0c445ab455d6`,
caches that commit at build time, and sets `HF_HUB_OFFLINE=1` for every run. The
copy that executes is the copy the image was built against. The patch does not
change which model SEMMA uses. It fixes which version of it.
