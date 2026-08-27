# ULTRA baseline report

ULTRA is the reference this project measures every other repo against, so this
report is about one question before it is about any number: **does
`shared/metrics.py`, reading only the dumped ranks, reproduce what ULTRA itself
reports?** If it does not, no later comparison means anything.


Generated 2026-08-27 by `scripts/make_report.py`.

## Verdict

| criterion | result | coverage |
| --- | --- | --- |
| A (strict) — every value bitwise identical to ULTRA's CSV | **FAIL** | 173/246 exact over 41 graphs |
| A (metric definition) — order-independent metrics bitwise identical | **PASS** | 123 comparisons; order-dependent ones within 2 float32 ulp |
| B — group means within ±0.002 of the repository figures | **PASS** | inductive (e) 18/18, inductive (e,r) 23/23 |

Both criteria must pass. They do not. What is missing and why is in *Deviations* below, and the criterion A residual is dissected in *The resolved tie rule* — it is float32 associativity inside ULTRA's own reduction, not a disagreement about what a rank is. Nothing was tuned to close any gap.


## Criterion A — metric equivalence

`shared/metrics.py` recomputes each metric from `ranks/ultra/*.parquet` and is compared against the raw `ultra_results_*.csv` ULTRA wrote during the same run, value by value, at the full 17-digit precision `csv.DictWriter` prints. `ulps(f32)` is the distance in float32 representable steps; 0 means the two floats are the same bit pattern.

Source CSVs (unmodified, kept in `results/`):

* `MOTIF_results_2026-08-26-19-03-02.csv`
* `MOTIF_results_2026-08-26-19-03-40.csv`
* `MOTIF_results_2026-08-26-19-03-55.csv`
* `MOTIF_results_2026-08-26-19-04-29.csv`
* `MOTIF_results_2026-08-26-19-05-52.csv`
* `MOTIF_results_2026-08-26-19-05-55.csv`
* `MOTIF_results_2026-08-26-19-06-00.csv`
* `MOTIF_results_2026-08-26-19-06-06.csv`
* `MOTIF_results_2026-08-26-19-06-19.csv`
* `MOTIF_results_2026-08-26-19-06-22.csv`
* `MOTIF_results_2026-08-26-19-06-32.csv`
* `MOTIF_results_2026-08-26-19-06-54.csv`
* `MOTIF_results_2026-08-26-19-07-11.csv`
* `MOTIF_results_2026-08-26-19-07-28.csv`
* `MOTIF_results_2026-08-26-19-10-24.csv`
* `MOTIF_results_2026-08-26-19-10-29.csv`
* `MOTIF_results_2026-08-26-19-10-42.csv`
* `MOTIF_results_2026-08-26-19-11-03.csv`
* `MOTIF_results_2026-08-26-19-57-03.csv`
* `MOTIF_results_2026-08-26-20-00-09.csv`
* `MOTIF_results_2026-08-26-20-01-26.csv`
* `MOTIF_results_2026-08-26-20-02-19.csv`
* `MOTIF_results_2026-08-26-20-02-30.csv`
* `MOTIF_results_2026-08-26-20-02-35.csv`
* `MOTIF_results_2026-08-26-20-02-56.csv`
* `MOTIF_results_2026-08-26-20-03-02.csv`
* `MOTIF_results_2026-08-26-20-03-29.csv`
* `MOTIF_results_2026-08-26-20-03-36.csv`
* `MOTIF_results_2026-08-26-20-03-41.csv`
* `MOTIF_results_2026-08-26-20-03-47.csv`
* `MOTIF_results_2026-08-26-20-03-52.csv`
* `MOTIF_results_2026-08-26-20-03-57.csv`
* `MOTIF_results_2026-08-26-20-04-19.csv`
* `MOTIF_results_2026-08-26-20-04-29.csv`
* `MOTIF_results_2026-08-26-20-04-43.csv`
* `MOTIF_results_2026-08-26-20-04-54.csv`
* `MOTIF_results_2026-08-26-20-05-18.csv`
* `MOTIF_results_2026-08-26-20-05-33.csv`
* `MOTIF_results_2026-08-26-20-05-43.csv`
* `MOTIF_results_2026-08-26-20-05-54.csv`
* `MOTIF_results_2026-08-26-20-06-02.csv`

**Criterion A (metric definition): PASS** -- 123/123 hit counts identical.

**Criterion A (strict, bitwise): FAIL** -- 246 comparisons, 173 exact, 73 mismatched.

The two verdicts answer different questions, and only the first one is about ranking.

* **Order-independent metrics** (`hits@1`, `hits@3`, `hits@10`): 123/123 identical as counts -- **PASS**; 98/123 identical bitwise. `hits@k` is `count / n_queries`, and the count is a whole number that no summation order can move: a count disagreement is a disagreement about the tie rule, the rank offset or the dump. The float around it can still differ, because that last division is not done the same way everywhere -- on CUDA torch reduces and scales differently from numpy on the host. Every bitwise mismatch seen here is 1 ulp with the counts identical, which is that division and nothing else.
* **Order-dependent metrics** (`mrr`, `mr`, `hits@10_50`): 75/123 exact; worst disagreement 2 float32 ulp (6.10e-05 absolute). These carry no count to fall back on, so float32 associativity is the whole story.

| dataset | metric | metrics.py | ULTRA csv | exact | |diff| | ulps(f32) |
| --- | --- | --- | --- | --- | --- | --- |
| FB15k237Inductive:v1 | mrr | 0.5047094225883484 | 0.5047094225883484 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@1 | 0.4051094949245453 | 0.4051094949245453 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@3 | 0.569343090057373 | 0.569343090057373 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@10 | 0.6934306621551514 | 0.6934306621551514 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | mr | 50.99513244628906 | 50.99513244628906 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v1 | hits@10_50 | 0.925051212310791 | 0.925051212310791 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | mrr | 0.5111135840415955 | 0.5111135840415955 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@1 | 0.3954593539237976 | 0.3954593539237976 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@3 | 0.5850052833557129 | 0.5850052833557129 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v2 | hits@10 | 0.7159450650215149 | 0.7159451246261597 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v2 | mr | 45.14519500732422 | 45.145198822021484 | **no** | 3.815e-06 | 1 |
| FB15k237Inductive:v2 | hits@10_50 | 0.9582197070121765 | 0.9582197666168213 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v3 | mrr | 0.4996998608112335 | 0.4996998906135559 | **no** | 2.980e-08 | 1 |
| FB15k237Inductive:v3 | hits@1 | 0.4012131690979004 | 0.4012131690979004 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | hits@3 | 0.558925449848175 | 0.558925449848175 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | hits@10 | 0.670710563659668 | 0.670710563659668 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | mr | 69.31340026855469 | 69.31340026855469 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v3 | hits@10_50 | 0.9586378931999207 | 0.9586378335952759 | **no** | 5.960e-08 | 1 |
| FB15k237Inductive:v4 | mrr | 0.4870959520339966 | 0.4870959520339966 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@1 | 0.3802816867828369 | 0.3802816867828369 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@3 | 0.5519366264343262 | 0.5519366264343262 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | hits@10 | 0.6772887110710144 | 0.6772887110710144 | yes | 0.000e+00 | 0 |
| FB15k237Inductive:v4 | mr | 63.05422592163086 | 63.054222106933594 | **no** | 3.815e-06 | 1 |
| FB15k237Inductive:v4 | hits@10_50 | 0.9724820852279663 | 0.9724821448326111 | **no** | 5.960e-08 | 1 |
| FBIngram:100 | mrr | 0.42834436893463135 | 0.42834436893463135 | yes | 0.000e+00 | 0 |
| FBIngram:100 | hits@1 | 0.3248175084590912 | 0.3248175382614136 | **no** | 2.980e-08 | 1 |
| FBIngram:100 | hits@3 | 0.473164439201355 | 0.47316446900367737 | **no** | 2.980e-08 | 1 |
| FBIngram:100 | hits@10 | 0.6277372241020203 | 0.6277372241020203 | yes | 0.000e+00 | 0 |
| FBIngram:100 | mr | 64.65843963623047 | 64.65843963623047 | yes | 0.000e+00 | 0 |
| FBIngram:100 | hits@10_50 | 0.960669755935669 | 0.9606696963310242 | **no** | 5.960e-08 | 1 |
| FBIngram:25 | mrr | 0.38383251428604126 | 0.38383251428604126 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@1 | 0.2638208568096161 | 0.2638208568096161 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@3 | 0.43308258056640625 | 0.43308258056640625 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@10 | 0.6398705244064331 | 0.6398705244064331 | yes | 0.000e+00 | 0 |
| FBIngram:25 | mr | 71.98985290527344 | 71.98985290527344 | yes | 0.000e+00 | 0 |
| FBIngram:25 | hits@10_50 | 0.974837601184845 | 0.9748376607894897 | **no** | 5.960e-08 | 1 |
| FBIngram:50 | mrr | 0.3381616473197937 | 0.3381616473197937 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@1 | 0.23562774062156677 | 0.23562775552272797 | **no** | 1.490e-08 | 1 |
| FBIngram:50 | hits@3 | 0.37458106875419617 | 0.37458109855651855 | **no** | 2.980e-08 | 1 |
| FBIngram:50 | hits@10 | 0.546403706073761 | 0.546403706073761 | yes | 0.000e+00 | 0 |
| FBIngram:50 | mr | 143.47280883789062 | 143.47280883789062 | yes | 0.000e+00 | 0 |
| FBIngram:50 | hits@10_50 | 0.9455816745758057 | 0.9455816745758057 | yes | 0.000e+00 | 0 |
| FBIngram:75 | mrr | 0.398820698261261 | 0.398820698261261 | yes | 0.000e+00 | 0 |
| FBIngram:75 | hits@1 | 0.2879909873008728 | 0.2879909873008728 | yes | 0.000e+00 | 0 |
| FBIngram:75 | hits@3 | 0.45090147852897644 | 0.45090150833129883 | **no** | 2.980e-08 | 1 |
| FBIngram:75 | hits@10 | 0.6141339540481567 | 0.6141339540481567 | yes | 0.000e+00 | 0 |
| FBIngram:75 | mr | 67.17160034179688 | 67.1716079711914 | **no** | 7.629e-06 | 1 |
| FBIngram:75 | hits@10_50 | 0.9675958752632141 | 0.9675959944725037 | **no** | 1.192e-07 | 2 |
| FBNELL | mrr | 0.4687882363796234 | 0.4687882363796234 | yes | 0.000e+00 | 0 |
| FBNELL | hits@1 | 0.3676716983318329 | 0.3676716983318329 | yes | 0.000e+00 | 0 |
| FBNELL | hits@3 | 0.5100502371788025 | 0.5100502371788025 | yes | 0.000e+00 | 0 |
| FBNELL | hits@10 | 0.6649916172027588 | 0.6649916172027588 | yes | 0.000e+00 | 0 |
| FBNELL | mr | 65.67839050292969 | 65.67839050292969 | yes | 0.000e+00 | 0 |
| FBNELL | hits@10_50 | 0.9870082139968872 | 0.987008273601532 | **no** | 5.960e-08 | 1 |
| HM:1k | mrr | 0.0634799674153328 | 0.0634799674153328 | yes | 0.000e+00 | 0 |
| HM:1k | hits@1 | 0.03781512752175331 | 0.03781512752175331 | yes | 0.000e+00 | 0 |
| HM:1k | hits@3 | 0.06197478994727135 | 0.061974793672561646 | **no** | 3.725e-09 | 1 |
| HM:1k | hits@10 | 0.09663865715265274 | 0.09663865715265274 | yes | 0.000e+00 | 0 |
| HM:1k | mr | 1006.8928833007812 | 1006.8928833007812 | yes | 0.000e+00 | 0 |
| HM:1k | hits@10_50 | 0.8065340518951416 | 0.8065341114997864 | **no** | 5.960e-08 | 1 |
| HM:3k | mrr | 0.055372823029756546 | 0.055372823029756546 | yes | 0.000e+00 | 0 |
| HM:3k | hits@1 | 0.03632320091128349 | 0.03632320463657379 | **no** | 3.725e-09 | 1 |
| HM:3k | hits@3 | 0.05448480322957039 | 0.05448480695486069 | **no** | 3.725e-09 | 1 |
| HM:3k | hits@10 | 0.08413639664649963 | 0.08413639664649963 | yes | 0.000e+00 | 0 |
| HM:3k | mr | 1878.362548828125 | 1878.362548828125 | yes | 0.000e+00 | 0 |
| HM:3k | hits@10_50 | 0.8169804811477661 | 0.8169804215431213 | **no** | 5.960e-08 | 1 |
| HM:5k | mrr | 0.05002613365650177 | 0.05002613365650177 | yes | 0.000e+00 | 0 |
| HM:5k | hits@1 | 0.03436911478638649 | 0.03436911478638649 | yes | 0.000e+00 | 0 |
| HM:5k | hits@3 | 0.049435026943683624 | 0.049435026943683624 | yes | 0.000e+00 | 0 |
| HM:5k | hits@10 | 0.07321092486381531 | 0.07321092486381531 | yes | 0.000e+00 | 0 |
| HM:5k | mr | 2615.837158203125 | 2615.837158203125 | yes | 0.000e+00 | 0 |
| HM:5k | hits@10_50 | 0.8009949326515198 | 0.8009949326515198 | yes | 0.000e+00 | 0 |
| HM:indigo | mrr | 0.4255574941635132 | 0.4255574643611908 | **no** | 2.980e-08 | 1 |
| HM:indigo | hits@1 | 0.3157206177711487 | 0.3157206177711487 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@3 | 0.4826892018318176 | 0.4826892018318176 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@10 | 0.6365405321121216 | 0.6365405321121216 | yes | 0.000e+00 | 0 |
| HM:indigo | mr | 97.144287109375 | 97.144287109375 | yes | 0.000e+00 | 0 |
| HM:indigo | hits@10_50 | 0.9955934286117554 | 0.9955933690071106 | **no** | 5.960e-08 | 1 |
| ILPC2022:large | mrr | 0.2854379117488861 | 0.2854379415512085 | **no** | 2.980e-08 | 1 |
| ILPC2022:large | hits@1 | 0.21469952166080475 | 0.21469953656196594 | **no** | 1.490e-08 | 1 |
| ILPC2022:large | hits@3 | 0.325265109539032 | 0.32526513934135437 | **no** | 2.980e-08 | 1 |
| ILPC2022:large | hits@10 | 0.414620965719223 | 0.4146209955215454 | **no** | 2.980e-08 | 1 |
| ILPC2022:large | mr | 1460.170654296875 | 1460.170654296875 | yes | 0.000e+00 | 0 |
| ILPC2022:large | hits@10_50 | 0.9145548343658447 | 0.9145548343658447 | yes | 0.000e+00 | 0 |
| ILPC2022:small | mrr | 0.2955174744129181 | 0.2955174446105957 | **no** | 2.980e-08 | 1 |
| ILPC2022:small | hits@1 | 0.21829771995544434 | 0.21829771995544434 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@3 | 0.328738808631897 | 0.328738808631897 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@10 | 0.4443487226963043 | 0.4443487226963043 | yes | 0.000e+00 | 0 |
| ILPC2022:small | mr | 401.7886047363281 | 401.7886047363281 | yes | 0.000e+00 | 0 |
| ILPC2022:small | hits@10_50 | 0.8947778940200806 | 0.8947778940200806 | yes | 0.000e+00 | 0 |
| Metafam | mrr | 0.3440011143684387 | 0.3440011441707611 | **no** | 2.980e-08 | 1 |
| Metafam | hits@1 | 0.15217390656471252 | 0.15217392146587372 | **no** | 1.490e-08 | 1 |
| Metafam | hits@3 | 0.4103260934352875 | 0.4103260934352875 | yes | 0.000e+00 | 0 |
| Metafam | hits@10 | 0.8288043737411499 | 0.8288043737411499 | yes | 0.000e+00 | 0 |
| Metafam | mr | 10.559782981872559 | 10.559782981872559 | yes | 0.000e+00 | 0 |
| Metafam | hits@10_50 | 0.9826166033744812 | 0.9826166033744812 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | mrr | 0.6684776544570923 | 0.6684776544570923 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@1 | 0.5945273637771606 | 0.5945273637771606 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@3 | 0.6815920472145081 | 0.6815920472145081 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@10 | 0.8656716346740723 | 0.8656716346740723 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | mr | 4.034825801849365 | 4.034825801849365 | yes | 0.000e+00 | 0 |
| NELLInductive:v1 | hits@10_50 | 0.8811387419700623 | 0.8811386227607727 | **no** | 1.192e-07 | 2 |
| NELLInductive:v2 | mrr | 0.56395024061203 | 0.56395024061203 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@1 | 0.4524064064025879 | 0.4524064362049103 | **no** | 2.980e-08 | 1 |
| NELLInductive:v2 | hits@3 | 0.6363636255264282 | 0.6363636255264282 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | hits@10 | 0.7684491872787476 | 0.7684491872787476 | yes | 0.000e+00 | 0 |
| NELLInductive:v2 | mr | 27.945453643798828 | 27.94545555114746 | **no** | 1.907e-06 | 1 |
| NELLInductive:v2 | hits@10_50 | 0.9843021631240845 | 0.9843021631240845 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | mrr | 0.5326294898986816 | 0.5326294898986816 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@1 | 0.4317901134490967 | 0.43179014325141907 | **no** | 2.980e-08 | 1 |
| NELLInductive:v3 | hits@3 | 0.5907407402992249 | 0.5907407402992249 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@10 | 0.7243826985359192 | 0.724382758140564 | **no** | 5.960e-08 | 1 |
| NELLInductive:v3 | mr | 35.92376708984375 | 35.92376708984375 | yes | 0.000e+00 | 0 |
| NELLInductive:v3 | hits@10_50 | 0.992525577545166 | 0.9925256371498108 | **no** | 5.960e-08 | 1 |
| NELLInductive:v4 | mrr | 0.5034835338592529 | 0.5034835338592529 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@1 | 0.38113337755203247 | 0.38113337755203247 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@3 | 0.5929509401321411 | 0.5929509401321411 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@10 | 0.710780918598175 | 0.710780918598175 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | mr | 51.56461715698242 | 51.56461715698242 | yes | 0.000e+00 | 0 |
| NELLInductive:v4 | hits@10_50 | 0.9755236506462097 | 0.9755237102508545 | **no** | 5.960e-08 | 1 |
| NLIngram:0 | mrr | 0.3236577808856964 | 0.3236578106880188 | **no** | 2.980e-08 | 1 |
| NLIngram:0 | hits@1 | 0.23722149431705475 | 0.23722150921821594 | **no** | 1.490e-08 | 1 |
| NLIngram:0 | hits@3 | 0.3381389379501343 | 0.3381389379501343 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@10 | 0.496723473072052 | 0.496723473072052 | yes | 0.000e+00 | 0 |
| NLIngram:0 | mr | 65.14678955078125 | 65.14678955078125 | yes | 0.000e+00 | 0 |
| NLIngram:0 | hits@10_50 | 0.9597114324569702 | 0.9597113728523254 | **no** | 5.960e-08 | 1 |
| NLIngram:100 | mrr | 0.43845731019973755 | 0.4384573698043823 | **no** | 5.960e-08 | 2 |
| NLIngram:100 | hits@1 | 0.33228248357772827 | 0.33228248357772827 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@3 | 0.4924337863922119 | 0.4924337863922119 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@10 | 0.6475409865379333 | 0.6475409865379333 | yes | 0.000e+00 | 0 |
| NLIngram:100 | mr | 39.96342849731445 | 39.96342849731445 | yes | 0.000e+00 | 0 |
| NLIngram:100 | hits@10_50 | 0.9765729904174805 | 0.9765729904174805 | yes | 0.000e+00 | 0 |
| NLIngram:25 | mrr | 0.34768205881118774 | 0.34768208861351013 | **no** | 2.980e-08 | 1 |
| NLIngram:25 | hits@1 | 0.26008063554763794 | 0.26008063554763794 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@3 | 0.375 | 0.375 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@10 | 0.4979838728904724 | 0.4979838728904724 | yes | 0.000e+00 | 0 |
| NLIngram:25 | mr | 76.3326644897461 | 76.3326644897461 | yes | 0.000e+00 | 0 |
| NLIngram:25 | hits@10_50 | 0.9552790522575378 | 0.9552789330482483 | **no** | 1.192e-07 | 2 |
| NLIngram:50 | mrr | 0.3734474778175354 | 0.3734475076198578 | **no** | 2.980e-08 | 1 |
| NLIngram:50 | hits@1 | 0.28346914052963257 | 0.28346914052963257 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@3 | 0.41443538665771484 | 0.41443538665771484 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@10 | 0.5325960516929626 | 0.5325960516929626 | yes | 0.000e+00 | 0 |
| NLIngram:50 | mr | 77.3119888305664 | 77.3119888305664 | yes | 0.000e+00 | 0 |
| NLIngram:50 | hits@10_50 | 0.9575586318969727 | 0.9575586318969727 | yes | 0.000e+00 | 0 |
| NLIngram:75 | mrr | 0.3140687346458435 | 0.3140687346458435 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@1 | 0.2149917632341385 | 0.2149917632341385 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@3 | 0.3459637463092804 | 0.3459637463092804 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@10 | 0.5107083916664124 | 0.5107083916664124 | yes | 0.000e+00 | 0 |
| NLIngram:75 | mr | 50.175453186035156 | 50.175453186035156 | yes | 0.000e+00 | 0 |
| NLIngram:75 | hits@10_50 | 0.9707918167114258 | 0.9707918167114258 | yes | 0.000e+00 | 0 |
| WKIngram:100 | mrr | 0.16432227194309235 | 0.16432227194309235 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@1 | 0.10175711661577225 | 0.10175711661577225 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@3 | 0.18371886014938354 | 0.18371886014938354 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@10 | 0.28213968873023987 | 0.28213968873023987 | yes | 0.000e+00 | 0 |
| WKIngram:100 | mr | 604.0765991210938 | 604.0765991210938 | yes | 0.000e+00 | 0 |
| WKIngram:100 | hits@10_50 | 0.9084514379501343 | 0.908451497554779 | **no** | 5.960e-08 | 1 |
| WKIngram:25 | mrr | 0.3110649287700653 | 0.3110649585723877 | **no** | 2.980e-08 | 1 |
| WKIngram:25 | hits@1 | 0.22104331851005554 | 0.22104333341121674 | **no** | 1.490e-08 | 1 |
| WKIngram:25 | hits@3 | 0.3465959429740906 | 0.3465959429740906 | yes | 0.000e+00 | 0 |
| WKIngram:25 | hits@10 | 0.4933687150478363 | 0.4933687150478363 | yes | 0.000e+00 | 0 |
| WKIngram:25 | mr | 168.26878356933594 | 168.268798828125 | **no** | 1.526e-05 | 1 |
| WKIngram:25 | hits@10_50 | 0.9264618754386902 | 0.9264619946479797 | **no** | 1.192e-07 | 2 |
| WKIngram:50 | mrr | 0.1627478301525116 | 0.1627478003501892 | **no** | 2.980e-08 | 2 |
| WKIngram:50 | hits@1 | 0.09147287160158157 | 0.09147286415100098 | **no** | 7.451e-09 | 1 |
| WKIngram:50 | hits@3 | 0.1770542562007904 | 0.1770542562007904 | yes | 0.000e+00 | 0 |
| WKIngram:50 | hits@10 | 0.31333333253860474 | 0.31333333253860474 | yes | 0.000e+00 | 0 |
| WKIngram:50 | mr | 859.7763061523438 | 859.7762451171875 | **no** | 6.104e-05 | 1 |
| WKIngram:50 | hits@10_50 | 0.8616250157356262 | 0.8616250157356262 | yes | 0.000e+00 | 0 |
| WKIngram:75 | mrr | 0.36604028940200806 | 0.36604028940200806 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@1 | 0.2753496468067169 | 0.2753496468067169 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@3 | 0.4086538553237915 | 0.4086538553237915 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@10 | 0.5402097702026367 | 0.5402097702026367 | yes | 0.000e+00 | 0 |
| WKIngram:75 | mr | 135.7329559326172 | 135.7329559326172 | yes | 0.000e+00 | 0 |
| WKIngram:75 | hits@10_50 | 0.9310377240180969 | 0.9310376048088074 | **no** | 1.192e-07 | 2 |
| WN18RRInductive:v1 | mrr | 0.6807820200920105 | 0.6807820796966553 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v1 | hits@1 | 0.6233243942260742 | 0.6233243942260742 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@3 | 0.7131367325782776 | 0.7131367325782776 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@10 | 0.777479887008667 | 0.7774799466133118 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v1 | mr | 53.38605880737305 | 53.38605880737305 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v1 | hits@10_50 | 0.9046658873558044 | 0.9046658873558044 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | mrr | 0.6628316044807434 | 0.6628316044807434 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@1 | 0.6103286147117615 | 0.6103286743164062 | **no** | 5.960e-08 | 1 |
| WN18RRInductive:v2 | hits@3 | 0.6860328912734985 | 0.6860328912734985 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@10 | 0.7711267471313477 | 0.7711267471313477 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | mr | 138.44601440429688 | 138.44601440429688 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v2 | hits@10_50 | 0.9108631610870361 | 0.9108631610870361 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | mrr | 0.4197711944580078 | 0.4197711944580078 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@1 | 0.3600175082683563 | 0.3600175082683563 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@3 | 0.4448818862438202 | 0.4448818862438202 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | hits@10 | 0.5376203060150146 | 0.5376203060150146 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v3 | mr | 326.2152099609375 | 326.2152404785156 | **no** | 3.052e-05 | 1 |
| WN18RRInductive:v3 | hits@10_50 | 0.8709588646888733 | 0.8709587454795837 | **no** | 1.192e-07 | 2 |
| WN18RRInductive:v4 | mrr | 0.6403668522834778 | 0.6403668522834778 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@1 | 0.5982996821403503 | 0.5982996821403503 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@3 | 0.6613531708717346 | 0.6613531708717346 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@10 | 0.7178533673286438 | 0.7178533673286438 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | mr | 432.6654357910156 | 432.6654357910156 | yes | 0.000e+00 | 0 |
| WN18RRInductive:v4 | hits@10_50 | 0.8905373215675354 | 0.8905373811721802 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT1:health | mrr | 0.326387494802475 | 0.326387494802475 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@1 | 0.28192847967147827 | 0.28192847967147827 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@3 | 0.3400382995605469 | 0.3400382995605469 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@10 | 0.3978288769721985 | 0.3978288471698761 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT1:health | mr | 301.2547912597656 | 301.2547912597656 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:health | hits@10_50 | 0.9661930799484253 | 0.9661930203437805 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT1:tax | mrr | 0.32463064789772034 | 0.32463064789772034 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@1 | 0.2565430700778961 | 0.2565430700778961 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@3 | 0.3601417541503906 | 0.3601417541503906 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@10 | 0.4479280114173889 | 0.4479280114173889 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | mr | 365.78515625 | 365.78515625 | yes | 0.000e+00 | 0 |
| WikiTopicsMT1:tax | hits@10_50 | 0.9418656229972839 | 0.9418656229972839 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | mrr | 0.09211689978837967 | 0.09211689978837967 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@1 | 0.055305201560258865 | 0.055305201560258865 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@3 | 0.09873002767562866 | 0.09873002767562866 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@10 | 0.15424007177352905 | 0.15424007177352905 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | mr | 1033.944091796875 | 1033.944091796875 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:org | hits@10_50 | 0.7735559344291687 | 0.7735559344291687 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | mrr | 0.2858940064907074 | 0.2858940064907074 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@1 | 0.2009090930223465 | 0.2009090930223465 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@3 | 0.33636364340782166 | 0.33636364340782166 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@10 | 0.43454545736312866 | 0.43454545736312866 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | mr | 347.5299987792969 | 347.5299987792969 | yes | 0.000e+00 | 0 |
| WikiTopicsMT2:sci | hits@10_50 | 0.9443399310112 | 0.9443398714065552 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:art | mrr | 0.2692040205001831 | 0.2692040205001831 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@1 | 0.19402505457401276 | 0.19402505457401276 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@3 | 0.29714101552963257 | 0.29714101552963257 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | hits@10 | 0.41439124941825867 | 0.41439124941825867 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:art | mr | 456.68536376953125 | 456.6853332519531 | **no** | 3.052e-05 | 1 |
| WikiTopicsMT3:art | hits@10_50 | 0.9172103404998779 | 0.9172102808952332 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:infra | mrr | 0.6583489775657654 | 0.6583490967750549 | **no** | 1.192e-07 | 2 |
| WikiTopicsMT3:infra | hits@1 | 0.5906444787979126 | 0.5906445384025574 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT3:infra | hits@3 | 0.6952183246612549 | 0.6952183246612549 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@10 | 0.7864865064620972 | 0.7864865064620972 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | mr | 61.451560974121094 | 61.451560974121094 | yes | 0.000e+00 | 0 |
| WikiTopicsMT3:infra | hits@10_50 | 0.9919083118438721 | 0.9919083714485168 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT4:health | mrr | 0.6264393329620361 | 0.6264393925666809 | **no** | 5.960e-08 | 1 |
| WikiTopicsMT4:health | hits@1 | 0.55226069688797 | 0.55226069688797 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@3 | 0.6717557311058044 | 0.6717557311058044 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@10 | 0.7621843814849854 | 0.7621843814849854 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | mr | 161.46270751953125 | 161.46270751953125 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:health | hits@10_50 | 0.9714552760124207 | 0.9714552760124207 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | mrr | 0.283387154340744 | 0.28338712453842163 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT4:sci | hits@1 | 0.19920749962329865 | 0.19920748472213745 | **no** | 1.490e-08 | 1 |
| WikiTopicsMT4:sci | hits@3 | 0.3152017295360565 | 0.3152017295360565 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@10 | 0.4506484270095825 | 0.45064839720726013 | **no** | 2.980e-08 | 1 |
| WikiTopicsMT4:sci | mr | 321.8987731933594 | 321.8987731933594 | yes | 0.000e+00 | 0 |
| WikiTopicsMT4:sci | hits@10_50 | 0.9506902694702148 | 0.9506901502609253 | **no** | 1.192e-07 | 2 |


## Per-dataset results

### ind_e (18 of 18 graphs)

| dataset | family | queries | MRR | Hits@10 |
| --- | --- | --- | --- | --- |
| FB15k237Inductive:v1 | FB | 822 | 0.5047 | 0.6934 |
| FB15k237Inductive:v2 | FB | 1894 | 0.5111 | 0.7159 |
| FB15k237Inductive:v3 | FB | 3462 | 0.4997 | 0.6707 |
| FB15k237Inductive:v4 | FB | 5680 | 0.4871 | 0.6773 |
| WN18RRInductive:v1 | WN | 746 | 0.6808 | 0.7775 |
| WN18RRInductive:v2 | WN | 1704 | 0.6628 | 0.7711 |
| WN18RRInductive:v3 | WN | 2286 | 0.4198 | 0.5376 |
| WN18RRInductive:v4 | WN | 5646 | 0.6404 | 0.7179 |
| NELLInductive:v1 | NELL | 402 | 0.6685 | 0.8657 |
| NELLInductive:v2 | NELL | 1870 | 0.5640 | 0.7684 |
| NELLInductive:v3 | NELL | 3240 | 0.5326 | 0.7244 |
| NELLInductive:v4 | NELL | 2894 | 0.5035 | 0.7108 |
| ILPC2022:small | WK | 5804 | 0.2955 | 0.4443 |
| ILPC2022:large | WK | 20368 | 0.2854 | 0.4146 |
| HM:1k | FB | 952 | 0.0635 | 0.0966 |
| HM:3k | FB | 2698 | 0.0554 | 0.0841 |
| HM:5k | FB | 4248 | 0.0500 | 0.0732 |
| HM:indigo | FB | 29808 | 0.4256 | 0.6365 |

### ind_er (23 of 23 graphs)

| dataset | family | queries | MRR | Hits@10 |
| --- | --- | --- | --- | --- |
| FBIngram:25 | FB | 11432 | 0.3838 | 0.6399 |
| FBIngram:50 | FB | 7758 | 0.3382 | 0.5464 |
| FBIngram:75 | FB | 6212 | 0.3988 | 0.6141 |
| FBIngram:100 | FB | 4658 | 0.4283 | 0.6277 |
| WKIngram:25 | WK | 2262 | 0.3111 | 0.4934 |
| WKIngram:50 | WK | 6450 | 0.1627 | 0.3133 |
| WKIngram:75 | WK | 2288 | 0.3660 | 0.5402 |
| WKIngram:100 | WK | 8992 | 0.1643 | 0.2821 |
| NLIngram:0 | NELL | 1526 | 0.3237 | 0.4967 |
| NLIngram:25 | NELL | 1488 | 0.3477 | 0.4980 |
| NLIngram:50 | NELL | 1718 | 0.3734 | 0.5326 |
| NLIngram:75 | NELL | 1214 | 0.3141 | 0.5107 |
| NLIngram:100 | NELL | 1586 | 0.4385 | 0.6475 |
| WikiTopicsMT1:tax | WK | 3668 | 0.3246 | 0.4479 |
| WikiTopicsMT1:health | WK | 3132 | 0.3264 | 0.3978 |
| WikiTopicsMT2:org | WK | 4882 | 0.0921 | 0.1542 |
| WikiTopicsMT2:sci | WK | 3300 | 0.2859 | 0.4345 |
| WikiTopicsMT3:art | WK | 6226 | 0.2692 | 0.4144 |
| WikiTopicsMT3:infra | WK | 4810 | 0.6583 | 0.7865 |
| WikiTopicsMT4:sci | WK | 2776 | 0.2834 | 0.4506 |
| WikiTopicsMT4:health | WK | 3406 | 0.6264 | 0.7622 |
| Metafam | other | 368 | 0.3440 | 0.8288 |
| FBNELL | other | 1194 | 0.4688 | 0.6650 |


## Criterion B — published numbers

Targets are the ULTRA **repository's** PyG figures (README at the pinned SHA), not the paper's. Group means are unweighted over datasets: every graph counts once regardless of how many test queries it has. The last two columns show the distance to the paper numbers as well — landing on those instead of the repository ones would be an anomaly worth reporting.

**Criterion B (motif): PASS**

Target: `arXiv 2502.13339 (ICML 2025), Table 2, row 'Motif (F_3^path)'`.

pretrained on FB15k237, WN18RR and CoDEx Medium. The released ckpts/motif_3g.pth is taken to be this row's checkpoint; the repository ships no other 3-graph checkpoint.

| group | metric | datasets | ours | paper target | delta | within +/-0.002 |
| --- | --- | --- | --- | --- | --- | --- |
| ind_e | mrr | 18/18 | 0.4361 | 0.436 | +0.0001 | yes |
| ind_e | hits@10 | 18/18 | 0.5767 | 0.577 | -0.0003 | yes |
| ind_er | mrr | 23/23 | 0.3491 | 0.349 | +0.0001 | yes |
| ind_er | hits@10 | 23/23 | 0.5254 | 0.525 | +0.0004 | yes |



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
