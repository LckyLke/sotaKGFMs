# Combination probe: propagation + in-context, and L1 + MX (2026-09-04, CPU only)

Two questions, one afternoon of CPU, nothing on the 41 test graphs
except a descriptive read of the existing rank dumps.

1. Luke's axis: does an architecture that combines the PROPAGATION
   family (TRIX, ULTRA, the INCITE trunk: NBFNet-style message passing
   over the relation structure) with the IN-CONTEXT family (KG-ICL's
   prompt graphs of example facts, KGPFN's in-context inference over
   example triples) promise the best of both? What do the dumps say
   about their complementarity, and which fusion designs have a
   premise that survives a cheap test?
2. The narrower axis the program already has: L1 (no synthetic data;
   best on evidence-rich, seen-answer candidates) against MX15 / MX1
   (15 / 30 percent synthetic rules-prior steps; best on evidence-poor,
   unseen-answer candidates). Does any inference-time, weight-level or
   in-trunk combination of two existing checkpoints recover the
   review's oracle (+0.0076 / +0.0079 over MX1 on the test suite)?

Scripts, dumps and result JSONs: `output/incite-combo/` (listed at the
end). Every number below is reproducible from them.

## Verdict

* **L1 + MX at the score level: nothing.** On eight held-out KGs (sixteen
  carved inductive splits, design half / held-out half), every ensemble,
  per-candidate switch, per-scenario mixture, rank fusion, learned
  combiner and per-query gate of L1 with MX1 or MX15 lands within
  ±0.001 of L1 alone (the best single model there), against an oracle
  headroom of +0.0065 over L1 (+0.0210 over MX1). The oracle's
  information is per query (which candidate population holds the true
  answer); the observable per-candidate indicator cannot deliver it, and
  a per-query gate that predicts it with AUC 0.84 does not help, because
  the queries it hands to MX are the ones where L1's margin is largest.
* **Weight-space soups reproduce the two-pass ensemble at one pass**
  (soup L1 / MX15 at 0.5: 0.2571 against the ensemble's 0.2579 and L1's
  0.2579; profile between the two models; 0.003 above the interpolation
  of the members' numbers), and the soup restricted to the entity steps
  does the same: the trade lives in the entity steps. The best one-pass
  model of the day is a round-wise swap, L1's rounds 0-2 and heads with
  MX15's entity and relation steps of rounds 3-5: 0.2603, +0.0024
  [+0.0009, +0.0039] over L1 on the held-out half (+0.0016 on the design
  half), with both SQSA (0.2913 vs 0.2869) and UQUA (0.1392 vs 0.1221)
  above L1's: the only combination of the day whose interval against the
  best single model is above zero and whose cells do not trade. The
  reverse assignment (MX's early rounds) is −0.015. Small, unfitted
  (one of seven pre-chosen partial soups), and the one lead.
* **In-trunk fusion of L1 and MX: the states carry the signal, the
  heads do not use it.** A linear probe reads the answer-half indicator
  from either trunk's final node states with AUC 1.00 and the query-half
  indicator from the relation states with AUC 0.998, so a scenario-
  conditioned trunk has a signal; but a fusion head on both frozen
  trunks' states fitted on 12k design queries is +0.001 over the
  ensemble at best, no better than a head on the two scores alone.
* **Propagation vs in-context on the 41 graphs: the in-context models
  are below the propagation models on every cell, and their wins are
  per graph family, not per scenario.** KG-ICL is under MX15 on all
  four cells of both groups (ind_e 0.4240 vs 0.4621; ind_er 0.3722 vs
  0.3893); it is above MX15 on 13 of 41 graphs (the three HM graphs,
  four of the five NLIngram, three of the four WKIngram, FB15k237 v2,
  FBIngram:100, WikiTopicsMT1:health) and above TRIX on 19; it wins the
  ind_er NELL family (+0.008, on the unseen-answer cells) and loses WN,
  WikiTopics and ind_e NELL by 0.05 to 0.11. KGPFN (29 graphs, exact
  per-query joins) is below MX15 on 24 of 29. The cell-switch oracle
  KG-ICL / MX15 is +0.007 / +0.011 over MX15; the per-query oracle
  KGPFN / MX15 is +0.032 / +0.035 (a loose bound). No plain rank fusion
  can be evaluated from the dumps: they hold the true answer's rank
  only, not the candidate lists.
* **Fusion designs:** of three designs that feed in-context relation
  evidence into the propagation trunk, the cheap premise tests find one
  premise already satisfied by the trunk (its relation states predict
  the relation's in-context statistics with R² 0.77 to 0.91; a
  handcrafted example-similarity feature adds nothing), one that works
  only with oracle examples (per-relation calibration of the seen /
  unseen candidate populations: +0.0016 [+0.0002, +0.0031] over L1 when
  the examples are other real queries of the relation, −0.006 when they
  are the relation's own message edges scored with the edge removed,
  the deployable version), and leave one untested (rule-body evidence
  mined from the example facts). Nothing here is ready for a GPU.
* **What deserves a GPU:** no training run. Two evaluations of existing
  one-pass checkpoints (config `incite_phase1.yaml`): the round-wise
  swap `output/incite-combo/soups/part_late100_mx15.pth` first, the
  soup `soup_L1_MX15_a50.pth` second, on the full stratified dev suite
  and, for one within 0.002 of MX15 there, on the 41 graphs, with the
  expectation stated in advance that they land near MX15's level and do
  not clear the recipe rule's +0.003 over it. Everything else here is a
  CPU-sized next step (section 9).

## 1. Setting and protocol

* Splits: `diagnostics/dev_eval.py::carve_inductive` on the train
  triples of YAGO310, CoDExSmall, CoDExLarge, Hetionet, ConceptNet100k,
  DBpedia100k, AristoV4, WDsinger (2,000 nodes, 4,800 message edges,
  1,200 queries; CoDExSmall capped at half its size), carve seeds 1024
  (`#0`) and 1025 (`#1`). The `#0` carves are the DESIGN half (every
  parameter is fitted there), the `#1` carves the HELD-OUT half (every
  headline number). Reverse fits (fit on `#1`, evaluate on `#0`) are
  reported where a method has parameters.
* Queries: the stratified sample of `dev_eval.py` (up to 300 per
  (direction, scenario) cell, seed 1024, the same queries for every
  model): 12,368 design queries, 12,506 held-out queries, 2,000
  candidates each. The number reported is the protocol's benchmark-
  weighted dev number (`inductive_v3`: eight cell MRRs combined with the
  41-graph suite's (direction, scenario) weights), the unweighted mean
  over the eight graphs; the four pooled cell MRRs are beside it.
* Validation: the per-model numbers recomputed from the score dumps
  reproduce `results/incite/dev/ind_L1.json` and `ind_MX1.json` cell for
  cell (sum of absolute differences over 16 graph numbers 0.0005, all
  rounding), and the container-side two-trunk evaluation reproduces the
  host-side ensemble number (0.2573 both ways).
* Cost of the probe: scoring all candidates of the 24,874 queries of the
  16 splits took 4 to 5 minutes per checkpoint at 6 CPU threads.
* Intervals: paired bootstrap over queries within every (direction,
  scenario) cell of every held-out split (1,000 to 2,000 resamples), 95
  percent, of the mean weighted difference over the eight graphs.

## 2. The propagation checkpoints and the oracle (held-out half)

| model | design | held-out | SQSA | SQUA | UQSA | UQUA | vs L1 (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 (4g + 10k decay) | 0.2487 | 0.2579 | 0.2869 | 0.0868 | 0.4409 | 0.1221 | |
| MX15 (15% synthetic) | 0.2433 | 0.2511 | 0.2761 | 0.0959 | 0.4115 | 0.1558 | −0.0067 [−0.0089, −0.0046] |
| MX1 (30% synthetic) | 0.2361 | 0.2433 | 0.2639 | 0.0978 | 0.3968 | 0.1633 | −0.0145 [−0.0172, −0.0118] |
| 20k start | 0.2434 | 0.2493 | 0.2674 | 0.0963 | 0.4162 | 0.1541 | −0.0086 [−0.0108, −0.0063] |
| oracle L1 (SA cells) / MX1 (UA cells) | 0.2552 | 0.2644 | 0.2869 | 0.0978 | 0.4409 | 0.1633 | +0.0065 [+0.0051, +0.0078]; +0.0210 over MX1 |
| oracle L1 / MX15 | 0.2547 | 0.2632 | 0.2869 | 0.0959 | 0.4409 | 0.1558 | +0.0054 [+0.0042, +0.0066]; +0.0121 over MX15 |

The trade is the test suite's (DEV_SUITE.md), but on these graphs the
seen-answer cost outweighs the unseen-answer gain, so L1 is the best
single model and every combination is read against L1 as well as
against MX. The oracle is the review's: per query, L1's reciprocal rank
when the true answer's half is seen, MX's when it is not.

## 3. Inference-time combinations of L1 with MX (two forward passes)

z-scores are per query over all candidates; `a_seen` is the per-
candidate answer-half indicator (the candidate has an incoming edge of
the query relation in the message graph; for head queries the mirror),
the quantity `scenario_features` computes in `incite/model.py`. Held-out
half; parameters fitted on the design half.

### L1 + MX1

| method | design | held-out | SQSA | SQUA | UQSA | UQUA | vs L1 | vs MX1 | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean logit 0.5 / 0.5 (`ScoreEnsemble`) | 0.2491 | 0.2573 | 0.2813 | 0.0947 | 0.4292 | 0.1520 | −0.0006 [−0.0023, +0.0010] | +0.0139 | |
| mean logit, weight fitted | 0.2501 | 0.2578 | 0.2825 | 0.0929 | 0.4333 | 0.1462 | −0.0001 | +0.0145 | w_L1 = 0.6 |
| mean z-score, weight fitted | 0.2500 | 0.2579 | 0.2826 | 0.0940 | 0.4315 | 0.1486 | −0.0000 | +0.0145 | w_L1 = 0.6 |
| reciprocal rank fusion, k = 60 (k = 1, 10 alike) | 0.2419 | 0.2492 | 0.2733 | 0.0923 | 0.4189 | 0.1348 | −0.0087 [−0.0106, −0.0069] | +0.0058 | |
| per-candidate switch on z-scores, plain | 0.2436 | 0.2525 | 0.2766 | 0.0905 | 0.4208 | 0.1581 | −0.0053 | +0.0092 | z_L1 if a_seen else z_MX1 |
| per-candidate switch, affine map fitted | 0.2464 | 0.2544 | 0.2808 | 0.0873 | 0.4280 | 0.1506 | −0.0035 [−0.0053, −0.0017] | +0.0111 | else 0.8 z_MX1 + 0.5 (reverse fit: 0.6, 1.0) |
| per-candidate switch on raw logits, offset fitted | 0.2447 | 0.2541 | 0.2930 | 0.0689 | 0.4329 | 0.1392 | −0.0037 | +0.0108 | else s_MX1 − 1.25 |
| per-scenario convex mixture of z-scores | 0.2500 | 0.2582 | 0.2770 | 0.1005 | 0.4342 | 0.1473 | +0.0004 [−0.0010, +0.0018] | +0.0149 | seen 1.0 z_L1; unseen 0.75 z_L1 + 0.25 z_MX1 + 0.25 (reverse: 0.75 / 0.75 / 0.0) |
| per-query switch on the query-half indicator | 0.2402 | 0.2483 | 0.2869 | 0.0868 | 0.3968 | 0.1633 | −0.0096 | +0.0049 | |
| logistic combiner, 12 features, listwise loss | 0.2485 | 0.2582 | 0.2865 | 0.0898 | 0.4378 | 0.1343 | +0.0004 [−0.0018, +0.0025] | +0.0149 | weights in `results_L1_MX1.json`; reverse fit 0.2486 vs L1 0.2487 |
| the same plus a 16-unit MLP | 0.2430 | 0.2476 | 0.2678 | 0.0856 | 0.4234 | 0.1447 | −0.0102 [−0.0125, −0.0080] | +0.0043 | overfits |

### L1 + MX15

| method | design | held-out | SQSA | SQUA | UQSA | UQUA | vs L1 | vs MX15 | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean logit 0.5 / 0.5 | 0.2491 | 0.2579 | 0.2841 | 0.0929 | 0.4317 | 0.1477 | +0.0000 [−0.0015, +0.0015] | +0.0067 [+0.0052, +0.0083] | |
| mean logit, weight fitted | 0.2500 | 0.2585 | 0.2848 | 0.0919 | 0.4350 | 0.1439 | +0.0007 | +0.0074 | w_L1 = 0.6 |
| mean z-score, weight fitted | 0.2504 | 0.2588 | 0.2853 | 0.0910 | 0.4367 | 0.1409 | +0.0009 | +0.0076 | w_L1 = 0.7 |
| reciprocal rank fusion, k = 60 | 0.2439 | 0.2528 | 0.2805 | 0.0909 | 0.4238 | 0.1321 | −0.0051 [−0.0065, −0.0037] | +0.0016 | |
| per-candidate switch, affine map fitted | 0.2460 | 0.2540 | 0.2794 | 0.0903 | 0.4260 | 0.1492 | −0.0039 [−0.0056, −0.0021] | +0.0028 | 0.8, 0.5 (reverse: 0.8, 0.0) |
| per-scenario convex mixture of z-scores | 0.2506 | 0.2583 | 0.2768 | 0.1018 | 0.4328 | 0.1487 | +0.0004 [−0.0011, +0.0020] | +0.0071 | 0.75 / 0.75 / +0.25 (reverse 0.75 / 0.75 / 0.0) |
| logistic combiner | 0.2470 | 0.2561 | 0.2838 | 0.0913 | 0.4326 | 0.1339 | −0.0018 [−0.0039, +0.0004] | +0.0049 | |
| three-way mean logit L1 + MX15 + MX1 | 0.2483 | 0.2559 | 0.2807 | 0.0957 | 0.4246 | 0.1522 | −0.0020 [−0.0036, +0.0001] | +0.0048 | |
| mean logit L1 + 20k start | 0.2493 | 0.2572 | 0.2832 | 0.0926 | 0.4317 | 0.1409 | −0.0007 [−0.0021, +0.0010] | | |

### Headroom recovered

| method | L1 + MX1: vs MX1 (oracle +0.0210) | vs L1 (oracle +0.0065) | L1 + MX15: vs MX15 (oracle +0.0121) | vs L1 (oracle +0.0054) |
| --- | --- | --- | --- | --- |
| mean logit 0.5 | 66% | −9% | 56% | 0% |
| mean, weight fitted (logit / z) | 69% | 0% | 61% / 63% | 12% / 17% |
| per-scenario mixture | 71% | 6% | 59% | 8% |
| logistic combiner | 71% | 5% | 41% | −33% |
| per-candidate switch (fitted) | 53% | −54% | 24% | −72% |
| reciprocal rank fusion | 28% | −134% | 13% | −95% |

Read against the weaker model of a pair, the plain mean "recovers" two
thirds of the oracle; read against the stronger model, nothing recovers
more than a sixth, and every interval includes zero. The recovery
against MX is the ensemble moving back to L1.

### Why the per-candidate switch cannot reach the oracle

The oracle knows which population the true answer is in. The switch
scores each population with the model that ranks it better, but the
true answer's rank is decided against the other population too: any
rule that lifts the unseen candidates (a positive offset, or MX1's
z-scores, which put unseen candidates higher than L1's) helps the 41
percent of queries whose answer is unseen and hurts the 59 percent whose
answer is seen. The fitted offsets (+0.5 / +1.0 on z-scores, −1.25 /
−1.5 on raw logits, unstable between the halves) are the point on that
one-dimensional trade-off the design half prefers, and the cells move
as the dose curve moves them (SQSA, UQSA down; UQUA up). A per-candidate
scalar correction is the lever SC1 already had.

### A per-query gate from observable query features

Features per query on the message graph: the relation's leave-one-out
unseen-answer prior p_UA(r) = share of r-edges whose tail has no other
incoming r-edge (an estimate of P(the held-out answer is unseen)), the
query-half indicator and its log count, the seen-candidate share and
log1p of the relation's edge count. A logistic regression fitted on the
design half predicts the true answer's status on the held-out half with
AUC 0.840 (p_UA alone 0.828; base rate 0.41 unseen); coefficients
[3.775, 0.339, 0.061, 3.494, −0.365], intercept −0.552
(`results_query.json`). Gating L1 / MX1 by it, held-out half:

| gate | held-out | SQSA | SQUA | UQSA | UQUA | vs L1 (95% CI) | vs MX1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| soft: (1 − p) z_L1 + p z_MX1 | 0.2563 | 0.2799 | 0.0956 | 0.4281 | 0.1529 | −0.0016 [−0.0033, +0.0004] | +0.0130 |
| hard, τ = 0.3 (TPR 0.86, FPR 0.36) | 0.2493 | 0.2714 | 0.0963 | 0.4116 | 0.1608 | −0.0086 [−0.0107, −0.0064] | +0.0060 |
| hard, τ = 0.5 (TPR 0.61, FPR 0.14) | 0.2545 | 0.2800 | 0.0944 | 0.4231 | 0.1506 | −0.0034 [−0.0052, −0.0015] | +0.0112 |
| hard, τ = 0.7 (TPR 0.47, FPR 0.07) | 0.2565 | 0.2826 | 0.0934 | 0.4294 | 0.1460 | −0.0014 [−0.0028, +0.0003] | +0.0132 |
| hard, τ = 0.8 (design-fitted) | 0.2571 | 0.2847 | 0.0915 | 0.4321 | 0.1372 | −0.0008 [−0.0021, +0.0006] | +0.0138 |
| p_UA alone, τ = 0.9 | 0.2573 | 0.2855 | 0.0896 | 0.4341 | 0.1356 | −0.0006 [−0.0017, +0.0006] | +0.0140 |
| confidence (larger top-2 logit margin) | 0.2523 | 0.2737 | 0.0932 | 0.4229 | 0.1523 | −0.0055 [−0.0074, −0.0035] | +0.0090 |
| oracle on z-scores | 0.2644 | 0.2869 | 0.0978 | 0.4409 | 0.1633 | +0.0065 [+0.0053, +0.0078] | +0.0210 |

(L1 / MX15: the same picture; soft gate 0.2578, −0.0001 vs L1.) Linear
cell arithmetic (cell means of L1 − MX1 times the gate's TPR / FPR)
predicts +0.0009 over L1 at τ = 0.5; the measurement is −0.0034: the
queries the gate sends to MX1 (sparse relations, high p_UA) are the
seen-answer queries on which L1's margin is far above its cell mean.
Under the benchmark's own cell effects (L1 − MX1 = +0.012 / +0.017 on
the seen-answer cells against MX1 − L1 = +0.020 / +0.068 on the unseen
ones, ind_e) the same arithmetic gives +0.004 over MX1 at τ = 0.3 to 0.5
(oracle +0.009); the dev measurement says to discount it by the same
0.004. Not recommended; the parameters are recorded because the gate
can be evaluated on the 41 graphs from the existing L1 and MX1 rank
dumps plus the inference graphs' edge counts, with no model run, if the
benchmark's own answer is wanted.

## 4. Weight-space soups (one forward pass)

θ = (1 − a) θ_L1 + a θ_MX, all parameters (`make_soups.py`); the partial
soups mix only the named parameter subset and keep L1's for the rest.
Held-out half:

| soup | design | held-out | SQSA | SQUA | UQSA | UQUA | vs L1 (95% CI) | vs the pair's mean-logit ensemble (2 passes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 / MX1, a = 0.25 | 0.2486 | 0.2574 | 0.2825 | 0.0912 | 0.4376 | 0.1345 | −0.0005 [−0.0016, +0.0009] | +0.0001 [−0.0012, +0.0015] |
| L1 / MX1, a = 0.5 | 0.2479 | 0.2545 | 0.2775 | 0.0947 | 0.4260 | 0.1490 | −0.0034 [−0.0052, −0.0013] | −0.0028 [−0.0039, −0.0017] |
| L1 / MX1, a = 0.75 | 0.2425 | 0.2498 | 0.2725 | 0.0955 | 0.4121 | 0.1605 | −0.0081 [−0.0104, −0.0057] | −0.0075 [−0.0091, −0.0061] |
| L1 / MX15, a = 0.25 | 0.2495 | 0.2579 | 0.2844 | 0.0909 | 0.4364 | 0.1350 | +0.0000 [−0.0011, +0.0012] | +0.0000 [−0.0011, +0.0011] |
| L1 / MX15, a = 0.5 | 0.2479 | 0.2571 | 0.2823 | 0.0942 | 0.4298 | 0.1457 | −0.0008 [−0.0023, +0.0009] | −0.0008 [−0.0017, +0.0001] |
| L1 / MX15, a = 0.75 | 0.2462 | 0.2542 | 0.2788 | 0.0955 | 0.4211 | 0.1528 | −0.0037 [−0.0055, −0.0015] | −0.0037 [−0.0049, −0.0025] |
| entity steps 0.5 L1 / MX1, rest L1 | 0.2485 | 0.2571 | 0.2838 | 0.0886 | 0.4347 | 0.1424 | −0.0007 [−0.0022, +0.0008] | −0.0001 [−0.0016, +0.0012] |
| entity steps from MX1, rest L1 | 0.2425 | 0.2539 | 0.2805 | 0.0908 | 0.4223 | 0.1554 | −0.0040 [−0.0062, −0.0015] | −0.0034 [−0.0051, −0.0016] |
| entity + relation steps 0.5, heads L1 | 0.2479 | 0.2554 | 0.2812 | 0.0923 | 0.4268 | 0.1473 | −0.0025 [−0.0042, −0.0006] | −0.0019 [−0.0030, −0.0008] |
| rounds 3-5 (entity + relation steps) from MX1, rounds 0-2 and heads L1 | 0.2498 | 0.2591 | 0.2901 | 0.0833 | 0.4394 | 0.1398 | +0.0013 [−0.0005, +0.0031] | +0.0019 [−0.0001, +0.0038] |
| rounds 3-5 from MX15, rounds 0-2 and heads L1 | 0.2503 | 0.2603 | 0.2913 | 0.0863 | 0.4390 | 0.1392 | +0.0024 [+0.0009, +0.0039] | +0.0024 [+0.0009, +0.0040] |
| rounds 0-2 from MX1, rounds 3-5 and heads L1 | 0.2351 | 0.2427 | 0.2483 | 0.1026 | 0.4129 | 0.1493 | −0.0151 [−0.0178, −0.0123] | −0.0145 [−0.0169, −0.0122] |
| relation steps 0.5 L1 / MX1, rest L1 | 0.2483 | 0.2553 | 0.2824 | 0.0883 | 0.4349 | 0.1293 | −0.0025 [−0.0040, −0.0011] | −0.0019 [−0.0036, −0.0003] |

Not scored within the budget: the three-way soup and the heads-only
swap. Reading:

* A soup's number is not the interpolation of its members': the L1 /
  MX15 soup at 0.5 is 0.2571 where the linear interpolation of the two
  numbers is 0.2545, and its cell profile is the two-pass ensemble's
  (SQSA 0.2823 vs 0.2841, UQUA 0.1457 vs 0.1477) at one pass, within
  0.001 of it. The L1 / MX1 soups fall short of their ensemble (−0.003
  at 0.5, −0.008 at 0.75; the checkpoints are 17 percent apart in
  parameter norm, MX15 15 percent), so the soup works for the near
  pair. At a = 0.25 both soups equal L1 exactly.
* The trade lives in the entity steps: mixing only them at 0.5
  reproduces the full soup's number and profile (0.2571; UQUA 0.1424),
  and taking MX1's entity steps outright moves the model most of the way
  to MX1's profile (UQUA 0.1554) at a cost of 0.004. The relation steps
  alone at 0.5 cost 0.0025 without any unseen-answer gain (UQUA 0.1293),
  and mixing them on top of the entity steps adds nothing (trunk 0.5:
  0.2554).
* The one family of partial soups with both a seen-answer AND an
  unseen-answer gain over L1 is the round-wise swap: L1's rounds 0 to 2
  and heads with the mixed model's entity and relation steps of rounds
  3 to 5. With MX15's late rounds: 0.2603, SQSA 0.2913 (L1 0.2869) and
  UQUA 0.1392 (L1 0.1221), +0.0024 [+0.0009, +0.0039] over L1 on the
  held-out half (+0.0016 on the design half, 0.2503 vs 0.2487) and
  +0.0024 [+0.0009, +0.0040] over the two-pass ensemble, at one pass;
  with MX1's late rounds 0.2591, +0.0013 [−0.0005, +0.0031]. The reverse
  assignment (MX1's rounds 0 to 2 under L1's late rounds) is −0.015: the
  early rounds must be L1's, and what the synthetic prior adds that
  survives the seen-answer cells sits in the late rounds' message
  functions. The MX15 swap is the only combination of the day with an
  interval above zero against the best single model, and the only one
  whose cells do not trade; nothing was fitted for it (a = 1 on a fixed
  parameter subset, chosen before the numbers were seen as one of seven
  partial soups, so the multiplicity is seven).
* On the dev suite no soup is above L1 beyond noise; on the benchmark,
  where MX15 is +0.006 over L1, a soup's profile would be expected near
  MX15's level (the interpolation plus the soup's own +0.003).

## 5. In-trunk fusion of the two propagation trunks (Luke's first list)

Premise tests with the two frozen checkpoints, CPU:

| idea | premise | test | result | inference cost | training it would need |
| --- | --- | --- | --- | --- | --- |
| scenario-conditioned rounds | the trunk's states separate evidence-rich and evidence-poor regimes | linear probes on the frozen final states, fitted on the design half, AUC on the held-out half (802k candidates / 12.5k queries) | candidate state x_t → answer-half indicator: AUC 1.000 (L1), 0.9999 (MX1); the model's own score as a probe: 0.937 / 0.865. [z_r, x_h] → query-half indicator: AUC 0.998 / 0.999. The signal is there; SC1's zero-init head did not move because the training graphs carry no such contrast (GATE_RESULT / SCENARIO_RESULT) | 1 pass, two scalars into the layers | a continuation with the two query indicators as inputs of the relation step's layer-norm scale, AND a training signal where the indicator matters (half-link masking or the generator's `unseen_answer_share`); without the second, SC1 again |
| two trunks, one readout | a head over both trunks' states, conditioned on the indicators, can pick per candidate | heads on [x_t, z_r of both trunks, both scores, 4 indicators], initialized at the mean-logit ensemble, fitted on the design half by a listwise loss, evaluated on the held-out half with all 2,000 candidates | first fit (target + top-32 of each model + 64 random negatives, 30 epochs): the MLP head −0.045 vs L1 on the full candidate set although +0.035 on its own subset (it reorders the long tail it never saw); second fit (target + 64 random negatives, 10 epochs, lr 1e-3, wd 1e-3): fusion MLP 0.2588 (+0.0009 [−0.0014, +0.0031] vs L1; +0.0015 vs the ensemble), late-linear 0.2591 (+0.0012 [−0.0009, +0.0038]), the score-only control 0.2586 (+0.0007). The states add nothing a head on the two scores lacks | 2 trunk passes (rounds could be shared: untested) | joint training of both trunks and the head; the frozen-trunk shortcut gives no signal |
| message-level mixture of experts | mixing the two checkpoints' message functions per round beats mixing everything | partial soups (section 4): entity steps only at 0.5 and 1.0, trunk only, rounds 3-5 from MX1 (rounds 0-2, relation steps only, heads only: see the table) | the entity steps carry the whole trade (entity-only 0.5 = the full soup: 0.2571, same profile); a hard per-round assignment (L1's rounds 0-2 and heads, the mixed model's rounds 3-5) is the best one-pass model of the day: with MX15's late rounds 0.2603, +0.0024 [+0.0009, +0.0039] over L1, with MX1's 0.2591, +0.0013 [−0.0005, +0.0031], BOTH SQSA and UQUA above L1's; the reverse assignment −0.015. So the premise "mix the message functions, not everything" holds at the round level, and a soft per-node gate has something to select between; the size is +0.002 | 1 pass, two message functions per round (about 1.5 to 2× the entity step) unless gated hard | joint training from the two checkpoints with a per-node gate |
| late fusion of final states, one linear head | a linear head on the concatenated states beats the ensemble | the `late_linear` head above | +0.0018 [−0.0002, +0.0037] vs the ensemble, +0.0012 vs L1 | 2 passes | none beyond the head; and the head at its best equals the ensemble |

Ranking of this list: (1) the message-level mixture, because the
partial soups show the trade lives in the entity steps and a hard
round-wise assignment already gives the day's best one-pass number
with both cells up; (2) scenario-conditioned rounds, whose premise test
is unambiguous but which needs the training-signal fix first (SC1's
post-mortem); (3) two trunks with a readout and (4) late fusion, which
cost a second trunk pass for +0.001.

## 6. Propagation vs in-context on the 41 test graphs (descriptive)

From the existing dumps only; no parameter fitted, no model run. Group
numbers are unweighted means over graphs.

Two caveats. KG-ICL's dumps carry its own entity, relation and query
ids, so they do not join per query with the other dumps; its cells are
computed in its own id space from its own inference graph
(`output/kgicl-data/<graph>/test/background.txt`,
`kgicl_cells.py`), the same definition. On the 12 GraIL graphs
(FB15k237 / WN18RR / NELL Inductive v1-v4) KG-ICL's test set is half
the suite's (205 of 411 queries on FB v1), so those rows are a
different sample. KGPFN's dumps (29 graphs; 11 of ind_e, 18 of ind_er)
join exactly.

### Per cell, by group

| ind_e (18 graphs) | SQSA | SQUA | UQSA | UQUA | MRR |
| --- | --- | --- | --- | --- | --- |
| KG-ICL | 0.4657 | 0.2350 | 0.5998 | 0.3686 | 0.4240 |
| TRIX | 0.4842 | 0.2932 | 0.6092 | 0.3428 | 0.4562 |
| ULTRA | 0.4581 | 0.2205 | 0.5871 | 0.2896 | 0.4158 |
| L1 | 0.4929 | 0.2856 | 0.6221 | 0.3187 | 0.4560 |
| MX15 | 0.4856 | 0.3020 | 0.6132 | 0.3756 | 0.4621 |
| MX1 | 0.4805 | 0.3051 | 0.6048 | 0.3864 | 0.4606 |

| ind_er (23 graphs) | SQSA | SQUA | UQSA | UQUA | MRR |
| --- | --- | --- | --- | --- | --- |
| KG-ICL | 0.3977 | 0.1540 | 0.5677 | 0.2302 | 0.3722 |
| TRIX | 0.3709 | 0.1696 | 0.5736 | 0.2361 | 0.3679 |
| ULTRA | 0.3798 | 0.1141 | 0.5447 | 0.1914 | 0.3421 |
| L1 | 0.4079 | 0.1724 | 0.5896 | 0.2032 | 0.3852 |
| MX15 | 0.4013 | 0.1892 | 0.5827 | 0.2518 | 0.3893 |
| MX1 | 0.3942 | 0.1876 | 0.5762 | 0.2684 | 0.3851 |

KGPFN on its 11 ind_e graphs (paired): SQSA 0.5417, SQUA 0.3213, UQSA
0.6237, UQUA 0.3044, MRR 0.4813 against MX15's 0.5234 / 0.3813 / 0.6157
/ 0.3352 / 0.4963 and L1's 0.5305 / 0.3644 / 0.6237 / 0.3008 / 0.4933;
on its 18 ind_er graphs 0.3889 / 0.1319 / 0.5251 / 0.1737 / 0.3415
against MX15's 0.3921 / 0.1935 / 0.5639 / 0.2030 / 0.3790. KGPFN is the
strongest model on SQSA of ind_e and the weakest on the unseen-answer
cells everywhere: the in-context family is not the evidence-poor
specialist; the synthetic prior is.

### Per family (KG-ICL, TRIX, L1, MX15)

| group | family | graphs | model | SQSA | SQUA | UQSA | UQUA | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ind_e | FB | 8 | KG-ICL | 0.3514 | 0.1604 | 0.4694 | 0.3669 | 0.3305 |
| ind_e | FB | 8 | TRIX | 0.3673 | 0.1497 | 0.4918 | 0.3114 | 0.3344 |
| ind_e | FB | 8 | MX15 | 0.3607 | 0.1532 | 0.4857 | 0.3530 | 0.3342 |
| ind_e | NELL | 4 | KG-ICL | 0.5562 | 0.2123 | 0.8716 | 0.4614 | 0.5414 |
| ind_e | NELL | 4 | MX15 | 0.6182 | 0.4292 | 0.8686 | 0.4456 | 0.6495 |
| ind_e | WN | 4 | KG-ICL | 0.7128 | 0.5050 | 0.6239 | 0.3518 | 0.5609 |
| ind_e | WN | 4 | MX15 | 0.7270 | 0.5974 | 0.6409 | 0.4120 | 0.6091 |
| ind_e | WK | 2 | KG-ICL | 0.2929 | 0.0392 | 0.5296 | 0.2703 | 0.2896 |
| ind_e | WK | 2 | MX15 | 0.3029 | 0.0522 | 0.5573 | 0.2879 | 0.3055 |
| ind_er | FB | 4 | KG-ICL | 0.4230 | 0.1416 | 0.5923 | 0.3264 | 0.3924 |
| ind_er | FB | 4 | MX15 | 0.4127 | 0.1511 | 0.6072 | 0.3844 | 0.3998 |
| ind_er | NELL | 5 | KG-ICL | 0.4229 | 0.1810 | 0.7132 | 0.2466 | 0.4235 |
| ind_er | NELL | 5 | TRIX | 0.3683 | 0.1641 | 0.7259 | 0.1633 | 0.3979 |
| ind_er | NELL | 5 | L1 | 0.4507 | 0.1163 | 0.7352 | 0.0829 | 0.4010 |
| ind_er | NELL | 5 | MX15 | 0.4398 | 0.1476 | 0.7314 | 0.1756 | 0.4152 |
| ind_er | WK | 12 | KG-ICL | 0.3688 | 0.1312 | 0.4790 | 0.1663 | 0.3279 |
| ind_er | WK | 12 | MX15 | 0.3722 | 0.1796 | 0.5031 | 0.2145 | 0.3519 |
| ind_er | other | 2 | KG-ICL | 0.5177 | 0.2484 | 0.6871 | 0.5293 | 0.4695 |
| ind_er | other | 2 | MX15 | 0.5125 | 0.4274 | 0.6391 | 0.5505 | 0.5277 |

(the full table with L1 and TRIX on every family:
`output/incite-combo/kgicl_cells.md`.) Where KG-ICL wins: ind_er NELL
(+0.008 over MX15, +0.026 over TRIX, and there on the unseen-answer
cells: SQUA 0.181 vs 0.148, UQUA 0.247 vs 0.176), ind_er FB on SQSA
(0.423 vs 0.413), ind_e FB on UQUA (0.367 vs 0.353). Per graph
(`kgicl_cells.md`) it is above MX15 on 13 of 41: the three HM graphs
(sparse, mostly unreachable answers), NLIngram 0 / 25 / 75 / 100,
WKIngram 25 / 75 / 100, FB15k237 v2 (+0.007), FBIngram:100 and
WikiTopicsMT1:health; above TRIX on 19 of 41, above L1 on 16. Where it
loses: ind_e NELL by 0.11 (SQUA 0.21 vs 0.43), WN by 0.05 (all four
graphs), the WikiTopics graphs, Metafam / FBNELL by 0.06. The wins are
the Ingram-style relation-inductive graphs and the sparse HM graphs,
the losses the entity-inductive GraIL graphs: a family pattern, not a
scenario pattern.

### Oracles

Cell-switch (per query by the TRUE cell, the review's construction;
needs only cell MRRs and shares):

| group | MX15 | KG-ICL | best cell of KG-ICL / MX15 per graph | KG-ICL on SA, MX15 on UA | MX15 on SA, KG-ICL on UA | best cell of KG-ICL / L1 | best cell of MX15 / L1 | best cell of KG-ICL / MX15 / L1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ind_e | 0.4621 | 0.4240 | 0.4693 | 0.4487 | 0.4381 | 0.4687 | 0.4670 | 0.4728 |
| ind_er | 0.3893 | 0.3722 | 0.4002 | 0.3863 | 0.3753 | 0.4000 | 0.3953 | 0.4041 |

Both fixed assignments are below MX15: there is no scenario in which
KG-ICL is the better model across graphs. The +0.007 / +0.011 of the
per-graph best-cell oracle is graph-specific complementarity (the
family table), and a fusion that wanted it would need a per-graph
selector, which nothing observable at inference provides yet.

Per-query max-reciprocal-rank oracles on the KGPFN-joinable graphs (a
loose bound: two models rarely fail on the same query):

| group | graphs | MX15 | KGPFN | KGPFN + MX15 | KGPFN + L1 | KGPFN + TRIX | MX15 + L1 | TRIX + MX15 | ULTRA + MX15 | KGPFN + MX15 + L1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ind_e | 11 | 0.4963 | 0.4813 | 0.5279 | 0.5256 | 0.5275 | 0.5109 | 0.5246 | 0.5230 | 0.5357 |
| ind_er | 18 | 0.3790 | 0.3415 | 0.4140 | 0.4090 | 0.4081 | 0.3961 | 0.4038 | 0.4100 | 0.4227 |

KGPFN + MX15 (+0.032 / +0.035) is twice MX15 + L1 (+0.015 / +0.017), but
TRIX + MX15 and even ULTRA + MX15 (+0.027 / +0.031) are nearly as large:
the per-query oracle mostly measures that two different models make
different mistakes, not that the in-context model is right where
propagation is wrong. KGPFN is above MX15 on 5 of its 29 graphs (WN v1,
v2, HM:1k, HM:3k, WikiTopicsMT2:org); per query it ranks the answer
strictly better than MX15 on 5 to 37 percent of a graph's queries and
strictly worse on 18 to 70 percent (`complementarity.md`).

### Rank fusion on the test dumps

Not computable: the dumps hold the true answer's rank per query, not the
candidate lists, and the fused rank of the answer depends on every other
candidate's ranks in both models. What is computable from the dumps is
above (per-cell numbers, cell-switch and per-query oracles). A
parameter-free RRF of KG-ICL and MX15 would need KG-ICL's candidate
scores on the 41 graphs (a KG-ICL evaluation run with a score dump) and
was not run.

### KG-ICL on the carved dev splits

Not run. Its evaluation needs its own dataset build (`scripts/
build_kgicl_datasets.py`: entity / relation dictionaries, background,
facts, filter files and the authors' case sampling) for every carved
split, then its 6-layer evaluation in the `kg-icl` container; with one
container at a time and the soup runs occupying it, this did not fit
the afternoon. It is the right next CPU step if a fitted propagation +
in-context combination is wanted off the test set (about half a day:
a converter from `carve_inductive`'s output to KG-ICL's directory
layout, then `evaluation.py` with a score dump).

## 7. Designs that fuse propagation with in-context relation evidence

PHASE22's same-encoder support readout (k example facts of the query
relation encoded by the trunk itself, cross-attention in the readout)
was functionally dead: the trunk's own representations carried whatever
same-graph support rows could add, and the optimizer routed the readout
to metric-neutrality. The designs below differ from it in where the
in-context evidence enters and in what it is. Premise tests
(`incontext_probe.py`, held-out half, CPU):

| test | question | result |
| --- | --- | --- |
| T1 | does the query relation's own examples carry calibration information the trunk lacks? Per relation, the offset β_r on the unseen-answer candidate population (score = z_L1 + β_r [answer half unseen]) chosen from the OTHER queries of the same relation in the same split (leave-one-out: oracle examples, since real queries with known answers are not available at test time) | +0.0014 [−0.0000, +0.0032] over L1 with at least 5 other queries, +0.0016 [+0.0002, +0.0031] with at least 10 (design 0.2505 vs 0.2487), against 0.0000 for the best global offset (β = 0.25). The chosen β_r spread over the whole grid (−1.5 for 31 percent of the held-out queries, +0.25 for 18 percent): relations differ in the seen / unseen calibration they want. The same per-relation offset on the L1 / MX1 switch: −0.0037, the switch's own loss dominating |
| T1' | the deployable version: the examples are the relation's own MESSAGE EDGES, up to 16 per direct relation, each scored as a query in both directions with that edge removed from the graph (`remove_easy_edges`, the training condition; `dump_pseudo.py`, about 1,700 pseudo-queries per split, 10 seconds per split on the CPU), β_r chosen to maximize the pseudo-queries' filtered MRR, applied to the real queries of the relation; no real-query label used | −0.0055 [−0.0079, −0.0027] on the design half, −0.0064 [−0.0090, −0.0040] on the held-out half (at least 8 pseudo-queries per relation; 95 percent of the queries covered; at least 16: −0.0053 / −0.0053). The chosen β_r pile up at the grid's ends (−1.5 for 45 percent of the relations, +1.5 for 21 percent): a pseudo-query's answer loses only its own edge and mostly stays a seen answer, whereas the real queries were carved out at 20 percent and their answers are unseen far more often, so the examples' scenario mix is not the queries' and the per-relation optimum is extreme and wrong. Untested fix: remove a random 20 percent of the relation's edges around each pseudo-query, or estimate β_r as a smooth function of the relation's LOO prior instead of a free per-relation grid |
| T2 | does a KG-ICL-style example-similarity feature add to the trunk? cosine between a candidate's relation signature (in / out degree per relation, the query relation's own entries removed) and the mean signature of the entities holding the answer half of the query relation in the message graph | alone it separates the answer from eligible negatives with AUC 0.851 and correlates 0.44 with z_L1; added to z_L1 with a fitted weight it gains 0.0000 (γ = 0 chosen), restricted to the unseen population +0.0001. The trunk already uses this |
| T3 | does the trunk's query-relation state already encode the relation's in-context statistics? ridge from z_r (L1, 32-d) to the LOO unseen-answer prior, the query-half count, the seen-candidate share and the edge count, fitted on the design half | R² on the held-out half 0.77 / 0.86 / 0.81 / 0.91 (q_seen 0.57). It does |

### Design A: per-relation in-context calibration of the score (premise holds with oracle examples, fails with real ones)

* What: for every relation r of the inference graph, take m of its own
  message edges (h, r, t) as pseudo-queries, score them with the trunk
  with that edge removed (the training condition, `remove_easy_edges`),
  and choose β_r, the offset between the unseen-answer and seen-answer
  candidate populations, to maximize the pseudo-queries' reciprocal
  rank; apply score(t) = s(t) + β_r [t's answer half unseen] to the real
  queries of r (head queries: the inverse relation, its own β).
* Premise tests: T1 (+0.0016 [+0.0002, +0.0031] with other real queries
  as the examples) says the relation-level calibration signal exists;
  T1' (−0.0064 [−0.0090, −0.0040] with the relation's own message edges
  as the examples, 16 per relation, the edge removed) says the
  deployable estimate of it is biased: the examples' scenario mix is
  not the queries'. A version whose examples are scored under the
  queries' held-out rate (a random 20 percent of the relation's edges
  removed around each pseudo-query) is the untested repair; with
  β_r free per relation the estimate also needs shrinkage (the chosen
  offsets sit at the grid's ends).
* Training: none (inference-time); or a learned version, a head that
  outputs β_r from the pseudo-queries' score statistics, trained on the
  pretraining graphs (a continuation), which would learn the bias
  correction the hand-made version lacks.
* Inference cost: R × m extra single-row trunk passes per graph (m = 16
  on a 50-relation graph: 800 passes; 10 seconds per carved split on
  the CPU here), one pass per query after that.
* Cheapest next test: `dump_pseudo.py` with the 20-percent removal and
  a shrunk β_r (CPU, an hour); only if that is at T1's +0.0016 is a
  learned head worth a continuation.

### Design B: prompt-graph relation evidence in the relation step (premise open)

* What: KG-ICL's prompt encoder (a small GNN over the union of the
  local subgraphs of k example facts of each relation, producing one
  vector per relation of the inference graph) feeds the TRIX-style
  relation step as its boundary condition: z_boundary[r] = query
  indicator + W · prompt(r) for every relation, not only the query one,
  so the relation states start from in-context evidence of what an
  r-edge looks like in THIS graph, and the alternating rounds refine
  them.
* Premise: the trunk's relation states are built from the relation
  graph's co-occurrence structure only. T3 says they already encode the
  relation's context statistics (R² up to 0.91), and T2 says a
  degree-signature similarity adds nothing; what neither tests is the
  example facts' PATH structure (the rule bodies connecting example
  heads to example tails), which is what a prompt graph encodes and the
  relation graph does not.
* Cheapest test of that premise (not run, CPU, about half a day): mine
  the length-2 relation paths connecting the heads and tails of r's
  message edges (an AnyBURL-lite over the message graph), score
  candidates by the number of mined bodies that connect h to t, and
  test the feature on z_L1 exactly as T2 tested the signature. If it
  adds nothing either, the trunk already propagates that evidence and
  design B has no premise; if it adds +0.003 or more on the held-out
  half, the prompt encoder is worth a continuation.
* Training: a continuation of the trunk with the prompt encoder,
  prompts sampled per relation per step (KG-ICL's recipe: k = 5 example
  facts, 2-hop subgraphs); the synthetic generator can supply
  relations whose example facts carry known rule bodies.
* Inference cost: one prompt encoding per relation per graph
  (amortized over the graph's queries, KG-ICL's cost) plus the
  unchanged trunk pass: about 1.2× on small graphs.

### Design C: example-pair analogy in the readout, own encoder (premise not supported by the cheap proxy)

* What: an encoder separate from the trunk embeds the (head, tail)
  pairs of k example facts of r (their local subgraphs); the readout
  attends from the candidate pair (h, t)'s embedding to the examples'
  and adds the similarity to the score.
* Premise: the answer is the candidate whose relation to h most
  resembles the examples' — analogy rather than propagation.
* Test: T2 is the degree-level proxy of that similarity and adds
  nothing to the trunk; PHASE22 is the same-encoder version and was
  dead. What is untested is the path-level analogy, which is design B's
  test with the feature attached per candidate pair. Only worth
  building if that test is positive.
* Training: joint; the second encoder from scratch. Inference cost: a
  second encoder over k subgraphs per relation (amortized) plus a
  cross-attention per candidate: about 1.3×.

Ranking: B (the one design whose premise is genuinely open and matches
the hypothesis; its test costs a CPU half-day) > A (the signal exists
at +0.0016 with oracle examples, the first deployable estimate of it is
−0.006; one more CPU hour decides it) > C (its cheap proxy is negative
and its predecessor died). None of the three has earned a GPU run
today.

## 8. Cost

| method | trunk passes per query | extra |
| --- | --- | --- |
| any score-level combination of two checkpoints (mean, switch, mixture, RRF, combiner, gate) | 2 | O(V) arithmetic |
| soup | 1 | none |
| two trunks + fusion head | 2 | a 134 → 32 → 1 MLP per candidate |
| design A (per-relation calibration) | 1, plus R × m pseudo-query passes per graph | none |
| design B (prompt-conditioned relation step) | 1, plus one prompt encoding per relation per graph | the prompt encoder |
| design C (analogy readout) | 1, plus the example encoder per relation per graph | a cross-attention per candidate |

## 9. Recommendation

1. **GPU evaluation (not training), two one-pass checkpoints, in this
   order:** the round-wise swap
   `output/incite-combo/soups/part_late100_mx15.pth` (L1's rounds 0-2
   and all heads, MX15's entity and relation steps of rounds 3-5; the
   day's best number, +0.0024 [+0.0009, +0.0039] over L1 on the held-out
   half, both cells up, nothing fitted), then the soup
   `output/incite-combo/soups/soup_L1_MX15_a50.pth` (θ = 0.5 θ_L1 +
   0.5 θ_MX15; the ensemble's profile at one pass). Both load with
   `load_members` under `configs/incite_phase1.yaml`. Full stratified
   dev suite first (`dev_eval.py`, the cost of one D0W reference each),
   then the 41 graphs for one that is within 0.002 of MX15 there.
   Expectation written down now: on the benchmark the swap keeps L1's
   seen-answer cells (SQSA / UQSA at or above L1's 0.4929 / 0.6221) and
   takes about a third of MX15's unseen-answer gain (UQUA between L1's
   0.3187 and MX15's 0.3756, nearer L1), which under the benchmark's
   weights puts it near MX15's 0.4621 / 0.3893, not +0.003 above it;
   the soup lands near MX15's level with SQSA / UQSA above and UQUA
   below MX15's. Neither is expected to clear the recipe rule's +0.003
   over MX15; both cost nothing at inference; the two-pass mean ensemble
   (w_L1 = 0.6) is the soup's upper bound and is not worth two passes.
   If the swap is at or above MX15 on the benchmark, the design worth a
   training run is the soft version (a per-node gate between the two
   late-round message functions, section 5), not more swaps.
2. **No fitted score-level combination of L1 with MX** deserves a run:
   the best is +0.0009 over L1 with an interval through zero, and the
   fitted parameters do not replicate between the halves.
3. **CPU next steps, in order:** design B's rule-body feature test
   (half a day); design A's repaired pseudo-query estimate (an hour);
   KG-ICL scored on the carved splits with a score dump (half a day),
   which is what any fitted propagation + in-context combination needs
   before it can be measured off the test set.
4. **The training run that the in-trunk tests point at** is not a
   fusion: it is the scenario-conditioned trunk WITH a training signal
   in which the indicator matters (half-link masking or the generator's
   unseen-answer share), since the states already carry the indicator
   with AUC 1.0 and the readout only failed for lack of contrast in the
   training graphs. That is a recipe modification for the queue after
   the seeds, not a combination of checkpoints.

## 10. What failed and why

* Every per-candidate rule (switch, affine switch, raw-logit switch,
  per-scenario mixture, logistic combiner) is a point on the seen-vs-
  unseen trade-off that the single models already span: none beats L1
  by more than +0.0009, and the fitted offsets do not agree between the
  halves (0.5 vs 1.0; 0.25 vs 0.0).
* Reciprocal rank fusion loses 0.005 to 0.009 to L1: ranks discard the
  size of the two models' agreement, and the disagreements sit where MX
  is wrong on these graphs.
* The 16-unit MLP combiner overfits 12k design queries by 0.008 to 0.010
  held-out although it starts at the ensemble; the linear combiner does
  not overfit and does not gain.
* The per-query gate at AUC 0.84 is not enough: gains and losses per
  query are of the same size, and the gated queries are adversely
  selected.
* The first fusion-head fit (hard negatives from both models' top-32 in
  the training subset) was −0.045 on the full candidate set while +0.035
  on its subset: a head trained on a candidate subset with a listwise
  loss reorders the long tail it never saw. Random negatives only, ten
  epochs, weight decay fixed it and left +0.001.
* The plain z-score switch per candidate is worse than the raw-logit
  switch on SQSA and better on UQUA: z-scoring per query removes the
  models' shared calibration, which the raw mean keeps.
* Design A's deployable estimate (pseudo-queries from the message
  edges, one edge removed) is −0.006 where its oracle-example version
  is +0.0016: the examples' answers are mostly still seen after one
  edge is removed, the real queries' answers are not, and the free
  per-relation offset is driven to the grid's ends by that mismatch.
* KG-ICL could not be joined per query (own id space); its complement-
  arity is read per cell in its own id space and per graph, and no
  rank fusion was possible from the dumps.

## 11. Files

* `output/incite-combo/dumps/`: per split `*.split.pt` (queries in tail
  form, strict masks, per-(relation, entity) in / out counts) and per
  model `*.<model>.pt` (scores over all candidates, ranks); soups and
  partial soups included.
* `output/incite-combo/feats/`: the two trunks' states on the candidate
  subsets; `heads/`, `heads2/`: the fitted fusion heads; `head_ranks*/`:
  their full-candidate ranks on the held-out half.
* `output/incite-combo/pseudo/`: the pseudo-query score dumps of design
  A (L1, 16 message edges per relation, both directions, edge removed).
* `output/incite-combo/results_L1_MX1.json`, `results_L1_MX15.json`,
  `results_query.json`, `results_gate_tau.json`, `results_three.json`,
  `results_soups_final.json`, `results_heads_fit_heads*.json`,
  `results_heads_eval_heads1.json`, `results_heads_eval_heads2.json`, `results_incontext_probe.json`,
  `results_pseudo.json`, `results_kgicl_cells.json`,
  `results_complementarity.json`; markdown dumps `kgicl_cells.md`,
  `complementarity.md`.
* `output/incite-combo/soups/*.pth`: the soup checkpoints (state dicts,
  loadable with `incite.run.load_members` under
  `configs/incite_phase1.yaml`).
* Scripts: `dump_scores.py` (container: score dumps), `combine.py`,
  `combine_query.py`, `tables.py`, `make_soups.py`, `soups_eval.py`,
  `soups_final.py`, `dump_features.py` / `fit_heads.py` /
  `eval_head.py` / `heads_report.py` (the fusion heads and probes),
  `incontext_probe.py` (T1-T3), `dump_pseudo.py` / `pseudo_eval.py`
  (T1', design A), `kgicl_cells.py`, `complementarity.py`.
