# The dev suite: D0 under the uniform protocol, and why it is stratified now (2026-09-03)

The independent review asked for a dev suite disjoint from the 41 test
graphs for every lever decision. Plan v15 built it from the VALID splits
of eight transductive graphs outside the pretraining diet and the test
suite (YAGO310, CoDExSmall, CoDExLarge, Hetionet, ConceptNet100k,
DBpedia100k, AristoV4, WDsinger; NELL23k left out as a NELL995
derivative), `diagnostics/dev_eval.py`, 1,000 valid triples per graph,
both query directions, filtered MRR, the unweighted mean over graphs.

## D0: the four references (22:48, uniform protocol)

| graph | L1 | MX1 | G1 | L2 |
| --- | --- | --- | --- | --- |
| YAGO310 | 0.4953 | 0.4894 | 0.4876 | 0.4756 |
| CoDExSmall | 0.4063 | 0.4045 | 0.4070 | 0.4082 |
| CoDExLarge | 0.3017 | 0.3030 | 0.3048 | 0.3147 |
| Hetionet | 0.1895 | 0.1778 | 0.1886 | 0.2017 |
| ConceptNet100k | 0.2282 | 0.2049 | 0.2298 | 0.2199 |
| DBpedia100k | 0.4043 | 0.4014 | 0.3952 | 0.4320 |
| AristoV4 | 0.2134 | 0.2001 | 0.2161 | 0.2027 |
| WDsinger | 0.4876 | 0.4781 | 0.4902 | 0.4711 |
| mean | 0.3408 | 0.3324 | 0.3399 | 0.3407 |

L1 = the 4-graph backbone plus 10k decay; MX1 = L1's schedule with 30
percent synthetic rules-prior steps (the best test model); G1 = L1 plus
the unary channel; L2 = the 3-graph floor plus 10k decay. Files:
`results/incite/dev/{L1,MX1,G1,L2}.uniform.json` (after stage D0W;
`*.json` before it).

MX1, which beats L1 on the 41-graph test suite (ind_e +0.0046 with an
interval above zero), is 0.0084 BELOW L1 here and loses on seven of the
eight graphs. The fourth diet graph adds nothing here (L2 equals L1).

## Why: the scenario mix

The half-link scenarios (Gregucci et al.) split queries by whether the
query half (h, r) and the answer half (r, t) occur in the inference
graph. On the test suite MX1 trades the seen-answer cells for the
unseen-answer cells:

| group | cell | L1 | MX1 | MX1 − L1 |
| --- | --- | --- | --- | --- |
| ind_e | SQSA | 0.4929 | 0.4805 | −0.012 |
| ind_e | SQUA | 0.2856 | 0.3051 | +0.020 |
| ind_e | UQSA | 0.6221 | 0.6048 | −0.017 |
| ind_e | UQUA | 0.3187 | 0.3864 | +0.068 |
| ind_er | SQSA | 0.4079 | 0.3942 | −0.014 |
| ind_er | SQUA | 0.1724 | 0.1876 | +0.015 |
| ind_er | UQSA | 0.5896 | 0.5762 | −0.013 |
| ind_er | UQUA | 0.2032 | 0.2684 | +0.065 |

The test suite is 61 (ind_e) / 66 (ind_er) percent seen-answer queries
(SQSA + UQSA). A transductive valid split is 80 to 90 percent (CoDExSmall:
SQSA 0.80, SQUA 0.10, UQSA 0.10, UQUA 0.008; WDsinger, sparse and
tail-only, is the exception at 0.27 / 0.14 / 0.42 / 0.17). A uniform
sample of it measures the cells where the prior costs and barely sees
the cells where it gains: −0.013 on 85 percent of the queries and +0.03
on the rest is about −0.009, the observed gap. A proxy that would have
rejected the best model cannot choose its successor.

## The stratified protocol (`protocol: stratified_v2`)

`diagnostics/dev_eval.py` now labels every valid query with its scenario
(`diagnostics/halflink.py`, the same definition as the test-suite
tables), samples up to 300 queries per (direction, cell) with a fixed
seed (the same queries for every model, so graphs are paired), scores
them, and combines the four cell MRRs with the benchmark's scenario
weights: SQSA 0.353, SQUA 0.284, UQSA 0.284, UQUA 0.079 (the mean over
the 41 test graphs of the per-graph shares, averaged over the two
groups; from `results/incite/halflink_labels.json`). A cell under 30
queries is left out and the weights are renormalized. The graph's own
mix is reported beside (`graphs_natural`, `mean_natural`: the quantity
the uniform sample measured), and the cell MRRs with their counts are in
the file. The plan's recipe decision (v16) refuses any file that is not
`stratified_v2`, so a uniform number is never compared with a stratified
one; stage D0W recomputes the four references.

What it does not fix: the dev graphs are transductive, so their
unseen-answer cells are small (tens of queries on CoDExSmall) and noisy;
the weighted number's resolution is about ±0.004 paired over the eight
graphs, the same order as the uniform number's. The test-set bootstrap
remains the second gate of the recipe rule.
