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
tables), samples up to 300 queries per (direction, scenario) cell with a
fixed seed (the same queries for every model, so graphs are paired),
scores them, and combines the eight cell MRRs with the benchmark's
(direction, scenario) weights: tail SQSA 0.177, SQUA 0.088, UQSA 0.196,
UQUA 0.039, and the head direction with SQUA and UQSA mirrored (the mean
over the 41 test graphs of the per-graph shares, averaged over the two
groups; from `results/incite/halflink_labels.json`; pooled over
directions SQSA 0.353, SQUA 0.284, UQSA 0.284, UQUA 0.079). A cell under
10 queries is left out and the weights are renormalized; a tail-only
graph (WDsinger) uses the tail weights. The graph's own mix is reported
beside (`graphs_natural`, `mean_natural`: the quantity the uniform
sample measured), and every cell's MRR and count is in the file, so any
other weighting is recomputable after the fact. The plan's recipe
decision (v16) refuses any file that is not `stratified_v2`, so a
uniform number is never compared with a stratified one; stage D0W
recomputes the four references.

Resolution (the verifier's estimate from the paired test dumps, where
the per-query spread of rank-reciprocal differences is about 0.13 per
cell): the standard error of a graph's paired weighted difference is
about 0.003 at 600 queries per pooled cell, and about 0.001 for the
eight-graph mean, so the rule's 0.003 is about three standard errors.
The unquantified part is the transfer of cell effects from transductive
valid splits to the 41 inductive test graphs. The test-set bootstrap
remains the second gate of the recipe rule.

## The expectation for D0W, written before it lands

Applying the benchmark weights to MX1's cell effects on the benchmark
itself gives MX1 − L1 of about +0.001 to +0.003 (the groups' own weights:
+0.0033 on ind_e, −0.0013 on ind_er; the averaged weights: +0.0016 /
+0.0002), against the actual +0.0046 / −0.0002. So under the stratified
protocol MX1 is expected to sit within ±0.002 of L1 on the dev suite,
not above it by the rule's margin, even with perfect transfer of the
cell effects. Consequences: MX1 itself would not clear the 0.003 bar
against L1, which is consistent with the review (the prior's gain is
small); a candidate must beat MX1, not L1; and if D0W puts MX1 more than
about 0.003 below L1 again, the proxy still disagrees with the benchmark
and the first gate is suspect.

## D0W: the references under the stratified protocol (02:30, 4 Sep)

| graph | L1 | MX1 | G1 | L2 | SC1 | MX1 − L1 |
| --- | --- | --- | --- | --- | --- | --- |
| YAGO310 | 0.4034 | 0.4102 | 0.4101 | 0.3930 | 0.4201 | +0.0068 |
| CoDExSmall | 0.3918 | 0.3892 | 0.3869 | 0.3894 | 0.3857 | −0.0026 |
| CoDExLarge | 0.3264 | 0.3237 | 0.3262 | 0.3301 | 0.3265 | −0.0027 |
| Hetionet | 0.1310 | 0.1246 | 0.1348 | 0.1323 | 0.1260 | −0.0064 |
| ConceptNet100k | 0.1765 | 0.1544 | 0.1760 | 0.1555 | 0.1512 | −0.0221 |
| DBpedia100k | 0.4059 | 0.3965 | 0.3990 | 0.4170 | 0.3991 | −0.0094 |
| AristoV4 | 0.1763 | 0.1668 | 0.1804 | 0.1635 | 0.1653 | −0.0095 |
| WDsinger | 0.4543 | 0.4459 | 0.4571 | 0.4422 | 0.4511 | −0.0084 |
| mean | 0.3082 | 0.3014 | 0.3088 | 0.3029 | 0.3031 | −0.0068 |
| graphs' own mix | 0.3378 | 0.3290 | 0.3375 | 0.3365 | 0.3318 | −0.0088 |

The expectation above FAILED: MX1 is 0.0068 below L1, on 7 of 8 graphs,
where the benchmark's cell effects predicted within ±0.002. The cells
(mean over graphs of the pooled cell MRR, MX1 − L1): SQSA −0.011, SQUA
+0.007, UQSA −0.017, UQUA +0.020. The seen-answer costs transfer from the
benchmark (there: −0.012 / −0.017); the unseen-answer gains do not (there:
+0.020 / +0.068). Re-weighting cannot fix a proxy whose cell EFFECTS
differ, only one whose cell SHARES differ.

## Where the benchmark gain lives: the diet's own families

MX1 − L1 on the 41 test graphs by the source of the graph (`suite.family`;
the pretraining diet is FB15k237, WN18RR, CoDExMedium, NELL995):

| group | family | graphs | MX1 − L1 | wins |
| --- | --- | --- | --- | --- |
| ind_e | FB (FB15k237Inductive v1-v4, HM) | 8 | +0.0066 | 5 |
| ind_e | NELL (NELLInductive v1-v4) | 4 | +0.0084 | 4 |
| ind_e | WK (ILPC) | 2 | −0.0001 | 1 |
| ind_e | WN (WN18RRInductive v1-v4) | 4 | −0.0009 | 1 |
| ind_er | FB (FBIngram) | 4 | +0.0047 | 4 |
| ind_er | NELL (NLIngram) | 5 | +0.0121 | 3 |
| ind_er | WK (WKIngram, WikiTopics) | 12 | −0.0031 | 2 |
| ind_er | other (Metafam, FBNELL) | 2 | −0.0231 | 0 |

The gain sits on the Freebase- and NELL-derived graphs (FB15k237Inductive
v1-v4: +0.010 to +0.023 each) and is absent on the Wikidata- and
WordNet-derived ones. Against the 20k start (before any decay), MX1 gains
on every family (+0.003 to +0.015) except Metafam, while L1's plain decay
gains on WN and WK and loses slightly on FB and NELL: the mix keeps and
extends the FB/NELL level that the plain continuation gives up. Nothing
in this pattern says "a general structural prior"; it says an interaction
with the diet's own graph families.

## A carved inductive dev suite says the same (pilot, CPU)

`dev_eval.py --split inductive` carves small sparse inference graphs out
of each dev graph's train triples (2,000 entities by random walks, 4,800
message edges, 1,200 queries, entities and relations re-indexed): the
benchmark's regime, on KGs outside the benchmark. Pilot at 100 queries
per cell, one split per graph, L1 versus MX1:

| split | L1 | MX1 | MX1 − L1 |
| --- | --- | --- | --- |
| YAGO310#0 | 0.4329 | 0.4275 | −0.0054 |
| CoDExLarge#0 | 0.2142 | 0.2054 | −0.0088 |
| ConceptNet100k#0 | 0.1788 | 0.1426 | −0.0362 |

Cells (MX1 − L1): SQSA −0.013 to −0.057, UQSA −0.024 to −0.075, SQUA
+0.003 to +0.013, UQUA +0.035 to +0.053. The sparse regime does not
rescue the prior outside the benchmark's families; the full comparison
(six checkpoints, two splits per graph, 300 per cell, protocol
`inductive_v3`, files `results/incite/dev/ind_*.json`) runs on the CPU
beside FMX and is added when it lands.

## What this means for the program

1. The claim "a synthetic rules prior teaches a KGFM something general"
   is not supported by any graph outside the benchmark: on eight
   held-out KGs the prior costs 0.007 to 0.009 (stratified, uniform), on
   carved sparse splits of them 0.005 to 0.036, always through the
   seen-answer cells. Its benchmark gain (+0.0046 ind_e, single seed) is
   concentrated on the diet's own Freebase and NELL families.
2. The recipe rule stands as recorded (MX1's recipe plus at most one
   modification, the stratified dev gate and the paired test gate). The
   dev gate will favour candidates that reduce the prior's cost (MX15)
   over ones that intensify it (MX45, MXS9); that is the conservative
   direction and it is disclosed here.
3. The decisive experiment is already queued: R0 (no mix) against R1
   (the recipe) from scratch at three seeds, on the benchmark AND on the
   dev suite. The paper's framing follows from it: if R1 − R0 is inside
   the seed noise on the benchmark, or negative on the dev suite, the
   prior is reported as what it is, a diet-family effect with a
   seen-answer cost, and the paper's contribution is the analysis (P1,
   the scenario tables, the family tables), not a recipe.
