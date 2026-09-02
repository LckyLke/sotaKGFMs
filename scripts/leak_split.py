#!/usr/bin/env python3
"""Leak split and paired statistics over committed rank dumps (H4 of
docs/KGFM_PLAN.md, done with what the harness already has).

A test graph is LEAKED for a model when its family (shared/suite.py) is
among the model's pretraining families, CLEAN otherwise. 3-graph models
(FB15k237, WN18RR, CoDExMedium) leak FB and WN: 16 of the 41 inductive
graphs; 4-graph models (plus NELL995) also leak NELL: 25 of 41. CoDEx has
no inductive graph. Group means are unweighted over graphs, as everywhere.

The second table compares pairs of models on the graphs clean for BOTH,
with the paired graph bootstrap (20,000 resamples), a two-sided Wilcoxon
signed-rank test, the sign test at a 0.005 tie band, and Holm-adjusted
p-values within each family of comparisons (one model against several
baselines). Single seed throughout: the intervals cover graph variation,
not seed variation.

    python3 scripts/leak_split.py > results/leak_split.md
"""
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))
import metrics  # noqa: E402
import suite  # noqa: E402

THREE = {"FB", "WN", "CoDEx"}
FOUR = THREE | {"NELL"}
#: rank dir -> (label, pretraining families). KG-ICL is left out: its
#: pretraining mix is its own and the matched convention differs.
MODELS = [
    ("ranks/ultra", "ULTRA 3g", THREE),
    ("ranks/ultra-4g", "ULTRA 4g", FOUR),
    ("ranks/motif", "MOTIF", THREE),
    ("ranks/trix", "TRIX", THREE),
    ("ranks/semma", "SEMMA", THREE),
    ("ranks/flock", "FLOCK", THREE),
    ("ranks/kgpfn", "KGPFN", THREE),
    ("ranks/incite", "INCITE floor 3g", THREE),
    ("ranks/incite-4g-last", "INCITE 4g last", FOUR),
]
#: comparison families for Holm: (model, [baselines])
COMPARISONS = [
    ("ranks/incite-4g-last", ["ranks/ultra-4g", "ranks/trix", "ranks/flock"]),
    ("ranks/incite", ["ranks/ultra", "ranks/motif", "ranks/semma", "ranks/trix", "ranks/flock"]),
    ("ranks/trix", ["ranks/ultra"]),
]
INDUCTIVE = [g for g in suite.ids() if suite.by_id(g).group != "transductive"]


def per_graph(rank_dir):
    out = metrics.compute_dir(os.path.join(ROOT, rank_dir), dtype="float64")
    return {g: out[g]["mrr"] for g in INDUCTIVE if g in out}


def mean_over(vals, gids):
    got = [vals[g] for g in gids if g in vals]
    return (float(np.mean(got)), len(got)) if got else (float("nan"), 0)


label = {d: l for d, l, _ in MODELS}
fams = {d: f for d, _, f in MODELS}
per = {d: per_graph(d) for d, _, _ in MODELS}

print("## Leak split (41 inductive graphs, test splits, seed 1024, MRR)\n")
print("Leaked: the graph's family is in the model's pretraining set. Means are over the graphs the model has.\n")
print("| model | pretraining families | leaked ind_e | clean ind_e | leaked ind_er | clean ind_er | leaked all | clean all |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for d, l, f in MODELS:
    cells = []
    for grp in ("ind_e", "ind_er", None):
        gids = [g for g in INDUCTIVE if grp is None or suite.by_id(g).group == grp]
        leaked = [g for g in gids if suite.by_id(g).family in f]
        clean = [g for g in gids if suite.by_id(g).family not in f]
        for sub in (leaked, clean):
            m, n = mean_over(per[d], sub)
            cells.append("%.4f (%d)" % (m, n) if n else "-")
    print("| %s | %s | %s |" % (l, ", ".join(sorted(f)), " | ".join(cells)))

print("\n## Paired comparisons on graphs clean for both models\n")
print("| model | baseline | clean graphs | mean delta | bootstrap 95% | wins/ties/losses | Wilcoxon p | Holm p |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
rng = np.random.default_rng(1024)
for model, baselines in COMPARISONS:
    rows = []
    for base in baselines:
        f = fams[model] | fams[base]
        gids = [g for g in INDUCTIVE if suite.by_id(g).family not in f
                and g in per[model] and g in per[base]]
        d = np.array([per[model][g] - per[base][g] for g in gids])
        if len(d) < 5:
            continue
        bs = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        wins, losses = int((d > 0.005).sum()), int((d < -0.005).sum())
        try:
            p = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        rows.append([base, len(d), d.mean(), lo, hi, wins, len(d) - wins - losses, losses, p])
    # Holm within this family of comparisons
    ps = np.array([r[-1] for r in rows])
    order = np.argsort(ps)
    holm = np.empty_like(ps)
    m = len(ps)
    running = 0.0
    for rank_i, idx in enumerate(order):
        adj = min(1.0, (m - rank_i) * ps[idx])
        running = max(running, adj)
        holm[idx] = running
    for r, hp in zip(rows, holm):
        print("| %s | %s | %d | %+.4f | [%+.4f, %+.4f] | %d/%d/%d | %.4f | %.4f |" % (
            label[model], label[r[0]], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], hp))
print("\nFamilies: FB 12 graphs, WN 4, NELL 9, WK 14, other 2 (Metafam, FBNELL). "
      "Per-graph values from shared/metrics.py over the committed parquets; "
      "FLOCK lacks FBIngram:25 (leaked for every model here), KGPFN has 25 of 41 graphs.")
