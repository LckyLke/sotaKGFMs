"""Paired bootstrap over graphs for the difference between two rank dumps.

Every group mean in this project is an unweighted mean over graphs, so
the natural unit of resampling is the graph. For models A and B with
per-graph MRR a_g and b_g on the same graphs, this resamples the graphs
with replacement (default 20,000 times) and reports the mean difference
with a 95 percent interval and the fraction of resamples where A > B.
It says whether a margin survives graph-level variation. It says nothing
about seed variation: that needs seed repeats (single-seed dumps here).

CPU, numpy + the shared metrics module. Usage:
  python3 scripts/paired_bootstrap.py ranks/trix ../sotaKGFMs-incite/ranks/incite-4g
  python3 scripts/paired_bootstrap.py A B --metric hits@10 --groups ind_e,ind_er
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import metrics  # noqa: E402
import suite  # noqa: E402


def per_graph(rank_dir, metric):
    out = metrics.compute_dir(rank_dir, dtype="float64")
    return {gid: row[metric] for gid, row in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default="mrr")
    ap.add_argument("--groups", default="ind_e,ind_er")
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1024)
    args = ap.parse_args()

    pa, pb = per_graph(args.a, args.metric), per_graph(args.b, args.metric)
    rng = np.random.default_rng(args.seed)
    na, nb = os.path.basename(args.a.rstrip("/")), os.path.basename(args.b.rstrip("/"))
    print("A = %s, B = %s, metric = %s, %d resamples" % (na, nb, args.metric, args.resamples))
    for grp in args.groups.split(","):
        gids = [g for g in suite.ids(grp) if g in pa and g in pb]
        if not gids:
            print("%s: no common graphs" % grp)
            continue
        a = np.array([pa[g] for g in gids])
        b = np.array([pb[g] for g in gids])
        d = a - b
        idx = rng.integers(0, len(gids), size=(args.resamples, len(gids)))
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print("%-7s n=%2d  A %.4f  B %.4f  A-B %+.4f  95%% CI [%+.4f, %+.4f]  P(A>B) %.3f  "
              "graphs A wins %d/%d" % (grp, len(gids), a.mean(), b.mean(), d.mean(), lo, hi,
                                       float((boot > 0).mean()), int((d > 0).sum()), len(gids)))


if __name__ == "__main__":
    main()
