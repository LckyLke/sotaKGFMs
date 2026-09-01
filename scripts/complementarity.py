"""Mechanism complementarity across models, from identical per-query ranks.

Joins the rank dumps of several models on (dataset, query id, direction) --
the harness guarantees identical query sets -- and reports: per-model MRR,
pairwise complementarity (how often A succeeds where B fails), the oracle
upper bound (best rank per query across models), and two trivial fusions
(min-rank, mean-reciprocal-rank). Fusions are ENSEMBLES and must always be
labeled as such on any table. Also maps the KGPFN-vs-TRIX delta per graph
against the graph's relation count -- the "where does in-context win" map.

CPU, pandas only. Usage:
  python3 scripts/complementarity.py [--models trix,incite,incite-4g,kgpfn]
Rank dirs resolve as ranks/<name>/ under this repo, except incite variants,
which resolve under ../sotaKGFMs-incite/ranks/. Writes
results/complementarity.md.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import suite  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INC = os.path.join(HERE, "..", "sotaKGFMs-incite")


def rank_dir(model):
    base = INC if model.startswith("incite") else HERE
    return os.path.join(base, "ranks", model)


def load(model, gid):
    g = suite.by_id(gid).id.replace(":", "_")
    p = os.path.join(rank_dir(model), g + ".parquet")
    if not os.path.exists(p):
        return None  # a model may lack a graph (FLOCK 40/41, KGPFN partial)
    df = pd.read_parquet(p, columns=["query_id", "direction", "rank"])
    df = df.rename(columns={"rank": model})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="trix,incite,incite-4g,flock")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "complementarity.md"))
    args = ap.parse_args()
    models = args.models.split(",")

    rows, per_graph = [], []
    for grp in ("ind_e", "ind_er"):
        for gid in suite.ids(grp):
            dfs = [load(m, gid) for m in models]
            if any(d is None for d in dfs):
                continue  # only graphs every model has enter the join
            j = dfs[0]
            for d in dfs[1:]:
                j = j.merge(d, on=["query_id", "direction"], validate="1:1")
            j["_grp"], j["_gid"] = grp, gid
            rows.append(j)
            rec = {"group": grp, "graph": gid,
                   "num_relations": None}
            for m in models:
                rec[m] = (1.0 / j[m]).mean()
            per_graph.append(rec)
    big = pd.concat(rows, ignore_index=True)

    lines = ["# Mechanism complementarity (%s)" % ", ".join(models), ""]
    lines.append("Queries joined 1:1 on (dataset, query id, direction): %d over %d graphs "
                 "(graphs any model lacks are dropped)" % (len(big), big["_gid"].nunique()))
    lines.append("")
    lines.append("## Group MRR (per-graph unweighted mean)")
    pg = pd.DataFrame(per_graph)
    for grp in ("ind_e", "ind_er"):
        sub = pg[pg.group == grp]
        cells = " | ".join("%s %.4f" % (m, sub[m].mean()) for m in models)
        lines.append("- %s: %s" % (grp, cells))

    lines.append("")
    lines.append("## Pairwise complementarity (hit@10 level)")
    for a in models:
        for b in models:
            if a >= b:
                continue
            a10, b10 = big[a] <= 10, big[b] <= 10
            only_a = (a10 & ~b10).mean()
            only_b = (b10 & ~a10).mean()
            lines.append("- %s vs %s: only-%s %.3f, only-%s %.3f, both %.3f"
                         % (a, b, a, only_a, b, only_b, (a10 & b10).mean()))

    lines.append("")
    lines.append("## Oracle and trivial fusions (ENSEMBLES -- label them so)")
    ranks = big[models]
    oracle = ranks.min(axis=1)
    minfuse = oracle  # min-rank fusion IS the per-query best here
    mrr_fuse = (1.0 / ranks).mean(axis=1)
    # per-graph means to keep the suite convention
    big["_oracle"], big["_mrrf"] = 1.0 / oracle, mrr_fuse
    for grp in ("ind_e", "ind_er"):
        sub = big[big._grp == grp]
        o = sub.groupby("_gid")["_oracle"].mean().mean()
        f = sub.groupby("_gid")["_mrrf"].mean().mean()
        lines.append("- %s: oracle(min-rank) %.4f, mean-RR fusion %.4f" % (grp, o, f))

    lines.append("")
    lines.append("## Where does KGPFN win vs TRIX (per graph, MRR delta)")
    if "kgpfn" in models and "trix" in models:
        pg["delta"] = pg["kgpfn"] - pg["trix"]
        top = pg.sort_values("delta", ascending=False)
        for _, r in top.head(8).iterrows():
            lines.append("- +%.4f %s (%s)" % (r.delta, r.graph, r.group))
        lines.append("- ...")
        for _, r in top.tail(4).iterrows():
            lines.append("- %.4f %s (%s)" % (r.delta, r.graph, r.group))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
