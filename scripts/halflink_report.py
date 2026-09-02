"""Per-scenario MRR table for any rank dumps, on the host.

Joins rank parquets with the per-query half-link labels that
diagnostics/halflink.py --labels_out wrote (incite branch,
results/incite/halflink_labels.json). Scenarios follow Gregucci et al.
(arXiv 2606.18001): SQSA / SQUA / UQSA / UQUA by whether the query half
(h, r, .) and the answer half (., r, t) have a witness in the inference
graph. Group numbers are unweighted means over graphs.

  python3 scripts/halflink_report.py --labels ../sotaKGFMs-incite/results/incite/halflink_labels.json \
      trix=ranks/trix flock=ranks/flock incite-4g=../sotaKGFMs-incite/ranks/incite-4g
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
import suite  # noqa: E402

SCEN = ("SQSA", "SQUA", "UQSA", "UQUA")


def per_scenario(rank_dir, gid, labels):
    p = os.path.join(rank_dir, gid.replace(":", "_") + ".parquet")
    if not os.path.exists(p):
        return None
    tb = pq.read_table(p, columns=["query_id", "direction", "rank"])
    qid = tb.column("query_id").to_numpy()
    dirn = tb.column("direction").to_numpy(zero_copy_only=False)
    rr = 1.0 / tb.column("rank").to_numpy().astype(np.float64)
    n_lab = len(labels["tail"])
    if qid.max() >= n_lab or len(qid) != 2 * n_lab and len(qid) != n_lab:
        # a different query set (KG-ICL dumps half the rows under its own
        # ids): the labels do not apply, skip rather than mislabel
        print("skip %s on %s: %d rows, %d labels" % (rank_dir, gid, len(qid), n_lab),
              file=sys.stderr)
        return None
    lab = np.array([labels[d][int(q)] for q, d in zip(qid, dirn)])
    out = {"mrr": float(rr.mean())}
    for s in SCEN:
        m = lab == s
        out[s] = float(rr[m].mean()) if m.any() else None
        out["n_" + s] = int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--groups", default="ind_e,ind_er")
    ap.add_argument("--out", default=None)
    ap.add_argument("models", nargs="+", help="name=rank_dir")
    args = ap.parse_args()
    labels = json.load(open(args.labels))
    models = [m.split("=", 1) for m in args.models]
    lines = []
    for grp in args.groups.split(","):
        gids = [g for g in suite.ids(grp) if g in labels]
        lines.append("## %s (%d graphs)" % (grp, len(gids)))
        lines.append("| model | n | MRR | SQSA | SQUA | UQSA | UQUA |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, d in models:
            rows = [per_scenario(d, g, labels[g]) for g in gids]
            rows = [r for r in rows if r is not None]
            if not rows:
                continue
            cell = lambda k: np.mean([r[k] for r in rows if r[k] is not None])
            lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
                name, len(rows), cell("mrr"), cell("SQSA"), cell("SQUA"), cell("UQSA"), cell("UQUA")))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
