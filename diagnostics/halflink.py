"""Half-link scenarios (Gregucci et al., arXiv 2606.18001) on this suite,
joined with our rank dumps: where is the MRR lost?

For a test triple (h, r, t) in inference graph G:
  query half seen   (SQ)  <=>  some (h, r, x) in G      (else UQ)
  answer half seen  (SA)  <=>  some (y, r, t) in G      (else UA)
giving SQSA, SQUA, UQSA, UQUA for the tail direction; for the head
direction the roles swap: query half = (r, t), answer half = (h, r).

Per graph and per model this prints the share of queries in each scenario
and the MRR inside each, from a rank dump joined on (query_id, direction).
Group means are unweighted over graphs, the suite convention.

CONTAINER-ONLY (pinned TRIX loaders), CPU. Usage:
  python diagnostics/halflink.py --data_root /kgfm-src/data/roots/trix \
      --ranks trix=/kgfm-src/../sotaKGFMs/ranks/trix,incite-4g=/kgfm-src/ranks/incite-4g \
      --out /kgfm-src/results/incite/halflink.json
"""
import argparse
import json
import os
import sys

import torch

TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
sys.path.insert(0, os.path.join(TRIX_ROOT, "src"))

from easydict import EasyDict  # noqa: E402
from trix import util  # noqa: E402

import suite  # noqa: E402

SCEN = ("SQSA", "SQUA", "UQSA", "UQUA")


def load_graph(root, gid):
    g = suite.by_id(gid)
    cfg = {"class": g.dataset, "root": root}
    if g.version is not None:
        cfg["version"] = util.literal_eval(g.version)
    elif g.id == "Metafam":
        cfg["version"] = "Metafam"
    elif g.id == "FBNELL":
        cfg["version"] = "FBNELL_v1"
    dataset = util.build_dataset(EasyDict({"dataset": cfg}))
    return dataset[2]


def scenarios(data):
    """Scenario label per test triple, for tail and head queries."""
    num_direct = int(data.num_relations) // 2
    ei, et = data.edge_index, data.edge_type
    direct = et < num_direct
    src, dst, rel = ei[0, direct], ei[1, direct], et[direct]
    R = num_direct
    # keys of observed (entity, relation) incidences in the inference graph
    head_keys = torch.unique(src * R + rel)      # (h, r, .) exists
    tail_keys = torch.unique(dst * R + rel)      # (., r, t) exists
    h, t, r = data.target_edge_index[0], data.target_edge_index[1], data.target_edge_type
    r = torch.where(r >= R, r - R, r)
    hr_seen = torch.isin(h * R + r, head_keys)
    rt_seen = torch.isin(t * R + r, tail_keys)

    def label(q_seen, a_seen):
        out = []
        for q, a in zip(q_seen.tolist(), a_seen.tolist()):
            out.append(("SQ" if q else "UQ") + ("SA" if a else "UA"))
        return out
    return {"tail": label(hr_seen, rt_seen), "head": label(rt_seen, hr_seen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/kgfm-src/data/roots/trix")
    ap.add_argument("--ranks", default="", help="name=dir,name=dir,...")
    ap.add_argument("--groups", default="ind_e,ind_er")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels_out", default=None,
                    help="also write per-query scenario labels (per graph: "
                         "tail and head lists) for host-side joins")
    args = ap.parse_args()
    models = dict(kv.split("=", 1) for kv in args.ranks.split(",")) if args.ranks else {}
    labels = {}
    import pyarrow.parquet as pq

    results = {}
    for grp in args.groups.split(","):
        for gid in suite.ids(grp):
            try:
                data = load_graph(args.data_root, gid)
            except Exception as e:
                print("SKIP", gid, repr(e)[:120], flush=True)
                continue
            lab = scenarios(data)
            labels[gid] = lab
            n = len(lab["tail"])
            rec = {"group": grp, "queries": n, "share": {}, "models": {}}
            allq = lab["tail"] + lab["head"]
            for s in SCEN:
                rec["share"][s] = round(sum(1 for x in allq if x == s) / len(allq), 4)
            for name, d in models.items():
                p = os.path.join(d, gid.replace(":", "_") + ".parquet")
                if not os.path.exists(p):
                    continue
                tb = pq.read_table(p, columns=["query_id", "direction", "rank"])
                qid = tb.column("query_id").to_numpy()
                dirn = tb.column("direction").to_numpy(zero_copy_only=False)
                rk = tb.column("rank").to_numpy()
                per = {s: [] for s in SCEN}
                for q, direction, rank in zip(qid.tolist(), dirn.tolist(), rk.tolist()):
                    per[lab[direction][int(q)]].append(1.0 / float(rank))
                rec["models"][name] = {
                    s: (round(sum(v) / len(v), 4) if v else None) for s, v in per.items()}
                rec["models"][name]["mrr"] = round(float((1.0 / rk).mean()), 4)
            results[gid] = rec
            print(gid, json.dumps(rec["share"]), json.dumps(rec["models"]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print("written", args.out, flush=True)
    if args.labels_out:
        os.makedirs(os.path.dirname(args.labels_out) or ".", exist_ok=True)
        json.dump(labels, open(args.labels_out, "w"))
        print("written", args.labels_out, flush=True)


if __name__ == "__main__":
    main()
