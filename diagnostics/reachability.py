"""Ceiling analysis: how many test answers lie beyond the model's reach?

A query-conditioned NBFNet-style trunk with L rounds gives a candidate a
non-zero state only if a path of length <= L connects it to the query
entity in the inference graph (both edge directions count). A target
outside that ball ties with every other unreachable candidate at the
bottom, so its reciprocal rank is at most 1 / (#unreachable eligible
candidates + 1). This script measures, per graph and direction:

* the fraction of test queries whose target is unreachable within L hops;
* the same for L = 2, 4, 6, 8 (how much a deeper trunk would recover);
* the model's actual mean reciprocal rank on the unreachable subset, from
  a rank dump joined on (query_id, direction), to confirm the mechanism.

CONTAINER-ONLY (pinned TRIX loaders), CPU is enough: BFS is a sparse
matrix-vector product per hop. Usage:
  python diagnostics/reachability.py --data_root /kgfm-src/data/roots/trix \
      --ranks /kgfm-src/ranks/incite-4g --hops 2,4,6,8 --out results/incite/reachability.json
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
    return dataset[2]  # the test (inference) graph with its target edges


def reach_matrix(edge_index, num_nodes):
    """Boolean sparse adjacency (both directions already materialized)."""
    idx = edge_index
    val = torch.ones(idx.shape[1])
    return torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce()


def hops_to_targets(adj, starts, targets, max_hops):
    """For each (start, target) pair: the smallest hop count <= max_hops at
    which target is reached from start, else max_hops + 1. Batched BFS via
    sparse products on one-hot frontiers (num_nodes x batch)."""
    n = adj.shape[0]
    out = torch.full((len(starts),), max_hops + 1, dtype=torch.long)
    bs = 256
    for s in range(0, len(starts), bs):
        st, tg = starts[s:s + bs], targets[s:s + bs]
        b = len(st)
        visited = torch.zeros(n, b, dtype=torch.bool)
        visited[st, torch.arange(b)] = True
        frontier = visited.clone().float()
        found = torch.full((b,), max_hops + 1, dtype=torch.long)
        hit0 = visited[tg, torch.arange(b)]
        found[hit0] = 0
        for hop in range(1, max_hops + 1):
            nxt = torch.sparse.mm(adj, frontier) > 0
            nxt &= ~visited
            visited |= nxt
            hit = nxt[tg, torch.arange(b)] & (found > max_hops)
            found[hit] = hop
            frontier = nxt.float()
            if not bool(frontier.any()):
                break
        out[s:s + b] = found
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/kgfm-src/data/roots/trix")
    ap.add_argument("--ranks", default=None, help="rank dump dir to join")
    ap.add_argument("--hops", default="2,4,6,8")
    ap.add_argument("--groups", default="ind_e,ind_er")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    hops = [int(h) for h in args.hops.split(",")]
    max_hops = max(hops)

    results = {}
    for grp in args.groups.split(","):
        for gid in suite.ids(grp):
            try:
                data = load_graph(args.data_root, gid)
            except Exception as e:  # a graph missing from the root
                print("SKIP", gid, repr(e)[:120])
                continue
            n = int(data.num_nodes)
            adj = reach_matrix(data.edge_index, n)
            h, t = data.target_edge_index[0], data.target_edge_index[1]
            # tail queries start at h and look for t; head queries the reverse
            d_tail = hops_to_targets(adj, h, t, max_hops)
            d_head = hops_to_targets(adj, t, h, max_hops)
            rec = {"group": grp, "num_nodes": n, "num_edges": int(data.edge_index.shape[1]),
                   "queries": int(len(h))}
            for L in hops:
                rec["unreach@%d" % L] = round(float(((d_tail > L).float().mean()
                                                     + (d_head > L).float().mean()) / 2), 4)
            if args.ranks:
                p = os.path.join(args.ranks, gid.replace(":", "_") + ".parquet")
                if os.path.exists(p):
                    import pyarrow.parquet as pq
                    tb = pq.read_table(p, columns=["query_id", "direction", "rank"])
                    qid = tb.column("query_id").to_numpy()
                    dirn = tb.column("direction").to_numpy(zero_copy_only=False)
                    rk = tb.column("rank").to_numpy()
                    rr = {}
                    for name, dist in (("tail", d_tail), ("head", d_head)):
                        sel = dirn == name
                        r = torch.full((len(dist),), float("nan"))
                        r[torch.as_tensor(qid[sel])] = torch.as_tensor(rk[sel], dtype=torch.float)
                        far = dist > 6
                        rr[name] = {
                            "mrr_reachable6": round(float((1 / r[~far]).mean()), 4) if int((~far).sum()) else None,
                            "mrr_unreachable6": round(float((1 / r[far]).mean()), 4) if int(far.sum()) else None,
                            "n_unreachable6": int(far.sum()),
                        }
                    rec["model"] = rr
                    # what the group MRR would be if every unreachable query were solved
                    r_all = torch.as_tensor(rk, dtype=torch.float)
                    rec["mrr"] = round(float((1 / r_all).mean()), 4)
            results[gid] = rec
            print(gid, json.dumps({k: v for k, v in rec.items() if k != "model"}),
                  json.dumps(rec.get("model", {})))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)
        print("written", args.out)


if __name__ == "__main__":
    main()
