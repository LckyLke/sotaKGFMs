"""The dev suite for lever decisions (2026-09-03, after the independent
review; stratified by half-link scenario since D0 landed): filtered MRR on
the VALID splits of transductive graphs that are neither in the pretraining
diet nor among the 41 test graphs, so a lever verdict read here never
touches the benchmark.

Why stratified. D0 (2026-09-03, uniform 1000-query samples) ranked MX1
0.0084 BELOW L1 on this suite while the 41-graph test suite ranks it
above: transductive valid queries are almost all "answer half seen"
(SQSA), the cells where the synthetic prior costs, and the benchmark is
only 61 / 66 percent such queries. A dev number that is to predict the
benchmark must weight the four scenarios the way the benchmark does. So
every graph now samples up to ``--per_cell`` queries per (direction,
scenario) cell, and the graph's dev number is the eight cell MRRs
combined with the benchmark's (direction, scenario) weights
(``BENCH_WEIGHTS``; a tail-only graph uses the tail weights, renormalized).
The graph's own mix is reported beside it (``natural``, the quantity the
uniform sample measured), and every cell's MRR and count is in the file,
so any other weighting is recomputable after the fact.

``--split inductive`` (analysis, not the plan's number) carves small
sparse inference graphs out of each dev graph's train triples
(``carve_inductive``: a random-walk entity sample, thinned to a target
density, a share of the triples held out as queries), the regime of the
41-graph suite, written with protocol ``inductive_v3`` so the plan's
decision never reads it. Its first use (2026-09-04, 02:45): MX1 is below
L1 there too, on every graph tried.

CONTAINER-ONLY (imports the pinned TRIX loaders through incite.pretrain).

    usage (from the prepared work tree, TRIX_ROOT set):
      python diagnostics/dev_eval.py --config configs/<cfg>.yaml \\
          --ckpt output/incite-pretrain-<run>/incite_last.pth \\
          [--graphs YAGO310,CoDExSmall,...] [--per_cell 300] \\
          [--batch_size 8] [--gpus "[0]"] [--out results/incite/dev/<name>.json]

Default graphs (DEV8T): YAGO310, CoDExSmall, CoDExLarge, Hetionet,
ConceptNet100k, DBpedia100k, AristoV4, WDsinger (NELL23k derives from a
diet graph and is left out). Tail-only graphs score tail queries only, as
the suite does. Output per graph: the weighted dev number (``graphs``),
the natural-mix number (``graphs_natural``), the eight (direction,
scenario) cell MRRs with their counts (``cells``), the same pooled over
directions (``cells4``) and the graph's scenario shares (``share``);
null for a graph that failed, with the error. ``mean`` is the unweighted
mean of ``graphs`` over the graphs that succeeded; ``complete`` says
whether every graph did. ``--val_samples`` is accepted and ignored (the
plan's call passes it; the stratified sample replaced it).
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml
from easydict import EasyDict
from torch_geometric.data import Data

TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
sys.path.insert(0, os.path.join(TRIX_ROOT, "src"))

from trix import tasks, util  # noqa: E402

import suite  # noqa: E402

from incite.pretrain import load_dev_graph  # noqa: E402
from incite.run import load_members  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halflink import SCEN, scenarios  # noqa: E402

# NELL23k is left out: it derives from NELL995, which is in the 4-graph diet.
DEV8T = ["YAGO310", "CoDExSmall", "CoDExLarge", "Hetionet", "ConceptNet100k",
         "DBpedia100k", "AristoV4", "WDsinger"]

#: The 41-graph suite's (direction, scenario) mix: the mean over graphs of
#: the per-graph shares, averaged over the two groups (ind_e tail
#: .163 / .109 / .179 / .049, ind_er tail .191 / .068 / .212 / .030; the head
#: direction mirrors SQUA and UQSA; from results/incite/halflink_labels.json,
#: 2026-09-03). Sums to 1; pooled over directions: SQSA .353, SQUA .284,
#: UQSA .284, UQUA .079.
BENCH_WEIGHTS = {"tail": {"SQSA": 0.177, "SQUA": 0.088, "UQSA": 0.196, "UQUA": 0.039},
                 "head": {"SQSA": 0.177, "SQUA": 0.196, "UQSA": 0.088, "UQUA": 0.039}}
#: A cell with fewer scored queries than this is left out of both
#: combinations (the weights are renormalized over the cells present).
#: Ten is low on purpose: the small cells are the UQUA ones, whose weight
#: is 0.039 per direction, so a noisy small cell moves the graph's number
#: by about 0.002 while dropping it would bias the number against the
#: levers that gain exactly there (the verifier's point, 2026-09-03).
MIN_CELL = 10
#: Protocol names written into the file (the plan's recipe decision reads
#: only files of the protocol it expects, never mixing two).
PROTOCOLS = {"inductive": "inductive_v3", "transductive": "stratified_v2"}


def carve_inductive(valid, num_nodes, edges_per_node, query_frac, seed,
                    walk_len=100):
    """An inductive dev split carved from a transductive graph's TRAIN
    triples, in the regime of the 41-graph suite (small, sparse inference
    graphs): a random-walk entity sample of up to ``num_nodes`` (at most
    half the graph), the induced direct triples thinned at random to
    ``edges_per_node`` per node, a ``query_frac`` share of them held out
    as the queries, the rest the message graph (with inverse edges, as the
    loaders build it). Entities and relations are re-indexed to the
    sample, so the model sees a graph it has never seen. Pure function of
    the seed."""
    R = int(valid.num_relations) // 2
    ei, et = valid.edge_index.cpu(), valid.edge_type.cpu()
    direct = et < R
    src, dst, rel = ei[0, direct], ei[1, direct], et[direct]
    n_all = int(valid.num_nodes)
    target = min(int(num_nodes), n_all // 2)
    gen = torch.Generator().manual_seed(int(seed))
    # undirected adjacency in CSR form for the walks
    u, v = torch.cat([src, dst]), torch.cat([dst, src])
    order = torch.argsort(u)
    u, v = u[order], v[order]
    ptr = torch.zeros(n_all + 1, dtype=torch.long)
    ptr[1:] = torch.bincount(u, minlength=n_all).cumsum(0)
    starts = (ptr[1:] > ptr[:-1]).nonzero().flatten()
    chosen = torch.zeros(n_all, dtype=torch.bool)
    count = 0
    while count < target:
        node = int(starts[int(torch.randint(starts.numel(), (1,), generator=gen))])
        for _ in range(walk_len):
            if not chosen[node]:
                chosen[node] = True
                count += 1
                if count >= target:
                    break
            a, b = int(ptr[node]), int(ptr[node + 1])
            if b <= a:
                break
            node = int(v[a + int(torch.randint(b - a, (1,), generator=gen))])
    nodes = chosen.nonzero().flatten()
    keep = chosen[src] & chosen[dst]
    s, d, r = src[keep], dst[keep], rel[keep]
    max_edges = int(edges_per_node * nodes.numel())
    if s.numel() > max_edges:
        sel = torch.randperm(s.numel(), generator=gen)[:max_edges]
        s, d, r = s[sel], d[sel], r[sel]
    node_map = torch.full((n_all,), -1, dtype=torch.long)
    node_map[nodes] = torch.arange(nodes.numel())
    rels = torch.unique(r)
    rel_map = torch.full((R,), -1, dtype=torch.long)
    rel_map[rels] = torch.arange(rels.numel())
    s, d, r = node_map[s], node_map[d], rel_map[r]
    n_rel = int(rels.numel())
    perm = torch.randperm(s.numel(), generator=gen)
    nq = max(1, int(round(float(query_frac) * s.numel())))
    q, m = perm[:nq], perm[nq:]
    msg_ei = torch.stack([torch.cat([s[m], d[m]]), torch.cat([d[m], s[m]])])
    msg_et = torch.cat([r[m], r[m] + n_rel])
    tgt_ei = torch.stack([s[q], d[q]])
    data = Data(edge_index=msg_ei, edge_type=msg_et, num_nodes=int(nodes.numel()),
                num_relations=2 * n_rel, target_edge_index=tgt_ei,
                target_edge_type=r[q])
    filt = Data(edge_index=torch.cat([msg_ei, tgt_ei], dim=1),
                edge_type=torch.cat([msg_et, r[q]]), num_nodes=int(nodes.numel()))
    stats = {"nodes": int(nodes.numel()), "relations": n_rel,
             "message_edges": int(m.numel()), "queries": int(nq),
             "induced_before_thinning": int(keep.sum())}
    return data, filt, stats


@torch.no_grad()
def rank_queries(model, graph, filter_graph, triplets, direction, batch_size):
    """Filtered ranks of the given triples, queried in one direction."""
    ranks = []
    for start in range(0, triplets.shape[0], batch_size):
        batch = triplets[start:start + batch_size]
        t_batch, h_batch = tasks.all_negative(graph, batch)
        pos_h_index, pos_t_index, _ = batch.t()
        t_mask, h_mask = tasks.strict_negative_mask(filter_graph, batch)
        if direction == "tail":
            pred = model(graph, t_batch)
            ranks.append(tasks.compute_ranking(pred, pos_t_index, t_mask))
        else:
            pred = model(graph, h_batch)
            ranks.append(tasks.compute_ranking(pred, pos_h_index, h_mask))
    return torch.cat(ranks).float()


def stratified_indices(labels, per_cell, gen):
    """Up to ``per_cell`` query indices per scenario, a seeded random subset
    when the cell is larger, every index when it is not."""
    lab = torch.tensor([SCEN.index(x) for x in labels])
    out = {}
    for ci, s in enumerate(SCEN):
        idx = (lab == ci).nonzero().flatten()
        if idx.numel() > per_cell:
            idx = idx[torch.randperm(idx.numel(), generator=gen)[:per_cell]]
        out[s] = idx
    return out


def combine(cells, weights):
    """Weighted mean of the (direction, scenario) cell MRRs over the cells
    with enough queries, the weights renormalized over them; None when no
    cell qualifies. ``cells`` and ``weights``: {direction: {scenario: ..}}."""
    present = [(d, s) for d in cells for s in SCEN
               if cells[d][s]["n"] >= MIN_CELL and weights.get(d, {}).get(s, 0) > 0]
    wsum = sum(weights[d][s] for d, s in present)
    if not present or wsum <= 0:
        return None
    return round(sum(weights[d][s] * cells[d][s]["mrr"] for d, s in present) / wsum, 4)


def eval_graph(model, valid, filt, per_cell, batch_size, seed, tail_only):
    labels = scenarios(valid)
    triplets = torch.cat(
        [valid.target_edge_index, valid.target_edge_type.unsqueeze(0)]).t()
    gen = torch.Generator().manual_seed(int(seed))
    directions = ["tail"] if tail_only else ["tail", "head"]
    total = sum(len(labels[d]) for d in directions)
    cells, share, rr = {}, {}, {}
    for direction in directions:
        lab = labels[direction]
        picked = stratified_indices(lab, per_cell, gen)
        cells[direction], share[direction] = {}, {}
        for s in SCEN:
            idx = picked[s]
            vals = []
            if idx.numel() > 0:
                ranks = rank_queries(model, valid, filt,
                                     triplets[idx.to(triplets.device)],
                                     direction, batch_size)
                vals = (1.0 / ranks).tolist()
            rr[(direction, s)] = vals
            cells[direction][s] = {"mrr": round(sum(vals) / len(vals), 4) if vals else None,
                                   "n": len(vals)}
            # the graph's own mix: this cell's share of ALL its queries
            # (both directions), so the natural combination estimates the
            # uniform-sample MRR
            share[direction][s] = round(sum(1 for x in lab if x == s) / total, 4)
    # the four scenarios pooled over directions, for reading
    cells4 = {}
    for s in SCEN:
        vals = sum((rr[(d, s)] for d in directions), [])
        cells4[s] = {"mrr": round(sum(vals) / len(vals), 4) if vals else None,
                     "n": len(vals)}
    return {"weighted": combine(cells, BENCH_WEIGHTS),
            "natural": combine(cells, share),
            "cells": cells, "cells4": cells4, "share": share,
            "queries": sum(c["n"] for c in cells4.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path, or comma list (ensemble)")
    ap.add_argument("--graphs", default=",".join(DEV8T))
    ap.add_argument("--split", default="transductive", choices=list(PROTOCOLS),
                    help="transductive (the plan's dev number): the graph's "
                         "own valid split; inductive (analysis): splits "
                         "carved from the graph's train triples in the "
                         "suite's regime, protocol inductive_v3")
    ap.add_argument("--samples", type=int, default=2,
                    help="inductive: carved splits per graph (seeds seed+i)")
    ap.add_argument("--sub_nodes", type=int, default=2000)
    ap.add_argument("--edges_per_node", type=float, default=3.0)
    ap.add_argument("--query_frac", type=float, default=0.2)
    ap.add_argument("--per_cell", type=int, default=300,
                    help="queries per (direction, scenario) cell")
    ap.add_argument("--val_samples", type=int, default=None,
                    help="ignored (legacy uniform sample size)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dev_root", default="/kgfm-src/data/roots/trix")
    ap.add_argument("--gpus", default="[0]")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = EasyDict(yaml.safe_load(open(args.config)))
    gpus = None if args.gpus in ("null", "None") else util.literal_eval(args.gpus)
    device = torch.device(gpus[0]) if gpus else torch.device("cpu")
    model, hashes = load_members(cfg, [p for p in args.ckpt.split(",") if p])
    model = model.to(device).eval()

    per, natural, cells, cells4, share, counts, failed = {}, {}, {}, {}, {}, {}, {}
    t0 = time.perf_counter()

    split_stats = {}

    def run(gid, retry):
        base, _, k = gid.partition("#")
        info = suite.by_id(base)
        valid, filt = load_dev_graph(args.dev_root, base)
        valid.num_relations = int(valid.num_relations)
        tail_only = bool(getattr(info, "tail_only", False))
        if args.split == "inductive":
            valid, filt, split_stats[gid] = carve_inductive(
                valid, args.sub_nodes, args.edges_per_node, args.query_frac,
                args.seed + int(k or 0))
            tail_only = False  # our construction: both directions are queries
        r = eval_graph(model, valid.to(device), filt.to(device), args.per_cell,
                       args.batch_size, args.seed, tail_only)
        per[gid], natural[gid] = r["weighted"], r["natural"]
        cells[gid], cells4[gid] = r["cells"], r["cells4"]
        share[gid], counts[gid] = r["share"], r["queries"]
        failed.pop(gid, None)
        line = {"graph": gid, "dev": r["weighted"], "natural": r["natural"],
                "cells4": {s: [c["mrr"], c["n"]] for s, c in r["cells4"].items()},
                "queries": r["queries"]}
        if gid in split_stats:
            line["split"] = split_stats[gid]
        if retry:
            line["retry"] = True
        print(json.dumps(line), flush=True)

    # one graph's failure (an OOM beside another job, a loader error) must
    # not void the whole verdict: record null and carry on; the recipe rule
    # compares candidates on the graphs both have. Then one retry of the
    # graphs that failed (a transient OOM), so a candidate is compared on
    # the full suite whenever possible.
    graphs = [g for g in args.graphs.split(",") if g]
    if args.split == "inductive":
        graphs = ["%s#%d" % (g, i) for g in graphs for i in range(args.samples)]
    for retry in (False, True):
        for gid in (graphs if not retry else list(failed)):
            try:
                run(gid, retry)
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                per[gid] = natural[gid] = None
                failed[gid] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
                print(json.dumps({"graph": gid, "dev": None, "error": failed[gid]}),
                      flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    ok = {g: v for g, v in per.items() if v is not None}
    nat = {g: v for g, v in natural.items() if v is not None}
    result = {"config": os.path.basename(args.config), "ckpt": args.ckpt,
              "hashes": hashes, "protocol": PROTOCOLS[args.split],
              "split": args.split, "samples": args.samples,
              "sub_nodes": args.sub_nodes, "edges_per_node": args.edges_per_node,
              "query_frac": args.query_frac, "split_stats": split_stats,
              "per_cell": args.per_cell,
              "weights": BENCH_WEIGHTS, "min_cell": MIN_CELL,
              "graphs": per, "graphs_natural": natural, "cells": cells,
              "cells4": cells4, "share": share, "queries": counts,
              "failed": failed,
              "complete": not failed, "graphs_ok": len(ok),
              "mean": round(sum(ok.values()) / len(ok), 4) if ok else None,
              "mean_natural": round(sum(nat.values()) / len(nat), 4) if nat else None,
              "seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps({"mean": result["mean"], "mean_natural": result["mean_natural"],
                      "seconds": result["seconds"]}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("written", args.out, flush=True)


if __name__ == "__main__":
    main()
