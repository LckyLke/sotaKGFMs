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
scenario) cell, and the graph's dev number is the cell MRRs combined with
the benchmark's scenario weights (``BENCH_WEIGHTS``). The graph's own mix
is reported beside it (``natural``, the quantity the uniform sample
measured).

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
the natural-mix number (``graphs_natural``), the four cell MRRs with
their counts (``cells``) and the graph's scenario shares (``share``);
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

#: The 41-graph suite's scenario mix: the mean over graphs of the per-graph
#: shares (tail and head queries together), averaged over the two groups
#: (ind_e .325 / .288 / .288 / .098, ind_er .381 / .280 / .280 / .059; from
#: results/incite/halflink_labels.json, 2026-09-03). Sums to 1.
BENCH_WEIGHTS = {"SQSA": 0.353, "SQUA": 0.284, "UQSA": 0.284, "UQUA": 0.079}
#: A cell with fewer scored queries than this is left out of both
#: combinations (the weights are renormalized over the cells present): a
#: transductive valid split can have a UQUA cell of thirty queries, and a
#: cell mean over fewer would move the graph's number by its own noise.
MIN_CELL = 30
PROTOCOL = "stratified_v2"


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
    """Weighted mean of the cell MRRs over the cells with enough queries;
    None when no cell qualifies."""
    present = [s for s in SCEN if cells[s]["n"] >= MIN_CELL and weights.get(s, 0) > 0]
    wsum = sum(weights[s] for s in present)
    if not present or wsum <= 0:
        return None
    return round(sum(weights[s] * cells[s]["mrr"] for s in present) / wsum, 4)


def eval_graph(model, valid, filt, per_cell, batch_size, seed, tail_only):
    labels = scenarios(valid)
    triplets = torch.cat(
        [valid.target_edge_index, valid.target_edge_type.unsqueeze(0)]).t()
    gen = torch.Generator().manual_seed(int(seed))
    rr = {s: [] for s in SCEN}
    counts = {s: 0 for s in SCEN}
    for direction in (["tail"] if tail_only else ["tail", "head"]):
        lab = labels[direction]
        for s in SCEN:
            counts[s] += sum(1 for x in lab if x == s)
        picked = stratified_indices(lab, per_cell, gen)
        for s in SCEN:
            idx = picked[s]
            if idx.numel() == 0:
                continue
            ranks = rank_queries(model, valid, filt,
                                 triplets[idx.to(triplets.device)],
                                 direction, batch_size)
            rr[s].extend((1.0 / ranks).tolist())
    total = sum(counts.values())
    cells = {s: {"mrr": round(sum(v) / len(v), 4) if v else None, "n": len(v)}
             for s, v in rr.items()}
    share = {s: round(counts[s] / total, 4) for s in SCEN}
    return {"weighted": combine(cells, BENCH_WEIGHTS),
            "natural": combine(cells, share),
            "cells": cells, "share": share,
            "queries": sum(c["n"] for c in cells.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path, or comma list (ensemble)")
    ap.add_argument("--graphs", default=",".join(DEV8T))
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

    per, natural, cells, share, counts, failed = {}, {}, {}, {}, {}, {}
    t0 = time.perf_counter()

    def run(gid, retry):
        info = suite.by_id(gid)
        valid, filt = load_dev_graph(args.dev_root, gid)
        valid.num_relations = int(valid.num_relations)
        r = eval_graph(model, valid.to(device), filt.to(device), args.per_cell,
                       args.batch_size, args.seed,
                       bool(getattr(info, "tail_only", False)))
        per[gid], natural[gid] = r["weighted"], r["natural"]
        cells[gid], share[gid], counts[gid] = r["cells"], r["share"], r["queries"]
        failed.pop(gid, None)
        line = {"graph": gid, "dev": r["weighted"], "natural": r["natural"],
                "cells": {s: [c["mrr"], c["n"]] for s, c in r["cells"].items()},
                "queries": r["queries"]}
        if retry:
            line["retry"] = True
        print(json.dumps(line), flush=True)

    # one graph's failure (an OOM beside another job, a loader error) must
    # not void the whole verdict: record null and carry on; the recipe rule
    # compares candidates on the graphs both have. Then one retry of the
    # graphs that failed (a transient OOM), so a candidate is compared on
    # the full suite whenever possible.
    graphs = [g for g in args.graphs.split(",") if g]
    for retry in (False, True):
        for gid in (graphs if not retry else list(failed)):
            try:
                run(gid, retry)
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                per[gid] = None
                failed[gid] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
                print(json.dumps({"graph": gid, "dev": None, "error": failed[gid]}),
                      flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    ok = {g: v for g, v in per.items() if v is not None}
    nat = {g: v for g, v in natural.items() if v is not None}
    result = {"config": os.path.basename(args.config), "ckpt": args.ckpt,
              "hashes": hashes, "protocol": PROTOCOL, "per_cell": args.per_cell,
              "weights": BENCH_WEIGHTS, "min_cell": MIN_CELL,
              "graphs": per, "graphs_natural": natural, "cells": cells,
              "share": share, "queries": counts, "failed": failed,
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
