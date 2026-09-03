"""The pruning curve of the proof-guided gate (PG1, 2026-09-03): DEV10
filtered MRR (VALID splits, never test) when the lowest-gated share x of
every query's edges is dropped in every round, for a list of x.

CONTAINER-ONLY (imports the pinned TRIX loaders through incite.pretrain).
The curve decides whether sparse propagation (per-query edge subsets with
their own kernels) is worth building: flat to 60 or 80 percent means yes.

    usage (from the prepared work tree, TRIX_ROOT set):
      python diagnostics/gate_prune_dev.py --config configs/<gated>.yaml \\
          --ckpt output/incite-pretrain-<run>/incite_last.pth \\
          --fracs 0,0.2,0.4,0.6,0.8,0.9,0.95 [--val_samples 500] \\
          [--batch_size 4] [--gpus "[0]"] [--out results/incite/gate_prune.json]

Also reports, per graph at x = 0, the share of raw gate products below 0.5
after the last round (how far the gates moved from their open start), the
REALIZED kept fraction of every cell (ties keep more; a saturated gate
keeps everything), and a random-pruning control curve at every fraction.
A gate curve is evidence only where it sits above the random curve at
the same realized kept fraction.
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


@torch.no_grad()
def validate_graph(model, graph, filter_graph, batch_size, num_samples, seed):
    triplets = torch.cat(
        [graph.target_edge_index, graph.target_edge_type.unsqueeze(0)]).t()
    if num_samples and triplets.shape[0] > num_samples:
        gen = torch.Generator().manual_seed(int(seed))
        keep = torch.randperm(triplets.shape[0], generator=gen)[:num_samples]
        triplets = triplets[keep.to(triplets.device)]
    rankings = []
    for start in range(0, triplets.shape[0], batch_size):
        batch = triplets[start:start + batch_size]
        t_batch, h_batch = tasks.all_negative(graph, batch)
        pos_h_index, pos_t_index, _ = batch.t()
        t_pred = model(graph, t_batch)
        h_pred = model(graph, h_batch)
        t_mask, h_mask = tasks.strict_negative_mask(filter_graph, batch)
        rankings.append(tasks.compute_ranking(t_pred, pos_t_index, t_mask))
        rankings.append(tasks.compute_ranking(h_pred, pos_h_index, h_mask))
    ranking = torch.cat(rankings).float()
    return float((1 / ranking).mean())


@torch.no_grad()
def gate_stats(model, graph, seed, num_queries=8):
    """Share of raw gate products below 0.5 in the last round, over a few
    queries: 0 means the gates never moved from their open start."""
    if getattr(model, "gates", None) is None:
        return None
    triplets = torch.cat(
        [graph.target_edge_index, graph.target_edge_type.unsqueeze(0)]).t()
    gen = torch.Generator().manual_seed(int(seed))
    keep = torch.randperm(triplets.shape[0], generator=gen)[:num_queries]
    batch = triplets[keep.to(triplets.device)]
    from incite.model import TASK_ENTITY, negative_sample_to_tail
    t_batch, _ = tasks.all_negative(graph, batch)
    h_index, t_index, r_index = t_batch.unbind(-1)
    num_direct = int(graph.num_relations) // 2
    h_index, t_index, r_index = negative_sample_to_tail(
        h_index, t_index, r_index, num_direct)
    pairs = model._pairs(graph)
    x, z, _ = model._trunk(graph, pairs, h_index[:, 0], r_index[:, 0], None,
                           TASK_ENTITY)
    b, d = x.shape[0], model.dim
    q = z.gather(1, r_index[:, 0].view(b, 1, 1).expand(b, 1, d)).squeeze(1)
    gn, gr = model.gates[-1](x, z, q)
    prod = gn[:, graph.edge_index[0]] * gr[:, graph.edge_type]
    return {"below_0.5": round(float((prod < 0.5).float().mean()), 4),
            "mean": round(float(prod.mean()), 4),
            "p10": round(float(prod.flatten().kthvalue(
                max(1, prod.numel() // 10)).values), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fracs", default="0,0.2,0.4,0.6,0.8,0.9,0.95")
    ap.add_argument("--val_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dev_root", default="/kgfm-src/data/roots/trix")
    ap.add_argument("--dev_graphs", default=None)
    ap.add_argument("--gpus", default="[0]")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = EasyDict(yaml.safe_load(open(args.config)))
    gpus = None if args.gpus in ("null", "None") else util.literal_eval(args.gpus)
    device = torch.device(gpus[0]) if gpus else torch.device("cpu")
    model, hashes = load_members(cfg, [p for p in args.ckpt.split(",") if p])
    model = model.to(device).eval()
    if getattr(model, "gates", None) is None:
        print("WARNING: the model has no gates; every fraction prunes at random ties")

    dev_ids = args.dev_graphs.split(",") if args.dev_graphs else list(suite.DEV10)
    dev = []
    for gid in dev_ids:
        valid, filt = load_dev_graph(args.dev_root, gid)
        valid.num_relations = int(valid.num_relations)
        dev.append((gid, valid.to(device), filt.to(device)))

    groups = suite.dev10_by_group()
    cells = []
    stats = {}
    # every fraction twice: the gate curve and the random-pruning control
    # (seeded uniform scores in place of the gate products). ``kept`` is
    # the realized kept fraction; ties at the threshold keep more than
    # 1 - x, and a gate that never left its open start keeps everything,
    # which would make a "flat curve" mean "nothing was pruned".
    for frac in [float(x) for x in args.fracs.split(",")]:
        for random_control in ((False, True) if frac > 0 else (False,)):
            model.prune_frac = frac
            model.prune_random = random_control
            model.prune_seed = args.seed
            t0 = time.perf_counter()
            per, kept = {}, {}
            for gid, vg, fg in dev:
                model.prune_kept = []
                per[gid] = round(validate_graph(model, vg, fg, args.batch_size,
                                                args.val_samples, args.seed), 4)
                kept[gid] = round(sum(model.prune_kept) / len(model.prune_kept), 4) \
                    if model.prune_kept else 1.0
                if frac == 0.0:
                    stats[gid] = gate_stats(model, vg, args.seed)
            gm = {g: round(sum(per[x] for x in ids if x in per) / len(ids), 4)
                  for g, ids in groups.items()}
            cell = {"prune_frac": frac, "random_control": random_control,
                    "dev10": per, "dev10_groups": gm, "kept": kept,
                    "kept_mean": round(sum(kept.values()) / len(kept), 4),
                    "seconds": round(time.perf_counter() - t0, 1)}
            print(json.dumps(cell))
            cells.append(cell)
    model.prune_frac = 0.0
    model.prune_random = False
    result = {"config": os.path.basename(args.config), "ckpt": args.ckpt,
              "hashes": hashes, "val_samples": args.val_samples,
              "gate_stats_at_0": stats, "cells": cells}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("written", args.out)


if __name__ == "__main__":
    main()
