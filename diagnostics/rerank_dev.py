"""DEV10 sweep for the test-time levers: re-ranking depth k and weight, and
score ensembles, scored on the DEV10 graphs' VALID splits (never test).

CONTAINER-ONLY (imports the pinned TRIX loaders through incite.pretrain).
The 41-graph test evaluation runs once, with the setting this sweep picks;
selecting k on test would be tuning on the benchmark.

    usage (from the prepared work tree, TRIX_ROOT set):
      python diagnostics/rerank_dev.py --config configs/incite_phase1.yaml \
          --ckpt output/incite-pretrain-4g/incite_best.pth[,more.pth] \
          --ks 0,4,8,16 --weights 1.0 [--val_samples 500] [--gpus "[0]"] \
          [--out results/incite/rerank_dev.json]

Reports filtered MRR per DEV10 graph and the three group means for every
(k, weight) cell, exactly as pretrain.py's selection reports them.
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
from incite.rerank import rerank_predictions  # noqa: E402
from incite.run import load_members  # noqa: E402


@torch.no_grad()
def validate_graph_rerank(model, graph, filter_graph, batch_size, num_samples,
                          seed, k, weight, chunk):
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
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_pred = model(graph, t_batch)
        h_pred = model(graph, h_batch)
        t_mask, h_mask = tasks.strict_negative_mask(filter_graph, batch)
        if k > 0:
            t_pred = rerank_predictions(model, graph, t_batch, t_pred,
                                        pos_t_index, t_mask, k, weight, chunk)
            h_pred = rerank_predictions(model, graph, h_batch, h_pred,
                                        pos_h_index, h_mask, k, weight, chunk)
        rankings.append(tasks.compute_ranking(t_pred, pos_t_index, t_mask))
        rankings.append(tasks.compute_ranking(h_pred, pos_h_index, h_mask))
    ranking = torch.cat(rankings).float()
    return float((1 / ranking).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path, or comma list (ensemble)")
    ap.add_argument("--ks", default="0,4,8,16")
    ap.add_argument("--weights", default="1.0")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--val_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
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

    dev_ids = args.dev_graphs.split(",") if args.dev_graphs else list(suite.DEV10)
    dev = []
    for gid in dev_ids:
        valid, filt = load_dev_graph(args.dev_root, gid)
        valid.num_relations = int(valid.num_relations)
        dev.append((gid, valid.to(device), filt.to(device)))

    ks = [int(x) for x in args.ks.split(",")]
    weights = [float(x) for x in args.weights.split(",")]
    groups = suite.dev10_by_group()
    cells = []
    for k in ks:
        for w in (weights if k > 0 else [1.0]):
            t0 = time.perf_counter()
            per = {}
            for gid, vg, fg in dev:
                per[gid] = round(validate_graph_rerank(
                    model, vg, fg, args.batch_size, args.val_samples, args.seed,
                    k, w, args.chunk), 4)
            gm = {g: round(sum(per[x] for x in ids if x in per) / len(ids), 4)
                  for g, ids in groups.items()}
            cell = {"k": k, "weight": w, "dev10": per, "dev10_groups": gm,
                    "seconds": round(time.perf_counter() - t0, 1)}
            print(json.dumps(cell))
            cells.append(cell)
    result = {"config": os.path.basename(args.config), "ckpt": args.ckpt,
              "hashes": hashes, "val_samples": args.val_samples, "cells": cells}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("written", args.out)


if __name__ == "__main__":
    main()
