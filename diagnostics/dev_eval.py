"""The dev suite for lever decisions (2026-09-03, after the independent
review): filtered MRR on the VALID splits of transductive graphs that are
neither in the pretraining diet nor among the 41 test graphs, so a lever
verdict read here never touches the benchmark.

CONTAINER-ONLY (imports the pinned TRIX loaders through incite.pretrain).

    usage (from the prepared work tree, TRIX_ROOT set):
      python diagnostics/dev_eval.py --config configs/<cfg>.yaml \\
          --ckpt output/incite-pretrain-<run>/incite_last.pth \\
          [--graphs YAGO310,CoDExSmall,...] [--val_samples 500] \\
          [--batch_size 16] [--gpus "[0]"] [--out results/incite/dev/<name>.json]

Default graphs (DEV8T): YAGO310, CoDExSmall, CoDExLarge, Hetionet,
ConceptNet100k, DBpedia100k, AristoV4, WDsinger (NELL23k derives from a
diet graph and is left out). Tail-only graphs score tail queries only, as
the suite does. Reports per-graph MRR (null for a graph that failed, with
the error), the unweighted mean over the graphs that succeeded, and
``complete``.
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

# NELL23k is left out: it derives from NELL995, which is in the 4-graph diet.
DEV8T = ["YAGO310", "CoDExSmall", "CoDExLarge", "Hetionet", "ConceptNet100k",
         "DBpedia100k", "AristoV4", "WDsinger"]
DEV9T = DEV8T + ["NELL23k"]


@torch.no_grad()
def validate_graph(model, graph, filter_graph, batch_size, num_samples, seed,
                   tail_only):
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
        t_mask, h_mask = tasks.strict_negative_mask(filter_graph, batch)
        t_pred = model(graph, t_batch)
        rankings.append(tasks.compute_ranking(t_pred, pos_t_index, t_mask))
        if not tail_only:
            h_pred = model(graph, h_batch)
            rankings.append(tasks.compute_ranking(h_pred, pos_h_index, h_mask))
    ranking = torch.cat(rankings).float()
    return float((1 / ranking).mean()), int(ranking.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path, or comma list (ensemble)")
    ap.add_argument("--graphs", default=",".join(DEV8T))
    ap.add_argument("--val_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=16)
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

    per, counts, failed = {}, {}, {}
    t0 = time.perf_counter()
    for gid in [g for g in args.graphs.split(",") if g]:
        # one graph's failure (an OOM beside another job, a loader error)
        # must not void the whole verdict: record null and carry on; the
        # recipe rule compares candidates on the graphs both have
        try:
            info = suite.by_id(gid)
            valid, filt = load_dev_graph(args.dev_root, gid)
            valid.num_relations = int(valid.num_relations)
            mrr, n = validate_graph(model, valid.to(device), filt.to(device),
                                    args.batch_size, args.val_samples, args.seed,
                                    bool(getattr(info, "tail_only", False)))
            per[gid] = round(mrr, 4)
            counts[gid] = n
            print(json.dumps({"graph": gid, "mrr": per[gid], "queries": n}))
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            per[gid] = None
            failed[gid] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            print(json.dumps({"graph": gid, "mrr": None, "error": failed[gid]}))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    ok = {g: v for g, v in per.items() if v is not None}
    mean = round(sum(ok.values()) / len(ok), 4) if ok else None
    non_nell = [v for g, v in ok.items() if "NELL" not in g]
    result = {"config": os.path.basename(args.config), "ckpt": args.ckpt,
              "hashes": hashes, "val_samples": args.val_samples,
              "graphs": per, "queries": counts, "failed": failed,
              "complete": not failed, "graphs_ok": len(ok), "mean": mean,
              "mean_non_nell": round(sum(non_nell) / len(non_nell), 4) if non_nell else None,
              "seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps({"mean": result["mean"], "mean_non_nell": result["mean_non_nell"],
                      "seconds": result["seconds"]}))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=1)
        print("written", args.out)


if __name__ == "__main__":
    main()
