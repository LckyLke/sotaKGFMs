"""Support usage probe -- the phase-2.2 kill switch (PLAN phase 2.2).

Does the trained model USE its support set? Score queries twice: once with
the real store, once with the same store's signed labels permuted across
rows (per relation, seeded). If the rankings barely move, the readout
ignores the labels -- the CREST failure signature -- and the lever is dead
per the plan's stop rule.

CONTAINER-ONLY. Probes the VALID split of a few DEV10 graphs (test stays
untouched for verdicts). Reports per graph: fraction of queries whose
candidate ranking changed at all, mean |score delta|, and MRR under both
stores.

    usage: python diagnostics/support_probe.py --ckpt X --config Y \
        [--graphs FBIngram:25,Metafam,FB15k237Inductive:v1] \
        [--samples 300] [--out result.json]
"""
import argparse
import copy
import json
import os
import sys

import torch
import yaml
from easydict import EasyDict

TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
sys.path.insert(0, os.path.join(TRIX_ROOT, "src"))

from trix import tasks  # noqa: E402

try:
    from incite import support as incite_support
    from incite.run import build_model, support_build_kwargs
    from incite.pretrain import load_dev_graph
except ImportError:  # flat work-tree invocation
    import support as incite_support  # type: ignore
    from run import build_model, support_build_kwargs  # type: ignore
    from pretrain import load_dev_graph  # type: ignore


def permuted_labels(store, generator):
    """A deep copy whose signed-label column is shuffled per relation."""
    twin = copy.deepcopy(store)
    for rid, feat in twin._feat.items():
        labels = feat[..., -1].flatten()
        perm = torch.randperm(labels.numel(), generator=generator)
        feat[..., -1] = labels[perm].view_as(feat[..., -1])
    return twin


@torch.no_grad()
def probe_graph(model, cfg, gid, dev_root, samples, seed):
    device = next(model.parameters()).device
    valid, filt = load_dev_graph(dev_root, gid)
    valid, filt = valid.to(device), filt.to(device)
    store = incite_support.SupportStore()
    incite_support.build_support(valid, model, seed=seed, store=store,
                                 **support_build_kwargs(cfg))
    gen = torch.Generator().manual_seed(seed)
    twin = permuted_labels(store, gen)

    store.to(device)
    twin.to(device)
    triplets = torch.cat(
        [valid.target_edge_index, valid.target_edge_type.unsqueeze(0)]).t()
    if triplets.shape[0] > samples:
        keep = torch.randperm(triplets.shape[0],
                              generator=torch.Generator().manual_seed(seed))
        triplets = triplets[keep[:samples].to(triplets.device)]

    changed, deltas, mrr = 0, [], {"real": [], "permuted": []}
    for start in range(0, triplets.shape[0], 8):
        batch = triplets[start:start + 8]
        t_batch, _ = tasks.all_negative(valid, batch)
        pos_t = batch[:, 1]
        t_mask, _ = tasks.strict_negative_mask(filt, batch)
        pred_a = model(valid, t_batch, support=store)
        pred_b = model(valid, t_batch, support=twin)
        changed += int((pred_a.argsort(-1) != pred_b.argsort(-1))
                       .any(-1).sum())
        deltas.append(float((pred_a - pred_b).abs().mean()))
        mrr["real"].append(tasks.compute_ranking(pred_a, pos_t, t_mask))
        mrr["permuted"].append(tasks.compute_ranking(pred_b, pos_t, t_mask))
    n = int(triplets.shape[0])
    return {
        "queries": n,
        "ranking_changed_frac": round(changed / n, 4),
        "mean_abs_score_delta": round(sum(deltas) / len(deltas), 6),
        "mrr_real": round(float((1.0 / torch.cat(mrr["real"]).float()).mean()), 4),
        "mrr_permuted": round(float((1.0 / torch.cat(mrr["permuted"]).float()).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--graphs",
                    default="FBIngram:25,Metafam,FB15k237Inductive:v1")
    ap.add_argument("--dev_root", default="/kgfm-src/data/roots/trix")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = EasyDict(yaml.safe_load(open(args.config)))
    assert cfg.support.enabled, "probe needs a support-enabled config"
    model = build_model(cfg)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    result = {"ckpt": os.path.basename(args.ckpt),
              "step": state.get("step"), "graphs": {}}
    for gid in args.graphs.split(","):
        result["graphs"][gid] = probe_graph(
            model, cfg, gid, args.dev_root, args.samples, args.seed)
        print(gid, result["graphs"][gid])
    fracs = [g["ranking_changed_frac"] for g in result["graphs"].values()]
    result["verdict"] = ("ALIVE: the readout reacts to support labels"
                        if max(fracs) > 0.05 else
                        "DEAD: rankings ignore the labels (CREST signature)")
    print(json.dumps(result, indent=1))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
