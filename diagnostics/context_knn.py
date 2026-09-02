"""Is the context signal present in the trunk's rows at all?

A non-parametric check beside diagnostics/context_necessity.py: take a
trained trunk (any mode's last.pth), build the same query and context rows,
and score each candidate by the mean cosine similarity of its row to the
context positives minus the mean to the context negatives. No parameters,
no training. If this beats the floor's MLP on the same held-out instances,
the rows carry what the context needs and a learned scorer that ignores
context has failed to use available information; if it does not, the
representation, not the readout, is the limit. Shuffled labels give the
chance level of the same scorer.

    python diagnostics/context_knn.py -c configs/context_necessity.yaml \\
        --ckpt output/context-necessity/floor_w1_s1024/last.pth [--gpus null]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
from easydict import EasyDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "shared"), os.path.join(REPO, "diagnostics")):
    if p not in sys.path:
        sys.path.insert(0, p)

from incite import synth as S  # noqa: E402
from incite.context import build_rows, shuffle_labels, LABEL_POSITIVE  # noqa: E402
import context_necessity as D  # noqa: E402


def knn_scores(query_rows, ctx_rows, ctx_labels):
    q = torch.nn.functional.normalize(query_rows, dim=-1)          # [k, C, f]
    c = torch.nn.functional.normalize(ctx_rows, dim=-1)            # [k, M, f]
    sim = q @ c.transpose(1, 2)                                    # [k, C, M]
    pos = (ctx_labels == LABEL_POSITIVE).float().unsqueeze(1)      # [k, 1, M]
    neg = 1.0 - pos
    return (sim * pos).sum(-1) / pos.sum(-1) - (sim * neg).sum(-1) / neg.sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpus", default="null")
    ap.add_argument("--eval_instances", type=int, default=None)
    args = ap.parse_args()
    cfg = EasyDict(yaml.safe_load(open(args.config)))
    state = torch.load(args.ckpt, map_location="cpu")
    model = D.ContextModel(cfg, state.get("mode", "floor"))
    model.load_state_dict(state["model"])
    model.eval()
    ccfg = S.context_config(dict(cfg.context))
    ranges = {k: tuple(v) if isinstance(v, list) else v for k, v in dict(cfg.get("synth", {}) or {}).items()}
    n = int(args.eval_instances or cfg.train.eval_instances)
    held = D.make_instances(int(cfg.train.eval_seed), n, ccfg, ranges)
    out = {}
    with torch.no_grad():
        for cond in ("knn", "knn_shuffled", "floor_mlp", "mlp_plus_knn"):
            gen = torch.Generator().manual_seed(int(cfg.seed))
            ranks = []
            for i in range(0, len(held), 4):
                chunk = held[i:i + 4]
                union, batch = S.context_batch(chunk, num_negative=None)
                k, P = len(chunk), batch["ctx_h"].shape[1]
                heads = torch.cat([batch["q_h"], batch["ctx_h"].reshape(-1)])
                rels = torch.cat([batch["q_r"], batch["q_r"].repeat_interleave(P)])
                x, z_q, _ = model.trunk.encode_queries(union, heads, rels)
                q_rows, c_rows, labels = build_rows(x, z_q, batch, k)
                if cond == "floor_mlp":
                    score = model.trunk.score_mlp(q_rows).squeeze(-1)
                elif cond == "mlp_plus_knn":
                    # the eval-time lever: both terms standardised within the
                    # query's candidate set (masked columns excluded), summed
                    def zs(t):
                        m = batch["cand_mask"].float()
                        mu = (t * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True)
                        sd = (((t - mu) ** 2) * m).sum(-1, keepdim=True).div(m.sum(-1, keepdim=True)).sqrt() + 1e-6
                        return (t - mu) / sd
                    score = zs(model.trunk.score_mlp(q_rows).squeeze(-1)) + zs(knn_scores(q_rows, c_rows, labels))
                else:
                    if cond == "knn_shuffled":
                        labels = shuffle_labels(labels, gen)
                    score = knn_scores(q_rows, c_rows, labels)
                ranks.append(D.ranks_from_scores(score, batch["cand_mask"]))
            r = torch.cat(ranks).double()
            out[cond] = {"mrr": float((1 / r).mean()), "hits@1": float((r <= 1).double().mean()),
                         "hits@10": float((r <= 10).double().mean()), "n": int(r.numel())}
    out["ckpt"] = args.ckpt
    out["mode"] = state.get("mode")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
