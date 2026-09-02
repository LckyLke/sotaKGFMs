"""PETALS diagnostic: can the model tell the two petal candidates apart?

CONTAINER-ONLY, CPU is enough (220 graphs of ~70 nodes). Each instance
holds a query (head, relation 0) and two candidate tails; the first is
the true one (FLOCK's generator convention). Relation-color symmetry
makes the candidates automorphic for deterministic models, which
therefore tie and score 50% with fair tie-breaking. The walks lever
exists to break exactly this tie (design section C; kill switch in
docs/INCITE_PLAN.md phase 2.1).

Scoring: model(data, batch) with batch [1, 2, 3]; accuracy counts
score(true) > score(false) as 1, a tie as 0.5. Walk features are seeded;
--offsets N averages the scores of N seeded passes (offset 0..N-1) and
also reports the 1-pass primary protocol.

    usage: python diagnostics/petals_eval.py --ckpt X --config Y \
        [--instances diagnostics/petals] [--offsets 8] [--out result.json]
"""
import argparse
import glob
import json
import os
import sys

import torch
import yaml
from easydict import EasyDict

TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
sys.path.insert(0, os.path.join(TRIX_ROOT, "src"))

try:
    from incite.run import build_model
except ImportError:  # flat invocation from the work tree
    from run import build_model  # type: ignore


def augment(data):
    """Add inverse edges, TRIX convention: type + num_direct, swapped ends."""
    ei, et = data.edge_index, data.edge_type
    num_direct = int(data.num_relations)
    from torch_geometric.data import Data
    return Data(
        edge_index=torch.cat([ei, ei.flip(0)], dim=1),
        edge_type=torch.cat([et, et + num_direct]),
        num_nodes=int(data.num_nodes),
        num_relations=2 * num_direct)


@torch.no_grad()
def score_instance(model, inst, offsets):
    g = augment(inst)
    h = int(inst.test_triplets[0][0])
    t_true = int(inst.test_triplets[0][1])
    t_false = int(inst.test_triplets[1][1])
    r = int(inst.test_triplets[0][2])
    batch = torch.tensor([[[h, t_true, r], [h, t_false, r]]])
    per_offset = []
    for k in range(max(offsets, 1)):
        pred = model(g, batch, walk_offset=k)
        per_offset.append(pred[0])
    one = per_offset[0]
    avg = torch.stack(per_offset).mean(0)
    def verdict(p):
        if p[0] > p[1]:
            return 1.0
        if p[0] < p[1]:
            return 0.0
        return 0.5
    return verdict(one), verdict(avg), float((one[0] - one[1]).abs())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--instances", default="diagnostics/petals")
    ap.add_argument("--offsets", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = EasyDict(yaml.safe_load(open(args.config)))
    model = build_model(cfg)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    assert not unexpected, unexpected
    if missing:
        print("WARNING: %d tensors at init (config/ckpt mismatch): %s"
              % (len(missing), missing[:3]))
    model.eval()

    files = sorted(glob.glob(os.path.join(args.instances, "_*.pth")))
    assert files, "no instances under %s -- run generate_petals.py" % args.instances
    ones, avgs, margins, ties = [], [], [], 0
    for f in files:
        inst = torch.load(f, map_location="cpu", weights_only=False)
        v1, va, m = score_instance(model, inst, args.offsets)
        ones.append(v1); avgs.append(va); margins.append(m)
        ties += (v1 == 0.5)
    result = {
        "ckpt": os.path.basename(args.ckpt),
        "config": os.path.basename(args.config),
        "instances": len(files),
        "accuracy_1pass": round(sum(ones) / len(ones), 4),
        "accuracy_%dpass_avg" % args.offsets: round(sum(avgs) / len(avgs), 4),
        "ties_1pass": ties,
        "mean_abs_margin_1pass": round(sum(margins) / len(margins), 6),
        "step": state.get("step"),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
