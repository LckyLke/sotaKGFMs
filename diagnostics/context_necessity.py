"""Context-necessity diagnostic: can a from-scratch in-context scorer use
its context at all, when the context is the only place the query relation's
facts exist?

Why (2026-09-02, results/incite/CONTEXT_NECESSITY.md when it lands): every
in-context readout measured here was a residual on a trained encoder's own
score and ended metric-neutral (CREST, INCITE 2.2, and the released KGPFN at
or below TRIX on 25 graphs). docs/KGFM_PLAN.md proposes the mechanism again,
from scratch and end to end. Before anyone builds that at plan scale, this
script answers the smaller question on synthetic graphs whose query relation
is withheld from the message graph (incite/synth.py::create_context_instance):

  * three scorers on one trunk, each trained from scratch on the same stream:
    floor (the trunk's MLP), context_only (the PFN-style scorer is the only
    scorer), residual (MLP + scorer, the design that died twice);
  * three evaluation conditions on the same held-out instances: full context,
    shuffled labels, no context -- the external plan's K3 ordering test.

Pass, per the plan's own criterion: full > shuffled > none, full minus none
at least 0.02 MRR, for the context-only model. The floor is the same number
under every condition by construction and bounds what structure alone gives.

CPU or GPU; no container is required (torch + PyG + the incite package).
Ranks here are not benchmark ranks -- nothing is written under ranks/.

    python diagnostics/context_necessity.py -c configs/context_necessity.yaml \\
        --mode context_only --out output/context-necessity/context_only \\
        [--steps N] [--withhold 1.0] [--seed 1024] [--gpus "[0]"|null]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import torch
import yaml
from easydict import EasyDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "shared")):
    if p not in sys.path:
        sys.path.insert(0, p)

from incite import synth as S  # noqa: E402
from incite.context import ContextScorer, build_rows, shuffle_labels  # noqa: E402
from incite.model import INCITE  # noqa: E402
from incite.train import self_adversarial_nll  # noqa: E402

MODES = ("floor", "context_only", "residual")
CONDITIONS = ("full", "shuffled", "none")


class ContextModel(torch.nn.Module):
    """One trunk, one optional scorer, one ``mode``."""

    def __init__(self, cfg: EasyDict, mode: str):
        super().__init__()
        assert mode in MODES, mode
        self.mode = mode
        m = cfg.model
        self.trunk = INCITE(dim=int(m.dim), rounds=int(m.rounds),
                            layer_norm=bool(m.layer_norm), short_cut=bool(m.short_cut),
                            count_channel=bool(m.count_channel), walks=None,
                            support_readout=False, num_mlp_layer=int(m.num_mlp_layer))
        self.scorer = None
        if mode != "floor":
            self.scorer = ContextScorer(2 * int(m.dim), width=int(cfg.scorer.width),
                                        depth=int(cfg.scorer.depth), heads=int(cfg.scorer.heads))
        self.detach_rows = bool(cfg.scorer.get("detach_rows", False))

    def forward(self, union, batch: dict, condition: str = "full",
                shuffle_gen: torch.Generator = None) -> torch.Tensor:
        """Scores ``[k, C]`` for the candidates of every query in the batch."""
        k = batch["q_h"].shape[0]
        P = batch["ctx_h"].shape[1]
        heads = torch.cat([batch["q_h"], batch["ctx_h"].reshape(-1)])
        rels = torch.cat([batch["q_r"], batch["q_r"].repeat_interleave(P)])
        x, z_q, _s0 = self.trunk.encode_queries(union, heads, rels)
        query_rows, ctx_rows, ctx_labels = build_rows(x, z_q, batch, k)
        score = None
        if self.mode in ("floor", "residual"):
            score = self.trunk.score_mlp(query_rows).squeeze(-1)
        if self.scorer is not None:
            if condition == "none":
                ctx_rows, ctx_labels = None, None
            else:
                if condition == "shuffled":
                    ctx_labels = shuffle_labels(ctx_labels, shuffle_gen)
                if self.detach_rows:
                    ctx_rows = ctx_rows.detach()
            s = self.scorer(query_rows, ctx_rows, ctx_labels)
            score = s if score is None else score + s
        return score


def to_device(union, batch, device):
    union.edge_index = union.edge_index.to(device)
    union.edge_type = union.edge_type.to(device)
    return union, {key: val.to(device) for key, val in batch.items()}


def make_instances(seed: int, n: int, ccfg: dict, ranges: dict):
    gen = torch.Generator().manual_seed(int(seed))
    return [S.create_context_instance(gen, ccfg, ranges) for _ in range(n)]


def ranks_from_scores(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """1-based, pessimistic ties, column 0 the answer, masked columns ignored
    (the shared rank definition of shared/metrics.py)."""
    pos = score[:, :1]
    beat = (score[:, 1:] >= pos) & mask[:, 1:]
    return beat.sum(dim=-1) + 1


@torch.no_grad()
def evaluate(model, held, ccfg, device, batch_size: int, shuffle_seed: int) -> dict:
    model.eval()
    out = {}
    for cond in CONDITIONS:
        if model.scorer is None and cond != "full":
            out[cond] = dict(out["full"])
            continue
        gen = torch.Generator().manual_seed(int(shuffle_seed))
        ranks = []
        for i in range(0, len(held), batch_size):
            chunk = held[i:i + batch_size]
            union, batch = S.context_batch(chunk, num_negative=None)
            union, batch = to_device(union, batch, device)
            score = model(union, batch, condition=cond, shuffle_gen=gen)
            ranks.append(ranks_from_scores(score, batch["cand_mask"]).cpu())
        r = torch.cat(ranks).double()
        out[cond] = {"mrr": float((1.0 / r).mean()), "hits@1": float((r <= 1).double().mean()),
                     "hits@3": float((r <= 3).double().mean()),
                     "hits@10": float((r <= 10).double().mean()), "n": int(r.numel())}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--withhold", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gpus", default="[0]")
    ap.add_argument("--eval_instances", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--instances_per_step", type=int, default=None)
    ap.add_argument("--detach_rows", action="store_true")
    args = ap.parse_args()

    cfg = EasyDict(yaml.safe_load(open(args.config)))
    if args.withhold is not None:
        cfg.context.withhold = float(args.withhold)
    if args.detach_rows:
        cfg.scorer.detach_rows = True
    steps = int(args.steps if args.steps is not None else cfg.train.steps)
    seed = int(args.seed if args.seed is not None else cfg.seed)
    n_eval = int(args.eval_instances if args.eval_instances is not None else cfg.train.eval_instances)
    eval_every = int(args.eval_every if args.eval_every is not None else cfg.train.eval_every)
    ips = int(args.instances_per_step if args.instances_per_step is not None
              else cfg.train.instances_per_step)
    ccfg = S.context_config(dict(cfg.context))
    ranges = {key: tuple(val) if isinstance(val, list) else val
              for key, val in dict(cfg.get("synth", {}) or {}).items()}
    device = torch.device("cpu") if args.gpus in ("null", "None", "") or not torch.cuda.is_available() \
        else torch.device("cuda:0")

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(seed)
    model = ContextModel(cfg, args.mode).to(device)
    model.trunk.checkpoint_activations = bool(cfg.train.checkpoint_activations) and device.type == "cuda"
    n_trunk = sum(p.numel() for p in model.trunk.parameters())
    n_scorer = sum(p.numel() for p in model.scorer.parameters()) if model.scorer is not None else 0
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr))

    held = make_instances(int(cfg.train.eval_seed), n_eval, ccfg, ranges)
    held_stats = {"withheld_all": all(int(i.num_removed) == int(i.num_obs_r) for i in held),
                  "mean_nodes": sum(int(i.num_nodes) for i in held) / len(held),
                  "mean_edges": sum(int(i.edge_index.shape[1]) for i in held) / len(held),
                  "mean_eval_pool": sum(int(i.num_eval_pool) for i in held) / len(held),
                  "mean_derivable_r": sum(int(i.num_derivable_r) for i in held) / len(held),
                  "mean_eval_hard": sum(int(i.num_eval_hard) for i in held) / len(held)}
    prov = {"mode": args.mode, "config": os.path.abspath(args.config), "seed": seed,
            "withhold": ccfg["withhold"], "steps": steps, "instances_per_step": ips,
            "context": ccfg, "scorer": dict(cfg.scorer), "synth_ranges": {k: list(v) if isinstance(v, tuple) else v for k, v in ranges.items()},
            "params": {"trunk": n_trunk, "scorer": n_scorer},
            "device": device.type, "torch": torch.__version__, "cuda": torch.version.cuda,
            "host": platform.platform(), "image": os.environ.get("KGFM_IMAGE"),
            "held_out": {"seed": int(cfg.train.eval_seed), "n": n_eval, **held_stats}}
    try:
        prov["git"] = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                              text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        prov["git"] = None
    try:
        prov["gpu"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    except Exception:
        prov["gpu"] = None
    json.dump(prov, open(os.path.join(args.out, "PROVENANCE.json"), "w"), indent=2, sort_keys=True)
    print("mode %s | trunk %d params | scorer %d params | device %s | held-out %d instances (%s)"
          % (args.mode, n_trunk, n_scorer, device, n_eval, held_stats), flush=True)

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    t0 = time.time()
    curve = []
    model.train()
    for step in range(1, steps + 1):
        gen = torch.Generator().manual_seed(int(cfg.train.train_seed) * 1000003 + step)
        insts = [S.create_context_instance(gen, ccfg, ranges) for _ in range(ips)]
        union, batch = S.context_batch(insts, num_negative=ccfg["num_negative"])
        union, batch = to_device(union, batch, device)
        score = model(union, batch, condition="full")
        loss = self_adversarial_nll(score, int(score.shape[1]) - 1,
                                    float(cfg.train.adversarial_temperature))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss.detach()), "seconds": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
        if step % eval_every == 0 or step == steps:
            ev = evaluate(model, held, ccfg, device, batch_size=max(1, ips), shuffle_seed=seed)
            rec = {"step": step, "eval": ev, "seconds": round(time.time() - t0, 1)}
            curve.append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("eval step %d: %s" % (step, {c: round(ev[c]["mrr"], 4) for c in CONDITIONS}), flush=True)
            torch.save({"model": model.state_dict(), "step": step, "mode": args.mode},
                       os.path.join(args.out, "last.pth"))
    final = curve[-1]["eval"] if curve else None
    result = {"provenance": prov, "final": final, "curve": curve,
              "seconds": round(time.time() - t0, 1)}
    if final is not None and model.scorer is not None:
        gap = final["full"]["mrr"] - final["none"]["mrr"]
        result["k3"] = {"ordered": final["full"]["mrr"] > final["shuffled"]["mrr"] > final["none"]["mrr"],
                        "full_minus_none": gap, "pass": bool(gap >= 0.02 and
                        final["full"]["mrr"] > final["shuffled"]["mrr"] > final["none"]["mrr"])}
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=2, sort_keys=True)
    print("done:", json.dumps(result.get("k3")), "final", json.dumps({c: round(final[c]["mrr"], 4) for c in CONDITIONS}) if final else None)


if __name__ == "__main__":
    main()
