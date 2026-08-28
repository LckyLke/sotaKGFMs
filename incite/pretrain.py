"""Container driver: pretrain INCITE on the TRIX mix, selecting checkpoints
by zero-shot DEV10 -- never by pretraining-mix validation.

CONTAINER-ONLY, like ``incite/run.py``. The recipe is TRIX's own pretraining
(``pretrain_entity.yaml``): FB15k237 + WN18RR + CoDExMedium through TRIX's
dataset classes, one graph per step sampled proportionally to edge count
(``multigraph_collator`` restated), 512 strict negatives, self-adversarial
BCE at temperature 1, AdamW 5e-4, batch 32. The relation loss (design D)
is added as ``L_entity + lambda * L_relation`` when ``relation.lambda`` > 0.

------------------------------------------------------------------------------
Checkpoint selection (docs/INCITE_PLAN.md lesson 2 -- BINDING)
------------------------------------------------------------------------------
Pretraining-mix validation rose +0.017..+0.021 in three CREST regimes and
transferred nothing; that signal is formally disqualified. Every
``val_interval`` steps this driver evaluates ZERO-SHOT on the DEV10 graphs'
validation splits (``shared/suite.py::DEV10``), reports one mean per suite
group (``dev10_by_group`` -- never one number), and logs all three. The
single scalar needed to order checkpoints is the unweighted mean of the
per-group means (each group counts once); the per-group numbers are what
gets reported and inspected, the scalar is only the argmax key, and both
land in the JSONL log for every validation.

Selection uses each DEV10 graph's VALID split, never test.

    usage (inside the container / prepared workdir):
      python -m incite.pretrain -c configs/incite_phase1.yaml --gpus "[0]" \
          --data_root /kgfm-src/data/roots/incite \
          --raw_root /kgfm-src/data/raw/ultra-pretrain \
          --dev_root /kgfm-src/data/roots/trix \
          --output_dir /kgfm-src/output/incite-pretrain \
          [--steps N] [--val_interval N] [--val_samples N] [--seed 1024] \
          [--dev_graphs FBIngram:25,...] [--resume ckpt.pth]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch
import yaml
from easydict import EasyDict
from torch_geometric.data import Data

TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
sys.path.insert(0, os.path.join(TRIX_ROOT, "src"))

from trix import tasks, util  # noqa: E402

import suite  # noqa: E402  (shared/suite.py on PYTHONPATH)

try:
    from . import support as incite_support
    from . import train as incite_train
    from .run import build_model, support_build_kwargs
except ImportError:  # pragma: no cover - flat invocation
    import support as incite_support  # type: ignore
    import train as incite_train  # type: ignore
    from run import build_model, support_build_kwargs  # type: ignore

#: Raw layout, byte-for-byte what the TRIX loaders expect (crest precedent).
PRETRAIN_RAW_DIRS = {
    "FB15k237": "fb15k237",
    "WN18RR": "wn18rr",
    "CoDExMedium": "codex-m",
}


def prepare_pretrain_root(pretrain_root: str, raw_root: str, names) -> None:
    """Copy the raw files under ``pretrain_root``. Copies, not symlinks."""
    for name in names:
        top = PRETRAIN_RAW_DIRS[name]
        src, dst = os.path.join(raw_root, top), os.path.join(pretrain_root, top)
        if os.path.isdir(dst):
            continue
        if not os.path.isdir(src):
            raise FileNotFoundError(
                "raw files for %s expected at %s; scripts/train_incite.sh "
                "seeds them from data/roots/crest/pretrain or "
                "data/raw/ultra-pretrain" % (name, src))
        os.makedirs(pretrain_root, exist_ok=True)
        shutil.copytree(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)


def load_mix_graphs(pretrain_root: str, names):
    """(train, valid, test) Data lists through TRIX's loaders, cached once
    (the FB15k237/WN18RR loaders rerun build_relation_graph per call)."""
    from trix import datasets as trix_datasets
    cache = os.path.join(pretrain_root, "incite_mix_%s.pt" % "-".join(names))
    if os.path.exists(cache):
        return torch.load(cache, map_location="cpu")
    graphs = [trix_datasets.JointDataset.datasets_map[n](root=pretrain_root)
              for n in names]
    splits = ([g[0] for g in graphs], [g[1] for g in graphs], [g[2] for g in graphs])
    torch.save(splits, cache + ".tmp")
    os.replace(cache + ".tmp", cache)
    return splits


def load_dev_graph(dev_root: str, gid: str):
    """(valid_data, filter_graph) for one DEV10 suite id, via TRIX loaders."""
    graph = suite.by_id(gid)
    cfg = {"class": graph.dataset, "root": dev_root}
    if graph.version is not None:
        cfg["version"] = util.literal_eval(graph.version)
    elif graph.id == "Metafam":
        cfg["version"] = "Metafam"
    elif graph.id == "FBNELL":
        cfg["version"] = "FBNELL_v1"
    dataset = util.build_dataset(EasyDict({"dataset": cfg}))
    valid = dataset[1]
    filter_graph = Data(
        edge_index=torch.cat([valid.edge_index, valid.target_edge_index], dim=1),
        edge_type=torch.cat([valid.edge_type, valid.target_edge_type]),
        num_nodes=valid.num_nodes)
    return valid, filter_graph


@torch.no_grad()
def validate_graph(model, graph, filter_graph, store, batch_size,
                   num_samples, seed) -> float:
    """Filtered MRR on (a subsample of) one graph's validation split, scored
    exactly as incite/run.py::evaluate scores."""
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
        t_pred = model(graph, t_batch, support=store)
        h_pred = model(graph, h_batch, support=store)
        t_mask, h_mask = tasks.strict_negative_mask(filter_graph, batch)
        rankings.append(tasks.compute_ranking(t_pred, pos_t_index, t_mask))
        rankings.append(tasks.compute_ranking(h_pred, pos_h_index, h_mask))
    ranking = torch.cat(rankings).float()
    return float((1 / ranking).mean())


def save_checkpoint(path, model, step, dev, seed):
    state = {"model": model.state_dict(), "step": int(step),
             "dev10": dev, "seed": int(seed)}
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--graphs", default=None,
                        help="comma list of pretrain mix names; default: the "
                             "config's train.graphs")
    parser.add_argument("--gpus", default="[0]")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--raw_root", default="/kgfm-src/data/raw/ultra-pretrain")
    parser.add_argument("--dev_root", default="/kgfm-src/data/roots/trix",
                        help="processed root holding the DEV10 suite graphs "
                             "(shared with the TRIX baseline)")
    parser.add_argument("--dev_graphs", default=None,
                        help="comma list of suite ids; default suite.DEV10")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        cfg = EasyDict(yaml.safe_load(handle))
    tcfg = cfg.train
    steps = args.steps if args.steps is not None else int(tcfg.steps)
    val_interval = (args.val_interval if args.val_interval is not None
                    else int(tcfg.val_interval))
    val_samples = (args.val_samples if args.val_samples is not None
                   else int(tcfg.val_samples))
    batch_size = int(tcfg.batch_size)
    # The 16 GiB GPU cannot backprop the full recipe batch in one piece; the
    # recipe's effective batch is batch_size * accum_steps micro-batches from
    # the same drawn graph, averaged before the optimizer step.
    accum = int(getattr(tcfg, "accum_steps", 1) or 1)
    num_negative = int(tcfg.num_negative)
    adv_temperature = float(tcfg.adversarial_temperature)
    strict = bool(tcfg.strict_negative)
    lam = float(cfg.relation.get("lambda", 0.0))
    mix_names = args.graphs.split(",") if args.graphs else list(tcfg.graphs)
    dev_ids = (args.dev_graphs.split(",") if args.dev_graphs
               else list(suite.DEV10))

    torch.manual_seed(args.seed)
    gpus = None if args.gpus in ("null", "None") else util.literal_eval(args.gpus)
    device = torch.device(gpus[0]) if gpus else torch.device("cpu")

    # ---- pretraining mix --------------------------------------------------
    pretrain_root = os.path.join(args.data_root, "pretrain")
    prepare_pretrain_root(pretrain_root, args.raw_root, mix_names)
    t0 = time.perf_counter()
    trains, valids, tests = load_mix_graphs(pretrain_root, mix_names)
    print("mix graphs ready in %.1fs" % (time.perf_counter() - t0))
    train_graphs = []
    for name, trg in zip(mix_names, trains):
        trg.num_relations = int(trg.num_relations)
        train_graphs.append(trg.to(device))
        print("%s: %d nodes, %d relation ids, %d train targets" % (
            name, trg.num_nodes, trg.num_relations,
            trg.target_edge_index.shape[1]))

    # ---- DEV10 (zero-shot checkpoint selection) ---------------------------
    dev = []
    for gid in dev_ids:
        valid, filt = load_dev_graph(args.dev_root, gid)
        valid.num_relations = int(valid.num_relations)
        dev.append((gid, valid.to(device), filt.to(device)))
        print("dev graph %s: %d nodes, %d valid targets" % (
            gid, valid.num_nodes, valid.target_edge_index.shape[1]))

    # ---- model ------------------------------------------------------------
    model = build_model(cfg).to(device)
    if bool(tcfg.get("checkpoint_activations", False)):
        # per-round activation checkpointing (results/incite/config_diff.md):
        # trades one trunk recompute in backward for the retained (b, V, d)
        # activations that OOM batch 16+. Default off; the launched phase-1
        # recipe ran without it and stays reproducible as launched.
        model.checkpoint_activations = True
        print("activation checkpointing: on")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        print("resumed from %s (step %s)" % (args.resume, state.get("step")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg.lr))

    # ---- support stores (training graphs), refreshed on an interval -------
    stores, refreshers = [None] * len(train_graphs), []
    if cfg.support.enabled:
        model.eval()  # rows must come from the same eval-mode forward that
        # scores them; train mode would drop the sampled edge twice
        kwargs = support_build_kwargs(cfg)
        for name, g in zip(mix_names, train_graphs):
            t0 = time.perf_counter()
            store = incite_support.build_support(g, model, seed=args.seed, **kwargs)
            stores[mix_names.index(name)] = store
            refreshers.append(incite_support.SupportRefresher(
                g, model, store, build_kwargs=kwargs,
                refresh_interval=int(tcfg.refresh_interval),
                cost_gate=float(tcfg.cost_gate)))
            print("support %s: %d relation ids in %.1fs" % (
                name, len(store), time.perf_counter() - t0))

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "pretrain.jsonl")
    log = open(log_path, "a")

    def validate_all(step):
        model.eval()
        per_graph = {}
        for gid, vg, fg in dev:
            store = None
            if cfg.support.enabled:
                # fresh store per validation: the weights moved. This is the
                # honest (and expensive) protocol; phase 1 runs support-off.
                store = incite_support.build_support(
                    vg, model, seed=args.seed, **support_build_kwargs(cfg))
            per_graph[gid] = round(validate_graph(
                model, vg, fg, store, batch_size, val_samples, args.seed), 4)
        model.train()
        groups = suite.dev10_by_group()
        group_means = {}
        for group, gids in groups.items():
            vals = [per_graph[g] for g in gids if g in per_graph]
            if vals:
                group_means[group] = round(sum(vals) / len(vals), 4)
        # the selection scalar: unweighted mean of the group means. Reported
        # quantities are the group means; this number only orders checkpoints.
        selection = round(sum(group_means.values()) / len(group_means), 4)
        entry = {"step": step, "dev10": per_graph,
                 "dev10_groups": group_means, "selection": selection}
        print(json.dumps(entry))
        log.write(json.dumps(entry) + "\n")
        log.flush()
        return selection, {"per_graph": per_graph, "groups": group_means,
                           "selection": selection}

    # TRIX's multigraph alternation: a graph per step, probability
    # proportional to its edge count (pretrain_entity.py::multigraph_collator)
    probs = torch.tensor([float(g.edge_index.shape[1]) for g in train_graphs])
    probs /= probs.sum()
    pick_gen = torch.Generator().manual_seed(args.seed + 1)
    pos_gen = torch.Generator().manual_seed(args.seed)

    model.train()
    best_sel, best_path = float("-inf"), None
    last_path = os.path.join(args.output_dir, "incite_last.pth")
    train_seconds = 0.0
    for step in range(1, steps + 1):
        gid = int(torch.multinomial(probs, 1, generator=pick_gen))
        graph, store = train_graphs[gid], stores[gid]
        t0 = time.perf_counter()
        optimizer.zero_grad()
        micro_triples, loss, loss_rel = [], 0.0, None
        for micro in range(accum):
            triples = incite_train.sample_positive_triples(
                graph, batch_size, pos_gen)
            micro_triples.append(triples)
            micro_loss = incite_train.entity_loss_from_triples(
                model, graph, triples, num_negative,
                adversarial_temperature=adv_temperature, strict=strict,
                support=store, walk_offset=step * accum + micro,
                sampler=tasks.negative_sampling)
            if lam > 0:
                micro_rel = incite_train.relation_loss_from_triples(
                    model, graph, triples, support=store,
                    walk_offset=step * accum + micro)
                micro_loss = micro_loss + lam * micro_rel
                loss_rel = (0.0 if loss_rel is None else loss_rel) \
                    + float(micro_rel) / accum
            (micro_loss / accum).backward()
            loss += float(micro_loss) / accum
        triples = torch.cat(micro_triples)
        optimizer.step()
        dt = time.perf_counter() - t0
        train_seconds += dt
        if refreshers:
            # touch only what the step drew: positives and their inverses;
            # the refresh itself runs under eval so refreshed rows match the
            # eval-mode forward that scores against them
            num_direct = int(graph.num_relations) // 2
            rids = triples[:, 2].unique().tolist()
            refreshers[gid].touch(
                rids + [r + num_direct if r < num_direct else r - num_direct
                        for r in rids])
            model.eval()
            refreshers[gid].after_step(dt)
            model.train()
        if args.log_every and step % args.log_every == 0:
            entry = {"step": step, "graph": mix_names[gid],
                     "loss": round(float(loss), 4),
                     "loss_rel": (None if loss_rel is None
                                  else round(float(loss_rel), 4)),
                     "it_per_s": round(step / train_seconds, 2)}
            print(json.dumps(entry))
            log.write(json.dumps(entry) + "\n")
            log.flush()
        if step % val_interval == 0 or step == steps:
            sel, dev_report = validate_all(step)
            save_checkpoint(last_path, model, step, dev_report, args.seed)
            if sel > best_sel:
                best_sel = sel
                best_path = os.path.join(args.output_dir, "incite_best.pth")
                save_checkpoint(best_path, model, step, dev_report, args.seed)
                print("new best: selection %.4f at step %d -> %s"
                      % (sel, step, best_path))

    summary = {"steps": steps, "it_per_s": round(steps / train_seconds, 2),
               "best_selection": None if best_path is None else round(best_sel, 4),
               "best_checkpoint": best_path, "last_checkpoint": last_path}
    print(json.dumps(summary))
    log.write(json.dumps(summary) + "\n")
    log.close()

    # prove the round trip: the file just written must load into a fresh model
    if best_path:
        probe = build_model(cfg)
        probe.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
        print("checkpoint reload OK: %s" % best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
