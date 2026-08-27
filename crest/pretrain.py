"""Container driver: pretrain CREST's readout (stage A) or the full model
(stage B) on the TRIX pretraining mix.

CONTAINER-ONLY, like ``crest/run.py``: this module imports TRIX from the
patched tree and PyG, so the host test suite never touches it.

The recipe is TRIX's own pretraining (``repos/trix/src/pretrain_entity.py``
with ``config/pretrain_entity.yaml``), with the encoder wrapped by CREST:

* graphs: FB15k237 + WN18RR + CoDExMedium, loaded through TRIX's own dataset
  classes (``JointDataset.datasets_map``) under ``data/roots/crest/pretrain``;
  the raw files land there by copy from ``data/raw/ultra-pretrain``, whose
  layout is already exactly what the loaders expect;
* alternation: one graph per step, sampled with probability proportional to
  its edge count -- TRIX's ``multigraph_collator``, restated;
* objective: TRIX's ``tasks.negative_sampling`` (strict) and its
  self-adversarial BCE at ``adversarial_temperature`` from the config,
  applied to ``s = s_v0 + s_pfn`` (``crest/train.py``), one encoder forward
  per batch;
* validation: every ``val_interval`` steps on ``val_samples`` triples of each
  graph's validation split (TRIX's ``fast_test``), filtered over the union of
  the graph's own target splits exactly as ``pretrain_entity.py`` builds its
  ``filtered_data``. Best checkpoint by mean MRR over the graphs.

Stage A freezes the encoder, so the banks stay valid for the whole stage and
the checkpoint carries the readout alone (``{"readout": ...}``, the shape
``crest/run.py --readout_ckpt`` loads). Stage B unfreezes the encoder at
``encoder_lr_scale`` times the base rate, refreshes banks through
``BankRefresher`` (touched relation ids only, 20% cost gate), and saves the
full model as well (``{"model": ..., "readout": ...}`` -- the ``model`` key
is TRIX's own checkpoint shape, so a stage-B file also serves as ``--ckpt``).

    usage (inside the container / prepared workdir):
      python -m crest.pretrain -c configs/crest_v1.yaml --stage a \
          --gpus "[0]" --ckpt entity_prediction.pth \
          --data_root /kgfm-src/data/roots/crest \
          --raw_root /kgfm-src/data/raw/ultra-pretrain \
          --output_dir /kgfm-src/output/crest-pretrain \
          [--graphs FB15k237,WN18RR,CoDExMedium | WN18RRInductive:v1,...] \
          [--steps N] [--val_interval N] [--val_samples N] [--seed 1024]

``--graphs`` also accepts ``Dataset:version`` suite specs, which load through
``util.build_dataset`` like ``crest/run.py`` and train on the *train* split's
graph -- that is the single-graph smoke-test path, not the pretraining mix.
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
from torch import nn

TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
sys.path.insert(0, os.path.join(TRIX_ROOT, "src"))

from trix import tasks, util  # noqa: E402
from trix.models_entity import TRIX  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

try:
    from . import bank as crest_bank
    from . import train as crest_train
    from .model import CRESTEntity
    from .pfn import Readout
    from .run import TrixEntityAdapter, sha256_file
except ImportError:  # pragma: no cover - flat invocation
    import bank as crest_bank  # type: ignore
    import train as crest_train  # type: ignore
    from model import CRESTEntity  # type: ignore
    from pfn import Readout  # type: ignore
    from run import TrixEntityAdapter, sha256_file  # type: ignore

#: The three pretraining graphs' top-level raw directories. The layout under
#: data/raw/ultra-pretrain is byte-for-byte what the TRIX loaders expect
#: relative to their root (RelLinkPredDataset wants fb15k237/FB15k-237/raw,
#: WordNet18RR wants wn18rr/raw, CoDEx wants codex-m/raw), so preparing the
#: pretrain root is a plain copy of each top-level directory.
PRETRAIN_RAW_DIRS = {
    "FB15k237": "fb15k237",
    "WN18RR": "wn18rr",
    "CoDExMedium": "codex-m",
}


def prepare_pretrain_root(pretrain_root: str, raw_root: str, names) -> None:
    """Copy the raw files the TRIX loaders will read under ``pretrain_root``.

    Copies rather than symlinks: an absolute symlink into the bind-mounted
    source tree would dangle when the root is read outside the container.
    Writes only under the CREST data root, per the data discipline.
    """
    for name in names:
        top = PRETRAIN_RAW_DIRS[name]
        src, dst = os.path.join(raw_root, top), os.path.join(pretrain_root, top)
        if os.path.isdir(dst):
            continue
        if not os.path.isdir(src):
            raise FileNotFoundError(
                "raw files for %s expected at %s; fetch data/raw/ultra-pretrain "
                "first" % (name, src))
        os.makedirs(pretrain_root, exist_ok=True)
        shutil.copytree(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)


def load_mix_graphs(pretrain_root: str, names):
    """(train, valid, test) Data lists for the mix, through TRIX's loaders.

    The FB15k237/WN18RR loader functions rerun ``build_relation_graph`` on
    every call -- a Python loop over half a million edges -- so the collated
    result is cached once under the pretrain root and reused. The cache holds
    exactly what ``JointDataset.process`` would have saved.
    """
    from trix import datasets as trix_datasets
    cache = os.path.join(pretrain_root, "crest_mix_%s.pt" % "-".join(names))
    if os.path.exists(cache):
        return torch.load(cache, map_location="cpu")
    graphs = [trix_datasets.JointDataset.datasets_map[n](root=pretrain_root)
              for n in names]
    splits = ([g[0] for g in graphs], [g[1] for g in graphs], [g[2] for g in graphs])
    torch.save(splits, cache + ".tmp")
    os.replace(cache + ".tmp", cache)
    return splits


def load_suite_graph(data_root: str, spec: str):
    """(train, valid, test) for one ``Dataset[:version]`` suite spec."""
    name, _, version = spec.partition(":")
    cfg = {"class": name, "root": data_root}
    if version:
        cfg["version"] = util.literal_eval(version)
    dataset = util.build_dataset(EasyDict({"dataset": cfg}))
    return dataset[0], dataset[1], dataset[2]


def mix_filter(train_g, valid_g, test_g) -> Data:
    """pretrain_entity.py's filtered_data: the union of the target splits."""
    return Data(
        edge_index=torch.cat([train_g.target_edge_index, valid_g.target_edge_index,
                              test_g.target_edge_index], dim=1),
        edge_type=torch.cat([train_g.target_edge_type, valid_g.target_edge_type,
                             test_g.target_edge_type]),
        num_nodes=train_g.num_nodes)


def suite_filter(valid_g) -> Data:
    """For a suite-spec smoke run: filter over the validation graph's message
    edges and its own targets. The mix path above is the reference protocol;
    this one only exists so single-graph smokes report an honest MRR."""
    return Data(
        edge_index=torch.cat([valid_g.edge_index, valid_g.target_edge_index], dim=1),
        edge_type=torch.cat([valid_g.edge_type, valid_g.target_edge_type]),
        num_nodes=valid_g.num_nodes)


@torch.no_grad()
def validate_graph(crest_model, adapter, graph, filter_graph, ctx_bank,
                   batch_size, chunk_size, num_samples, seed) -> float:
    """Filtered MRR on (a subsample of) one graph's validation split,
    scored exactly the way crest/run.py::evaluate scores: encoder batch
    forward, residual added, TRIX's own masks and ranking."""
    triplets = torch.cat(
        [graph.target_edge_index, graph.target_edge_type.unsqueeze(0)]).t()
    if num_samples and triplets.shape[0] > num_samples:
        gen = torch.Generator().manual_seed(int(seed))
        keep = torch.randperm(triplets.shape[0], generator=gen)[:num_samples]
        triplets = triplets[keep.to(triplets.device)]
    num_direct = int(graph.num_relations) // 2
    rankings = []
    for start in range(0, triplets.shape[0], batch_size):
        batch = triplets[start:start + batch_size]
        t_batch, h_batch = tasks.all_negative(graph, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        x, z, s0 = adapter.encode_batch(graph, t_batch)
        t_pred = crest_model.score(x, z, s0, pos_r_index, ctx_bank,
                                   chunk_size=chunk_size)
        x, z, s0 = adapter.encode_batch(graph, h_batch)
        h_pred = crest_model.score(x, z, s0, pos_r_index + num_direct, ctx_bank,
                                   chunk_size=chunk_size)
        t_mask, h_mask = tasks.strict_negative_mask(filter_graph, batch)
        rankings.append(tasks.compute_ranking(t_pred, pos_t_index, t_mask))
        rankings.append(tasks.compute_ranking(h_pred, pos_h_index, h_mask))
    ranking = torch.cat(rankings).float()
    return float((1 / ranking).mean())


def save_checkpoint(path, stage, readout, trix_model, step, val_mrr, seed):
    state = {"readout": readout.state_dict(), "step": int(step),
             "val_mrr": val_mrr, "seed": int(seed), "stage": stage}
    if stage == "b":
        # TRIX's own checkpoint key, so a stage-B file also serves as --ckpt
        state["model"] = trix_model.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--stage", choices=("a", "b"), default="a")
    parser.add_argument("--graphs", default=None,
                        help="comma list; pretrain names (FB15k237) or suite "
                             "specs (WN18RRInductive:v1). Default: the "
                             "config's train.graphs mix")
    parser.add_argument("--gpus", default="[0]")
    parser.add_argument("--ckpt", required=True, help="TRIX entity_prediction.pth "
                        "(or a stage-B checkpoint, which carries the same key)")
    parser.add_argument("--readout_ckpt", default=None,
                        help="resume/init for the readout; stage B starts "
                             "from the stage-A best")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--raw_root", default="/kgfm-src/data/raw/ultra-pretrain")
    parser.add_argument("--bank_root", default=None,
                        help="default: <data_root>/banks")
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
    num_negative = int(tcfg.num_negative)
    adv_temperature = float(tcfg.adversarial_temperature)
    strict = bool(tcfg.strict_negative)
    bank_root = args.bank_root or os.path.join(args.data_root, "banks")
    graph_names = (args.graphs.split(",") if args.graphs
                   else list(tcfg.graphs))

    torch.manual_seed(args.seed)
    gpus = util.literal_eval(args.gpus)
    device = torch.device(gpus[0]) if gpus else torch.device("cpu")

    # ---- graphs -----------------------------------------------------------
    mix_names = [n for n in graph_names if n in PRETRAIN_RAW_DIRS]
    suite_specs = [n for n in graph_names if n not in PRETRAIN_RAW_DIRS]
    train_graphs, valid_graphs, filters, bank_ids = [], [], [], []
    if mix_names:
        pretrain_root = os.path.join(args.data_root, "pretrain")
        prepare_pretrain_root(pretrain_root, args.raw_root, mix_names)
        t0 = time.perf_counter()
        trains, valids, tests = load_mix_graphs(pretrain_root, mix_names)
        print("mix graphs ready in %.1fs" % (time.perf_counter() - t0))
        for name, trg, vag, teg in zip(mix_names, trains, valids, tests):
            train_graphs.append(trg)
            valid_graphs.append(vag)
            filters.append(mix_filter(trg, vag, teg))
            bank_ids.append("pretrain:%s" % name)
    for spec in suite_specs:
        trg, vag, _ = load_suite_graph(args.data_root, spec)
        train_graphs.append(trg)
        valid_graphs.append(vag)
        filters.append(suite_filter(vag))
        # keyed off the *train* graph: run.py's banks for the same spec come
        # from the inference graph and must never be conflated with these
        bank_ids.append("%s:train" % spec)
    names = mix_names + suite_specs

    for i in range(len(train_graphs)):
        # collated PyG attributes come back as 0-dim tensors; everything
        # downstream indexes with ints
        train_graphs[i].num_relations = int(train_graphs[i].num_relations)
        valid_graphs[i].num_relations = int(valid_graphs[i].num_relations)
        train_graphs[i] = train_graphs[i].to(device)
        valid_graphs[i] = valid_graphs[i].to(device)
        filters[i] = filters[i].to(device)
        print("%s: %d nodes, %d relation ids, %d train targets" % (
            names[i], train_graphs[i].num_nodes, train_graphs[i].num_relations,
            train_graphs[i].target_edge_index.shape[1]))

    # ---- model ------------------------------------------------------------
    model = TRIX(rel_model_cfg=cfg.trix.model.relation_model,
                 entity_model_1_cfg=cfg.trix.model.entity_model_1,
                 entity_model_2_cfg=cfg.trix.model.entity_model_2)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device)
    adapter = TrixEntityAdapter(model)

    d = cfg.trix.model.relation_model.input_dim
    readout = Readout(crest_bank.row_dim(d), width=cfg.readout.width,
                      depth=cfg.readout.depth, heads=cfg.readout.heads)
    if args.readout_ckpt:
        readout.load_state_dict(
            torch.load(args.readout_ckpt, map_location="cpu")["readout"])
    readout = readout.to(device)
    crest_model = CRESTEntity(adapter, readout)

    # ---- banks (disk-cached, built from each *training* graph) ------------
    ckpt_hash = sha256_file(args.ckpt)
    banks, bank_seconds = [], {}
    model.eval()  # bank rows must come from the same eval-mode forward
                  # crest/run.py scores with; train mode would drop the
                  # sampled edge twice and skew s0
    for name, graph, bank_id in zip(names, train_graphs, bank_ids):
        t0 = time.perf_counter()
        ctx_bank = crest_bank.ContextBank.load_or_build(
            bank_root,
            lambda empty, g=graph: crest_bank.build_bank_entity(
                g, adapter, seed=args.seed,
                num_positive=cfg.bank.num_positive,
                neg_per_pos=cfg.bank.neg_per_pos, bank=empty),
            bank_id, ckpt_hash, args.seed)
        ctx_bank.to(device)
        dt = time.perf_counter() - t0
        bank_seconds[name] = round(dt, 1)
        banks.append(ctx_bank)
        print("bank %s: %d relation ids in %.1fs" % (bank_id, len(ctx_bank), dt))

    # ---- optimizer, freezing ----------------------------------------------
    if args.stage == "a":
        for p in model.parameters():
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(readout.parameters(), lr=float(tcfg.lr))
        refreshers = None
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.AdamW([
            {"params": list(readout.parameters()), "lr": float(tcfg.lr)},
            {"params": list(model.parameters()),
             "lr": float(tcfg.lr) * float(tcfg.encoder_lr_scale)},
        ])
        refreshers = [crest_train.BankRefresher(
            g, adapter, b, refresh_interval=int(tcfg.refresh_interval),
            cost_gate=float(tcfg.cost_gate)) for g, b in zip(train_graphs, banks)]

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_stage%s.jsonl" % args.stage)
    log = open(log_path, "a")

    def validate_all(step):
        model.eval()
        crest_model.eval()
        per_graph = {}
        for name, vg, fg, b in zip(names, valid_graphs, filters, banks):
            per_graph[name] = round(validate_graph(
                crest_model, adapter, vg, fg, b, cfg.trix.batch_size,
                cfg.chunk_size, val_samples, args.seed), 4)
        model.train()
        crest_model.train()
        mean = sum(per_graph.values()) / len(per_graph)
        entry = {"step": step, "val_mrr": per_graph, "mean_mrr": round(mean, 4)}
        print(json.dumps(entry))
        log.write(json.dumps(entry) + "\n")
        log.flush()
        return mean

    # TRIX's multigraph alternation: a graph per step, probability
    # proportional to its edge count (pretrain_entity.py::multigraph_collator)
    probs = torch.tensor([float(g.edge_index.shape[1]) for g in train_graphs])
    probs /= probs.sum()
    pick_gen = torch.Generator().manual_seed(args.seed + 1)
    pos_gen = torch.Generator().manual_seed(args.seed)

    # train mode: TRIX's easy-edge removal lives behind self.training, and
    # the objective is meaningless without it (the answer is an edge)
    model.train()
    crest_model.train()
    best_mrr, best_path = float("-inf"), None
    last_path = os.path.join(args.output_dir, "crest_stage%s_last.pth" % args.stage)
    train_seconds = 0.0
    for step in range(1, steps + 1):
        gid = int(torch.multinomial(probs, 1, generator=pick_gen))
        graph, ctx_bank = train_graphs[gid], banks[gid]
        t0 = time.perf_counter()
        optimizer.zero_grad()
        triples = crest_train.sample_positive_triples(graph, batch_size, pos_gen)
        loss = crest_train.entity_loss_from_triples(
            crest_model, graph, ctx_bank, triples, num_negative,
            adversarial_temperature=adv_temperature, strict=strict,
            encoder_no_grad=(args.stage == "a"),
            sampler=tasks.negative_sampling)
        loss.backward()
        optimizer.step()
        dt = time.perf_counter() - t0
        train_seconds += dt
        if refreshers is not None:
            num_direct = int(graph.num_relations) // 2
            rids = triples[:, 2].unique().tolist()
            refreshers[gid].touch(rids + [r + num_direct for r in rids])
            model.eval()  # refreshed rows must match eval-mode scoring too
            refreshers[gid].after_step(dt)
            model.train()
        if args.log_every and step % args.log_every == 0:
            entry = {"step": step, "graph": names[gid],
                     "loss": round(float(loss), 4),
                     "it_per_s": round(step / train_seconds, 2)}
            print(json.dumps(entry))
            log.write(json.dumps(entry) + "\n")
            log.flush()
        if step % val_interval == 0 or step == steps:
            mean = validate_all(step)
            save_checkpoint(last_path, args.stage, readout, model, step,
                            mean, args.seed)
            if mean > best_mrr:
                best_mrr = mean
                best_path = os.path.join(
                    args.output_dir, "crest_stage%s_best.pth" % args.stage)
                save_checkpoint(best_path, args.stage, readout, model, step,
                                mean, args.seed)
                print("new best: mean MRR %.4f at step %d -> %s"
                      % (mean, step, best_path))

    if refreshers is not None:
        merged = {n: r.log for n, r in zip(names, refreshers)}
        with open(os.path.join(args.output_dir, "phase2_cost.json"), "w") as fh:
            json.dump(merged, fh, indent=2)

    summary = {"stage": args.stage, "steps": steps,
               "it_per_s": round(steps / train_seconds, 2),
               "bank_build_seconds": bank_seconds,
               "best_mean_mrr": None if best_path is None else round(best_mrr, 4),
               "best_checkpoint": best_path, "last_checkpoint": last_path}
    print(json.dumps(summary))
    log.write(json.dumps(summary) + "\n")
    log.close()

    # prove the round trip: the file just written must load into a fresh
    # readout without key or shape complaints
    if best_path:
        probe = Readout(crest_bank.row_dim(d), width=cfg.readout.width,
                        depth=cfg.readout.depth, heads=cfg.readout.heads)
        probe.load_state_dict(torch.load(best_path, map_location="cpu")["readout"])
        print("checkpoint reload OK: %s" % best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
