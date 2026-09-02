"""Container driver: evaluate INCITE on one graph, dump ranks, write its CSV.

CONTAINER-ONLY. Imports the patched TRIX tree (datasets, tasks, rank_dump)
from ``TRIX_ROOT``; the skeleton is crest/run.py (crest branch) -- same
dataset construction, same filtered-data logic, same evaluation loop as
``repos/trix/src/run_entity.py::test`` (same DistributedSampler, same
``all_negative`` expansion, TRIX's own strict masks and ``compute_ranking``)
with INCITE's forward supplying the scores. Support precompute runs inside
this process, hence inside the timed section of scripts/run_incite.sh --
the PLAN's amendment: a cost claim that excludes precompute is the kind
this harness exists to catch.

    usage (inside the container / prepared workdir):
      python -m incite.run -c configs/incite_v1.yaml --dataset Metafam \
          --version Metafam --gpus "[0]" --ckpt out/incite_best.pth \
          --data_root /kgfm-src/data/roots/trix --output_dir /kgfm/output/incite \
          [--support build|skip] [--task entity|relation] [--seed 1024]

``--ckpt none`` runs RANDOM weights -- smoke-testing the dump path only;
the CSV row and the printed banner both say so, and such ranks must never
remain under ranks/incite/ (task rule).

Rank dump directory arrives via ``INCITE_RANK_DUMP_DIR``, mirroring TRIX.
Relation-task dumps (``--task relation``) write ``direction == "relation"``
rows with the UNFILTERED protocol (rank = #candidates scoring >= the true
relation, target's own tie included, no +1 -- TRIX's
``compute_ranking_relation`` without mask; the baseline protocol recorded
in shared/published.json trix.relation).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import time

import torch
import yaml
from easydict import EasyDict

TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
sys.path.insert(0, os.path.join(TRIX_ROOT, "src"))

from trix import tasks, util  # noqa: E402
from trix.rank_dump import Dumper  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch.utils import data as torch_data  # noqa: E402

try:
    from . import support as incite_support
    from .model import INCITE
    from .rerank import ScoreEnsemble, rerank_predictions
except ImportError:  # pragma: no cover - flat invocation
    import support as incite_support  # type: ignore
    from model import INCITE  # type: ignore
    from rerank import ScoreEnsemble, rerank_predictions  # type: ignore

#: state_dict prefixes of lever modules a config may leave unbuilt; a
#: checkpoint carrying them still loads as an ensemble member (the trunk
#: and the score head are what the entity forward uses)
LEVER_PREFIXES = ("readout.", "walk_module.", "unary_mlp.")


def load_members(cfg: EasyDict, ckpt_paths):
    """Build one model per checkpoint path from ``cfg``; strict on the trunk.

    Returns (module, hashes). A single path returns the plain model; several
    return a ``ScoreEnsemble`` over them (support-off by construction).
    """
    members, hashes = [], []
    for path in ckpt_paths:
        model = build_model(cfg)
        # weights_only=False (2026-09-02): torch 2.6 flipped the default and
        # our checkpoints carry config dicts beside the tensors; they are our
        # own files, the cu128 stack is the first to hit this
        state = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        bad_missing = [k for k in missing if not k.startswith(LEVER_PREFIXES)]
        bad_unexpected = [k for k in unexpected if not k.startswith(LEVER_PREFIXES)]
        assert not bad_missing, "%s: trunk tensors missing: %s" % (path, bad_missing[:5])
        assert not bad_unexpected, "%s: tensors with no home: %s" % (path, bad_unexpected[:5])
        if missing or unexpected:
            print("%s: %d lever tensors at init, %d ignored" % (
                os.path.basename(path), len(missing), len(unexpected)))
        members.append(model)
        hashes.append(sha256_file(path))
    if len(members) == 1:
        return members[0], hashes
    print("score ensemble of %d checkpoints" % len(members))
    return ScoreEnsemble(members), hashes


def build_model(cfg: EasyDict) -> INCITE:
    """One place that turns a config into a model, shared with pretrain.py."""
    walks = None
    if cfg.walks.enabled:
        walks = dict(num_walks=int(cfg.walks.num_walks),
                     walk_length=int(cfg.walks.walk_length),
                     seed=int(cfg.walks.seed))
    return INCITE(dim=int(cfg.model.dim), rounds=int(cfg.model.rounds),
                  layer_norm=bool(cfg.model.layer_norm),
                  short_cut=bool(cfg.model.short_cut),
                  count_channel=bool(cfg.model.count_channel),
                  walks=walks,
                  support_readout=bool(cfg.support.enabled),
                  support_k=int(cfg.support.num_positive),
                  num_mlp_layer=int(cfg.model.num_mlp_layer),
                  unary=bool(cfg.model.get("unary", False)))


def support_build_kwargs(cfg: EasyDict) -> dict:
    return dict(per_relation_cap=int(cfg.support.per_relation_cap),
                neg_per_pos=int(cfg.support.neg_per_pos),
                prototype_k=int(cfg.support.prototype_k),
                hops=int(cfg.support.hops),
                ball_cap=int(cfg.support.ball_cap),
                class_prior=float(cfg.support.class_prior),
                build_batch_size=int(cfg.support.build_batch_size))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:16]


def build_filtered_data(cfg, dataset, train_data, valid_data, test_data):
    """Verbatim logic of run_entity.py __main__: which graph filters ranks."""
    if cfg.task.name == "InductiveInference":
        if "ILPC" in cfg.dataset["class"] or "Ingram" in cfg.dataset["class"]:
            full_inference_edges = torch.cat(
                [valid_data.edge_index, valid_data.target_edge_index,
                 test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat(
                [valid_data.edge_type, valid_data.target_edge_type,
                 test_data.target_edge_type])
        else:
            full_inference_edges = torch.cat(
                [test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat(
                [test_data.edge_type, test_data.target_edge_type])
        return Data(edge_index=full_inference_edges,
                    edge_type=full_inference_etypes,
                    num_nodes=test_data.num_nodes)
    return Data(edge_index=dataset._data.target_edge_index,
                edge_type=dataset._data.target_edge_type,
                num_nodes=dataset[0].num_nodes)


@torch.no_grad()
def evaluate(cfg, model, test_data, filtered_data, store, dump_spec, metric_list,
             rerank_k: int = 0, rerank_weight: float = 1.0, rerank_chunk: int = 32):
    """run_entity.py::test with INCITE's forward supplying the scores.

    ``rerank_k > 0`` applies bidirectional re-ranking (incite/rerank.py) to
    the top-k eligible candidates of each direction before ranking.
    """
    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, 1, 0)
    test_loader = torch_data.DataLoader(test_triplets, cfg.batch_size, sampler=sampler)
    dumper = Dumper(dump_spec, test_triplets, 1, 0, cfg.batch_size) if dump_spec else None

    model.eval()
    rankings, num_negatives = [], []
    tail_rankings, num_tail_negs = [], []
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        t_pred = model(test_data, t_batch, support=store)
        h_pred = model(test_data, h_batch, support=store)

        t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        if rerank_k > 0:
            t_pred = rerank_predictions(model, test_data, t_batch, t_pred,
                                        pos_t_index, t_mask, rerank_k,
                                        rerank_weight, rerank_chunk, support=store)
            h_pred = rerank_predictions(model, test_data, h_batch, h_pred,
                                        pos_h_index, h_mask, rerank_k,
                                        rerank_weight, rerank_chunk, support=store)
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        if dumper is not None:
            dumper.add(batch, t_ranking, h_ranking, num_t_negative, num_h_negative)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]
        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

    if dumper is not None:
        print("Rank dump written to %s" % dumper.write())

    ranking = torch.cat(rankings).float()
    num_negative = torch.cat(num_negatives)
    tail_ranking = torch.cat(tail_rankings).float()
    num_tail_neg = torch.cat(num_tail_negs)

    # the model's own metric values, computed the way TRIX computes them
    metrics = {}
    for metric in metric_list:
        name = metric
        _ranking, _num_neg = ranking, num_negative
        if "-tail" in metric:
            name, _ = metric.split("-")
            _ranking, _num_neg = tail_ranking, num_tail_neg
        if name == "mr":
            score = _ranking.mean()
        elif name == "mrr":
            score = (1 / _ranking).mean()
        elif name.startswith("hits@"):
            values = name[5:].split("_")
            threshold = int(values[0])
            if len(values) > 1:
                num_sample = int(values[1])
                fp_rate = (_ranking - 1) / _num_neg
                score = 0
                for i in range(threshold):
                    num_comb = (math.factorial(num_sample - 1) /
                                math.factorial(i) / math.factorial(num_sample - i - 1))
                    score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                score = score.mean()
            else:
                score = (_ranking <= threshold).float().mean()
        metrics[metric] = score
    return metrics


@torch.no_grad()
def evaluate_relation(cfg, model, test_data, store, dump_spec):
    """Relation prediction, UNFILTERED protocol (module docstring)."""
    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    num_direct = int(test_data.num_relations) // 2
    model.eval()
    columns = {"query_id": [], "h": [], "r": [], "t": [], "rank": []}
    rankings = []
    for start in range(0, test_triplets.shape[0], cfg.batch_size):
        batch = test_triplets[start:start + cfg.batch_size]
        pred = model.forward_relation(test_data, batch, support=store)
        pos = pred.gather(-1, batch[:, 2].unsqueeze(-1))
        # unfiltered: the target's own tie contributes the 1; no +1 offset
        ranking = torch.sum(pos <= pred, dim=-1)
        rankings.append(ranking)
        for i in range(len(batch)):
            columns["query_id"].append(start + i)
            columns["h"].append(int(batch[i, 0]))
            columns["t"].append(int(batch[i, 1]))
            columns["r"].append(int(batch[i, 2]))
            columns["rank"].append(int(ranking[i]))
    ranking = torch.cat(rankings).float()
    metrics = {"mrr": (1 / ranking).mean(), "mr": ranking.mean()}
    for k in (1, 3, 10):
        metrics["hits@%d" % k] = (ranking <= k).float().mean()

    if dump_spec:
        import pyarrow
        import pyarrow.parquet
        import suite  # shared/suite.py on PYTHONPATH

        graph = suite.by_run_id(dump_spec["dataset"])
        n = len(columns["rank"])
        arrow_types = {"string": pyarrow.string(), "int64": pyarrow.int64()}
        table = {
            "dataset": [graph.id] * n, "model": [dump_spec["model"]] * n,
            "seed": [dump_spec["seed"]] * n, "direction": ["relation"] * n,
            "query_id": columns["query_id"], "h": columns["h"],
            "r": columns["r"], "t": columns["t"], "rank": columns["rank"],
            "n_candidates": [num_direct - 1] * n,
        }
        schema = pyarrow.schema([(name, arrow_types[suite.RANK_COLUMN_TYPES[name]])
                                 for name in suite.RANK_COLUMNS])
        os.makedirs(dump_spec["dir"], exist_ok=True)
        path = os.path.join(dump_spec["dir"],
                            "%s.parquet" % graph.id.replace(":", "_"))
        pyarrow.parquet.write_table(
            pyarrow.table({k: table[k] for k in suite.RANK_COLUMNS}, schema=schema), path)
        print("Rank dump written to %s" % path)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", default="")
    parser.add_argument("--gpus", default="[0]")
    parser.add_argument("--ckpt", required=True,
                        help="INCITE checkpoint ({'model': state_dict}); the "
                             "literal string 'none' runs random weights "
                             "(smoke-testing the dump path only). A comma "
                             "list builds a score ensemble (incite/rerank.py)")
    parser.add_argument("--rerank_k", type=int, default=0,
                        help="bidirectional re-ranking of the top-k eligible "
                             "candidates per direction (0 = off)")
    parser.add_argument("--rerank_weight", type=float, default=1.0,
                        help="weight of the reverse-direction logit")
    parser.add_argument("--rerank_chunk", type=int, default=32,
                        help="reverse queries per trunk pass")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_name", choices=("InductiveInference", "TransductiveInference"),
                        default="InductiveInference")
    parser.add_argument("--task", choices=("entity", "relation"), default="entity")
    parser.add_argument("--support", choices=("build", "skip"), default="build")
    parser.add_argument("--support_root", default="/kgfm-src/data/roots/incite/support")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        cfg = EasyDict(yaml.safe_load(handle))

    version = util.literal_eval(args.version) if args.version else None
    # "null" spells CPU in every runner here; ast.literal_eval would hand the
    # string through and torch.device("null") does not exist
    gpus = None if args.gpus in ("null", "None") else util.literal_eval(args.gpus)
    data_cfg = EasyDict({"dataset": {"class": args.dataset, "root": args.data_root},
                         "train": {"gpus": gpus}})
    # `is not None`, not truthiness: literal_eval turns "0" into the integer 0
    # and NLIngram:0 is a real graph (see patches/trix/0001)
    if version is not None:
        data_cfg.dataset["version"] = version
    data_cfg.task = {"name": args.task_name}

    torch.manual_seed(args.seed)
    dataset = util.build_dataset(data_cfg)
    device = util.get_device(data_cfg)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    test_data.num_relations = int(test_data.num_relations)

    random_weights = args.ckpt.lower() == "none"
    if random_weights:
        model = build_model(cfg)
        print("WARNING: --ckpt none, scoring with RANDOM weights. This run "
              "only proves the dump path; its ranks must not be kept.")
        ckpt_hash = "random-seed%d" % args.seed
    else:
        paths = [p for p in args.ckpt.split(",") if p]
        model, hashes = load_members(cfg, paths)
        ckpt_hash = hashes[0] if len(hashes) == 1 else "ens%d-%s" % (
            len(hashes), hashlib.sha256("".join(hashes).encode()).hexdigest()[:16])
    model = model.to(device).eval()
    if args.rerank_k > 0:
        print("bidirectional re-ranking: k=%d weight=%g chunk=%d" % (
            args.rerank_k, args.rerank_weight, args.rerank_chunk))

    dataset_id = args.dataset + (":" + str(version) if version is not None else "")

    store = None
    if cfg.support.enabled and args.support == "build":
        assert not isinstance(model, ScoreEnsemble), "ensembles run support-off"
        t0 = time.perf_counter()
        store = incite_support.SupportStore.load_or_build(
            args.support_root,
            lambda empty: incite_support.build_support(
                test_data, model, seed=args.seed, store=empty,
                **support_build_kwargs(cfg)),
            dataset_id, ckpt_hash, args.seed)
        store.to(device)
        print("support ready in %.1fs (%d relation ids)" % (
            time.perf_counter() - t0, len(store)))

    dump_dir = os.environ.get("INCITE_RANK_DUMP_DIR")
    dump_spec = None if not dump_dir else {
        "dir": dump_dir, "dataset": dataset_id, "model": "incite",
        "seed": args.seed,
    }

    eval_cfg = EasyDict({"batch_size": int(cfg.train.batch_size)})
    if args.task == "entity":
        filtered_data = build_filtered_data(
            data_cfg, dataset, train_data, valid_data, test_data).to(device)
        metric_list = ["mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@10_50"]
        metrics = evaluate(eval_cfg, model, test_data, filtered_data, store,
                           dump_spec, metric_list, rerank_k=args.rerank_k,
                           rerank_weight=args.rerank_weight,
                           rerank_chunk=args.rerank_chunk)
    else:
        metrics = evaluate_relation(eval_cfg, model, test_data, store, dump_spec)
    for k, v in metrics.items():
        print("%s: %g" % (k, float(v)))

    # one timestamped CSV per invocation, the shape read_ultra_csv parses
    out_dir = os.path.join(args.output_dir, "INCITE_results")
    os.makedirs(out_dir, exist_ok=True)
    row = {"dataset": dataset_id}
    row.update({k: float(v) for k, v in metrics.items()})
    if random_weights:
        row["random_weights"] = 1
    if args.rerank_k > 0:
        row["rerank_k"] = args.rerank_k
        row["rerank_weight"] = args.rerank_weight
    if isinstance(model, ScoreEnsemble):
        row["ensemble_members"] = len(model.members)
    path = os.path.join(out_dir, "INCITE_results_%s.csv" % time.strftime("%Y-%m-%d-%H-%M-%S"))
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    print("Results written to %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
