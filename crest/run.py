"""Container driver: evaluate CREST on one graph, dump ranks, write its CSV.

CONTAINER-ONLY. This is the single module in ``crest/`` that imports TRIX
(from the patched tree ``scripts/prepare_crest_workdir.sh`` produces); the
host test suite never touches it.

The evaluation loop is ``repos/trix/src/run_entity.py::test`` line for line
-- same ``DistributedSampler``, same batch size, same ``all_negative``
expansion, same strict masks, TRIX's own ``compute_ranking`` -- with exactly
one insertion: CREST's residual is added to the score tensor before ranking.
That discipline is what makes phase 0's gate meaningful: with the readout's
last layer at zero the added residual is the constant 0 and every rank must
equal ``ranks/trix/`` row by row (``scripts/verify_crest_identity.py``).
Running at TRIX's own batch shape matters too -- GPU reductions are shape-
dependent in their low-order bits, and a near-tie flipped is a rank moved.

    usage (inside the container / prepared workdir):
      python -m crest.run -c configs/crest_v1.yaml --dataset FB15k237Inductive \
          --version v1 --gpus "[0]" --ckpt entity_prediction.pth \
          --data_root /kgfm/data/roots/crest --output_dir /kgfm/output/crest \
          [--readout_ckpt stageb.pth] [--bank build|skip] [--seed 1024]

Rank dump directory arrives via ``CREST_RANK_DUMP_DIR``, mirroring TRIX.
``--bank skip`` skips bank building; with the zero residual the scores are
unchanged, so phase 0 may use it to save the 20-edges-per-relation encoder
sweeps that a built bank costs. ``--bank build`` (default) exercises the full
path, cached under ``--bank_root`` by (graph, checkpoint hash, seed).
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

# the patched TRIX tree; the layout matches containers/crest/Dockerfile and
# prepare_crest_workdir.sh
TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
sys.path.insert(0, os.path.join(TRIX_ROOT, "src"))

from trix import tasks, util  # noqa: E402
from trix.models_entity import TRIX  # noqa: E402
from trix.rank_dump import Dumper  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch.utils import data as torch_data  # noqa: E402

try:
    from . import bank as crest_bank
    from .model import CRESTEntity
    from .pfn import Readout
except ImportError:  # pragma: no cover - flat invocation
    import bank as crest_bank  # type: ignore
    from model import CRESTEntity  # type: ignore
    from pfn import Readout  # type: ignore


class TrixEntityAdapter:
    """CREST's encoder protocol over a loaded TRIX model.

    ``encode_batch`` reproduces ``TRIX.forward`` call for call -- the same
    two submodule invocations, in the same order, on the same tensors -- and
    additionally keeps the per-node features and the query embedding that
    ``EntityNet.forward`` already computes. ``x`` is the last hidden state
    (``node_feature[..., :d]``) and ``z`` is the query embedding TRIX itself
    scored with (``node_feature[..., d:]``, constant over nodes), so nothing
    is recomputed and nothing can drift from what produced ``s0``.
    """

    def __init__(self, model: TRIX):
        self.model = model

    def encode_batch(self, data, batch):
        """``batch [b, c, 3]`` (a TRIX t_batch or h_batch) ->
        ``(x [b, n, d], z [b, d], s0 [b, c])``."""
        relation_representations = self.model.relation_model(
            data, batch, self.model.entity_model_1)
        out = self.model.entity_model_2(data, relation_representations, batch)
        feature = out["feature"]  # [b, num_nodes, 2d]
        d = feature.shape[-1] // 2
        x = feature[..., :d]
        z = feature[:, 0, d:]  # node_query is the same for every node
        return x, z, out["score"]

    def encode_single(self, graph, u, r):
        """The bank builder's single-query view: (u, r, ?) over all nodes."""
        n = graph.num_nodes
        device = graph.edge_index.device
        all_index = torch.arange(n, device=device)
        h_index = torch.full((1, n), int(u), device=device)
        r_index = torch.full((1, n), int(r), device=device)
        batch = torch.stack([h_index, all_index.unsqueeze(0), r_index], dim=-1)
        x, z, s0 = self.encode_batch(graph, batch)
        return x[0], z[0], s0[0]


def sha256_file(path: str, limit: int = 1 << 26) -> str:
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
def evaluate(cfg, crest_model, adapter, test_data, filtered_data, ctx_bank,
             dump_spec, chunk_size):
    """run_entity.py::test with the residual added before ranking."""
    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, 1, 0)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)
    dumper = Dumper(dump_spec, test_triplets, 1, 0, cfg.train.batch_size) if dump_spec else None
    num_direct = test_data.num_relations // 2

    crest_model.eval()
    rankings, num_negatives = [], []
    tail_rankings, num_tail_negs = [], []
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        # ---- the one departure from upstream: s = s_v0 + residual ---------
        x, z, s0 = adapter.encode_batch(test_data, t_batch)
        x_cand = x  # candidates of a t_batch are all nodes in id order
        t_pred = crest_model.score(x_cand, z, s0, pos_r_index, ctx_bank,
                                   chunk_size=chunk_size)
        x, z, s0 = adapter.encode_batch(test_data, h_batch)
        # head queries read the inverse relation's bank, mirroring TRIX's
        # negative_sample_to_tail; candidates are again all nodes in id order
        h_pred = crest_model.score(x, z, s0, pos_r_index + num_direct, ctx_bank,
                                   chunk_size=chunk_size)
        # -------------------------------------------------------------------

        t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
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
    # (torch reductions on device) -- this is what criterion A compares
    # shared/metrics.py against, so the arithmetic must be the model's own
    metrics = {}
    for metric in cfg.task.metric:
        name = metric
        _ranking, _num_neg = ranking, num_negative
        if "-tail" in metric:
            name, direction = metric.split("-")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", default="")
    parser.add_argument("--gpus", default="[0]")
    parser.add_argument("--ckpt", required=True, help="TRIX entity_prediction.pth")
    parser.add_argument("--readout_ckpt", default=None,
                        help="CREST readout state; omitted = zero residual (phase 0)")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task_name", choices=("InductiveInference", "TransductiveInference"),
                        default="InductiveInference",
                        help="which filtering scheme applies; run_crest.sh sets "
                             "this per suite group, as run_trix.sh does via its "
                             "two configs")
    parser.add_argument("--bank", choices=("build", "skip"), default="build")
    parser.add_argument("--bank_root", default="/kgfm/data/roots/crest/banks")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        crest_cfg = EasyDict(yaml.safe_load(handle))

    version = util.literal_eval(args.version) if args.version else None
    cfg = EasyDict({
        "dataset": {"class": args.dataset, "root": args.data_root},
        "model": crest_cfg.trix.model,
        "task": dict(crest_cfg.task, name=args.task_name),
        "train": {"gpus": util.literal_eval(args.gpus),
                  "batch_size": crest_cfg.trix.batch_size},
        "checkpoint": args.ckpt,
        "output_dir": args.output_dir,
    })
    # `is not None`, not truthiness: literal_eval turns "0" into the integer 0
    # and NLIngram:0 is a real graph (see patches/trix/0001)
    if version is not None:
        cfg.dataset["version"] = version

    torch.manual_seed(args.seed)
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    # all three move to the device: build_filtered_data concatenates valid and
    # test tensors, which must not straddle devices (run_entity.py moves all
    # three for the same reason)
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    model = TRIX(rel_model_cfg=cfg.model.relation_model,
                 entity_model_1_cfg=cfg.model.entity_model_1,
                 entity_model_2_cfg=cfg.model.entity_model_2)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    adapter = TrixEntityAdapter(model)

    d = cfg.model.relation_model.input_dim
    readout = Readout(crest_bank.row_dim(d),
                      width=crest_cfg.readout.width,
                      depth=crest_cfg.readout.depth,
                      heads=crest_cfg.readout.heads)
    if args.readout_ckpt:
        readout.load_state_dict(
            torch.load(args.readout_ckpt, map_location="cpu")["readout"])
    else:
        readout.zero_residual()  # phase 0: readout present, residual exactly 0
    readout = readout.to(device)
    crest_model = CRESTEntity(adapter, readout)

    dataset_id = args.dataset + (":" + str(version) if version is not None else "")
    ckpt_hash = sha256_file(args.ckpt)
    if args.readout_ckpt:
        ckpt_hash += "+" + sha256_file(args.readout_ckpt)

    if args.bank == "build":
        t0 = time.perf_counter()
        ctx_bank = crest_bank.ContextBank.load_or_build(
            args.bank_root,
            lambda empty: crest_bank.build_bank_entity(
                test_data, adapter, seed=args.seed,
                num_positive=crest_cfg.bank.num_positive,
                neg_per_pos=crest_cfg.bank.neg_per_pos,
                bank=empty),
            dataset_id, ckpt_hash, args.seed)
        ctx_bank.to(device)  # a cache hit loads CPU tensors
        print("bank ready in %.1fs (%d relation ids)" % (
            time.perf_counter() - t0, len(ctx_bank)))
    else:
        ctx_bank = crest_bank.ContextBank(dataset_id, ckpt_hash, args.seed)
        assert readout.residual_is_zero(), (
            "--bank skip with a live readout would silently score without "
            "context; skip is a phase 0 (zero residual) convenience only")

    filtered_data = build_filtered_data(cfg, dataset, train_data, valid_data,
                                        test_data).to(device)

    dump_spec = None if not os.environ.get("CREST_RANK_DUMP_DIR") else {
        "dir": os.environ["CREST_RANK_DUMP_DIR"],
        "dataset": dataset_id,
        "model": "crest", "seed": args.seed,
    }
    metrics = evaluate(cfg, crest_model, adapter, test_data, filtered_data,
                       ctx_bank, dump_spec, crest_cfg.chunk_size)
    for k, v in metrics.items():
        print("%s: %g" % (k, float(v)))

    # one timestamped CSV per invocation, the shape read_ultra_csv parses
    out_dir = os.path.join(args.output_dir, "CREST_results")
    os.makedirs(out_dir, exist_ok=True)
    row = {"dataset": dataset_id}
    row.update({k: float(v) for k, v in metrics.items()})
    path = os.path.join(out_dir, "CREST_results_%s.csv" % time.strftime("%Y-%m-%d-%H-%M-%S"))
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    print("Results written to %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
