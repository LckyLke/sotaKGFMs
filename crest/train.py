"""Training stages, with the bank-refresh cost control the plan requires.

Stage A trains the readout (and channel, if any) with the encoder frozen;
stage B unfreezes the encoder at a reduced learning rate; ``joint`` (track C)
alternates entity and relation batches one to one.

------------------------------------------------------------------------------
Bank refresh (docs/CREST_PLAN.md 4.2)
------------------------------------------------------------------------------
Rebuilding a whole bank every 500 steps was the unbudgeted cost of the
original plan (~18,960 encoder forwards per rebuild on FB15k-237). Three
controls, all implemented in ``BankRefresher``:

1. banks are cached on disk keyed by (graph id, checkpoint hash, seed) --
   ``ContextBank.load_or_build`` -- so an identical rebuild never runs;
2. a refresh rebuilds only the relation ids touched since the last one;
3. the gate: bank build time per window must stay under ``cost_gate`` (20%)
   of training step time over the same window, logged to
   ``results/phase2_cost.json``; on failure the refresh interval is raised
   *before* anything else is tried.

------------------------------------------------------------------------------
The objective is TRIX's own, unchanged
------------------------------------------------------------------------------
docs/CREST_PLAN.md inherits section 3.4 of the original plan: TRIX's loss and
negative sampling are used as they are, because the readout is a residual on
TRIX's score and a different objective would make the comparison against
``ranks/trix`` a comparison of objectives, not of models. Concretely, per
``repos/trix/src/run_entity.py::train_and_validate``:

* positives are sampled from the training graph's target edges;
* ``negative_sampling`` (strict) replaces tails on the first half of the
  batch and heads on the second, ``num_negative`` per query;
* the loss is binary cross-entropy with logits over ``s = s_v0 + s_pfn``,
  with self-adversarial negative weighting at ``adversarial_temperature``
  (1 in TRIX's pretraining config, and here).

``edge_match``/``strict_negative_mask``/``negative_sampling`` below are
verbatim copies of ``trix/tasks.py``, copied rather than imported so the host
test suite samples exactly the way the container does without needing TRIX --
the same precedent as ``crest/model.py::compute_ranking``.

The whole batch goes through the encoder in **one** forward
(``encoder.encode_batch``, the same call ``crest/run.py`` evaluates with);
the readout then attends with a batch dimension, so no per-query Python loop
survives on the training path. Encoders that only offer ``encode_single``
(the host-test toys) fall back to one call per query row, which is fine at
toy scale and never runs in the container.

Seed 1024 for every run until phase 4, like every other model here.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import bank as _bank
except ImportError:  # pragma: no cover - flat sys.path use inside the container
    import bank as _bank  # type: ignore


class BankRefresher:
    """Owns one graph's bank during training: touch, refresh, gate."""

    def __init__(self, graph, encoder, ctx_bank: _bank.ContextBank,
                 refresh_interval: int = 500, cost_gate: float = 0.20,
                 relation_task: bool = False):
        self.graph = graph
        self.encoder = encoder
        self.bank = ctx_bank
        self.refresh_interval = refresh_interval
        self.cost_gate = cost_gate
        self.relation_task = relation_task
        self.touched: set = set()
        self.step = 0
        self.window_train_seconds = 0.0
        self.window_bank_seconds = 0.0
        self.log: List[Dict] = []

    def touch(self, relation_ids: Iterable[int]) -> None:
        self.touched.update(int(r) for r in relation_ids)

    def after_step(self, step_seconds: float) -> None:
        self.step += 1
        self.window_train_seconds += step_seconds
        if self.step % self.refresh_interval == 0:
            self._refresh()

    def _refresh(self) -> None:
        if not self.touched:
            return
        build = _bank.build_bank_relation if self.relation_task else _bank.build_bank_entity
        t0 = time.perf_counter()
        build(self.graph, self.encoder, seed=self.bank.seed,
              relation_ids=sorted(self.touched), bank=self.bank)
        self.window_bank_seconds = time.perf_counter() - t0
        ratio = (self.window_bank_seconds / self.window_train_seconds
                 if self.window_train_seconds > 0 else float("inf"))
        entry = {
            "step": self.step,
            "refresh_interval": self.refresh_interval,
            "touched": len(self.touched),
            "bank_seconds": round(self.window_bank_seconds, 3),
            "train_seconds": round(self.window_train_seconds, 3),
            "ratio": round(ratio, 4),
            "gate": self.cost_gate,
            "gate_ok": ratio <= self.cost_gate,
        }
        self.log.append(entry)
        if ratio > self.cost_gate:
            # the plan's ordered remedy: raise the interval before anything else
            self.refresh_interval *= 2
            entry["refresh_interval_raised_to"] = self.refresh_interval
        self.touched.clear()
        self.window_train_seconds = 0.0

    def write_cost_log(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.log, handle, indent=2)


def _remove_batch_edges(graph, heads, rels, tails):
    """The training-time analogue of TRIX's remove_easy_edges: drop the
    batch's positive edges and their inverses from the message graph.

    Only the fallback encoder path uses this; ``encode_batch`` encoders (the
    TRIX adapter) remove their own easy edges inside the forward when the
    wrapped model is in training mode, exactly as TRIX training does. Under
    strict negatives the two removals are the same set: TRIX passes the whole
    [b, c] batch to ``remove_easy_edges``, but a strict negative is never an
    edge of the graph, so only the positives (and inverses) actually match.
    """
    ei, et = graph.edge_index, graph.edge_type
    num_direct = int(graph.num_relations) // 2
    keep = torch.ones(et.shape[0], dtype=torch.bool, device=et.device)
    for u, r, v in zip(heads.tolist(), rels.tolist(), tails.tolist()):
        r_inv = r + num_direct if r < num_direct else r - num_direct
        keep &= ~(((ei[0] == u) & (ei[1] == v) & (et == r)) |
                  ((ei[0] == v) & (ei[1] == u) & (et == r_inv)))
    import copy
    out = copy.copy(graph)
    out.edge_index = ei[:, keep]
    out.edge_type = et[keep]
    return out


# ---------------------------------------------------------------------------
# Verbatim copies of trix/tasks.py (pin 7596e14e): edge_match,
# strict_negative_mask, negative_sampling. Copied, not imported, so the host
# tests sample exactly the way the container does without needing TRIX; the
# container driver may still pass TRIX's own module through ``sampler``.
# ---------------------------------------------------------------------------

def edge_match(edge_index, query_index):
    base = edge_index.max(dim=1)[0] + 1
    from functools import reduce
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start
    offset = num_match.cumsum(0) - num_match
    range_ = torch.arange(num_match.sum(), device=edge_index.device)
    range_ = range_ + (start - offset).repeat_interleave(num_match)
    return order[range_], num_match


def strict_negative_mask(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    query_index = torch.stack([pos_h_index, pos_r_index])
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    query_index = torch.stack([pos_t_index, pos_r_index])
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)


# ---------------------------------------------------------------------------
# The batched CREST training path
# ---------------------------------------------------------------------------

def sample_positive_triples(graph, batch_size: int,
                            generator: torch.Generator) -> torch.Tensor:
    """``[batch_size, 3]`` rows of (h, t, r), the shape TRIX's train loader
    yields. Target edges are the training split when the graph carries one
    (TRIX datasets do); the duck-typed toy graphs fall back to their direct
    edges, since their inverse half restates the same facts."""
    if getattr(graph, "target_edge_index", None) is not None:
        ei, et = graph.target_edge_index, graph.target_edge_type
    else:
        num_direct = int(graph.num_relations) // 2
        direct = (graph.edge_type < num_direct).nonzero(as_tuple=True)[0]
        ei, et = graph.edge_index[:, direct], graph.edge_type[direct]
    picks = torch.randint(et.shape[0], (batch_size,), generator=generator).to(ei.device)
    return torch.stack([ei[0, picks], ei[1, picks], et[picks]], dim=-1)


def self_adversarial_nll(pred: torch.Tensor, num_negative: int,
                         adversarial_temperature: float) -> torch.Tensor:
    """TRIX's loss, verbatim from run_entity.py::train_and_validate:
    BCE-with-logits over [positive | negatives], negatives weighted by a
    softmax of their own scores (self-adversarial) at ``adversarial_temperature``,
    or uniformly at temperature 0."""
    target = torch.zeros_like(pred)
    target[:, 0] = 1
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    neg_weight = torch.ones_like(pred)
    if adversarial_temperature > 0:
        with torch.no_grad():
            neg_weight[:, 1:] = F.softmax(pred[:, 1:] / adversarial_temperature, dim=-1)
    else:
        neg_weight[:, 1:] = 1 / num_negative
    loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
    return loss.mean()


def entity_batch_scores(model, graph, batch: torch.Tensor,
                        ctx_bank: _bank.ContextBank,
                        encoder_no_grad: bool = False,
                        chunk_size=None) -> torch.Tensor:
    """``s = s_v0 + s_pfn`` for a raw TRIX batch ``[b, c, 3]`` -> ``[b, c]``.

    One encoder forward for the whole batch. The per-row conversion to tail
    form (candidates, effective relation id) restates TRIX's
    ``negative_sample_to_tail`` so the residual reads the same bank the score
    columns were produced under: rows whose head varies are head queries and
    read the inverse relation's bank.

    ``encoder_no_grad`` runs the encoder under ``torch.no_grad()`` -- stage A
    freezes it, so building its autograd graph would only cost memory.
    """
    h_index, t_index, r_index = batch.unbind(-1)
    num_direct = int(graph.num_relations) // 2
    is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
    cand = torch.where(is_t_neg, t_index, h_index)
    r_eff = torch.where(is_t_neg.squeeze(-1), r_index[:, 0], r_index[:, 0] + num_direct)

    encoder = model._encoder_ref
    grad_ctx = torch.no_grad() if encoder_no_grad else torch.enable_grad()
    with grad_ctx:
        if hasattr(encoder, "encode_batch"):
            # the TRIX adapter: easy-edge removal happens inside the wrapped
            # model when it is in training mode, exactly as in TRIX training
            x, z, s0_cand = encoder.encode_batch(graph, batch)
            x_cand = x.gather(1, cand.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        else:
            # host-test fallback: encode_single per query row on one shared
            # message graph with the batch positives (and inverses) removed
            message_graph = _remove_batch_edges(
                graph, h_index[:, 0], r_index[:, 0], t_index[:, 0])
            head_eff = torch.where(is_t_neg.squeeze(-1), h_index[:, 0], t_index[:, 0])
            xs, zs, s0s = [], [], []
            for i in range(batch.shape[0]):
                x_i, z_i, s0_i = encoder.encode_single(
                    message_graph, int(head_eff[i]), int(r_eff[i]))
                xs.append(x_i)
                zs.append(z_i)
                s0s.append(s0_i)
            x, z, s0 = torch.stack(xs), torch.stack(zs), torch.stack(s0s)
            x_cand = x.gather(1, cand.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            s0_cand = s0.gather(1, cand)
    return model.score(x_cand, z, s0_cand, r_eff, ctx_bank, chunk_size=chunk_size)


def entity_loss_from_triples(model, graph, ctx_bank: _bank.ContextBank,
                             triples: torch.Tensor, num_negative: int,
                             adversarial_temperature: float = 1.0,
                             strict: bool = True,
                             encoder_no_grad: bool = False,
                             sampler=negative_sampling) -> torch.Tensor:
    """TRIX's objective on given positives: sample negatives, score in one
    batched forward, weight self-adversarially. ``sampler`` defaults to the
    verbatim copy above; the container driver passes ``tasks.negative_sampling``
    from the patched TRIX tree so the sampling code is literally TRIX's own."""
    batch = sampler(graph, triples, num_negative, strict=strict)
    pred = entity_batch_scores(model, graph, batch, ctx_bank, encoder_no_grad)
    return self_adversarial_nll(pred, num_negative, adversarial_temperature)


def entity_batch_loss(model, graph, ctx_bank: _bank.ContextBank, batch_size: int,
                      num_negative: int, generator: torch.Generator,
                      adversarial_temperature: float = 1.0, strict: bool = True,
                      encoder_no_grad: bool = False,
                      sampler=negative_sampling) -> torch.Tensor:
    """One training step's loss: TRIX's sampling and loss, batched forward."""
    triples = sample_positive_triples(graph, batch_size, generator)
    return entity_loss_from_triples(model, graph, ctx_bank, triples, num_negative,
                                    adversarial_temperature, strict,
                                    encoder_no_grad, sampler)


def relation_batch_loss(model, graph, ctx_bank: _bank.ContextBank, batch_size: int,
                        generator: torch.Generator) -> torch.Tensor:
    """Cross-entropy over all direct relations for queries (u, ?, v).

    Still per-query: the relation task enters at track C, which has no driver
    yet, so this loop only ever runs on toy graphs in host tests. Batching it
    belongs to the track C work, alongside TRIX's ``negative_sampling_relation``.
    """
    n_edges = graph.edge_type.shape[0]
    num_direct = graph.num_relations // 2
    # sample direct edges only; the inverse of a triple is the same fact
    direct = (graph.edge_type < num_direct).nonzero(as_tuple=True)[0]
    picks = direct[torch.randint(len(direct), (batch_size,), generator=generator)]
    heads = graph.edge_index[0, picks]
    tails = graph.edge_index[1, picks]
    rels = graph.edge_type[picks]
    message_graph = _remove_batch_edges(graph, heads, rels, tails)

    losses = []
    for u, r, v in zip(heads.tolist(), rels.tolist(), tails.tolist()):
        scores = model(message_graph, u, v, ctx_bank)
        losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([r])))
    return torch.stack(losses).mean()


def _trainable(model: nn.Module, freeze_encoder: bool) -> List[torch.nn.Parameter]:
    params = []
    for name, p in model.named_parameters():
        if freeze_encoder and name.startswith("encoder."):
            p.requires_grad_(False)
            continue
        p.requires_grad_(True)
        params.append(p)
    return params


def stage_a(model, graph, ctx_bank: _bank.ContextBank, *, steps: int = 1000,
            batch_size: int = 32, num_negative: int = 32, lr: float = 5e-4,
            adversarial_temperature: float = 1.0, strict: bool = True,
            seed: int = 1024, refresher: Optional[BankRefresher] = None,
            log_every: int = 100) -> List[float]:
    """Readout-only training; the encoder is frozen so the bank stays valid
    for the whole stage and no refresh is needed unless one is passed in."""
    generator = torch.Generator().manual_seed(int(seed))
    optimizer = torch.optim.AdamW(_trainable(model, freeze_encoder=True), lr=lr)
    losses = []
    for step in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = entity_batch_loss(model, graph, ctx_bank, batch_size, num_negative,
                                 generator, adversarial_temperature, strict,
                                 encoder_no_grad=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if refresher is not None:
            refresher.after_step(time.perf_counter() - t0)
        if log_every and step % log_every == 0:
            print("stage_a step {} loss {:.4f}".format(step, float(loss)))
    return losses


def stage_b(model, graph, ctx_bank: _bank.ContextBank, *, steps: int = 1000,
            batch_size: int = 32, num_negative: int = 32, lr: float = 5e-4,
            encoder_lr_scale: float = 0.1,
            adversarial_temperature: float = 1.0, strict: bool = True,
            seed: int = 1024,
            refresher: Optional[BankRefresher] = None, log_every: int = 100
            ) -> List[float]:
    """Full finetune. The encoder moves, so bank rows go stale: the caller
    must pass a ``BankRefresher`` -- stage B without one silently trains
    against a bank the encoder no longer produces."""
    if refresher is None:
        raise ValueError("stage_b requires a BankRefresher; see the docstring")
    generator = torch.Generator().manual_seed(int(seed))
    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    other_params = _trainable(model, freeze_encoder=False)
    other_params = [p for p in other_params
                    if not any(p is q for q in encoder_params)]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": lr},
        {"params": encoder_params, "lr": lr * encoder_lr_scale},
    ])
    losses = []
    for step in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = entity_batch_loss(model, graph, ctx_bank, batch_size, num_negative,
                                 generator, adversarial_temperature, strict)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        refresher.touch(graph.edge_type.unique().tolist())
        refresher.after_step(time.perf_counter() - t0)
        if log_every and step % log_every == 0:
            print("stage_b step {} loss {:.4f}".format(step, float(loss)))
    return losses


def joint(joint_model, graph, entity_bank: _bank.ContextBank,
          relation_bank: _bank.ContextBank, *, steps: int = 1000,
          batch_size: int = 32, num_negative: int = 32, lr: float = 5e-4,
          adversarial_temperature: float = 1.0, strict: bool = True,
          seed: int = 1024, freeze_encoder: bool = True, log_every: int = 100
          ) -> List[float]:
    """Track C: alternate entity and relation batches one to one."""
    generator = torch.Generator().manual_seed(int(seed))
    optimizer = torch.optim.AdamW(_trainable(joint_model, freeze_encoder), lr=lr)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        if step % 2 == 0:
            loss = entity_batch_loss(joint_model.entity, graph, entity_bank,
                                     batch_size, num_negative, generator,
                                     adversarial_temperature, strict,
                                     encoder_no_grad=freeze_encoder)
        else:
            loss = relation_batch_loss(joint_model.relation, graph, relation_bank,
                                       batch_size, generator)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if log_every and step % log_every == 0:
            print("joint step {} loss {:.4f}".format(step, float(loss)))
    return losses
