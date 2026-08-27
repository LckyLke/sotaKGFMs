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

Training here is self-supervised link prediction on the training graph: the
batch's positive edges are removed from the message graph (TRIX's
remove_easy_edges, reimplemented on the duck-typed graph), negatives are
uniform, and the loss is binary cross-entropy over 1 positive and
``num_negative`` negatives per query -- the NBFNet family's objective.
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
    batch's positive edges and their inverses from the message graph."""
    ei, et = graph.edge_index, graph.edge_type
    num_direct = graph.num_relations // 2
    keep = torch.ones(et.shape[0], dtype=torch.bool)
    for u, r, v in zip(heads.tolist(), rels.tolist(), tails.tolist()):
        r_inv = r + num_direct if r < num_direct else r - num_direct
        keep &= ~(((ei[0] == u) & (ei[1] == v) & (et == r)) |
                  ((ei[0] == v) & (ei[1] == u) & (et == r_inv)))
    import copy
    out = copy.copy(graph)
    out.edge_index = ei[:, keep]
    out.edge_type = et[keep]
    return out


def entity_batch_loss(model, graph, ctx_bank: _bank.ContextBank, batch_size: int,
                      num_negative: int, generator: torch.Generator) -> torch.Tensor:
    """BCE over 1 positive + ``num_negative`` uniform negatives per query."""
    n_edges = graph.edge_type.shape[0]
    picks = torch.randint(n_edges, (batch_size,), generator=generator)
    heads = graph.edge_index[0, picks]
    tails = graph.edge_index[1, picks]
    rels = graph.edge_type[picks]
    message_graph = _remove_batch_edges(graph, heads, rels, tails)

    losses = []
    for u, r, v in zip(heads.tolist(), rels.tolist(), tails.tolist()):
        negs = torch.randint(graph.num_nodes, (num_negative,), generator=generator)
        candidates = torch.cat([torch.tensor([v]), negs])
        scores = model(message_graph, u, r, candidates, ctx_bank)
        target = torch.zeros_like(scores)
        target[0] = 1.0
        losses.append(F.binary_cross_entropy_with_logits(scores, target))
    return torch.stack(losses).mean()


def relation_batch_loss(model, graph, ctx_bank: _bank.ContextBank, batch_size: int,
                        generator: torch.Generator) -> torch.Tensor:
    """Cross-entropy over all direct relations for queries (u, ?, v)."""
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
        loss = entity_batch_loss(model, graph, ctx_bank, batch_size, num_negative, generator)
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
            encoder_lr_scale: float = 0.1, seed: int = 1024,
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
        loss = entity_batch_loss(model, graph, ctx_bank, batch_size, num_negative, generator)
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
                                     batch_size, num_negative, generator)
        else:
            loss = relation_batch_loss(joint_model.relation, graph, relation_bank,
                                       batch_size, generator)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if log_every and step % log_every == 0:
            print("joint step {} loss {:.4f}".format(step, float(loss)))
    return losses
