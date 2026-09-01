"""Test-time levers (2026-09-01): bidirectional re-ranking and score ensembles.

------------------------------------------------------------------------------
Bidirectional re-ranking
------------------------------------------------------------------------------
For a tail query (h, r, ?) the trunk conditions on (h, r) and scores every
candidate t from h's side of the fact. The same trained network can score
the same fact from t's side: under TRIX's interface the head query
(?, r, t) is the tail query (t, r^-1, ?) read at position h, and the
training loss covers both directions (``negative_sample_to_tail``). Both
numbers are logits of the same fact from the same BCE-trained head, so
their sum is a product-of-experts estimate of the fact.

A reverse score for every candidate would cost |V| trunk passes per query.
Only the top-k candidates of the forward pass are re-scored:

    final[c] = forward[c] + weight * reverse[c]        c in the top-k block
    final[c] = forward[c] + weight * floor             c outside the block

with ``floor`` the smallest reverse score inside the block (a per-query
imputation). Ranks are then computed from ``final`` under the shared rank
definition, unchanged (1-based, pessimistic ties, strict filtering).

Consequences, provable from the two lines above:

* a target OUTSIDE the block keeps its forward rank exactly: every block
  member had forward >= its forward and gets reverse >= floor; every
  non-member gets the same floor;
* a target INSIDE the block can only be outranked by non-members that tie
  it on the forward score with reverse == floor, which the pessimistic rule
  counts against it as before. No tie is resolved in the model's favour.

The block is chosen among the candidates that matter for the rank: the
strict-negative mask plus the target itself (filtered-out true answers do
not spend block slots). ``k`` is fixed per run and stated on every table.

Cost: k extra single-row trunk passes per query and direction, so a k=8
run costs about 9x the base evaluation.

------------------------------------------------------------------------------
Score ensemble
------------------------------------------------------------------------------
``ScoreEnsemble`` averages the logits of several INCITE checkpoints built
from one config. Support stores are per-encoder objects, so an ensemble
runs support-off only.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .model import negative_sample_to_tail

__all__ = ["rerank_predictions", "ScoreEnsemble"]


@torch.no_grad()
def rerank_predictions(model, data, batch: torch.Tensor, pred: torch.Tensor,
                       target: torch.Tensor, mask: Optional[torch.Tensor],
                       k: int, weight: float = 1.0, chunk: int = 32,
                       support=None, walk_offset: int = 0) -> torch.Tensor:
    """Return ``final`` scores ``[b, c]`` for the ``all_negative`` batch
    ``batch [b, c, 3]`` whose forward logits are ``pred [b, c]``.

    ``target [b]`` is the position of the true answer, ``mask [b, c]`` the
    strict-negative mask (True = counts for the rank), or None for no mask.
    ``k <= 0`` returns ``pred`` unchanged.
    """
    if k <= 0:
        return pred
    h_index, t_index, r_index = batch.unbind(-1)
    num_direct = int(data.num_relations) // 2
    # normalize head queries into tail form: h and r constant per row, the
    # candidate entity varies along the second axis (TRIX's own convention)
    h_index, t_index, r_index = negative_sample_to_tail(
        h_index, t_index, r_index, num_direct)
    b, c = pred.shape
    k = min(int(k), c)

    eligible = torch.zeros_like(pred, dtype=torch.bool) if mask is None else mask.clone()
    if mask is None:
        eligible[:] = True
    eligible.scatter_(1, target.view(b, 1), True)
    selectable = pred.masked_fill(~eligible, float("-inf"))
    top_val, top_idx = selectable.topk(k, dim=-1)                   # [b, k]
    cand = t_index.gather(1, top_idx)                              # [b, k] entity ids
    h0 = h_index[:, 0]
    r0 = r_index[:, 0]
    r_inv = torch.where(r0 < num_direct, r0 + num_direct, r0 - num_direct)

    # reverse queries (cand, r_inv, ?) read at position h0: rows [b*k, 1, 3]
    rev_batch = torch.stack([cand.reshape(-1),
                             h0.repeat_interleave(k),
                             r_inv.repeat_interleave(k)], dim=-1).unsqueeze(1)
    rev = []
    for start in range(0, rev_batch.shape[0], int(chunk)):
        out = model(data, rev_batch[start:start + int(chunk)],
                    support=support, walk_offset=walk_offset)
        rev.append(out.reshape(-1))
    rev = torch.cat(rev).view(b, k)

    # a block slot may hold -inf when fewer than k candidates are eligible;
    # such slots carry no candidate and must not define the floor
    valid = torch.isfinite(top_val)
    rev = torch.where(valid, rev, torch.full_like(rev, float("inf")))
    floor = rev.min(dim=-1, keepdim=True).values                   # [b, 1]
    floor = torch.where(torch.isfinite(floor), floor, torch.zeros_like(floor))
    final = pred + float(weight) * floor
    block = pred.gather(1, top_idx) + float(weight) * torch.where(
        valid, rev, floor.expand_as(rev))
    final = final.scatter(1, top_idx, block)
    return final


class ScoreEnsemble(nn.Module):
    """Mean of member logits. Members share one config and one interface."""

    def __init__(self, members: Sequence[nn.Module]):
        super().__init__()
        assert len(members) >= 1, "an ensemble needs at least one member"
        self.members = nn.ModuleList(list(members))
        self.checkpoint_activations = False

    def forward(self, data, batch, support=None, walk_offset: int = 0):
        assert support is None, "score ensembles run support-off"
        outs = [m(data, batch, support=None, walk_offset=walk_offset)
                for m in self.members]
        return torch.stack(outs).mean(0)

    def forward_relation(self, data, batch, support=None, walk_offset: int = 0):
        assert support is None, "score ensembles run support-off"
        outs = [m.forward_relation(data, batch, support=None,
                                   walk_offset=walk_offset)
                for m in self.members]
        return torch.stack(outs).mean(0)
