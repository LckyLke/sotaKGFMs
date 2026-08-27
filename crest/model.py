"""CREST models: encoder score plus an in-context residual.

``s = s_v0 + residual`` where ``s_v0`` is the encoder's own score and the
residual comes from ``crest/pfn.Readout`` reading the query's bank
(``crest/bank.py``). With the reader's last layer at zero the residual is the
exact constant 0 and CREST *is* the encoder -- phase 0's identity gate.

------------------------------------------------------------------------------
Encoder protocol
------------------------------------------------------------------------------
CREST does not subclass TRIX; it wraps any object satisfying:

    encode_single(graph, u, r)          -> (x [n, d], z [d], s0 [n])
    encode_relation_single(graph, u, v) -> (w [num_direct, d], c [d],
                                            s0 [num_direct])          (relation task)

``x``/``w`` are per-candidate features, ``z``/``c`` the query representation
actually used to score, ``s0`` the encoder's own scores. The TRIX adapter
lives in ``crest/run.py`` (container-only, since TRIX needs PyG); host tests
inject a toy encoder. The adapter must return these from the *same* forward
pass that produced ``s0``, at the same batch shape TRIX evaluates with --
identity at zero residual is arithmetic, but rank identity on a GPU also
needs the reductions to run in the same shape they ran for ``ranks/trix``.

------------------------------------------------------------------------------
Which bank a query reads
------------------------------------------------------------------------------
Entity queries read the bank of their *effective* relation id: ``r`` for a
tail query (h, r, ?), ``r + num_relations // 2`` for a head query (?, r, t),
mirroring TRIX's ``negative_sample_to_tail``. Inverse relations are separate
ids with separate banks. A relation id absent from the bank (no inference
edge of that type) contributes residual 0 -- the encoder score stands alone,
because examples for it could only be taken from edges the model must not
see. Relation queries read the stacked bank of all direct relations: a
relation query cannot know its own relation, that being the prediction.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:  # package import (host tests, container)
    from . import bank as _bank
    from .pfn import Readout
except ImportError:  # pragma: no cover - flat sys.path use inside the container
    import bank as _bank  # type: ignore
    from pfn import Readout  # type: ignore


def compute_ranking(pred, target, mask=None):
    """Verbatim copy of ``trix/tasks.py::compute_ranking`` (the entity one).

    1-based, pessimistic ties (non-strict ``<=`` with the target masked out),
    strict filtering under ``mask``. Copied rather than imported so host tests
    rank exactly the way the container does without needing TRIX; the
    container driver still calls TRIX's own. Note the ``+ 1`` on *both*
    branches -- upstream's ``compute_ranking_relation`` omits it on the
    unfiltered branch, which patches/trix/0004 corrects.
    """
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


class CRESTEntity(nn.Module):
    """Entity prediction: TRIX-shaped scores plus the bank residual."""

    def __init__(self, encoder, readout: Readout, channel: Optional[nn.Module] = None,
                 num_passes: int = 8):
        super().__init__()
        self.encoder = encoder if isinstance(encoder, nn.Module) else None
        self._encoder_ref = encoder
        self.readout = readout
        self.channel = channel
        # test-time averaging for the random channel (track B); the
        # deterministic model ignores it
        self.num_passes = num_passes

    # -- core: add the residual to precomputed encoder outputs --------------
    def residual(self, x_cand: torch.Tensor, z: torch.Tensor, s0: torch.Tensor,
                 r_eff: torch.Tensor, ctx_bank: _bank.ContextBank,
                 channel_cand: Optional[torch.Tensor] = None,
                 chunk_size: Optional[int] = None) -> torch.Tensor:
        """Residual for a batch: ``x_cand [b, c, d]``, ``z [b, d]``,
        ``s0 [b, c]``, ``r_eff [b]`` -> ``[b, c]``.

        Queries are grouped by effective relation id so each group attends
        over its own bank; ids without a bank keep residual 0.

        ``chunk_size`` bounds how many candidates one readout call carries.
        It is a memory parameter, not a model parameter (docs/CREST_PLAN.md
        4.1): candidates never attend to each other, so chunking is exact and
        the chosen value is recorded in PROVENANCE.json, never tuned against.
        """
        rows = _bank.row_features(x_cand, z, s0)
        if channel_cand is not None:
            rows = torch.cat([rows, channel_cand], dim=-1)
        out = torch.zeros_like(s0)
        for rid in r_eff.unique().tolist():
            if rid not in ctx_bank:
                continue
            sel = (r_eff == rid).nonzero(as_tuple=True)[0]
            feats = ctx_bank.features(rid)
            labels = ctx_bank.labels(rid)
            if channel_cand is not None:
                # bank rows carry no walk information: the channel describes
                # the live query, not the stored examples
                pad = torch.zeros(feats.shape[0], channel_cand.shape[-1],
                                  dtype=feats.dtype, device=feats.device)
                feats = torch.cat([feats, pad], dim=-1)
            group = rows[sel]
            step = group.shape[1] if chunk_size is None else int(chunk_size)
            pieces = [self.readout(group[:, i:i + step], feats, labels)
                      for i in range(0, group.shape[1], step)]
            out[sel] = torch.cat(pieces, dim=1)
        return out

    def score(self, x_cand: torch.Tensor, z: torch.Tensor, s0: torch.Tensor,
              r_eff: torch.Tensor, ctx_bank: _bank.ContextBank,
              channel_cand: Optional[torch.Tensor] = None,
              chunk_size: Optional[int] = None) -> torch.Tensor:
        return s0 + self.residual(x_cand, z, s0, r_eff, ctx_bank, channel_cand,
                                  chunk_size)

    # -- convenience: single-query path through the encoder ------------------
    def forward(self, graph, h: int, r_eff: int, candidates: torch.Tensor,
                ctx_bank: _bank.ContextBank) -> torch.Tensor:
        """Scores for one query (h, r_eff, ?) over ``candidates [c]``.

        ``r_eff`` is the effective relation id (inverse id for head queries).
        With a random channel active, averages ``num_passes`` passes at test
        time (track B); the residual is recomputed per pass, the encoder runs
        once.
        """
        x, z, s0 = self._encoder_ref.encode_single(graph, int(h), int(r_eff))
        x_cand = x[candidates].unsqueeze(0)
        s0_cand = s0[candidates].unsqueeze(0)
        zb = z.unsqueeze(0)
        rid = torch.tensor([int(r_eff)])
        if self.channel is None:
            return self.score(x_cand, zb, s0_cand, rid, ctx_bank)[0]
        passes = self.num_passes if not self.training else 1
        total = torch.zeros_like(s0_cand)
        for _ in range(passes):
            chan = self.channel(graph, int(h))[candidates].unsqueeze(0)
            total = total + self.score(x_cand, zb, s0_cand, rid, ctx_bank, chan)
        return (total / passes)[0]


class CRESTRelation(nn.Module):
    """Relation prediction: candidates are the direct relations.

    The context is ``bank.stacked()`` -- every direct relation's rows at once
    -- because the query's own relation is the unknown. The zero-residual
    identity holds here exactly as for entities.
    """

    def __init__(self, encoder, readout: Readout):
        super().__init__()
        self.encoder = encoder if isinstance(encoder, nn.Module) else None
        self._encoder_ref = encoder
        self.readout = readout

    def forward(self, graph, u: int, v: int, ctx_bank: _bank.ContextBank) -> torch.Tensor:
        w, c, s0 = self._encoder_ref.encode_relation_single(graph, int(u), int(v))
        rows = _bank.row_features(w.unsqueeze(0), c.unsqueeze(0), s0.unsqueeze(0))
        if len(ctx_bank) == 0:
            return s0
        feats, labels = ctx_bank.stacked()
        return s0 + self.readout(rows, feats, labels)[0]


class CRESTJoint(nn.Module):
    """Track C: one encoder, both readouts, both banks.

    The encoder is shared; each task keeps its own readout and bank. Batch
    alternation (one to one) lives in ``crest/train.py::joint``; the keep
    rule is each task within 0.005 MRR of its separate model on all 41.
    """

    def __init__(self, encoder, entity_readout: Readout, relation_readout: Readout,
                 channel: Optional[nn.Module] = None, num_passes: int = 8):
        super().__init__()
        self.entity = CRESTEntity(encoder, entity_readout, channel, num_passes)
        self.relation = CRESTRelation(encoder, relation_readout)
        self._encoder_ref = encoder

    def zero_residual(self) -> None:
        self.entity.readout.zero_residual()
        self.relation.readout.zero_residual()
