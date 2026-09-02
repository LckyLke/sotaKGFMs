"""The in-context scorer for the context-necessity diagnostic (2026-09-02).

Why this module exists
------------------------------------------------------------------------------
Three in-context readouts have been measured in this harness and none moved a
zero-shot number: CREST's PFN-style context transformer over a bank of
leave-one-out rows (results/STOP.md, crest branch), INCITE's one-head
cross-attention over retrieved support rows (results/incite/PHASE22_RESULT.md),
and the released KGPFN (25 graphs, at or below TRIX). All three share two
things: the readout is a RESIDUAL on the encoder's own score, and the rows are
functions of the same encoder representations as that score, so a readout at
zero is a local optimum the loss never leaves. The external plan
(docs/KGFM_PLAN.md, phase 5) proposes the same mechanism again, trained from
scratch and end to end.

This module is the smallest thing that can tell those two situations apart.
It is a PFN-style scorer -- context self-attention, query cross-attention,
per-candidate reader, the CREST shape -- that can be the ONLY scorer
(``mode: context_only``), a residual beside the trunk's MLP (``mode:
residual``, the failed design), or absent (``mode: floor``). The diagnostic
(diagnostics/context_necessity.py) trains each from scratch on synthetic
graphs where the query relation is withheld from the message graph and the
context is the only place its facts appear, and evaluates with full context,
shuffled labels and no context -- the external plan's own K3 ordering test.

Properties kept from crest/pfn.py: query rows never attend to each other
(candidate-permutation equivariance), context attention is a weighted sum
(row-order invariance). Dropped: the exact-zero initialisation of the reader
(this scorer trains from scratch and must not start silent).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["ContextScorer", "LABEL_NEGATIVE", "LABEL_POSITIVE", "LABEL_QUERY"]

#: Label vocabulary of the rows: context negatives, context positives, and
#: the query candidates themselves (whose label is what is being estimated).
LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_QUERY = 0, 1, 2


class ContextScorer(nn.Module):
    """Rows ``[.., in_dim]`` plus labels -> one scalar per query candidate.

    ``forward(query_rows [b, c, in], ctx_rows [b, m, in], ctx_labels [b, m])``
    returns ``[b, c]``. With ``m == 0`` (the no-context condition) the query
    rows pass through the per-block MLPs only: the module still scores, and
    whatever it scores with is then, by construction, not context.
    """

    def __init__(self, in_dim: int, width: int = 64, depth: int = 2,
                 heads: int = 4):
        super().__init__()
        self.in_dim, self.width, self.depth = int(in_dim), int(width), int(depth)
        self.proj = nn.Linear(self.in_dim, self.width)
        self.label = nn.Embedding(3, self.width)
        mk_attn = lambda: nn.MultiheadAttention(self.width, int(heads), batch_first=True)
        mk_mlp = lambda: nn.Sequential(nn.LayerNorm(self.width),
                                       nn.Linear(self.width, 2 * self.width),
                                       nn.ReLU(),
                                       nn.Linear(2 * self.width, self.width))
        self.ctx_norm = nn.ModuleList(nn.LayerNorm(self.width) for _ in range(self.depth))
        self.ctx_attn = nn.ModuleList(mk_attn() for _ in range(self.depth))
        self.ctx_mlp = nn.ModuleList(mk_mlp() for _ in range(self.depth))
        self.qry_norm = nn.ModuleList(nn.LayerNorm(self.width) for _ in range(self.depth))
        self.qry_attn = nn.ModuleList(mk_attn() for _ in range(self.depth))
        self.qry_mlp = nn.ModuleList(mk_mlp() for _ in range(self.depth))
        self.reader = nn.Sequential(nn.LayerNorm(self.width),
                                    nn.Linear(self.width, self.width), nn.ReLU(),
                                    nn.Linear(self.width, 1))

    def forward(self, query_rows: torch.Tensor, ctx_rows: Optional[torch.Tensor],
                ctx_labels: Optional[torch.Tensor],
                ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c = query_rows.shape[:2]
        q_ids = torch.full((b, c), LABEL_QUERY, dtype=torch.long,
                           device=query_rows.device)
        query = self.proj(query_rows) + self.label(q_ids)
        have_ctx = ctx_rows is not None and ctx_rows.shape[1] > 0
        if have_ctx:
            context = self.proj(ctx_rows) + self.label(ctx_labels.long())
            pad = None if ctx_mask is None else ~ctx_mask
        for i in range(self.depth):
            if have_ctx:
                ctx_n = self.ctx_norm[i](context)
                context = context + self.ctx_attn[i](
                    ctx_n, ctx_n, ctx_n, key_padding_mask=pad, need_weights=False)[0]
                context = context + self.ctx_mlp[i](context)
                qry_n = self.qry_norm[i](query)
                ctx_kv = self.ctx_norm[i](context)
                query = query + self.qry_attn[i](
                    qry_n, ctx_kv, ctx_kv, key_padding_mask=pad, need_weights=False)[0]
            query = query + self.qry_mlp[i](query)
        return self.reader(query).squeeze(-1)


# ---------------------------------------------------------------------------
# Row assembly (shared by the diagnostic and its tests)
# ---------------------------------------------------------------------------
def build_rows(x: torch.Tensor, z_q: torch.Tensor, batch: dict, k: int):
    """From one ``encode_queries`` call over ``[query heads ; context heads]``
    to the scorer's inputs.

    ``x [k + k*P, V, d]``, ``z_q [k + k*P, d]`` (the first ``k`` row-sets are
    the queries, then the ``P`` context heads of each instance in order).
    Returns ``(query_rows [k, C, 2d], ctx_rows [k, P*(1+Nn), 2d],
    ctx_labels [k, P*(1+Nn)])`` where a row is ``[x_candidate ; z_query]`` --
    exactly the feature the trunk's own score MLP consumes, so the floor,
    the residual and the context-only scorer all read the same vectors.
    """
    d = x.shape[-1]
    cands, ctx_v, ctx_neg = batch["cands"], batch["ctx_v"], batch["ctx_neg"]
    P, Nn = ctx_neg.shape[1], ctx_neg.shape[2]
    C = cands.shape[1]
    xq, zq = x[:k], z_q[:k]
    q_feat = xq.gather(1, cands.unsqueeze(-1).expand(k, C, d))
    query_rows = torch.cat([q_feat, zq.unsqueeze(1).expand(k, C, d)], dim=-1)
    xc = x[k:].reshape(k, P, x.shape[1], d)
    zc = z_q[k:].reshape(k, P, d)
    pos = xc.gather(2, ctx_v.view(k, P, 1, 1).expand(k, P, 1, d))        # [k,P,1,d]
    neg = xc.gather(2, ctx_neg.unsqueeze(-1).expand(k, P, Nn, d))        # [k,P,Nn,d]
    feats = torch.cat([pos, neg], dim=2)                                  # [k,P,1+Nn,d]
    rows = torch.cat([feats, zc.unsqueeze(2).expand(k, P, 1 + Nn, d)], dim=-1)
    labels = torch.zeros(k, P, 1 + Nn, dtype=torch.long, device=x.device)
    labels[:, :, 0] = LABEL_POSITIVE
    return (query_rows, rows.reshape(k, P * (1 + Nn), 2 * d),
            labels.reshape(k, P * (1 + Nn)))


def shuffle_labels(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permute each instance's context labels among its rows (the K3
    'shuffled' condition): the same rows, the same label counts, no
    row-label correspondence."""
    out = labels.clone()
    for i in range(labels.shape[0]):
        perm = torch.randperm(labels.shape[1], generator=generator).to(labels.device)
        out[i] = labels[i, perm]
    return out
