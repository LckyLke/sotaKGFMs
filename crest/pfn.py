"""The in-context readout: row encoder, context transformer, query reader.

PFN-style: the bank rows (labelled examples) are the context, the query's
candidate rows are read against it, and the output is a per-candidate scalar
residual added to the encoder's own score. Three properties are load-bearing:

1. **Exact identity at zero.** ``QueryReader``'s last linear layer is
   initialised to zero, so the residual is the constant 0.0 tensor and
   ``s = s_v0 + 0`` -- arithmetically TRIX. This is phase 0's gate
   (docs/CREST_PLAN.md section 2) and must hold exactly, not approximately:
   multiplying by a zero weight matrix and adding a zero bias produce exact
   zeros for every finite input, and ``s_v0 + 0.0`` returns ``s_v0``'s value
   bit for bit (the lone exception, ``-0.0 + 0.0 == +0.0``, changes no
   comparison and therefore no rank).
2. **Candidate-permutation equivariance.** Query rows never attend to each
   other -- each candidate reads the bank independently -- so permuting the
   candidates permutes the residuals, and the model is insensitive to how
   many candidates a batch carries.
3. **Bank-order invariance.** Attention over the bank is a weighted sum, so
   the readout does not depend on the order rows were inserted in.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

#: Label-embedding vocabulary: bank negatives, bank positives, and the query
#: rows themselves, whose label is what the readout is estimating.
LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_QUERY = 0, 1, 2


class RowEncoder(nn.Module):
    """Embed a raw row ``[3d + 2 (+ channel)]`` plus its label into ``width``."""

    def __init__(self, row_dim: int, width: int = 128):
        super().__init__()
        self.row_dim = row_dim
        self.proj = nn.Linear(row_dim, width)
        self.label = nn.Embedding(3, width)

    def forward(self, rows: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(rows) + self.label(label_ids)


class ContextTransformer(nn.Module):
    """Alternating context self-attention and query cross-attention blocks.

    Per block: bank rows self-attend (examples talk to each other), then each
    query row cross-attends to the bank -- never to other query rows, see the
    module docstring -- and both pass through a pre-norm MLP.
    """

    def __init__(self, width: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        self.depth = depth
        # nn.MultiheadAttention(batch_first=True) behaves identically on
        # torch 2.1 (container) and 2.5 (host); nothing here relies on
        # scaled_dot_product_attention backends added after 2.1.
        self.ctx_attn = nn.ModuleList(
            nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(depth))
        self.qry_attn = nn.ModuleList(
            nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(depth))
        self.ctx_norm = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.qry_norm = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.ctx_mlp = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                          nn.ReLU(), nn.Linear(width * 2, width))
            for _ in range(depth))
        self.qry_mlp = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                          nn.ReLU(), nn.Linear(width * 2, width))
            for _ in range(depth))

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """``query [b, c, width]``, ``context [b, m, width]`` -> ``[b, c, width]``."""
        for i in range(self.depth):
            ctx_n = self.ctx_norm[i](context)
            context = context + self.ctx_attn[i](ctx_n, ctx_n, ctx_n, need_weights=False)[0]
            context = context + self.ctx_mlp[i](context)
            qry_n = self.qry_norm[i](query)
            ctx_kv = self.ctx_norm[i](context)
            query = query + self.qry_attn[i](qry_n, ctx_kv, ctx_kv, need_weights=False)[0]
            query = query + self.qry_mlp[i](query)
        return query


class QueryReader(nn.Module):
    """Per-candidate scalar residual; the last layer starts (and can be
    re-forced) at zero, which is what makes CREST collapse to TRIX exactly."""

    def __init__(self, width: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, width)
        self.out = nn.Linear(width, 1)
        self.zero_residual()

    def zero_residual(self) -> None:
        """Force the residual to the exact constant 0. Phase 0 runs with this."""
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def residual_is_zero(self) -> bool:
        return bool((self.out.weight == 0).all() and (self.out.bias == 0).all())

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.hidden(self.norm(query)))).squeeze(-1)


class Readout(nn.Module):
    """RowEncoder -> ContextTransformer -> QueryReader, as one module.

    ``forward`` takes raw query rows ``[b, c, row_dim]``, raw bank rows
    ``[m, row_dim]`` and bank labels ``[m]`` and returns the residual
    ``[b, c]``. The caller (crest/model.py) owns building rows, choosing the
    bank per relation id, and adding the residual to the encoder score.
    """

    def __init__(self, row_dim: int, width: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        self.rows = RowEncoder(row_dim, width)
        self.transformer = ContextTransformer(width, depth, heads)
        self.reader = QueryReader(width)

    def zero_residual(self) -> None:
        self.reader.zero_residual()

    def residual_is_zero(self) -> bool:
        return self.reader.residual_is_zero()

    def forward(self, query_rows: torch.Tensor, bank_rows: torch.Tensor,
                bank_labels: torch.Tensor) -> torch.Tensor:
        b = query_rows.shape[0]
        # bank labels are {0, 1} and so already are LABEL_NEGATIVE/LABEL_POSITIVE;
        # the comparison keeps this correct for any nonzero positive encoding
        # and stays on bank_labels' device
        labels = (bank_labels > 0).to(torch.long)
        context = self.rows(bank_rows, labels).unsqueeze(0).expand(b, -1, -1)
        query_ids = torch.full(query_rows.shape[:-1], LABEL_QUERY,
                               dtype=torch.long, device=query_rows.device)
        query = self.rows(query_rows, query_ids)
        return self.reader(self.transformer(query, context))
