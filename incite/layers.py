"""The two INCITE steps: an NBFNet-style entity conv and the factorized
relation step of docs/INCITE_DESIGN.md section A.

------------------------------------------------------------------------------
Direction convention -- mirrors TRIX AS EXECUTED, not as commented
------------------------------------------------------------------------------
TRIX's fused rspmm kernel aggregates into ``edge_index[0]`` gathering from
``edge_index[1]`` (``rspmm.cpp``: ``row_ind = edge_index[0]``,
``out[row] += rel[type] * input[col]``), and TRIX passes its edge lists
unflipped. On the relation graph that means the executed hh message into
r1 is sum over co-incident r2 of f(x_v) * z[r2] -- the transpose of the
docstring's direction. hh and tt edge sets are symmetric so the sums agree;
ht and th are each other's transposes, so as executed the "ht" weights serve
the tails-into-heads direction. The factorized step below reproduces the
EXECUTED semantics channel by channel, because the layer gate compares
against the pinned TRIX layer as it actually runs:

    m_hh[r1] = sum_{(v,r1) in P_h} f_hh(v) * (s_h[v] - z[r1])
    m_ht[r1] = sum_{(v,r1) in P_h} f_ht(v) *  s_t[v]
    m_th[r1] = sum_{(v,r1) in P_t} f_th(v) *  s_h[v]
    m_tt[r1] = sum_{(v,r1) in P_t} f_tt(v) * (s_t[v] - z[r1])

with s_h[v] = sum_{(v,r) in P_h} z[r] and s_t[v] likewise over P_t. Each sum
touches every unique incidence pair once: O(|E|), never O(|V| alpha^2), and
the identity with the materialized double sum is exact in real arithmetic
(float reordering only -- hence the 1e-5 allclose gate, docs/INCITE_PLAN.md).

The count channel sets f(v) = 1 in the same four formulas (ULTRA's count
message, section A "keep both channels"); its diagonal correction becomes
deg_role(r1) * z[r1] with deg the unique-pair degree.

------------------------------------------------------------------------------
Entity step direction
------------------------------------------------------------------------------
For an edge (h, r, t) the entity message flows h -> t: out[t] += rel[r] * x[h].
Both directions of every fact are materialized in ``edge_index``, so this is a
convention, not a modeling choice; it is applied identically on the fallback
and the rspmm path (which receives the FLIPPED edge index, because the kernel
aggregates into row 0).

------------------------------------------------------------------------------
pair_sum: one primitive, two implementations
------------------------------------------------------------------------------
``pair_sum`` computes out[:, dst] += input[:, src] over a pair list. The
torch fallback materializes the gathered rows (O(P*b*d) transient memory,
fine on CPU tests and single-graph eval); on CUDA, when TRIX's compiled
rspmm extension is importable, the same sum runs through the kernel with a
constant-ones relation table, which keeps backward memory at O(N*b*d) --
the difference between fitting batch-32 training in 16 GiB and not.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["pair_sum", "distmult_sum", "EntityStep", "FactorizedRelationStep"]

_RSPMM = None


def _rspmm():
    """TRIX's generalized_rspmm, or None outside the container."""
    global _RSPMM
    if _RSPMM is None:
        try:
            from trix.rspmm import generalized_rspmm  # noqa: WPS433
            _RSPMM = generalized_rspmm
        except Exception:
            _RSPMM = False
    return _RSPMM or None


def _use_kernel(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and _rspmm() is not None


def pair_sum(input: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
             n_out: int) -> torch.Tensor:
    """``out[b, dst[p]] += input[b, src[p]]`` -> ``[b, n_out, d]``."""
    b, n_in, d = input.shape
    if _use_kernel(input):
        n = max(int(n_out), n_in)
        flat = input.transpose(0, 1).reshape(n_in, b * d)
        if n > n_in:
            flat = F.pad(flat, (0, 0, 0, n - n_in))
        edge_index = torch.stack([dst, src])
        edge_type = torch.zeros_like(src)
        edge_weight = torch.ones(len(src), device=input.device, dtype=input.dtype)
        ones = torch.ones(1, b * d, device=input.device, dtype=input.dtype)
        out = _rspmm()(edge_index, edge_type, edge_weight, ones, flat,
                       sum="add", mul="mul")
        return out[:int(n_out)].view(int(n_out), b, d).transpose(0, 1)
    out = input.new_zeros(b, int(n_out), d)
    out.index_add_(1, dst, input.index_select(1, src))
    return out


def distmult_sum(x: torch.Tensor, rel: torch.Tensor, edge_index: torch.Tensor,
                 edge_type: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    """``out[t] = boundary[t] + sum_{(h,r,t)} rel[r] * x[h]`` -> ``[b, V, d]``."""
    b, n, d = x.shape
    src, dst = edge_index[0], edge_index[1]
    if _use_kernel(x):
        flat_x = x.transpose(0, 1).reshape(n, b * d)
        flat_rel = rel.transpose(0, 1).reshape(rel.shape[1], b * d)
        edge_weight = torch.ones(len(edge_type), device=x.device, dtype=x.dtype)
        # flipped: the kernel aggregates into edge_index[0]
        out = _rspmm()(torch.stack([dst, src]), edge_type, edge_weight,
                       flat_rel, flat_x, sum="add", mul="mul")
        return out.view(n, b, d).transpose(0, 1) + boundary
    msg = x.index_select(1, src) * rel.index_select(1, edge_type)
    out = boundary.clone()
    out.index_add_(1, dst, msg)
    return out


class _Update(nn.Module):
    """TRIX's per-conv update: linear([input, agg]) -> layer_norm -> relu."""

    def __init__(self, dim: int, layer_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim) if layer_norm else None

    def forward(self, layer_input: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        out = self.linear(torch.cat([layer_input, agg], dim=-1))
        if self.norm is not None:
            out = self.norm(out)
        return F.relu(out)


def _mlp(dim: int) -> nn.Sequential:
    """TRIX's relation_projection shape: linear-relu-linear, dim preserved."""
    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))


class EntityStep(nn.Module):
    """One entity conv: distmult message, sum aggregation, boundary self-loop,
    TRIX update. Relation features are a per-step projection of the current
    relation states (``project_relations``, as in every pinned TRIX layer).
    The residual shortcut lives in the caller, as in BaseNBFNet."""

    def __init__(self, dim: int, layer_norm: bool = True):
        super().__init__()
        self.rel_proj = _mlp(dim)
        self.update = _Update(dim, layer_norm)

    def forward(self, x, z, boundary, edge_index, edge_type):
        rel = self.rel_proj(z)
        agg = distmult_sum(x, rel, edge_index, edge_type, boundary)
        return self.update(x, agg)


class _Channel(nn.Module):
    """One role channel of the relation step. ``count=True`` drops the node
    projection and uses f(v) = 1 (the ULTRA count message)."""

    def __init__(self, dim: int, layer_norm: bool, count: bool):
        super().__init__()
        self.count = count
        self.rel_proj = None if count else _mlp(dim)
        self.update = _Update(dim, layer_norm)

    def message(self, z, node_repr, agg_v, agg_r, s2, diag_z, num_relations):
        """The factorized channel message ``[b, R, d]``.

        ``agg_v/agg_r``: the aggregation-side incidence pairs (P_h for
        hh/ht, P_t for th/tt); ``s2``: the source-side per-entity sum
        (s_h or s_t); ``diag_z``: z when the roles coincide (hh, tt), else
        None -- the r1 != r2 exclusion of build_relation_graph.
        """
        if self.count:
            t1 = pair_sum(s2, agg_v, agg_r, num_relations)
            if diag_z is None:
                return t1
            deg = torch.bincount(agg_r, minlength=num_relations).to(s2.dtype)
            return t1 - deg.view(1, -1, 1) * diag_z
        f = self.rel_proj(node_repr)
        t1 = pair_sum(f * s2, agg_v, agg_r, num_relations)
        if diag_z is None:
            return t1
        t2 = pair_sum(f, agg_v, agg_r, num_relations)
        return t1 - t2 * diag_z


class FactorizedRelationStep(nn.Module):
    """One relation round: the four role channels (plus, optionally, their
    count twins), each with TRIX's boundary self-loop and update, summed --
    exactly RelNet's ``hidden_hh + hidden_ht + hidden_th + hidden_tt`` with
    the O(|V| alpha^2) pair materialization replaced by the exact O(|E|)
    factorization. The caller owns the shortcut."""

    ROLES = ("hh", "ht", "th", "tt")

    def __init__(self, dim: int, layer_norm: bool = True, count_channel: bool = True):
        super().__init__()
        self.channels = nn.ModuleDict(
            {r: _Channel(dim, layer_norm, count=False) for r in self.ROLES})
        self.count_channels = nn.ModuleDict(
            {r: _Channel(dim, layer_norm, count=True) for r in self.ROLES}
        ) if count_channel else None

    def forward(self, z, node_repr, pairs, boundary):
        """``z [b, R, d]``, ``node_repr [b, V, d]`` (already node_mlp'd),
        ``pairs``: an ``IncidencePairs``, ``boundary [b, R, d]``."""
        num_nodes, num_relations = pairs.num_nodes, pairs.num_relations
        s_h = pair_sum(z, pairs.head_r, pairs.head_v, num_nodes)
        s_t = pair_sum(z, pairs.tail_r, pairs.tail_v, num_nodes)
        spec = {
            "hh": (pairs.head_v, pairs.head_r, s_h, z),
            "ht": (pairs.head_v, pairs.head_r, s_t, None),
            "th": (pairs.tail_v, pairs.tail_r, s_h, None),
            "tt": (pairs.tail_v, pairs.tail_r, s_t, z),
        }
        hidden = None
        for role in self.ROLES:
            agg_v, agg_r, s2, diag_z = spec[role]
            m = self.channels[role].message(
                z, node_repr, agg_v, agg_r, s2, diag_z, num_relations)
            h = self.channels[role].update(z, m + boundary)
            hidden = h if hidden is None else hidden + h
            if self.count_channels is not None:
                mc = self.count_channels[role].message(
                    z, node_repr, agg_v, agg_r, s2, diag_z, num_relations)
                hidden = hidden + self.count_channels[role].update(z, mc + boundary)
        return hidden
