"""Materialized reference for the layer gate (docs/INCITE_PLAN.md phase 1.1).

``materialized_relation_step`` computes the SAME quantity as
``FactorizedRelationStep.forward`` -- same module, same weights -- but the
slow way: it first materializes the TRIX-style relation-graph pair lists
{(r1, r2, v)} per role channel (the O(|V| alpha^2) object INCITE exists to
avoid) and then sums per-pair messages f_c(v) * z[r2] into r1, mirroring the
executed direction of TRIX's rspmm kernel (see incite/layers.py). The
factorization is exact in real arithmetic, so the two must agree to float
reordering: the gate asserts allclose at 1e-5.

``build_role_pairs`` restates ``trix/tasks.py::build_relation_graph``'s pair
construction (unique incidences, r1 != r2 excluded on hh and tt) in
vectorized form; the tests also compare it against the pinned
build_relation_graph itself where TRIX is importable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .graphs import IncidencePairs, incidence_pairs
from .layers import FactorizedRelationStep

__all__ = ["build_role_pairs", "materialized_relation_step"]


def _pairs_by_entity(v: torch.Tensor, r: torch.Tensor, num_nodes: int):
    """Group unique incidence pairs by entity: sorted (v, r) plus per-entity
    slice offsets, for the cartesian expansion below."""
    order = v.argsort(stable=True)
    v_sorted, r_sorted = v[order], r[order]
    counts = torch.bincount(v_sorted, minlength=num_nodes)
    offsets = counts.cumsum(0) - counts
    return v_sorted, r_sorted, counts, offsets


def build_role_pairs(graph) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The four materialized pair lists {(r1, r2, v)} of build_relation_graph.

    Returns ``{role: (r1, r2, v)}`` with r1 the aggregation side in the
    executed direction (edge_index[0] of the TRIX relation graph). hh and tt
    exclude r1 == r2; ht and th do not, exactly as upstream.
    """
    pairs = incidence_pairs(graph)
    num_nodes = pairs.num_nodes
    grouped = {
        "h": _pairs_by_entity(pairs.head_v, pairs.head_r, num_nodes),
        "t": _pairs_by_entity(pairs.tail_v, pairs.tail_r, num_nodes),
    }
    out = {}
    for role, (side1, side2) in (("hh", ("h", "h")), ("ht", ("h", "t")),
                                 ("th", ("t", "h")), ("tt", ("t", "t"))):
        _, r1_sorted, c1, o1 = grouped[side1]
        _, r2_sorted, c2, o2 = grouped[side2]
        r1_list, r2_list, v_list = [], [], []
        for v in range(num_nodes):
            n1, n2 = int(c1[v]), int(c2[v])
            if n1 == 0 or n2 == 0:
                continue
            a = r1_sorted[int(o1[v]):int(o1[v]) + n1]
            b = r2_sorted[int(o2[v]):int(o2[v]) + n2]
            rr1 = a.repeat_interleave(n2)
            rr2 = b.repeat(n1)
            if side1 == side2:
                keep = rr1 != rr2
                rr1, rr2 = rr1[keep], rr2[keep]
            r1_list.append(rr1)
            r2_list.append(rr2)
            v_list.append(torch.full_like(rr1, v))
        cat = lambda xs: (torch.cat(xs) if xs else
                          torch.zeros(0, dtype=torch.long, device=graph.edge_index.device))
        out[role] = (cat(r1_list), cat(r2_list), cat(v_list))
    return out


def materialized_relation_step(step: FactorizedRelationStep, z: torch.Tensor,
                               node_repr: torch.Tensor, graph,
                               boundary: torch.Tensor) -> torch.Tensor:
    """Reference forward of ``step`` over the materialized pair lists.

    Uses the step's own submodules (rel_proj, update) so the only difference
    from ``step.forward`` is HOW the message sums are formed.
    """
    role_pairs = build_role_pairs(graph)
    num_relations = int(graph.num_relations)
    hidden = None
    for role in step.ROLES:
        r1, r2, v = role_pairs[role]
        channel = step.channels[role]
        f = channel.rel_proj(node_repr)  # [b, V, d]
        msg = f.index_select(1, v) * z.index_select(1, r2)  # [b, P, d]
        agg = z.new_zeros(z.shape[0], num_relations, z.shape[-1])
        agg.index_add_(1, r1, msg)
        h = channel.update(z, agg + boundary)
        hidden = h if hidden is None else hidden + h
        if step.count_channels is not None:
            cmsg = z.index_select(1, r2)  # f(v) = 1
            cagg = z.new_zeros(z.shape[0], num_relations, z.shape[-1])
            cagg.index_add_(1, r1, cmsg)
            hidden = hidden + step.count_channels[role].update(z, cagg + boundary)
    return hidden
