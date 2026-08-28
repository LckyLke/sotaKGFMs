"""Graph utilities: incidence pairs, edge removal, n-hop balls, answer sets.

Graphs are duck-typed as everywhere in this project: anything with
``edge_index [2, E]``, ``edge_type [E]``, ``num_nodes`` and ``num_relations``
works, so PyG ``Data`` in the container and the plain ``Graph`` below in
tests are interchangeable. ``edge_index``/``edge_type`` always contain both
edge directions, with the inverse of relation ``r`` stored as
``r + num_relations // 2``.

------------------------------------------------------------------------------
Incidence pairs -- the object the factorized relation step runs on
------------------------------------------------------------------------------
TRIX materializes a relation graph whose hh edge set is
{(r1, r2, v) : v heads an r1-edge and an r2-edge, r1 != r2} (and ht/th/tt
analogously; ``trix/tasks.py::build_relation_graph``). INCITE never builds
those pairs. It stores only the *unique incidence pairs*

    P_h = {(v, r) : some edge (v, r, .) exists}     head role
    P_t = {(v, r) : some edge (., r, v) exists}     tail role

which is all the factorization in docs/INCITE_DESIGN.md section A needs.
Uniqueness matters: build_relation_graph iterates over the key sets of its
per-node dicts, so an entity heading five r-edges contributes ONE (v, r)
incidence, not five, and the factorized sums must count the same way.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch

__all__ = [
    "Graph",
    "IncidencePairs",
    "incidence_pairs",
    "inverse_relation",
    "remove_edge_and_inverse",
    "ans",
    "n_hop_ball",
    "sorted_by_source",
]


@dataclasses.dataclass
class Graph:
    """Minimal graph container for tests; PyG ``Data`` replaces it in the
    container. ``num_relations`` counts inverses too and must be even."""

    edge_index: torch.Tensor  # [2, E] long
    edge_type: torch.Tensor   # [E] long
    num_nodes: int
    num_relations: int

    def __post_init__(self):
        assert self.num_relations % 2 == 0, (
            "num_relations counts inverses too and must be even, got %d"
            % self.num_relations)


@dataclasses.dataclass
class IncidencePairs:
    """Unique (entity, relation) incidences of one graph, per role.

    ``head_v/head_r``: entity and relation id of every unique pair in P_h;
    ``tail_v/tail_r``: the same for P_t. All 1-d long tensors on the graph's
    device. ``num_nodes``/``num_relations`` are carried so consumers never
    have to re-derive them from a possibly edge-removed graph.
    """

    head_v: torch.Tensor
    head_r: torch.Tensor
    tail_v: torch.Tensor
    tail_r: torch.Tensor
    num_nodes: int
    num_relations: int


def _unique_pairs(v: torch.Tensor, r: torch.Tensor, num_relations: int
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    key = v * int(num_relations) + r
    key = torch.unique(key)
    return key // int(num_relations), key % int(num_relations)


def incidence_pairs(graph) -> IncidencePairs:
    """Extract the unique incidence pairs of ``graph``.

    O(E log E). Callers cache the result per graph where the edge set is
    stable (evaluation); training paths that remove batch edges recompute --
    but note that, mirroring TRIX, the relation step normally runs on the
    *full* graph's incidences (TRIX's relation_adj is built once at dataset
    processing time and remove_easy_edges never touches it), so the cached
    pairs are what the model consumes even while the entity step's edge list
    has the batch edges dropped.
    """
    ei, et = graph.edge_index, graph.edge_type
    num_relations = int(graph.num_relations)
    hv, hr = _unique_pairs(ei[0], et, num_relations)
    tv, tr = _unique_pairs(ei[1], et, num_relations)
    return IncidencePairs(head_v=hv, head_r=hr, tail_v=tv, tail_r=tr,
                          num_nodes=int(graph.num_nodes),
                          num_relations=num_relations)


def inverse_relation(r: int, num_relations: int) -> int:
    """The id of r's inverse under the r <-> r + num_direct convention."""
    num_direct = num_relations // 2
    return r + num_direct if r < num_direct else r - num_direct


def remove_edge_and_inverse(graph, u: int, r: int, v: int):
    """A copy of ``graph`` without (u, r, v) and without (v, inv(r), u).

    Removes every copy of both, in case the graph carries duplicates, and
    asserts the edge was present: support edges must come from the inference
    graph (the leakage contract, ported from crest/bank.py).
    """
    r_inv = inverse_relation(int(r), int(graph.num_relations))
    ei, et = graph.edge_index, graph.edge_type
    drop = ((ei[0] == int(u)) & (ei[1] == int(v)) & (et == int(r))) | \
           ((ei[0] == int(v)) & (ei[1] == int(u)) & (et == r_inv))
    assert bool(drop.any()), (
        "edge (%d, %d, %d) is not in the graph it is being removed from -- "
        "support edges must come from the inference graph" % (u, r, v))
    import copy as _copy
    out = _copy.copy(graph)
    out.edge_index = ei[:, ~drop]
    out.edge_type = et[~drop]
    return out


def remove_edges_batch(graph, triples):
    """One copy of ``graph`` missing every (u, r, v) in ``triples`` and their
    inverses. ``triples`` is an iterable of (u, r, v) ints. Each edge is
    asserted present, as in ``remove_edge_and_inverse``."""
    ei, et = graph.edge_index, graph.edge_type
    drop = torch.zeros(et.shape[0], dtype=torch.bool, device=et.device)
    for u, r, v in triples:
        r_inv = inverse_relation(int(r), int(graph.num_relations))
        this = ((ei[0] == int(u)) & (ei[1] == int(v)) & (et == int(r))) | \
               ((ei[0] == int(v)) & (ei[1] == int(u)) & (et == r_inv))
        assert bool(this.any()), (
            "edge (%d, %d, %d) is not in the graph it is being removed from -- "
            "support edges must come from the inference graph" % (u, r, v))
        drop |= this
    import copy as _copy
    out = _copy.copy(graph)
    out.edge_index = ei[:, ~drop]
    out.edge_type = et[~drop]
    return out


def ans(graph, u: int, r: int) -> torch.Tensor:
    """Ans(u, r): every tail v with (u, r, v) in ``graph``, sorted unique.

    ``graph`` must be the inference graph. Negatives are drawn from the
    complement of this set; computing it over any graph containing test
    edges would leak them. Single place the set is formed (crest precedent).
    """
    mask = (graph.edge_index[0] == int(u)) & (graph.edge_type == int(r))
    return graph.edge_index[1, mask].unique()


def n_hop_ball(graph, start: int, hops: int = 3,
               cap: Optional[int] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Entities within ``hops`` edges of ``start`` (start excluded).

    BFS over the typed multigraph, both directions already materialized in
    ``edge_index``. ``cap`` bounds the returned set on dense graphs: when the
    ball exceeds it, a seeded subsample of size ``cap`` is returned (the
    design's WN18RR estimate is ~100 entities; Hetionet-like graphs explode
    and the cap keeps support building O(1) per head).
    """
    device = graph.edge_index.device
    visited = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    frontier = torch.zeros_like(visited)
    frontier[int(start)] = True
    visited[int(start)] = True
    src, dst = graph.edge_index[0], graph.edge_index[1]
    for _ in range(int(hops)):
        hit = frontier[src]
        nxt = torch.zeros_like(visited)
        nxt[dst[hit]] = True
        frontier = nxt & ~visited
        visited |= nxt
        if not bool(frontier.any()):
            break
    visited[int(start)] = False
    ball = visited.nonzero(as_tuple=True)[0]
    if cap is not None and len(ball) > int(cap):
        gen = generator if generator is not None else torch.Generator(device="cpu")
        perm = torch.randperm(len(ball), generator=gen)[:int(cap)].to(device)
        ball = ball[perm]
    return ball


def sorted_by_source(edge_index: torch.Tensor):
    """Edges sorted by (source, target) -- what the walk sampler's CSR wants.

    Returns (sorted_src, sorted_dst, order) where ``order`` indexes back into
    the original edge list (for edge types).
    """
    src, dst = edge_index[0], edge_index[1]
    key = src * (dst.max() + 1 if len(dst) else 1) + dst
    order = key.argsort()
    return src[order], dst[order], order
