"""Context banks: per-relation tables of scored example triples.

A bank row is one (candidate, query) pair from the *inference graph*: the
encoder is run on a sampled edge (u, r, v) as if it were a query, and the row
records how a candidate entity looked to the encoder, together with a label
saying whether that candidate was the edge's actual tail. The readout
(``crest/pfn.py``) attends over these rows in context, PFN-style.

------------------------------------------------------------------------------
Leakage contract (docs/CREST_PLAN.md 4.1) -- the property the design rests on
------------------------------------------------------------------------------
* Bank rows are built from the **inference graph only**. The builder never
  sees validation or test edges, and asserts that every sampled edge is an
  inference-graph edge.
* For each sampled edge, that edge **and its inverse** are removed from the
  message graph before the encoder runs, so no row encodes the trivial
  answer of reading the queried edge off the graph.
* ``Ans(u, r)`` -- the set a negative must avoid -- is computed over the
  **inference graph only**, never over a graph containing test edges. A test
  triple (u, r, x) therefore *can* appear as a negative here, which is
  correct: at bank-build time the model cannot know test answers, and
  filtering them out would leak the test set into the bank.

The relation-graph statistics TRIX derives from the inference graph
(``data.relation_adj``) are deliberately *not* rebuilt per sampled edge: the
sampled edge is itself an inference-graph edge, so nothing unseen leaks, and
the removal above only exists to stop the entity-level message passing from
copying the answer out of the adjacency.

------------------------------------------------------------------------------
Shapes
------------------------------------------------------------------------------
Row feature (``row_features``)::

    f = [ x_v ; z_r ; x_v * z_r ; cos(x_v, z_r) ; s_v0 ]      size 3d + 2

which is 98 for the TRIX hidden size d = 32. A bank holds
``N_POSITIVE * (1 + NEG_PER_POS) = 20 * 4 = 80`` rows per relation id, so
bank tensors are ``[80, 98]``. Inverse relations are separate relation ids
(``r + num_relations // 2``), each with its own bank.

Graphs are duck-typed: anything with ``edge_index [2, E]``, ``edge_type [E]``,
``num_nodes`` and ``num_relations`` works, so PyG ``Data`` in the container
and the plain ``Graph`` below in host tests are interchangeable.  As
everywhere in this project, ``edge_index``/``edge_type`` already contain both
edge directions, with the inverse of relation ``r`` stored as
``r + num_relations // 2``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.nn import functional as F

#: Bank composition. 20 sampled edges per relation id (docs/CREST_PLAN.md 4.2
#: budgets exactly this), 3 negatives per positive.
N_POSITIVE = 20
NEG_PER_POS = 3
ROWS_PER_RELATION = N_POSITIVE * (1 + NEG_PER_POS)  # 80


def row_dim(d: int) -> int:
    """Row feature width for encoder dimension ``d``: 3d + 2 (98 for d=32)."""
    return 3 * d + 2


@dataclasses.dataclass
class Graph:
    """Minimal inference-graph container for host-side use.

    ``edge_index``/``edge_type`` hold both directions; the inverse of relation
    ``r`` is ``r + num_relations // 2``. ``num_relations`` is the total count
    including inverses and must therefore be even.
    """

    edge_index: torch.Tensor  # [2, E] long
    edge_type: torch.Tensor   # [E] long
    num_nodes: int
    num_relations: int

    def __post_init__(self):
        assert self.num_relations % 2 == 0, (
            "num_relations counts inverses too and must be even, got %d"
            % self.num_relations)


def inverse_relation(r: int, num_relations: int) -> int:
    """The id of r's inverse under the r <-> r + num_direct convention."""
    num_direct = num_relations // 2
    return r + num_direct if r < num_direct else r - num_direct


def relation_edges(graph, r: int) -> torch.Tensor:
    """Indices into ``edge_index`` of the edges typed ``r``."""
    return (graph.edge_type == int(r)).nonzero(as_tuple=True)[0]


def ans(graph, u: int, r: int) -> torch.Tensor:
    """Ans(u, r): every tail v with (u, r, v) in ``graph``, as a sorted tensor.

    ``graph`` must be the inference graph. Callers sampling negatives use the
    complement of this set; computing it over any graph containing test edges
    would leak them (docs/CREST_PLAN.md 4.1), so this function is the single
    place the set is formed and its contract is stated here.
    """
    mask = (graph.edge_index[0] == int(u)) & (graph.edge_type == int(r))
    return graph.edge_index[1, mask].unique()


def remove_edge_and_inverse(graph, u: int, r: int, v: int):
    """A copy of ``graph`` without (u, r, v) and without (v, inv(r), u).

    Removes *every* copy of both, in case the graph carries duplicate edges.
    Returns the same graph type it was given (PyG ``Data`` supports the same
    attribute assignment as the dataclass above).
    """
    r_inv = inverse_relation(int(r), graph.num_relations)
    ei, et = graph.edge_index, graph.edge_type
    drop = ((ei[0] == int(u)) & (ei[1] == int(v)) & (et == int(r))) | \
           ((ei[0] == int(v)) & (ei[1] == int(u)) & (et == r_inv))
    assert bool(drop.any()), (
        "edge (%d, %d, %d) is not in the graph it is being removed from -- "
        "bank edges must come from the inference graph" % (u, r, v))
    keep = ~drop
    import copy as _copy
    out = _copy.copy(graph)
    out.edge_index = ei[:, keep]
    out.edge_type = et[keep]
    return out


def row_features(x: torch.Tensor, z: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
    """``f = [x ; z ; x * z ; cos(x, z) ; s0]`` -> ``[..., n, 3d + 2]``.

    ``x``: candidate features ``[..., n, d]``; ``z``: the query's relation
    representation ``[..., d]``; ``s0``: the encoder's own score ``[..., n]``.
    """
    zb = z.unsqueeze(-2).expand_as(x)
    # F.cosine_similarity's eps guard keeps a zero vector finite; identical
    # semantics on torch 2.1 (container) and 2.5 (host).
    cos = F.cosine_similarity(x, zb, dim=-1)
    return torch.cat([x, zb, x * zb, cos.unsqueeze(-1), s0.unsqueeze(-1)], dim=-1)


class ContextBank:
    """Per-relation-id row tables: ``features [80, 3d+2]``, ``labels [80]``.

    Also carries the identity of what built it (graph id, checkpoint hash,
    seed), which is the disk-cache key: a rebuild that would produce an
    identical bank must not run (docs/CREST_PLAN.md 4.2).
    """

    def __init__(self, graph_id: str = "", checkpoint_hash: str = "", seed: int = 0):
        self.graph_id = graph_id
        self.checkpoint_hash = checkpoint_hash
        self.seed = int(seed)
        self._features: Dict[int, torch.Tensor] = {}
        self._labels: Dict[int, torch.Tensor] = {}

    def __contains__(self, relation_id: int) -> bool:
        return int(relation_id) in self._features

    def __len__(self) -> int:
        return len(self._features)

    def relation_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._features))

    def put(self, relation_id: int, features: torch.Tensor, labels: torch.Tensor) -> None:
        assert features.dim() == 2 and len(features) == len(labels), (features.shape, labels.shape)
        self._features[int(relation_id)] = features
        self._labels[int(relation_id)] = labels

    def features(self, relation_id: int) -> torch.Tensor:
        return self._features[int(relation_id)]

    def labels(self, relation_id: int) -> torch.Tensor:
        return self._labels[int(relation_id)]

    def stacked(self, relation_ids: Optional[Iterable[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenation of several relations' rows; the relation model's context."""
        ids = sorted(self._features) if relation_ids is None else [int(i) for i in relation_ids]
        feats = torch.cat([self._features[i] for i in ids], dim=0)
        labels = torch.cat([self._labels[i] for i in ids], dim=0)
        return feats, labels

    # -- disk cache ---------------------------------------------------------
    def cache_key(self) -> str:
        raw = "{}::{}::{}".format(self.graph_id, self.checkpoint_hash, self.seed)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def cache_path(self, root: str) -> str:
        safe = self.graph_id.replace(":", "_") or "unnamed"
        return os.path.join(root, "{}__{}.pt".format(safe, self.cache_key()))

    def save(self, root: str) -> str:
        os.makedirs(root, exist_ok=True)
        path = self.cache_path(root)
        tmp = path + ".tmp"
        torch.save({
            "graph_id": self.graph_id,
            "checkpoint_hash": self.checkpoint_hash,
            "seed": self.seed,
            "features": self._features,
            "labels": self._labels,
        }, tmp)
        os.replace(tmp, path)  # atomic: a crashed writer leaves no partial cache
        return path

    @classmethod
    def load(cls, path: str) -> "ContextBank":
        # weights_only=True: the cache is plain tensors, so opt out of pickle
        # arbitrary-code loading. Supported since torch 2.0, so identical on
        # the 2.1 container stack and the 2.5 host.
        state = torch.load(path, map_location="cpu", weights_only=True)
        bank = cls(state["graph_id"], state["checkpoint_hash"], int(state["seed"]))
        for rid, feats in state["features"].items():
            bank.put(int(rid), feats, state["labels"][rid])
        return bank

    @classmethod
    def load_or_build(cls, root: str, builder, graph_id: str, checkpoint_hash: str,
                      seed: int) -> "ContextBank":
        """The cost-control entry point: hit the cache or call ``builder()``.

        ``builder`` receives the ready-keyed empty bank and must fill it.
        """
        probe = cls(graph_id, checkpoint_hash, seed)
        path = probe.cache_path(root)
        if os.path.exists(path):
            return cls.load(path)
        builder(probe)
        probe.save(root)
        return probe


def _sample_edges(edge_ids: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    """``k`` edge indices; without replacement when the relation has enough
    edges, with replacement otherwise so the bank tensor keeps its [80, .]
    shape for every relation id present in the graph."""
    n = len(edge_ids)
    if n >= k:
        perm = torch.randperm(n, generator=generator)[:k]
    else:
        perm = torch.randint(n, (k,), generator=generator)
    return edge_ids[perm]


def _sample_negatives(graph, u: int, r: int, k: int,
                      generator: torch.Generator) -> torch.Tensor:
    """``k`` entities outside Ans(u, r) -- Ans over the inference graph only."""
    known = ans(graph, u, r)
    candidate_mask = torch.ones(graph.num_nodes, dtype=torch.bool)
    candidate_mask[known] = False
    candidates = candidate_mask.nonzero(as_tuple=True)[0]
    if len(candidates) == 0:
        raise ValueError(
            "every entity is an answer of (%d, %d); no negative exists. This "
            "cannot happen on a real graph and is not silently absorbed." % (u, r))
    idx = torch.randint(len(candidates), (k,), generator=generator)
    return candidates[idx]


def build_bank_entity(graph, encoder, seed: int = 1024,
                      num_positive: int = N_POSITIVE,
                      neg_per_pos: int = NEG_PER_POS,
                      relation_ids: Optional[Iterable[int]] = None,
                      bank: Optional[ContextBank] = None) -> ContextBank:
    """Build (or partially refresh) the entity-task bank from ``graph``.

    ``graph`` MUST be the inference graph -- see the module docstring's
    leakage contract. ``encoder.encode_single(graph, u, r)`` must return
    ``(x [num_nodes, d], z [d], s0 [num_nodes])`` for the query (u, r, ?);
    ``crest/run.py`` provides the TRIX adapter, tests provide a toy one.

    ``relation_ids`` restricts the build to the given ids -- this is the
    partial-refresh path of docs/CREST_PLAN.md 4.2: during training only the
    ids touched since the last refresh are rebuilt, into the same ``bank``.
    Relation ids with no inference-graph edge get no bank entry; the model
    falls back to the raw encoder score for them, because examples for such a
    relation could only come from edges the model must not see.
    """
    out = bank if bank is not None else ContextBank(seed=seed)
    generator = torch.Generator().manual_seed(int(seed))
    wanted = range(graph.num_relations) if relation_ids is None else sorted(set(int(i) for i in relation_ids))
    for rid in wanted:
        edge_ids = relation_edges(graph, rid)
        if len(edge_ids) == 0:
            continue
        chosen = _sample_edges(edge_ids, num_positive, generator)
        rows, labels = [], []
        for e in chosen.tolist():
            u = int(graph.edge_index[0, e])
            v = int(graph.edge_index[1, e])
            message_graph = remove_edge_and_inverse(graph, u, rid, v)
            x, z, s0 = encoder.encode_single(message_graph, u, rid)
            feats = row_features(x, z, s0)  # [num_nodes, 3d + 2]
            rows.append(feats[v])
            labels.append(1)
            for neg in _sample_negatives(graph, u, rid, neg_per_pos, generator).tolist():
                rows.append(feats[neg])
                labels.append(0)
        out.put(rid, torch.stack(rows), torch.tensor(labels, dtype=torch.long))
    return out


def build_bank_relation(graph, encoder, seed: int = 1024,
                        num_positive: int = N_POSITIVE,
                        neg_per_pos: int = NEG_PER_POS,
                        relation_ids: Optional[Iterable[int]] = None,
                        bank: Optional[ContextBank] = None) -> ContextBank:
    """Relation-task bank. Same leakage contract, candidates are relations.

    For a sampled edge (u, r, v) the query is (u, ?, v);
    ``encoder.encode_relation_single(graph, u, v)`` must return
    ``(w [num_direct, d], c [d], s0 [num_direct])`` -- per-candidate relation
    features, a query context vector and the encoder's own relation scores.
    Candidates are direct relations only, matching TRIX's
    ``all_negative_relation`` (which ranks over ``num_relations // 2``).

    The positive row is the true relation r; negatives are relations that do
    not hold between u and v in the inference graph. Rows are keyed by the
    positive's relation id; at query time the reader attends over
    ``bank.stacked()``, since a relation query does not know its own relation.
    """
    out = bank if bank is not None else ContextBank(seed=seed)
    generator = torch.Generator().manual_seed(int(seed))
    num_direct = graph.num_relations // 2
    wanted = range(num_direct) if relation_ids is None else sorted(set(int(i) for i in relation_ids))
    for rid in wanted:
        assert rid < num_direct, "relation banks are keyed by direct relation ids"
        edge_ids = relation_edges(graph, rid)
        if len(edge_ids) == 0:
            continue
        chosen = _sample_edges(edge_ids, num_positive, generator)
        rows, labels = [], []
        for e in chosen.tolist():
            u = int(graph.edge_index[0, e])
            v = int(graph.edge_index[1, e])
            message_graph = remove_edge_and_inverse(graph, u, rid, v)
            w, c, s0 = encoder.encode_relation_single(message_graph, u, v)
            feats = row_features(w, c, s0)  # [num_direct, 3d + 2]
            rows.append(feats[rid])
            labels.append(1)
            # relations that hold between u and v in the inference graph
            holds = graph.edge_type[(graph.edge_index[0] == u) & (graph.edge_index[1] == v)]
            holds = holds[holds < num_direct].unique()
            candidate_mask = torch.ones(num_direct, dtype=torch.bool)
            candidate_mask[holds] = False
            candidates = candidate_mask.nonzero(as_tuple=True)[0]
            if len(candidates) == 0:
                raise ValueError(
                    "every direct relation holds between %d and %d; no negative "
                    "relation exists" % (u, v))
            idx = torch.randint(len(candidates), (neg_per_pos,), generator=generator)
            for neg in candidates[idx].tolist():
                rows.append(feats[neg])
                labels.append(0)
        out.put(rid, torch.stack(rows), torch.tensor(labels, dtype=torch.long))
    return out
