"""Track B: an optional random channel that breaks relation symmetries.

FLOCK is *not* imported (docs/CREST_PLAN.md section 5, track B). Its walk
sampler is expensive because it samples per candidate entity -- measured here
at 24x to 412x ULTRA -- so this module samples a handful of walks from the
query head only, per query, in plain torch on the TRIX stack. The recording
protocol is FLOCK's as a *specification*: anonymised node ids (first-visit
order), relation direction bits, and a query flag marking the head.

A channel appends ``feature_dim`` extra columns to the query rows; bank rows
receive zeros there, since walks are a property of the live query, not of the
stored examples. ``active = True`` on every channel is what
``crest/tests/test_equivariance.py`` keys its skip on: a random channel
deliberately breaks the symmetry that test asserts.

Determinism: every sampler draws from an explicit ``torch.Generator``. FLOCK
ships ``np.random.seed`` commented out in its ``set_seed`` -- the reason its
instances could not be regenerated -- and this module does not repeat that
mistake: there is no path through here that touches a global RNG.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def sample_walks(graph, head: int, num_walks: int = 8, walk_length: int = 8,
                 generator: Optional[torch.Generator] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``num_walks`` uniform random walks out of ``head``.

    Returns ``(nodes [w, L+1], relations [w, L], direction_bits [w, L])``.
    A node with no outgoing edge ends its walk in place: the remaining steps
    repeat the node with relation -1. Direction bit 1 marks an inverse-typed
    edge (``r >= num_relations // 2``); since the graph stores both
    directions, walking it as-is already walks both ways.

    Cost: ``num_walks * walk_length`` categorical draws per query -- a few
    hundred steps, no C++ extension, per the track B budget.
    """
    ei, et = graph.edge_index, graph.edge_type
    num_direct = graph.num_relations // 2
    # CSR-style adjacency: edges sorted by source, offsets by cumulative count
    order = ei[0].argsort()
    src_sorted = ei[0][order]
    counts = torch.bincount(src_sorted, minlength=graph.num_nodes)
    offsets = counts.cumsum(0) - counts

    nodes = torch.full((num_walks, walk_length + 1), int(head), dtype=torch.long)
    rels = torch.full((num_walks, walk_length), -1, dtype=torch.long)
    bits = torch.zeros(num_walks, walk_length, dtype=torch.long)
    for w in range(num_walks):
        cur = int(head)
        for step in range(walk_length):
            n_out = int(counts[cur])
            if n_out == 0:
                nodes[w, step + 1:] = cur
                break
            pick = int(torch.randint(n_out, (1,), generator=generator))
            e = int(order[int(offsets[cur]) + pick])
            rels[w, step] = et[e]
            bits[w, step] = int(int(et[e]) >= num_direct)
            cur = int(ei[1, e])
            nodes[w, step + 1] = cur
    return nodes, rels, bits


def anonymise(nodes: torch.Tensor, head: int) -> torch.Tensor:
    """Map raw node ids to first-visit order across all walks; head is 0.

    This is the recording protocol's anonymisation: the channel must carry
    the *shape* of the walks, never entity identity, or it would break the
    entity-permutation equivariance the deterministic model has.
    """
    mapping = {int(head): 0}
    out = torch.zeros_like(nodes)
    for w in range(nodes.shape[0]):
        for i in range(nodes.shape[1]):
            v = int(nodes[w, i])
            if v not in mapping:
                mapping[v] = len(mapping)
            out[w, i] = mapping[v]
    return out


class NoiseChannel(nn.Module):
    """Variant B1-i: iid Gaussian noise per node, resampled every pass."""

    active = True

    def __init__(self, feature_dim: int = 8, seed: int = 1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.generator = torch.Generator().manual_seed(int(seed))

    def forward(self, graph, head: int) -> torch.Tensor:
        """Per-node channel features ``[num_nodes, feature_dim]``."""
        return torch.randn(graph.num_nodes, self.feature_dim, generator=self.generator)


class WalkChannel(nn.Module):
    """Variant B1-iii: features from ``num_walks`` walks out of the head.

    Each visited node accumulates an embedding of (anonymised id, direction
    bit, step position, query flag); unvisited nodes stay at zero. The
    embedding tables are integer-free learnable parameters, so the channel
    trains with the readout.
    """

    active = True

    #: anonymised ids beyond this bucket share one embedding
    MAX_ANON = 32

    def __init__(self, feature_dim: int = 8, num_walks: int = 8,
                 walk_length: int = 8, seed: int = 1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.generator = torch.Generator().manual_seed(int(seed))
        self.anon_embed = nn.Embedding(self.MAX_ANON + 1, feature_dim)
        self.bit_embed = nn.Embedding(2, feature_dim)
        self.step_embed = nn.Embedding(walk_length + 1, feature_dim)
        self.query_flag = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, graph, head: int) -> torch.Tensor:
        nodes, rels, bits = sample_walks(
            graph, head, self.num_walks, self.walk_length, self.generator)
        anon = anonymise(nodes, head).clamp(max=self.MAX_ANON)
        out = torch.zeros(graph.num_nodes, self.feature_dim)
        for w in range(self.num_walks):
            for i in range(self.walk_length + 1):
                v = int(nodes[w, i])
                feat = self.anon_embed(anon[w, i]) + self.step_embed(torch.tensor(i))
                if i > 0 and int(rels[w, i - 1]) >= 0:
                    feat = feat + self.bit_embed(bits[w, i - 1])
                out[v] = out[v] + feat
        out[int(head)] = out[int(head)] + self.query_flag
        return out
