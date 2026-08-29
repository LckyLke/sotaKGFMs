"""Walk-equality features (docs/INCITE_DESIGN.md section C).

n = 32 anonymized walks of length 8 from each start node, recorded with
FLOCK's protocol: within one walk, entities are renamed by first-visit index
and relations by first-appearance index. The record keeps relation IDENTITY
inside the walk -- two steps over the same relation share one symbol, two
different relations get two symbols -- which is what separates the PETALS
candidates that every deterministic invariant collapses (section C's
b0->a1->a3 = (alpha, alpha) vs b0->a2->a4 = (alpha, beta) example). Global
entity and relation ids never enter the encoder, so the module transfers
across relation vocabularies (ind_er) unchanged.

Each walk is encoded by a small GRU over per-step tokens
(anon-relation embedding + anon-entity embedding); the GRU outputs are
mean-pooled into the states of the entities visited and the relations
traversed, and the model adds those pooled states to its initial entity and
relation states before round 1.

------------------------------------------------------------------------------
Seeding (docs/INCITE_PLAN.md amendment: one seeded pass, seed 1024)
------------------------------------------------------------------------------
Sampling draws from a fresh ``torch.Generator`` seeded with
``seed + seed_offset`` on every call, so identical (graph, starts, seed)
always yield identical walks and therefore identical features -- the
determinism a test asserts. Evaluation uses offset 0 everywhere; training
passes the step index as the offset so consecutive steps see different
walks while the whole run stays reproducible. Test-time walk averaging is
cut from the primary protocol (PLAN amendment); unseeded spread is measured
separately by passing varying offsets.

CPU and CUDA generators draw different sequences for the same seed; that is
the existing device rule (ranks from CPU and GPU runs never mix), not a new
hazard.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .graphs import sorted_by_source

__all__ = ["sample_walks", "anonymize", "WalkModule"]


def _csr(graph):
    """Outgoing-edge CSR of the graph: sorted dst/type + per-node offsets."""
    src, dst, order = sorted_by_source(graph.edge_index)
    etype = graph.edge_type[order]
    counts = torch.bincount(src, minlength=graph.num_nodes)
    offsets = counts.cumsum(0) - counts
    return dst, etype, counts, offsets


def sample_walks(graph, starts: torch.Tensor, num_walks: int, walk_length: int,
                 seed: int, seed_offset: int = 0
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform random walks over outgoing edges.

    Returns ``(entities [B, n, L+1], relations [B, n, L], mask [B, n, L])``
    where B = len(starts); ``mask`` is False past a dead end (a node with no
    outgoing edge -- rare, since inverses are materialized, but a start node
    can be isolated) and the padded entries repeat the last node with
    relation 0.
    """
    device = starts.device
    dst, etype, counts, offsets = _csr(graph)
    b = len(starts)
    n, length = int(num_walks), int(walk_length)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(seed_offset))

    cur = starts.view(b, 1).expand(b, n).contiguous()
    ents = [cur]
    rels, masks = [], []
    alive = torch.ones(b, n, dtype=torch.bool, device=device)
    for _ in range(length):
        deg = counts[cur]
        alive = alive & (deg > 0)
        # a dead walk keeps drawing on a clamped degree so the generator
        # consumption stays shape-stable; its draws are masked out below
        draw = torch.rand(b, n, device=device, generator=generator)
        # Dead lanes are masked below, but their gather still executes; for
        # a zero-degree node whose CSR offset sits at the END of the edge
        # array (trailing edgeless node), offsets[cur] == num_edges and
        # dst[pick] reads out of bounds -- a device-side assert that took
        # ~290k queries to hit. The clamp is value-neutral: alive lanes have
        # deg > 0 and never reach it, dead lanes are overwritten by where.
        pick = ((draw * deg.clamp(min=1)).long() + offsets[cur]).clamp(
            max=max(dst.numel() - 1, 0))
        nxt = torch.where(alive, dst[pick], cur)
        rel = torch.where(alive, etype[pick], torch.zeros_like(etype[pick]))
        ents.append(nxt)
        rels.append(rel)
        masks.append(alive.clone())
        cur = nxt
    return (torch.stack(ents, dim=-1), torch.stack(rels, dim=-1),
            torch.stack(masks, dim=-1))


def _first_visit_index(seq: torch.Tensor) -> torch.Tensor:
    """Per-walk anonymization: rename each id to its first-visit rank.

    ``seq [..., L]`` -> same shape, values in [0, L). Vectorized: for each
    position t, find the first position with the same id, then rank those
    first positions.
    """
    eq = seq.unsqueeze(-1) == seq.unsqueeze(-2)          # [..., L, L]
    length = seq.shape[-1]
    pos = torch.arange(length, device=seq.device)
    # first position j with seq[j] == seq[t]
    big = torch.where(eq, pos.expand_as(eq), torch.full_like(pos.expand_as(eq), length))
    first = big.min(dim=-1).values                        # [..., L]
    is_new = first == pos                                 # position introduces a new id
    rank = is_new.long().cumsum(-1) - 1                   # rank of each introduction
    return rank.gather(-1, first)


def anonymize(ents: torch.Tensor, rels: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """FLOCK's record: entities by first-visit index, relations by
    first-appearance index, both per walk."""
    return _first_visit_index(ents), _first_visit_index(rels)


class WalkModule(nn.Module):
    """Sample, anonymize, GRU-encode, and pool walks into node/relation states."""

    def __init__(self, dim: int, num_walks: int = 32, walk_length: int = 8,
                 seed: int = 1024):
        super().__init__()
        self.dim = int(dim)
        self.num_walks = int(num_walks)
        self.walk_length = int(walk_length)
        self.seed = int(seed)
        # anon vocabularies: at most L+1 entity symbols, L relation symbols
        self.ent_emb = nn.Embedding(self.walk_length + 1, self.dim)
        self.rel_emb = nn.Embedding(self.walk_length, self.dim)
        self.gru = nn.GRU(self.dim, self.dim, batch_first=True)
        self.out_ent = nn.Linear(self.dim, self.dim)
        self.out_rel = nn.Linear(self.dim, self.dim)

    def forward(self, graph, starts: torch.Tensor, seed_offset: int = 0
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pooled walk states ``(w_ent [B, V, dim], w_rel [B, R, dim])`` for
        walks started at ``starts [B]``."""
        b = len(starts)
        num_nodes, num_relations = int(graph.num_nodes), int(graph.num_relations)
        ents, rels, mask = sample_walks(graph, starts, self.num_walks,
                                        self.walk_length, self.seed, seed_offset)
        anon_e, anon_r = anonymize(ents, rels)
        # step token t: the relation taken at step t plus the entity arrived at
        tokens = self.rel_emb(anon_r) + self.ent_emb(anon_e[..., 1:])
        flat = tokens.view(b * self.num_walks, self.walk_length, self.dim)
        outputs, _ = self.gru(flat)
        outputs = outputs.view(b, self.num_walks, self.walk_length, self.dim)
        outputs = outputs * mask.unsqueeze(-1)

        w_ent = outputs.new_zeros(b, num_nodes, self.dim)
        w_rel = outputs.new_zeros(b, num_relations, self.dim)
        n_ent = outputs.new_zeros(b, num_nodes, 1)
        n_rel = outputs.new_zeros(b, num_relations, 1)
        flat_out = outputs.reshape(b, -1, self.dim)
        flat_mask = mask.reshape(b, -1, 1).to(outputs.dtype)
        ent_idx = ents[..., 1:].reshape(b, -1)
        rel_idx = rels.reshape(b, -1)
        for i in range(b):  # V-sized scatters; b is the (small) query batch
            w_ent[i].index_add_(0, ent_idx[i], flat_out[i] * flat_mask[i])
            n_ent[i].index_add_(0, ent_idx[i], flat_mask[i])
            w_rel[i].index_add_(0, rel_idx[i], flat_out[i] * flat_mask[i])
            n_rel[i].index_add_(0, rel_idx[i], flat_mask[i])
        w_ent = self.out_ent(w_ent / n_ent.clamp(min=1.0))
        w_rel = self.out_rel(w_rel / n_rel.clamp(min=1.0))
        return w_ent, w_rel
