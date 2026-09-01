"""The INCITE network: alternating entity / factorized-relation rounds with
walk features, support readout, and joint task heads.

docs/INCITE_DESIGN.md sections A (network), C (walks), D (heads), with the
PLAN's amendments. One round = one entity step then one relation step; 6
rounds. The TRIX-reduction mode of docs/INCITE_PLAN.md phase 1 is reached by
constructing with ``walks=None`` and scoring with ``support=None`` -- the
configs (incite_phase1.yaml) spell that out; nothing here special-cases it.

------------------------------------------------------------------------------
Task flag (design D)
------------------------------------------------------------------------------
The label (query) vector carries a learned task embedding: entity prediction
labels the query relation with ``1 + task_emb[0]``; relation prediction
labels h with ``+q``, t with ``-q`` and every relation with ``q = 1 +
task_emb[1]``, so one pass scores all relations. The unlabeled encoding used
by the support module labels everything with plain ones and no flag.

------------------------------------------------------------------------------
What mirrors TRIX verbatim
------------------------------------------------------------------------------
``remove_easy_edges``, ``negative_sample_to_tail`` and ``compute_ranking``
are restated from the pinned tree (base_nbfnet.py / tasks.py) so the host-
side pieces behave identically without importing TRIX; the container driver
still uses TRIX's own tasks module for masks and ranking. As in TRIX, the
relation-level structure (here: the incidence pairs) is built from the FULL
graph and is not touched by easy-edge removal -- TRIX's relation_adj is a
dataset-processing-time object -- while the entity steps and the walk
sampler run on the edge-removed message graph.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .graphs import IncidencePairs, incidence_pairs
from .layers import EntityStep, FactorizedRelationStep
from .support import SupportStore, row_dim
from .walks import WalkModule

__all__ = ["INCITE", "SupportReadout", "compute_ranking",
           "remove_easy_edges", "negative_sample_to_tail", "mask_halflinks"]

TASK_ENTITY = 0
TASK_RELATION = 1


# ---------------------------------------------------------------------------
# Restated TRIX helpers (pin 7596e14e); copied, not imported -- crest precedent
# ---------------------------------------------------------------------------
def _edge_match(edge_index, query_index):
    from functools import reduce
    base = edge_index.max(dim=1)[0] + 1
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start
    offset = num_match.cumsum(0) - num_match
    range_ = torch.arange(num_match.sum(), device=edge_index.device)
    range_ = range_ + (start - offset).repeat_interleave(num_match)
    return order[range_], num_match


def remove_easy_edges(data, h_index, t_index, r_index):
    """base_nbfnet.remove_easy_edges, remove_one_hop=False branch."""
    h_ext = torch.cat([h_index, t_index], dim=-1)
    t_ext = torch.cat([t_index, h_index], dim=-1)
    r_ext = torch.cat([r_index, r_index + data.num_relations // 2], dim=-1)
    edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
    easy_edge = torch.stack([h_ext, t_ext, r_ext]).flatten(1)
    index = _edge_match(edge_index, easy_edge)[0]
    mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool,
                      device=data.edge_index.device)
    mask[index] = False
    out = copy.copy(data)
    out.edge_index = data.edge_index[:, mask]
    out.edge_type = data.edge_type[mask]
    return out


def negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel):
    """base_nbfnet.negative_sample_to_tail, verbatim semantics."""
    is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
    new_h_index = torch.where(is_t_neg, h_index, t_index)
    new_t_index = torch.where(is_t_neg, t_index, h_index)
    new_r_index = torch.where(is_t_neg, r_index, r_index + num_direct_rel)
    return new_h_index, new_t_index, new_r_index


def mask_halflinks(graph, h, r, t, mask_answer, mask_query, num_direct):
    """Half-link masking (2026-09-01): a copy of ``graph`` without the
    answer half and/or the query half of the given tail-form positives.

    Gregucci et al. (arXiv 2606.18001) show KGFMs lean on the answer half:
    a candidate that already has an incoming r-edge is scored up, and on
    the 28 percent of test queries whose true answer has none (SQUA) MRR
    falls below 0.2. Pretraining graphs are dense, so training positives
    almost always carry a seen answer half. Masking makes the training
    scenario mix look like the test mix.

    Rows are in tail form (``negative_sample_to_tail``): query ``(h_i, r_i)``
    with target ``t_i``, relation ids in the doubled vocabulary. For row i:

    * ``mask_answer[i]``: drop every edge ``(x, r_i, t_i)`` and its inverse
      copy ``(t_i, inv(r_i), x)`` -- t_i loses its incoming r-edges;
    * ``mask_query[i]``: drop every edge ``(h_i, r_i, x)`` and its inverse
      copy ``(x, inv(r_i), h_i)`` -- h_i loses its outgoing r-edges.

    The positive edge itself is among the dropped ones, so this subsumes
    ``remove_easy_edges`` for masked rows. The relation-level incidence
    pairs (built from the full graph, as in TRIX) are untouched: they carry
    a per-relation bulk statistic, not a per-entity signal.
    """
    if not (bool(mask_answer.any()) or bool(mask_query.any())):
        return graph
    ei, et = graph.edge_index, graph.edge_type
    R2 = int(graph.num_relations)

    def inv(rr):
        return torch.where(rr < num_direct, rr + num_direct, rr - num_direct)

    src_key = ei[0] * R2 + et
    dst_key = ei[1] * R2 + et
    drop = torch.zeros(et.shape[0], dtype=torch.bool, device=et.device)
    if bool(mask_answer.any()):
        ta, ra = t[mask_answer], r[mask_answer]
        drop |= torch.isin(dst_key, ta * R2 + ra)          # (x, r, t)
        drop |= torch.isin(src_key, ta * R2 + inv(ra))     # (t, inv r, x)
    if bool(mask_query.any()):
        hq, rq = h[mask_query], r[mask_query]
        drop |= torch.isin(src_key, hq * R2 + rq)          # (h, r, x)
        drop |= torch.isin(dst_key, hq * R2 + inv(rq))     # (x, inv r, h)
    out = copy.copy(graph)
    out.edge_index = ei[:, ~drop]
    out.edge_type = et[~drop]
    return out


def compute_ranking(pred, target, mask=None):
    """trix/tasks.py::compute_ranking: 1-based, pessimistic, strict."""
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


# ---------------------------------------------------------------------------
# Support readout
# ---------------------------------------------------------------------------
class SupportReadout(nn.Module):
    """Cross-attention of per-candidate features over retrieved support rows.

    One head, one small key space: the design budgets attention over at most
    80 rows per query, so anything larger would be decoration. The output is
    a scalar residual added to the base score. Rows arrive detached
    (support.py's gradient contract); gradients flow into the projections
    and into the query-side features only.
    """

    def __init__(self, dim: int, rdim: int, att_dim: int = 32):
        super().__init__()
        self.q = nn.Linear(2 * dim, att_dim)
        self.k = nn.Linear(rdim, att_dim)
        self.v = nn.Linear(rdim, att_dim)
        self.out = nn.Sequential(nn.Linear(att_dim, att_dim), nn.ReLU(),
                                 nn.Linear(att_dim, 1))
        self.scale = att_dim ** -0.5

    def forward(self, feature: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """``feature [c, 2d]``, ``rows [S, rdim]`` -> residual ``[c]``."""
        att = (self.q(feature) @ self.k(rows).t()) * self.scale
        ctx = F.softmax(att, dim=-1) @ self.v(rows)
        return self.out(ctx).squeeze(-1)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class INCITE(nn.Module):

    def __init__(self, dim: int = 32, rounds: int = 6, layer_norm: bool = True,
                 short_cut: bool = True, count_channel: bool = True,
                 walks: Optional[dict] = None,
                 support_readout: bool = True,
                 support_k: int = 16,
                 num_mlp_layer: int = 2,
                 unary: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.rounds = int(rounds)
        self.short_cut = bool(short_cut)
        self.support_k = int(support_k)
        # Unary channel (2026-09-01): a query-INDEPENDENT state for every
        # entity from the unlabeled pass (``encode_unlabeled``), read at the
        # head and at each candidate and scored with the query relation's
        # state by a second head that is ADDED to the path score. Every
        # candidate gets evidence of its own -- its relation signature --
        # even when no path from the head reaches it within the rounds
        # (17 percent of ind_e answers, results/incite/reachability.json)
        # or when it lacks an edge of the query relation (the SQUA case).
        self.unary_mlp = None
        if unary:
            ufeat = 3 * self.dim
            umlp = []
            for _ in range(num_mlp_layer - 1):
                umlp += [nn.Linear(ufeat, ufeat), nn.ReLU()]
            umlp.append(nn.Linear(ufeat, 1))
            self.unary_mlp = nn.Sequential(*umlp)
        # per-round activation checkpointing (train.checkpoint_activations):
        # default OFF so behavior is unchanged; the pretrain driver flips it.
        # Applied only where gradients are live -- no_grad passes (eval,
        # support building) take the plain path.
        self.checkpoint_activations = False
        self.entity_steps = nn.ModuleList(
            [EntityStep(dim, layer_norm) for _ in range(self.rounds)])
        self.relation_steps = nn.ModuleList(
            [FactorizedRelationStep(dim, layer_norm, count_channel)
             for _ in range(self.rounds)])
        # TRIX's node_mlp shape: entity feature [x ; node_query] -> dim
        self.node_mlps = nn.ModuleList(
            [nn.Linear(2 * dim, dim) for _ in range(self.rounds)])
        self.task_emb = nn.Embedding(2, dim)
        feat = 2 * dim
        mlp = []
        for _ in range(num_mlp_layer - 1):
            mlp += [nn.Linear(feat, feat), nn.ReLU()]
        mlp.append(nn.Linear(feat, 1))
        self.score_mlp = nn.Sequential(*mlp)
        # relation head: TRIX-style MLP on (p(h,t), z_r) -- design D
        rfeat = 3 * dim
        rmlp = []
        for _ in range(num_mlp_layer - 1):
            rmlp += [nn.Linear(rfeat, rfeat), nn.ReLU()]
        rmlp.append(nn.Linear(rfeat, 1))
        self.relation_mlp = nn.Sequential(*rmlp)
        self.walk_module = WalkModule(dim, **walks) if walks else None
        self.readout = SupportReadout(dim, row_dim(dim)) if support_readout else None

    # -- plumbing -----------------------------------------------------------
    def _pairs(self, data) -> IncidencePairs:
        """Incidence pairs of the FULL graph, cached on the data object.

        Cached under a key that includes the edge count, so a stale cache
        from a differently-sized predecessor object cannot be reused.
        """
        key = int(data.edge_index.shape[1])
        cached = getattr(data, "incite_pairs", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        pairs = incidence_pairs(data)
        try:
            data.incite_pairs = (key, pairs)
        except AttributeError:  # a frozen container; just recompute next time
            pass
        return pairs

    def _trunk(self, msg_graph, pairs: IncidencePairs, h_index: torch.Tensor,
               r_query: Optional[torch.Tensor], t_index: Optional[torch.Tensor],
               task: int, walk_offset: int = 0):
        """The alternating rounds. Returns (x [b, V, d], z [b, R, d])."""
        b = len(h_index)
        num_nodes, num_relations = pairs.num_nodes, pairs.num_relations
        device = h_index.device
        d = self.dim
        ones = torch.ones(b, d, device=device)
        if task == TASK_ENTITY:
            q = ones + self.task_emb.weight[TASK_ENTITY]
            z = torch.zeros(b, num_relations, d, device=device)
            z.scatter_add_(1, r_query.view(b, 1, 1).expand(b, 1, d), q.unsqueeze(1))
        else:
            q = ones + self.task_emb.weight[TASK_RELATION]
            z = q.unsqueeze(1).expand(b, num_relations, d).contiguous()
        z_boundary = z.clone()

        x = torch.zeros(b, num_nodes, d, device=device)
        if self.walk_module is not None:
            w_ent, w_rel = self.walk_module(msg_graph, h_index, walk_offset)
            if task == TASK_RELATION and t_index is not None:
                w_ent_t, w_rel_t = self.walk_module(msg_graph, t_index,
                                                    walk_offset)
                w_ent, w_rel = w_ent + w_ent_t, w_rel + w_rel_t
            x = x + w_ent
            z = z + w_rel

        # checkpointing trades the retained (b, V, d)/(b, R, d) activations
        # of each round for one recompute during backward. The recompute is
        # deterministic: the trunk holds no dropout (a test asserts it) and
        # the walk features -- sampled once, above, from a fresh seeded
        # Generator -- are inputs to round 1, not part of any round.
        use_ckpt = self.checkpoint_activations and torch.is_grad_enabled()
        for k in range(self.rounds):
            if use_ckpt:
                x, z = checkpoint(self._round, k, task, x, z, z_boundary,
                                  msg_graph.edge_index, msg_graph.edge_type,
                                  pairs, h_index, t_index, r_query, q,
                                  use_reentrant=False)
            else:
                x, z = self._round(k, task, x, z, z_boundary,
                                   msg_graph.edge_index, msg_graph.edge_type,
                                   pairs, h_index, t_index, r_query, q)
        return x, z

    def _round(self, k: int, task: int, x, z, z_boundary, edge_index,
               edge_type, pairs: IncidencePairs, h_index, t_index, r_query, q):
        """Round k: boundary from the CURRENT z, entity step, node MLP,
        relation step. Under activation checkpointing this whole function is
        recomputed in backward, so it must stay RNG-free (no dropout in the
        trunk -- asserted by a test) and free of hidden mutable state."""
        b, d = x.shape[0], self.dim
        num_nodes = pairs.num_nodes
        device = x.device
        boundary_x = torch.zeros(b, num_nodes, d, device=device)
        if task == TASK_ENTITY:
            q_k = z.gather(1, r_query.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
            boundary_x.scatter_add_(
                1, h_index.view(b, 1, 1).expand(b, 1, d), q_k.unsqueeze(1))
        else:
            q_k = q
            boundary_x.scatter_add_(
                1, h_index.view(b, 1, 1).expand(b, 1, d), q_k.unsqueeze(1))
            boundary_x.scatter_add_(
                1, t_index.view(b, 1, 1).expand(b, 1, d), (-q_k).unsqueeze(1))
        hidden = self.entity_steps[k](x, z, boundary_x, edge_index, edge_type)
        x = hidden + x if self.short_cut else hidden
        node_repr = self.node_mlps[k](
            torch.cat([x, q_k.unsqueeze(1).expand_as(x)], dim=-1))
        z_hidden = self.relation_steps[k](z, node_repr, pairs, z_boundary)
        z = z_hidden + z if self.short_cut else z_hidden
        return x, z

    # -- entity task --------------------------------------------------------
    def forward(self, data, batch: torch.Tensor,
                support: Optional[SupportStore] = None,
                walk_offset: int = 0, halflink=None) -> torch.Tensor:
        """TRIX's entity interface: ``batch [b, c, 3]`` (h, t, r) -> ``[b, c]``.

        ``halflink``: optional ``(mask_answer [b], mask_query [b])`` bool
        tensors, training only -- see ``mask_halflinks``.
        """
        h_index, t_index, r_index = batch.unbind(-1)
        num_direct = int(data.num_relations) // 2
        pairs = self._pairs(data)
        msg_graph = data
        if self.training:
            msg_graph = remove_easy_edges(data, h_index, t_index, r_index)
        h_index, t_index, r_index = negative_sample_to_tail(
            h_index, t_index, r_index, num_direct)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        if self.training and halflink is not None:
            mask_answer, mask_query = halflink
            msg_graph = mask_halflinks(msg_graph, h_index[:, 0], r_index[:, 0],
                                       t_index[:, 0], mask_answer, mask_query,
                                       num_direct)

        x, z = self._trunk(msg_graph, pairs, h_index[:, 0], r_index[:, 0],
                           None, TASK_ENTITY, walk_offset)
        b, d = x.shape[0], self.dim
        node_query = z.gather(
            1, r_index[:, 0].view(b, 1, 1).expand(b, 1, d)).expand_as(x)
        feature = torch.cat([x, node_query], dim=-1)  # [b, V, 2d]
        cand_feature = feature.gather(
            1, t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1]))
        score = self.score_mlp(cand_feature).squeeze(-1)

        if self.unary_mlp is not None:
            g = self._global_states(msg_graph)                      # [V, d]
            g_h = g[h_index[:, 0]]                                  # [b, d]
            g_t = g[t_index]                                        # [b, c, d]
            z_q = node_query[:, 0]                                  # [b, d]
            c = t_index.shape[1]
            ufeat = torch.cat([g_h.unsqueeze(1).expand(b, c, d), g_t,
                               z_q.unsqueeze(1).expand(b, c, d)], dim=-1)
            score = score + self.unary_mlp(ufeat).squeeze(-1)

        if support is not None and self.readout is not None:
            residual = torch.zeros_like(score)
            for i in range(b):
                # in training the first column is the positive and the query
                # is a graph edge: its own support row would hand the answer
                # back, so it is excluded (the no-leak contract)
                exclude = ((int(h_index[i, 0]), int(t_index[i, 0]))
                           if self.training else None)
                rows = support.retrieve(int(r_index[i, 0]), int(h_index[i, 0]),
                                        k=self.support_k, exclude_pair=exclude)
                if rows is None:
                    continue
                residual[i] = self.readout(cand_feature[i], rows)
            score = score + residual
        return score

    # -- relation task ------------------------------------------------------
    def forward_relation(self, data, batch: torch.Tensor,
                         support: Optional[SupportStore] = None,
                         walk_offset: int = 0) -> torch.Tensor:
        """``batch [b, 3]`` (h, t, r) -> scores over direct relations
        ``[b, num_direct]``. One pass scores all relations (design D)."""
        h_index, t_index, r_index = batch.unbind(-1)
        num_direct = int(data.num_relations) // 2
        pairs = self._pairs(data)
        msg_graph = data
        if self.training:
            msg_graph = remove_easy_edges(data, h_index.unsqueeze(-1),
                                          t_index.unsqueeze(-1),
                                          r_index.unsqueeze(-1))
        x, z = self._trunk(msg_graph, pairs, h_index, None, t_index,
                           TASK_RELATION, walk_offset)
        b, d = x.shape[0], self.dim
        x_h = x.gather(1, h_index.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
        x_t = x.gather(1, t_index.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
        p_ht = torch.cat([x_h, x_t], dim=-1)                      # [b, 2d]
        z_direct = z[:, :num_direct]                              # [b, Rd, d]
        pair_exp = p_ht.unsqueeze(1).expand(b, num_direct, 2 * d)
        score = self.relation_mlp(
            torch.cat([pair_exp, z_direct], dim=-1)).squeeze(-1)  # [b, Rd]
        if support is not None:
            protos = torch.zeros(num_direct, 2 * d, device=x.device)
            have = torch.zeros(num_direct, dtype=torch.bool, device=x.device)
            for rid in range(num_direct):
                c = support.prototype(rid)
                if c is not None:
                    protos[rid] = c
                    have[rid] = True
            # the prototype term of design D: p(h,t) . c_r, only where a
            # prototype exists (a relation with no inference edge has none)
            score = score + (p_ht @ protos.t()) * have.to(score.dtype)
        return score

    # -- support-module encoder protocol ------------------------------------
    def encode_queries(self, graph, heads: torch.Tensor, rels: torch.Tensor):
        """(x [m, V, d], z_q [m, d], s0 [m, V]) for queries (u, r, ?).

        The support builder's view; runs whatever mode (train/eval) the
        module is in -- builders call it under no_grad and eval().
        """
        pairs = self._pairs(graph)
        x, z = self._trunk(graph, pairs, heads, rels, None, TASK_ENTITY)
        m, d = x.shape[0], self.dim
        z_q = z.gather(1, rels.view(m, 1, 1).expand(m, 1, d)).squeeze(1)
        feature = torch.cat([x, z_q.unsqueeze(1).expand_as(x)], dim=-1)
        s0 = self.score_mlp(feature).squeeze(-1)
        return x, z_q, s0

    def _global_states(self, graph) -> torch.Tensor:
        """``encode_unlabeled`` for the unary channel: recomputed on every
        training forward (the weights move, and the message graph carries
        that step's edge removals), cached per graph object under no_grad
        at evaluation. The cache key holds the edge count, like ``_pairs``."""
        if self.training or torch.is_grad_enabled():
            return self.encode_unlabeled(graph)
        key = int(graph.edge_index.shape[1])
        cached = getattr(graph, "incite_global", None)
        if cached is not None and cached[0] == key and cached[1] is self:
            return cached[2]
        g = self.encode_unlabeled(graph)
        try:
            graph.incite_global = (key, self, g)
        except AttributeError:
            pass
        return g

    def encode_unlabeled(self, graph) -> torch.Tensor:
        """The once-per-graph unlabeled entity encoding (design B step 1).

        No query, no task flag: every relation is labeled with ones and
        every entity's boundary is ones, so the states are structural. One
        batch row; walks are not sampled (they are query-conditioned).
        """
        pairs = self._pairs(graph)
        num_nodes, num_relations = pairs.num_nodes, pairs.num_relations
        device = graph.edge_index.device
        d = self.dim
        z = torch.ones(1, num_relations, d, device=device)
        z_boundary = z.clone()
        x = torch.zeros(1, num_nodes, d, device=device)
        boundary_x = torch.ones(1, num_nodes, d, device=device)
        q = torch.ones(1, d, device=device)
        for k in range(self.rounds):
            hidden = self.entity_steps[k](x, z, boundary_x,
                                          graph.edge_index, graph.edge_type)
            x = hidden + x if self.short_cut else hidden
            node_repr = self.node_mlps[k](
                torch.cat([x, q.unsqueeze(1).expand_as(x)], dim=-1))
            z_hidden = self.relation_steps[k](z, node_repr, pairs, z_boundary)
            z = z_hidden + z if self.short_cut else z_hidden
        return x[0]
