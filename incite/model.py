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


def mask_halflinks(graph, h, r, t, mask_answer, mask_query, num_direct,
                   max_answer_degree: int = 0):
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

    ``max_answer_degree > 0`` (2026-09-02, after M1) restricts answer
    masking to targets with at most that many incoming query-relation
    edges in ``graph``. M1 masked hubs too (one target lost 484 edges of a
    relation) and the model learned a popularity prior from the stripped
    hubs; real unseen-answer targets are rare entities.
    """
    if not (bool(mask_answer.any()) or bool(mask_query.any())):
        return graph
    ei, et = graph.edge_index, graph.edge_type
    R2 = int(graph.num_relations)

    def inv(rr):
        return torch.where(rr < num_direct, rr + num_direct, rr - num_direct)

    src_key = ei[0] * R2 + et
    dst_key = ei[1] * R2 + et
    if max_answer_degree > 0 and bool(mask_answer.any()):
        # in-degree of each row's target under its query relation
        counts = torch.bincount(dst_key, minlength=int(graph.num_nodes) * R2)
        deg = counts[t * R2 + r]
        mask_answer = mask_answer & (deg <= int(max_answer_degree))
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
def scenario_features(graph, h: torch.Tensor, r: torch.Tensor,
                      t_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """The half-link scenario indicators of every candidate, ``[b, c, 4]``:
    ``[answer half present, log1p(its count), query half present,
    log1p(its count)]`` where the answer half of candidate t is an edge
    ``(x, r, t)`` of the message graph (rows are in tail form, so r may be
    an inverse id) and the query half is an edge ``(h, r, x)``. Computed on
    the graph the model propagates over, so in training the removed query
    edges do not count, exactly as at evaluation."""
    ei, et = graph.edge_index, graph.edge_type
    b, c = t_index.shape
    out = torch.zeros(b, c, 4, device=ei.device)
    for i in range(b):
        m = et == r[i]
        dst = ei[1, m]
        src = ei[0, m]
        in_r = torch.bincount(dst, minlength=int(num_nodes)).to(out.dtype)
        cnt = in_r[t_index[i]]
        out_h = (src == h[i]).sum().to(out.dtype)
        out[i, :, 0] = (cnt > 0).to(out.dtype)
        out[i, :, 1] = torch.log1p(cnt)
        out[i, :, 2] = (out_h > 0).to(out.dtype)
        out[i, :, 3] = torch.log1p(out_h)
    return out


class EdgeGate(nn.Module):
    """The proof-guided propagation gate of one round (2026-09-03, PG1).

    A per-(query, source node) scale and a per-(query, relation) scale;
    an edge's message is multiplied by ``node[b, src] * rel[b, r]``. The
    factorization is what lets the gate ride the fused kernel unchanged:
    the node scale folds into x, the relation scale into the projected
    relation features, and no ``[b, E, d]`` tensor is materialized. Each
    logit is linear in the state plus a query term. Weights start at zero
    and every scale is divided by sigmoid(BIAS) computed on the same
    device, so a freshly attached gate is EXACTLY the identity and a warm
    start leaves the trunk's function untouched. ``forward`` returns the
    raw sigmoids (what the proof loss pushes and the pruning measurement
    thresholds); ``scales`` turns them into the multipliers.
    """
    BIAS = 6.0

    def __init__(self, dim: int):
        super().__init__()
        self.node = nn.Linear(dim, 1)
        self.node_q = nn.Linear(dim, 1, bias=False)
        self.rel = nn.Linear(dim, 1)
        self.rel_q = nn.Linear(dim, 1, bias=False)
        for lin in (self.node, self.node_q, self.rel, self.rel_q):
            nn.init.zeros_(lin.weight)
        nn.init.constant_(self.node.bias, self.BIAS)
        nn.init.constant_(self.rel.bias, self.BIAS)

    def forward(self, x, z, q):
        """x [b, V, d], z [b, R, d], q [b, d] -> (node [b, V], rel [b, R])."""
        qn = self.node_q(q).squeeze(-1).unsqueeze(1)                 # [b, 1]
        qr = self.rel_q(q).squeeze(-1).unsqueeze(1)                   # [b, 1]
        gn = torch.sigmoid(self.node(x).squeeze(-1) + qn)             # [b, V]
        gr = torch.sigmoid(self.rel(z).squeeze(-1) + qr)              # [b, R]
        return gn, gr

    def scales(self, gn, gr):
        norm = torch.sigmoid(torch.full((1, 1), self.BIAS, dtype=gn.dtype,
                                        device=gn.device))
        return gn / norm, gr / norm


class RuleHead(nn.Module):
    """Rule recovery from relation states (2026-09-03, RR1).

    Every rules-prior instance carries its latent rule system. This head
    reads the trunk's relation states ``z [b, R, d]`` and scores rule
    hypotheses with certain labels: hierarchy ``r2 <- r1`` and inversion
    ``r2(y,x) <- r1(x,y)`` as bilinear forms on the two states, symmetry
    as a linear form on one state, composition ``r3 <- r1 . r2`` as an
    MLP on the three states. The loss trains the relation encoder to make
    the relational algebra of an unseen vocabulary linearly readable; no
    KGFM does this. ``idx [N, 5]`` rows are ``(row, kind, r1, r2, r3)``
    with kind 0 hier, 1 inv, 2 sym, 3 comp and unused slots -1.
    """
    KINDS = 4

    def __init__(self, dim: int):
        super().__init__()
        self.bilinear = nn.ModuleList([nn.Bilinear(dim, dim, 1) for _ in range(2)])
        self.sym = nn.Linear(dim, 1)
        self.comp = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(),
                                  nn.Linear(dim, 1))

    def forward(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        row, kind, r1, r2, r3 = idx.unbind(-1)
        a = z[row, r1]
        b = z[row, r2.clamp(min=0)]
        c = z[row, r3.clamp(min=0)]
        out = z.new_zeros(idx.shape[0])
        for k in (0, 1):
            m = kind == k
            if bool(m.any()):
                out = out.masked_scatter(m, self.bilinear[k](a[m], b[m]).squeeze(-1))
        m = kind == 2
        if bool(m.any()):
            out = out.masked_scatter(m, self.sym(a[m]).squeeze(-1))
        m = kind == 3
        if bool(m.any()):
            out = out.masked_scatter(
                m, self.comp(torch.cat([a[m], b[m], c[m]], dim=-1)).squeeze(-1))
        return out


def rule_recovery_loss(head: RuleHead, z: torch.Tensor, idx: torch.Tensor,
                       label: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits per rule kind, averaged over the kinds present, so
    the frequent kinds do not drown the rare ones."""
    logits = head(z, idx)
    label = label.to(logits.dtype)
    per = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
    kind = idx[:, 1]
    parts = []
    for k in range(RuleHead.KINDS):
        m = kind == k
        if bool(m.any()):
            parts.append(per[m].mean())
    return torch.stack(parts).mean()


class INCITE(nn.Module):

    def __init__(self, dim: int = 32, rounds: int = 6, layer_norm: bool = True,
                 short_cut: bool = True, count_channel: bool = True,
                 walks: Optional[dict] = None,
                 support_readout: bool = True,
                 support_k: int = 16,
                 num_mlp_layer: int = 2,
                 unary: bool = False,
                 gate: bool = False,
                 rule_head: bool = False,
                 scenario: bool = False):
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
        # Proof-guided propagation gate (2026-09-03, PG1): one EdgeGate per
        # round, None when off. ``prune_frac`` > 0 at evaluation drops that
        # share of each query's edges by gate value (the measurement in
        # diagnostics/gate_prune_dev.py); training never prunes.
        self.gates = nn.ModuleList(
            [EdgeGate(dim) for _ in range(self.rounds)]) if gate else None
        self.prune_frac = 0.0
        # pruning bookkeeping (measurement only): the realized kept fraction
        # of every pruned round is appended to ``prune_kept`` (ties at the
        # threshold keep MORE than 1 - prune_frac; a saturated gate keeps
        # everything, and the record makes that visible); ``prune_random``
        # replaces the gate products by seeded uniform noise, the control
        # curve any gate curve must beat.
        self.prune_kept = []
        self.prune_random = False
        self.prune_seed = 0
        # Rule recovery (2026-09-03, RR1): a head on the relation states,
        # trained on synthetic steps only (the rules are known there).
        self.rule_head = RuleHead(dim) if rule_head else None
        # Scenario-conditioned readout (2026-09-03, SC1, the reviewer's first
        # direction). Every lever so far moved the model ALONG the trade-off
        # between seen-answer and unseen-answer candidates. Both scenario
        # indicators are observable per candidate at inference: does the
        # candidate already have an incoming edge of the query relation
        # (and how many), does the head have an outgoing one. A second head
        # reads the candidate feature plus these four scalars and ADDS a
        # correction to the path score, so the two candidate populations
        # can be calibrated separately. Its last layer starts at zero: a
        # freshly attached head is exactly the identity.
        self.scenario_mlp = None
        if scenario:
            self.scenario_mlp = nn.Sequential(
                nn.Linear(2 * dim + 4, dim), nn.ReLU(), nn.Linear(dim, 1))
            nn.init.zeros_(self.scenario_mlp[-1].weight)
            nn.init.zeros_(self.scenario_mlp[-1].bias)
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
               task: int, walk_offset: int = 0, proof=None):
        """The alternating rounds. Returns (x [b, V, d], z [b, R, d], aux):
        ``aux`` is the per-round gate loss on ``proof`` summed over rounds,
        a zero scalar without gates or proof."""
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
        aux = x.new_zeros(())
        for k in range(self.rounds):
            if use_ckpt:
                x, z, a = checkpoint(self._round, k, task, x, z, z_boundary,
                                     msg_graph.edge_index, msg_graph.edge_type,
                                     pairs, h_index, t_index, r_query, q, proof,
                                     use_reentrant=False)
            else:
                x, z, a = self._round(k, task, x, z, z_boundary,
                                      msg_graph.edge_index, msg_graph.edge_type,
                                      pairs, h_index, t_index, r_query, q, proof)
            aux = aux + a
        return x, z, aux

    def _round(self, k: int, task: int, x, z, z_boundary, edge_index,
               edge_type, pairs: IncidencePairs, h_index, t_index, r_query, q,
               proof=None):
        """Round k: boundary from the CURRENT z, entity step, node MLP,
        relation step. Under activation checkpointing this whole function is
        recomputed in backward, so it must stay RNG-free (no dropout in the
        trunk -- asserted by a test) and free of hidden mutable state.

        With gates, the round's node and relation scales multiply the
        messages. ``proof`` = (rows [m], edges [m]) adds the one-sided gate
        loss on those (query, edge) pairs to the returned aux scalar: proof
        edges are pushed open, nothing is pushed shut (type context flows
        through non-proof edges). At evaluation ``prune_frac`` > 0 zeroes
        the lowest-gated share of each query's edges through a per-edge
        weight (measurement only, see diagnostics/gate_prune_dev.py)."""
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
        aux = x.new_zeros(())
        node_scale = rel_scale = edge_weight = None
        if self.gates is not None:
            gn, gr = self.gates[k](x, z, q_k)
            node_scale, rel_scale = self.gates[k].scales(gn, gr)
            if proof is not None:
                rows, edges = proof
                src, typ = edge_index[0, edges], edge_type[edges]
                aux = -(torch.log(gn[rows, src] + 1e-6)
                        + torch.log(gr[rows, typ] + 1e-6)).mean()
            if not self.training and self.prune_frac > 0:
                g = gn[:, edge_index[0]] * gr[:, edge_type]            # [b, E]
                if self.prune_random:
                    gen = torch.Generator(device=g.device)
                    gen.manual_seed(int(self.prune_seed) * 1009 + k)
                    g = torch.rand(g.shape, generator=gen, device=g.device,
                                   dtype=g.dtype)
                num_edges = g.shape[1]
                keep = max(1, int(round((1.0 - float(self.prune_frac))
                                        * num_edges)))
                thr = g.kthvalue(num_edges - keep + 1, dim=1,
                                 keepdim=True).values
                edge_weight = (g >= thr).to(x.dtype)
                self.prune_kept.append(float(edge_weight.mean()))
        hidden = self.entity_steps[k](x, z, boundary_x, edge_index, edge_type,
                                      node_scale, rel_scale, edge_weight)
        x = hidden + x if self.short_cut else hidden
        node_repr = self.node_mlps[k](
            torch.cat([x, q_k.unsqueeze(1).expand_as(x)], dim=-1))
        z_hidden = self.relation_steps[k](z, node_repr, pairs, z_boundary)
        z = z_hidden + z if self.short_cut else z_hidden
        return x, z, aux

    # -- entity task --------------------------------------------------------
    def forward(self, data, batch: torch.Tensor,
                support: Optional[SupportStore] = None,
                walk_offset: int = 0, halflink=None, proof=None,
                return_aux: bool = False, return_states: bool = False):
        """TRIX's entity interface: ``batch [b, c, 3]`` (h, t, r) -> ``[b, c]``.

        ``halflink``: optional ``(mask_answer [b], mask_query [b])`` bool
        tensors, training only -- see ``mask_halflinks``. ``proof``:
        optional ``(rows [m], edges [m])`` proof pairs in ``data``'s edge
        numbering for the gate loss; ``return_aux`` returns ``(score,
        aux)`` with that loss (zero without gates); ``return_states``
        returns ``(score, aux, z)`` with the final relation states (the
        rule-recovery head reads them).
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
            mask_answer, mask_query = halflink[0], halflink[1]
            maxdeg = int(halflink[2]) if len(halflink) > 2 else 0
            msg_graph = mask_halflinks(msg_graph, h_index[:, 0], r_index[:, 0],
                                       t_index[:, 0], mask_answer, mask_query,
                                       num_direct, max_answer_degree=maxdeg)

        x, z, aux = self._trunk(msg_graph, pairs, h_index[:, 0], r_index[:, 0],
                                None, TASK_ENTITY, walk_offset, proof=proof)
        b, d = x.shape[0], self.dim
        node_query = z.gather(
            1, r_index[:, 0].view(b, 1, 1).expand(b, 1, d)).expand_as(x)
        feature = torch.cat([x, node_query], dim=-1)  # [b, V, 2d]
        cand_feature = feature.gather(
            1, t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1]))
        score = self.score_mlp(cand_feature).squeeze(-1)

        if self.scenario_mlp is not None:
            feat = scenario_features(msg_graph, h_index[:, 0], r_index[:, 0],
                                     t_index, pairs.num_nodes)      # [b, c, 4]
            score = score + self.scenario_mlp(
                torch.cat([cand_feature, feat.to(cand_feature.dtype)],
                          dim=-1)).squeeze(-1)

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
        if return_states:
            return score, aux, z
        return (score, aux) if return_aux else score

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
        x, z, _ = self._trunk(msg_graph, pairs, h_index, None, t_index,
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
        x, z, _ = self._trunk(graph, pairs, heads, rels, None, TASK_ENTITY)
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
            node_scale = rel_scale = None
            if self.gates is not None:
                gn, gr = self.gates[k](x, z, q)
                node_scale, rel_scale = self.gates[k].scales(gn, gr)
            hidden = self.entity_steps[k](x, z, boundary_x,
                                          graph.edge_index, graph.edge_type,
                                          node_scale, rel_scale)
            x = hidden + x if self.short_cut else hidden
            node_repr = self.node_mlps[k](
                torch.cat([x, q.unsqueeze(1).expand_as(x)], dim=-1))
            z_hidden = self.relation_steps[k](z, node_repr, pairs, z_boundary)
            z = z_hidden + z if self.short_cut else z_hidden
        return x[0]
