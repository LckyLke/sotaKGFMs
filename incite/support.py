"""Support sets with retrieval and hard negatives (docs/INCITE_DESIGN.md B).

A support row records how a candidate entity looked to the network when a
sampled inference-graph edge (u, r, v) was posed as the query (u, r, ?) with
that edge and its inverse removed from the message graph. Per relation id
the store keeps up to ``per_relation_cap`` positives; at query time the
K = 16 positives whose head is nearest the query head in the unlabeled
embedding space are retrieved, each carrying its hard negatives from the
3-hop ball of its head.

------------------------------------------------------------------------------
Leakage contract (ported from crest/bank.py, the same rules bind here)
------------------------------------------------------------------------------
* Support edges come from the INFERENCE graph only; every sampled edge is
  asserted present in the graph it is removed from.
* The sampled edge and its inverse are removed from the message graph
  before the encoder runs (chunked: a chunk's edges are removed together,
  at most one edge per relation id per chunk, direct and inverse ids in
  separate rounds -- crest/bank.py's construction, same reasoning).
* ``Ans(u, r)`` is computed over the inference graph only. A test triple
  can appear as a negative; filtering it out would leak the test set.
* At retrieval, rows whose support pair equals the query pair (u == h and
  v == t) are excluded: during training the query IS a graph edge and its
  own support row would hand the answer back (the no-leak test asserts
  this exclusion).

------------------------------------------------------------------------------
Gradient contract (docs/INCITE_PLAN.md lesson 3 -- the 14.85 GiB lesson)
------------------------------------------------------------------------------
Builders run under ``torch.no_grad()`` and every stored tensor is
``.detach()``-ed. Full backprop through K = 16 support passes per query
cannot fit 16 GiB; rows are data, refreshed on an interval by
``SupportRefresher`` (the BankRefresher pattern), and this is a recorded
deviation and the first suspect if the support pathway underperforms.
A regression test asserts no stored tensor carries ``grad_fn``.

------------------------------------------------------------------------------
PU down-weighting
------------------------------------------------------------------------------
Some hard negatives are true facts absent from the graph. Rows carry a
signed label feature: +1 for positives, ``-(1 - class_prior)`` for
negatives, the positive-unlabeled correction the design prescribes; the
class prior is a config value (see results/incite/config_diff.md for the
chosen default -- the design fixes the mechanism, not the number).

Encoder protocol (duck-typed; INCITE's model and the test toys both
implement it):

    encode_queries(graph, heads [m], rels [m]) -> (x [m, V, d], z [m, d],
                                                   s0 [m, V])
    encode_unlabeled(graph) -> [V, d]
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from . import graphs as G

__all__ = ["SupportStore", "build_support", "row_dim", "SupportRefresher"]


def row_dim(d: int) -> int:
    """Row feature width: [x_u ; x_cand ; s0 ; signed label] = 2d + 2."""
    return 2 * d + 2


class SupportStore:
    """Per-relation-id support tables plus the unlabeled entity embedding.

    Keyed by (graph_id, checkpoint_hash, seed) for the disk cache, exactly
    like crest's ContextBank: an identical rebuild must never run.
    """

    def __init__(self, graph_id: str = "", checkpoint_hash: str = "", seed: int = 0):
        self.graph_id = graph_id
        self.checkpoint_hash = checkpoint_hash
        self.seed = int(seed)
        self.ent_emb: Optional[torch.Tensor] = None  # [V, d], unlabeled
        self._heads: Dict[int, torch.Tensor] = {}
        self._tails: Dict[int, torch.Tensor] = {}
        self._head_emb: Dict[int, torch.Tensor] = {}
        self._feat: Dict[int, torch.Tensor] = {}      # [m, 1 + neg, 2d + 2]
        self._proto: Dict[int, torch.Tensor] = {}     # [2d]

    def __contains__(self, rid: int) -> bool:
        return int(rid) in self._feat

    def __len__(self) -> int:
        return len(self._feat)

    def relation_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._feat))

    def put(self, rid: int, heads, tails, head_emb, feat, proto) -> None:
        rid = int(rid)
        assert feat.dim() == 3 and len(heads) == len(tails) == len(head_emb) == len(feat)
        self._heads[rid] = heads
        self._tails[rid] = tails
        self._head_emb[rid] = head_emb
        self._feat[rid] = feat
        self._proto[rid] = proto

    def prototype(self, rid: int) -> Optional[torch.Tensor]:
        return self._proto.get(int(rid))

    def tensors(self):
        """Every stored tensor, for the detached-support regression test."""
        out = [] if self.ent_emb is None else [self.ent_emb]
        for table in (self._heads, self._tails, self._head_emb, self._feat, self._proto):
            out.extend(table.values())
        return out

    def to(self, device) -> "SupportStore":
        if self.ent_emb is not None:
            self.ent_emb = self.ent_emb.to(device)
        for table in (self._heads, self._tails, self._head_emb, self._feat, self._proto):
            for k in list(table):
                table[k] = table[k].to(device)
        return self

    def retrieve(self, rid: int, h: int, k: int,
                 exclude_pair: Optional[Tuple[int, int]] = None
                 ) -> Optional[torch.Tensor]:
        """Rows of the k nearest support positives to head ``h``, flattened
        to ``[k * (1 + neg), 2d + 2]``; None when the relation has no rows.

        Nearness is L2 distance in the unlabeled embedding space (retrieval
        sets the kernel-average bias delta to the nearest neighbours,
        design B). ``exclude_pair`` drops rows equal to the query edge.
        """
        rid = int(rid)
        if rid not in self._feat or self.ent_emb is None:
            return None
        heads, tails = self._heads[rid], self._tails[rid]
        keep = torch.ones(len(heads), dtype=torch.bool, device=heads.device)
        if exclude_pair is not None:
            keep &= ~((heads == int(exclude_pair[0])) & (tails == int(exclude_pair[1])))
        if not bool(keep.any()):
            return None
        idx = keep.nonzero(as_tuple=True)[0]
        dist = (self._head_emb[rid][idx] - self.ent_emb[int(h)]).norm(dim=-1)
        take = idx[dist.argsort()[:int(k)]]
        return self._feat[rid][take].flatten(0, 1)

    # -- disk cache (ContextBank pattern, verbatim mechanics) ---------------
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
            "graph_id": self.graph_id, "checkpoint_hash": self.checkpoint_hash,
            "seed": self.seed, "ent_emb": self.ent_emb,
            "heads": self._heads, "tails": self._tails,
            "head_emb": self._head_emb, "feat": self._feat, "proto": self._proto,
        }, tmp)
        os.replace(tmp, path)  # atomic: a crashed writer leaves no partial cache
        return path

    @classmethod
    def load(cls, path: str) -> "SupportStore":
        state = torch.load(path, map_location="cpu", weights_only=True)
        store = cls(state["graph_id"], state["checkpoint_hash"], int(state["seed"]))
        store.ent_emb = state["ent_emb"]
        for rid in state["feat"]:
            store.put(int(rid), state["heads"][rid], state["tails"][rid],
                      state["head_emb"][rid], state["feat"][rid], state["proto"][rid])
        return store

    @classmethod
    def load_or_build(cls, root: str, builder, graph_id: str,
                      checkpoint_hash: str, seed: int) -> "SupportStore":
        probe = cls(graph_id, checkpoint_hash, seed)
        path = probe.cache_path(root)
        if os.path.exists(path):
            return cls.load(path)
        builder(probe)
        probe.save(root)
        return probe


def _sample_edges(edge_ids: torch.Tensor, k: int,
                  generator: torch.Generator) -> torch.Tensor:
    """k edge indices; without replacement when the relation has enough."""
    n = len(edge_ids)
    if n >= k:
        perm = torch.randperm(n, generator=generator)[:k]
    else:
        perm = torch.arange(n)  # a support set repeats nothing; short is short
    return edge_ids[perm.to(edge_ids.device)]


@torch.no_grad()
def build_support(graph, encoder, seed: int = 1024,
                  per_relation_cap: int = 64,
                  neg_per_pos: int = 4,
                  prototype_k: int = 16,
                  hops: int = 3,
                  ball_cap: int = 1024,
                  class_prior: float = 0.1,
                  relation_ids: Optional[Iterable[int]] = None,
                  store: Optional[SupportStore] = None,
                  build_batch_size: int = 16) -> SupportStore:
    """Build (or partially refresh) the support store from ``graph``.

    ``graph`` MUST be the inference graph (module docstring). One unlabeled
    encoder pass gives the retrieval space; then one query pass per sampled
    support edge, chunked ``build_batch_size`` at a time with at most one
    edge per relation id per chunk and direct/inverse ids in separate
    rounds (crest/bank.py's construction). ``relation_ids`` restricts to a
    partial refresh into ``store``. All outputs are detached (no_grad).
    """
    out = store if store is not None else SupportStore(seed=seed)
    generator = torch.Generator().manual_seed(int(seed))
    device = graph.edge_index.device
    num_relations = int(graph.num_relations)
    num_direct = num_relations // 2
    wanted = (range(num_relations) if relation_ids is None
              else sorted(set(int(i) for i in relation_ids)))

    out.ent_emb = encoder.encode_unlabeled(graph).detach()

    # sampling plan first, in one fixed generator order (chunk-size invariant)
    plan: Dict[int, List[Tuple[int, int]]] = {}
    for rid in wanted:
        edge_ids = (graph.edge_type == int(rid)).nonzero(as_tuple=True)[0]
        if len(edge_ids) == 0:
            continue  # no example edge may exist for an inference relation
        chosen = _sample_edges(edge_ids, int(per_relation_cap), generator)
        plan[rid] = [(int(graph.edge_index[0, e]), int(graph.edge_index[1, e]))
                     for e in chosen.tolist()]

    ball_gen = torch.Generator().manual_seed(int(seed) + 1)
    rows_by_rid: Dict[int, List[torch.Tensor]] = {rid: [] for rid in plan}
    pairs_by_rid: Dict[int, List[torch.Tensor]] = {rid: [] for rid in plan}
    present = sorted(plan)
    halves = ([r for r in present if r < num_direct],
              [r for r in present if r >= num_direct])
    max_len = max(len(v) for v in plan.values())
    rounds = [[(rid, plan[rid][k]) for rid in half if k < len(plan[rid])]
              for k in range(max_len) for half in halves if half]
    for round_jobs in rounds:
        for start in range(0, len(round_jobs), int(build_batch_size)):
            jobs = round_jobs[start:start + int(build_batch_size)]
            message_graph = G.remove_edges_batch(
                graph, [(u, rid, v) for rid, (u, v) in jobs])
            heads = torch.tensor([u for _, (u, _) in jobs], device=device)
            rels = torch.tensor([rid for rid, _ in jobs], device=device)
            x, z, s0 = encoder.encode_queries(message_graph, heads, rels)
            for j, (rid, (u, v)) in enumerate(jobs):
                known = G.ans(graph, u, rid)
                ball = G.n_hop_ball(graph, u, hops=hops, cap=ball_cap,
                                    generator=ball_gen)
                keep = torch.ones(len(ball), dtype=torch.bool, device=device)
                keep &= ~torch.isin(ball, known)
                negs = ball[keep]
                # hard negatives ranked by the current score (design B.3)
                order = s0[j, negs].argsort(descending=True)
                negs = negs[order[:int(neg_per_pos)]]
                if len(negs) < int(neg_per_pos):
                    # sparse ball: pad with seeded uniform non-answers so the
                    # row block keeps its shape
                    mask = torch.ones(graph.num_nodes, dtype=torch.bool, device=device)
                    mask[known] = False
                    mask[negs] = False
                    pool = mask.nonzero(as_tuple=True)[0]
                    extra = pool[torch.randint(len(pool),
                                               (int(neg_per_pos) - len(negs),),
                                               generator=ball_gen).to(device)]
                    negs = torch.cat([negs, extra])
                cand = torch.cat([torch.tensor([v], device=device), negs])
                x_u = x[j, u].unsqueeze(0).expand(len(cand), -1)
                x_c = x[j, cand]
                label = torch.full((len(cand), 1), -(1.0 - float(class_prior)),
                                   device=device)
                label[0, 0] = 1.0
                feat = torch.cat([x_u, x_c, s0[j, cand].unsqueeze(-1), label], dim=-1)
                rows_by_rid[rid].append(feat.detach())
                pairs_by_rid[rid].append(
                    torch.cat([x[j, u], x[j, v]]).detach())  # pair state, edge removed

    for rid in present:
        if not rows_by_rid[rid]:
            continue
        heads = torch.tensor([u for u, _ in plan[rid]], device=device)
        tails = torch.tensor([v for _, v in plan[rid]], device=device)
        feat = torch.stack(rows_by_rid[rid])          # [m, 1 + neg, 2d + 2]
        pair_states = torch.stack(pairs_by_rid[rid])  # [m, 2d]
        # prototype over the first K pairs in seed order, once per graph
        proto = pair_states[:int(prototype_k)].mean(dim=0)
        out.put(rid, heads.detach(), tails.detach(),
                out.ent_emb[heads].detach(), feat.detach(), proto.detach())
    return out


class SupportRefresher:
    """Owns one graph's support store during training: touch, refresh, gate.

    The BankRefresher pattern from crest/train.py: rebuild only the relation
    ids touched since the last refresh, every ``refresh_interval`` steps;
    when rebuild time exceeds ``cost_gate`` of training time over the same
    window, double the interval before anything else is tried.
    """

    def __init__(self, graph, encoder, store: SupportStore,
                 build_kwargs: Optional[dict] = None,
                 refresh_interval: int = 500, cost_gate: float = 0.20):
        self.graph = graph
        self.encoder = encoder
        self.store = store
        self.build_kwargs = dict(build_kwargs or {})
        self.refresh_interval = int(refresh_interval)
        self.cost_gate = float(cost_gate)
        self.touched: set = set()
        self.step = 0
        self.window_train_seconds = 0.0
        self.log: List[dict] = []

    def touch(self, relation_ids: Iterable[int]) -> None:
        self.touched.update(int(r) for r in relation_ids)

    def after_step(self, step_seconds: float) -> None:
        self.step += 1
        self.window_train_seconds += float(step_seconds)
        if self.step % self.refresh_interval == 0:
            self._refresh()

    def _refresh(self) -> None:
        if not self.touched:
            return
        t0 = time.perf_counter()
        build_support(self.graph, self.encoder, seed=self.store.seed,
                      relation_ids=sorted(self.touched), store=self.store,
                      **self.build_kwargs)
        bank_seconds = time.perf_counter() - t0
        ratio = (bank_seconds / self.window_train_seconds
                 if self.window_train_seconds > 0 else float("inf"))
        entry = {"step": self.step, "refresh_interval": self.refresh_interval,
                 "touched": len(self.touched),
                 "bank_seconds": round(bank_seconds, 3),
                 "train_seconds": round(self.window_train_seconds, 3),
                 "ratio": round(ratio, 4), "gate": self.cost_gate,
                 "gate_ok": ratio <= self.cost_gate}
        self.log.append(entry)
        if ratio > self.cost_gate:
            self.refresh_interval *= 2
            entry["refresh_interval_raised_to"] = self.refresh_interval
        self.touched.clear()
        self.window_train_seconds = 0.0
