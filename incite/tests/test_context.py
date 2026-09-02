"""Context-necessity pieces: instances, batching, rows, scorer, conditions."""

import sys

import pytest
import torch

from incite import synth as S
from incite.context import (ContextScorer, LABEL_POSITIVE, LABEL_NEGATIVE,
                            build_rows, shuffle_labels)
from incite.model import INCITE

SMALL = {"entities": (60, 200)}
#: seed 2048, neg_per_pos 3, pre-refactor module: (nodes, edges, base, kept, dropped, first query row)
GOLDEN_2048 = (774, 1233, 885, 348, 167, [579, 314, 24])
CCFG = {"k_pos": 4, "k_neg": 3, "num_negative": 8, "withhold": 1.0, "eval_cap": 31}


def _edges(inst):
    return {(int(h), int(r), int(t)) for h, r, t in
            zip(inst.edge_index[0], inst.edge_type, inst.edge_index[1])}


def test_rules_stream_golden():
    """The 2026-09-02 refactor (_sample_world) must not move the rules
    prior's draw stream: P1 on the first machine trains on it. Golden values
    were computed with the pre-refactor module at seed 2048."""
    inst = S.create_rules_instance(torch.Generator().manual_seed(2048), 3)
    assert (inst.num_nodes, int(inst.edge_index.shape[1]), inst.num_base,
            inst.num_derived_kept, inst.num_dropped) == GOLDEN_2048[:5]
    assert inst.test_triplets[0].tolist() == GOLDEN_2048[5]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_context_instance_contract(seed):
    inst = S.create_context_instance(torch.Generator().manual_seed(seed), CCFG, SMALL)
    edges = _edges(inst)
    r = int(inst.rel)
    # withhold 1.0: the query relation has no edge in the message graph
    assert not any(e[1] == r for e in edges)
    assert inst.num_removed == inst.num_obs_r
    # the query and every context positive are absent facts
    assert (int(inst.q_h), r, int(inst.q_t)) not in edges
    for u, v in zip(inst.ctx_h.tolist(), inst.ctx_v.tolist()):
        assert (u, r, v) not in edges and u != v
    # negatives are never the answer and never an edge
    assert int(inst.q_t) not in inst.q_negs.tolist()
    assert int(inst.q_t) not in inst.eval_cands.tolist()
    for n in inst.q_negs.tolist():
        assert (int(inst.q_h), r, n) not in edges
    assert inst.ctx_h.shape == (4,) and inst.ctx_neg.shape == (4, 3)
    assert inst.q_negs.shape == (8,) and 1 <= inst.eval_cands.numel() <= 31
    assert int(inst.edge_index.max()) < inst.num_nodes
    assert int(inst.ctx_neg.max()) < inst.num_nodes


def test_context_instance_reproducible():
    a = S.create_context_instance(torch.Generator().manual_seed(9), CCFG, SMALL)
    b = S.create_context_instance(torch.Generator().manual_seed(9), CCFG, SMALL)
    assert torch.equal(a.edge_index, b.edge_index) and torch.equal(a.ctx_neg, b.ctx_neg)
    assert (a.q_h, a.q_t, a.rel) == (b.q_h, b.q_t, b.rel)


def test_visible_mode_keeps_relation_edges():
    ccfg = dict(CCFG, withhold=0.0)
    inst = S.create_context_instance(torch.Generator().manual_seed(5), ccfg, SMALL)
    assert inst.num_removed == 0
    edges = _edges(inst)
    assert (int(inst.q_h), int(inst.rel), int(inst.q_t)) not in edges


def _batch(k=3, num_negative=8):
    gen = torch.Generator().manual_seed(11)
    insts = [S.create_context_instance(gen, CCFG, SMALL) for _ in range(k)]
    return insts, S.context_batch(insts, num_negative=num_negative)


def test_context_batch_offsets_and_eval_mask():
    insts, (union, batch) = _batch()
    assert batch["cands"].shape == (3, 9) and batch["cand_mask"].all()
    assert int(union.num_nodes) == sum(int(i.num_nodes) for i in insts)
    assert int(union.num_relations) == 2 * max(int(i.num_relations) for i in insts)
    # second instance's head is offset by the first instance's node count
    assert int(batch["q_h"][1]) == int(insts[1].q_h) + int(insts[0].num_nodes)
    union_e, batch_e = S.context_batch(insts, num_negative=None)
    assert batch_e["cand_mask"][:, 0].all()
    for i, inst in enumerate(insts):
        assert int(batch_e["cand_mask"][i].sum()) == 1 + inst.eval_cands.numel()


def test_rows_and_scorer_conditions():
    insts, (union, batch) = _batch()
    k, P, Nn = 3, 4, 3
    model = INCITE(dim=16, rounds=2, walks=None, support_readout=False)
    heads = torch.cat([batch["q_h"], batch["ctx_h"].reshape(-1)])
    rels = torch.cat([batch["q_r"], batch["q_r"].repeat_interleave(P)])
    x, z_q, _ = model.encode_queries(union, heads, rels)
    q_rows, c_rows, labels = build_rows(x, z_q, batch, k)
    assert q_rows.shape == (k, 9, 32) and c_rows.shape == (k, P * (1 + Nn), 32)
    assert labels.shape == (k, P * (1 + Nn))
    assert (labels.view(k, P, 1 + Nn)[:, :, 0] == LABEL_POSITIVE).all()
    assert (labels.view(k, P, 1 + Nn)[:, :, 1:] == LABEL_NEGATIVE).all()
    # a context row is the same feature the trunk's MLP sees for that pass
    i, p = 1, 2
    pass_idx = k + i * P + p
    expect = torch.cat([x[pass_idx, batch["ctx_v"][i, p]], z_q[pass_idx]])
    assert torch.allclose(c_rows[i, p * (1 + Nn)], expect)

    scorer = ContextScorer(32, width=16, depth=2, heads=2)
    full = scorer(q_rows, c_rows, labels)
    assert full.shape == (k, 9)
    shuffled = scorer(q_rows, c_rows, shuffle_labels(labels, torch.Generator().manual_seed(1)))
    none = scorer(q_rows, None, None)
    assert none.shape == (k, 9)
    assert not torch.allclose(full, shuffled) and not torch.allclose(full, none)
    # candidate-permutation equivariance: query rows never attend each other
    perm = torch.randperm(9, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(scorer(q_rows[:, perm], c_rows, labels), full[:, perm], atol=1e-5)
    # row-order invariance of the context
    cperm = torch.randperm(c_rows.shape[1], generator=torch.Generator().manual_seed(4))
    assert torch.allclose(scorer(q_rows, c_rows[:, cperm], labels[:, cperm]), full, atol=1e-5)


def test_gradients_reach_trunk_through_rows():
    insts, (union, batch) = _batch()
    model = INCITE(dim=16, rounds=2, walks=None, support_readout=False)
    scorer = ContextScorer(32, width=16, depth=1, heads=2)
    P = 4
    heads = torch.cat([batch["q_h"], batch["ctx_h"].reshape(-1)])
    rels = torch.cat([batch["q_r"], batch["q_r"].repeat_interleave(P)])
    x, z_q, _ = model.encode_queries(union, heads, rels)
    q_rows, c_rows, labels = build_rows(x, z_q, batch, 3)
    scorer(q_rows, c_rows, labels).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.entity_steps.parameters())


