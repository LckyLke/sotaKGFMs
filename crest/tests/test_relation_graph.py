"""Relation-level graph bookkeeping that bank building rests on.

Scope decision, recorded here because the plan names this test without
specifying it: CREST does not rebuild TRIX's relation-interaction graph (that
stays inside TRIX, untested from the host where TRIX cannot import), so what
this file pins down is the relation arithmetic CREST itself performs --
inverse-id mapping, per-relation edge partitioning, Ans(u, r), and the
edge-plus-inverse removal that guards the bank against trivial answers.
"""

import pytest
import torch

from crest import bank as crest_bank

from conftest import make_toy_graph


def test_inverse_relation_is_an_involution():
    num_relations = 6
    for r in range(num_relations):
        r_inv = crest_bank.inverse_relation(r, num_relations)
        assert r_inv != r
        assert crest_bank.inverse_relation(r_inv, num_relations) == r
    assert crest_bank.inverse_relation(0, 6) == 3
    assert crest_bank.inverse_relation(3, 6) == 0


def test_relation_edges_partition_the_graph():
    graph = make_toy_graph()
    seen = torch.zeros(graph.edge_type.shape[0], dtype=torch.long)
    for rid in range(graph.num_relations):
        seen[crest_bank.relation_edges(graph, rid)] += 1
    # every edge belongs to exactly one relation id
    assert bool((seen == 1).all())


def test_every_direct_edge_has_its_inverse():
    graph = make_toy_graph()
    num_direct = graph.num_relations // 2
    triples = set(zip(graph.edge_index[0].tolist(),
                      graph.edge_type.tolist(),
                      graph.edge_index[1].tolist()))
    for (u, r, v) in triples:
        if r < num_direct:
            assert (v, r + num_direct, u) in triples


def test_ans_matches_brute_force():
    graph = make_toy_graph()
    triples = list(zip(graph.edge_index[0].tolist(),
                       graph.edge_type.tolist(),
                       graph.edge_index[1].tolist()))
    for u in range(graph.num_nodes):
        for r in range(graph.num_relations):
            expected = sorted({v for (h, rr, v) in triples if h == u and rr == r})
            got = crest_bank.ans(graph, u, r).tolist()
            assert got == expected, (u, r)


def test_remove_edge_and_inverse_removes_exactly_that_pair():
    graph = make_toy_graph()
    num_direct = graph.num_relations // 2
    u = int(graph.edge_index[0, 0])
    v = int(graph.edge_index[1, 0])
    r = int(graph.edge_type[0])
    out = crest_bank.remove_edge_and_inverse(graph, u, r, v)

    def multiset(g):
        return sorted(zip(g.edge_index[0].tolist(), g.edge_type.tolist(),
                          g.edge_index[1].tolist()))

    r_inv = crest_bank.inverse_relation(r, graph.num_relations)
    removed = [t for t in multiset(graph) if t not in multiset(out)]
    # every removed triple is the edge or its inverse, and both are fully gone
    assert removed, "nothing was removed"
    assert all(t in ((u, r, v), (v, r_inv, u)) for t in removed)
    assert (u, r, v) not in multiset(out)
    assert (v, r_inv, u) not in multiset(out)
    # the original graph is untouched (copy semantics, not in-place)
    assert (u, r, v) in multiset(graph)


def test_remove_edge_refuses_a_missing_edge():
    graph = make_toy_graph()
    with pytest.raises(AssertionError, match="inference graph"):
        # relation id num_relations - 1 from node 0 to 0 does not exist
        crest_bank.remove_edge_and_inverse(graph, 0, graph.num_relations - 1, 0)
