"""The layer gate (docs/INCITE_PLAN.md phase 1.1).

1. The factorized relation step equals a straightforward materialized
   reference -- same module, same random weights, message sums formed over
   the explicit TRIX-style pair lists -- allclose at 1e-5, on the toy graph
   and on a small real graph (Metafam).
2. Where the pinned TRIX tree is importable, the factorized step (count
   channel off) equals one round of TRIX's own four GeneralizedRelationalConv
   role layers with matched weights, on the relation graph built by TRIX's
   own ``build_relation_graph``.

The identity is exact in real arithmetic; 1e-5 covers float reordering
(docs/INCITE_PLAN.md, verified-claims section).
"""

import copy

import pytest
import torch

from conftest import load_real_graph, make_toy_graph

from incite.graphs import incidence_pairs
from incite.layers import FactorizedRelationStep
from incite.reference import build_role_pairs, materialized_relation_step

ATOL = 1e-5


def _random_inputs(graph, b=3, d=16, seed=5):
    generator = torch.Generator().manual_seed(seed)
    num_relations = int(graph.num_relations)
    z = torch.randn(b, num_relations, d, generator=generator)
    node_repr = torch.randn(b, graph.num_nodes, d, generator=generator)
    boundary = torch.randn(b, num_relations, d, generator=generator)
    return z, node_repr, boundary


@pytest.mark.parametrize("count_channel", [False, True])
def test_factorized_equals_materialized_toy(count_channel):
    graph = make_toy_graph()
    torch.manual_seed(2)
    step = FactorizedRelationStep(16, layer_norm=True, count_channel=count_channel)
    z, node_repr, boundary = _random_inputs(graph)
    with torch.no_grad():
        fact = step(z, node_repr, incidence_pairs(graph), boundary)
        ref = materialized_relation_step(step, z, node_repr, graph, boundary)
    assert torch.allclose(fact, ref, atol=ATOL), (fact - ref).abs().max()


def test_factorized_equals_materialized_real_graph():
    graph = load_real_graph()  # Metafam; skips without the data root
    torch.manual_seed(2)
    step = FactorizedRelationStep(16, layer_norm=True, count_channel=True)
    z, node_repr, boundary = _random_inputs(graph, b=2)
    with torch.no_grad():
        fact = step(z, node_repr, incidence_pairs(graph), boundary)
        ref = materialized_relation_step(step, z, node_repr, graph, boundary)
    assert torch.allclose(fact, ref, atol=ATOL), (fact - ref).abs().max()


def test_role_pairs_match_pinned_build_relation_graph():
    """The vectorized pair construction equals trix.tasks.build_relation_graph
    edge for edge (as unordered multisets per role)."""
    tasks = pytest.importorskip("trix.tasks")
    graph = make_toy_graph()
    upstream = copy.copy(graph)
    upstream = tasks.build_relation_graph(upstream)
    ours = build_role_pairs(graph)
    for role in ("hh", "ht", "th", "tt"):
        their = upstream.relation_adj[role]
        theirs = set(zip(their.edge_index[0].tolist(),
                         their.edge_index[1].tolist(),
                         their.edge_type.tolist()))
        r1, r2, v = ours[role]
        mine = set(zip(r1.tolist(), r2.tolist(), v.tolist()))
        assert mine == theirs, role


def test_factorized_equals_pinned_trix_layer():
    """One relation round vs TRIX's four role convs with matched weights."""
    tasks = pytest.importorskip("trix.tasks")
    layers = pytest.importorskip("trix.layers")
    graph = make_toy_graph()
    d, b = 16, 3
    torch.manual_seed(2)
    step = FactorizedRelationStep(d, layer_norm=True, count_channel=False)
    z, node_repr, boundary = _random_inputs(graph, b=b, d=d)

    upstream = tasks.build_relation_graph(copy.copy(graph))
    rel_graph = upstream.relation_adj
    num_relations = int(graph.num_relations)
    size = (num_relations, num_relations)

    total = None
    for role in ("hh", "ht", "th", "tt"):
        conv = layers.GeneralizedRelationalConv(
            d, d, rel_graph[role].num_relations, d,
            message_func="distmult", aggregate_func="sum", layer_norm=True,
            activation="relu", dependent=False, project_relations=True)
        channel = step.channels[role]
        with torch.no_grad():
            conv.linear.load_state_dict(channel.update.linear.state_dict())
            conv.layer_norm.load_state_dict(channel.update.norm.state_dict())
            conv.relation_projection.load_state_dict(channel.rel_proj.state_dict())
        conv.relation = node_repr  # what RelNet assigns each round
        edge_weight = torch.ones(rel_graph[role].edge_index.shape[1])
        with torch.no_grad():
            hidden = conv(z, torch.ones(b, d), boundary,
                          rel_graph[role].edge_index, rel_graph[role].edge_type,
                          size, edge_weight)
        total = hidden if total is None else total + hidden

    with torch.no_grad():
        fact = step(z, node_repr, incidence_pairs(graph), boundary)
    assert torch.allclose(fact, total, atol=ATOL), (fact - total).abs().max()
