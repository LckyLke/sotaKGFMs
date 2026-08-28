"""Equivariance smoke: relabeling entity ids permutes but does not change
scores.

Every quantity in the reduction-mode model is a sum over edges, incidence
pairs, or boundary scatters, so a permutation of entity ids permutes the
candidate axis and nothing else. Float sums re-associate under permutation
(scatter order moves), hence allclose, not equal. Walk features are only
distributionally equivariant (the sampler's draws depend on edge order), so
this test runs the reduction-mode model -- the property the incidence core
must have unconditionally.
"""

import torch

from conftest import make_model, make_toy_graph

from incite.graphs import Graph


def _permute(graph, perm):
    return Graph(edge_index=perm[graph.edge_index],
                 edge_type=graph.edge_type.clone(),
                 num_nodes=graph.num_nodes,
                 num_relations=graph.num_relations)


def test_entity_scores_are_permutation_equivariant():
    graph = make_toy_graph()
    model = make_model(dim=16, rounds=3, walks=False, support=False)
    model.eval()
    torch.manual_seed(0)
    perm = torch.randperm(graph.num_nodes)
    graph2 = _permute(graph, perm)

    # a TRIX-shaped batch: 2 queries, all nodes as candidates
    num_direct = graph.num_relations // 2
    h = torch.tensor([int(graph.edge_index[0, 0]), int(graph.edge_index[0, 3])])
    r = torch.tensor([int(graph.edge_type[0]) % num_direct,
                      int(graph.edge_type[3]) % num_direct])
    cand = torch.arange(graph.num_nodes)
    batch = torch.stack([
        h.unsqueeze(-1).expand(-1, graph.num_nodes),
        cand.unsqueeze(0).expand(2, -1),
        r.unsqueeze(-1).expand(-1, graph.num_nodes)], dim=-1)
    batch2 = torch.stack([
        perm[h].unsqueeze(-1).expand(-1, graph.num_nodes),
        perm[cand].unsqueeze(0).expand(2, -1),
        r.unsqueeze(-1).expand(-1, graph.num_nodes)], dim=-1)

    with torch.no_grad():
        s1 = model(graph, batch)
        s2 = model(graph2, batch2)
    assert torch.allclose(s1, s2, atol=1e-5), (s1 - s2).abs().max()


def test_relation_scores_are_permutation_invariant():
    graph = make_toy_graph()
    model = make_model(dim=16, rounds=3, walks=False, support=False)
    model.eval()
    torch.manual_seed(0)
    perm = torch.randperm(graph.num_nodes)
    graph2 = _permute(graph, perm)
    h = torch.tensor([int(graph.edge_index[0, 0])])
    t = torch.tensor([int(graph.edge_index[1, 0])])
    r = torch.tensor([int(graph.edge_type[0]) % (graph.num_relations // 2)])
    batch = torch.stack([h, t, r], dim=-1)
    batch2 = torch.stack([perm[h], perm[t], r], dim=-1)
    with torch.no_grad():
        s1 = model.forward_relation(graph, batch)
        s2 = model.forward_relation(graph2, batch2)
    assert torch.allclose(s1, s2, atol=1e-5), (s1 - s2).abs().max()
