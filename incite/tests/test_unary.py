"""The unary channel: shapes, equivariance, warm start, and the eval cache.

* the forward with ``unary=True`` returns the same shape and equals the
  plain path score plus the unary head (checked against a manual
  recomputation from ``encode_unlabeled``);
* entity relabeling permutes and does not change the scores (the channel
  is built from sums over edges and gathered at head and candidates);
* a floor checkpoint loads into a unary model with strict=False and only
  ``unary_mlp.*`` tensors stay fresh;
* under no_grad the global states are cached on the graph object and a
  second call reuses them.
"""

import torch

from conftest import make_model, make_toy_graph
from incite.graphs import Graph


def _batch(graph, h, r):
    v = graph.num_nodes
    return torch.tensor([[[h, t, r] for t in range(v)]])


def test_unary_adds_a_second_head():
    g = make_toy_graph()
    torch.manual_seed(5)
    model = make_model(support=False, unary=True).eval()
    batch = _batch(g, 0, 1)
    with torch.no_grad():
        full = model(g, batch)
        model.unary_mlp, saved = None, model.unary_mlp
        base = model(g, batch)
        model.unary_mlp = saved
        # manual unary term
        glob = model.encode_unlabeled(g)
        b, c, d = 1, g.num_nodes, model.dim
        pairs = model._pairs(g)
        x, z = model._trunk(g, pairs, torch.tensor([0]), torch.tensor([1]), None, 0)
        z_q = z[:, 1]
        feat = torch.cat([glob[0].expand(c, d), glob, z_q.expand(c, d)], dim=-1)
        manual = model.unary_mlp(feat).squeeze(-1)
    assert full.shape == base.shape == (1, g.num_nodes)
    assert torch.allclose(full - base, manual.unsqueeze(0), atol=1e-5)


def test_unary_is_permutation_equivariant():
    g = make_toy_graph()
    model = make_model(dim=16, rounds=3, support=False, unary=True).eval()
    torch.manual_seed(0)
    perm = torch.randperm(g.num_nodes)
    g2 = Graph(edge_index=perm[g.edge_index], edge_type=g.edge_type.clone(),
               num_nodes=g.num_nodes, num_relations=g.num_relations)
    h, r = int(g.edge_index[0, 0]), int(g.edge_type[0]) % (g.num_relations // 2)
    cand = torch.arange(g.num_nodes)
    b1 = torch.stack([torch.full_like(cand, h), cand, torch.full_like(cand, r)], -1).unsqueeze(0)
    b2 = torch.stack([torch.full_like(cand, int(perm[h])), perm[cand],
                      torch.full_like(cand, r)], -1).unsqueeze(0)
    with torch.no_grad():
        s1, s2 = model(g, b1), model(g2, b2)
    assert torch.allclose(s1, s2, atol=1e-5), (s1 - s2).abs().max()


def test_warm_start_leaves_only_unary_fresh():
    floor = make_model(support=False, unary=False)
    unary = make_model(support=False, unary=True)
    missing, unexpected = unary.load_state_dict(floor.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(k.startswith("unary_mlp.") for k in missing), missing


def test_global_states_cached_under_no_grad():
    g = make_toy_graph()
    model = make_model(support=False, unary=True).eval()
    with torch.no_grad():
        a = model._global_states(g)
        assert getattr(g, "incite_global", None) is not None
        b = model._global_states(g)
    assert a is b
    model.train()
    c = model._global_states(g)   # training: recomputed, with grad
    assert c is not a and c.requires_grad
