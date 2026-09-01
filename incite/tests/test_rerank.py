"""Bidirectional re-ranking: the rank invariants its docstring promises.

Toy graph, toy model, CPU. Checks:

* k = 0 returns the forward scores untouched;
* a target outside the top-k block keeps its forward rank exactly;
* inside the block the combined score is forward + weight * reverse, and
  the reverse score is what the model itself returns for the reverse
  query (cand, r^-1, ?) read at the query head;
* non-members are imputed the block's minimum reverse score;
* head-direction batches (h varies, t fixed) are handled through the
  same tail-form normalization the model uses;
* the score ensemble is the mean of its members.
"""

import torch

from conftest import make_model, make_toy_graph
from incite.model import compute_ranking, negative_sample_to_tail
from incite.rerank import ScoreEnsemble, rerank_predictions


def _all_negative_tail(graph, h, r):
    v = graph.num_nodes
    return torch.tensor([[[h, t, r] for t in range(v)]])


def _all_negative_head(graph, t, r):
    v = graph.num_nodes
    return torch.tensor([[[h, t, r] for h in range(v)]])


def test_k0_is_identity():
    g = make_toy_graph()
    model = make_model(support=False).eval()
    batch = _all_negative_tail(g, 0, 1)
    pred = model(g, batch)
    out = rerank_predictions(model, g, batch, pred, torch.tensor([3]), None, k=0)
    assert torch.equal(out, pred)


def test_block_semantics_tail_query():
    g = make_toy_graph()
    model = make_model(support=False).eval()
    h, r = 0, 1
    batch = _all_negative_tail(g, h, r)
    pred = model(g, batch)                                    # [1, V]
    k, w = 3, 0.7
    target = torch.tensor([int(pred.argmin())])               # a weak target
    final = rerank_predictions(model, g, batch, pred, target, None, k=k, weight=w)
    top_idx = pred.topk(k, dim=-1).indices[0]
    # the target is far outside the block: rank unchanged
    assert int(target) not in top_idx.tolist()
    assert int(compute_ranking(final, target)) == int(compute_ranking(pred, target))
    # block members carry forward + w * reverse, reverse = the model's own
    # score of the reverse query read at h
    num_direct = g.num_relations // 2
    r_inv = r + num_direct
    revs = []
    for c in top_idx.tolist():
        rb = torch.tensor([[[c, h, r_inv]]])
        revs.append(float(model(g, rb)[0, 0]))
    revs = torch.tensor(revs)
    expect = pred[0, top_idx] + w * revs
    assert torch.allclose(final[0, top_idx], expect, atol=1e-6)
    # non-members: forward + w * min reverse of the block
    non = torch.ones(g.num_nodes, dtype=torch.bool)
    non[top_idx] = False
    assert torch.allclose(final[0, non], pred[0, non] + w * revs.min(), atol=1e-6)


def test_head_query_uses_tail_form():
    g = make_toy_graph()
    model = make_model(support=False).eval()
    t, r = 2, 0
    batch = _all_negative_head(g, t, r)
    pred = model(g, batch)
    k = 2
    target = torch.tensor([int(pred.argmax())])
    final = rerank_predictions(model, g, batch, pred, target, None, k=k)
    top_idx = pred.topk(k, dim=-1).indices[0]
    # tail-form of a head query: (t, r + num_direct, candidate); its reverse
    # is the plain forward direction (candidate, r, ?) read at t
    revs = torch.tensor([float(model(g, torch.tensor([[[c, t, r]]]))[0, 0])
                         for c in top_idx.tolist()])
    assert torch.allclose(final[0, top_idx], pred[0, top_idx] + revs, atol=1e-6)


def test_mask_keeps_filtered_answers_out_of_the_block():
    g = make_toy_graph()
    model = make_model(support=False).eval()
    batch = _all_negative_tail(g, 0, 1)
    pred = model(g, batch)
    order = pred[0].argsort(descending=True)
    # filter out the two strongest candidates; the target is the third
    mask = torch.ones_like(pred, dtype=torch.bool)
    mask[0, order[0]] = False
    mask[0, order[1]] = False
    target = order[2:3]
    mask[0, target] = False                  # the mask never holds the target
    final = rerank_predictions(model, g, batch, pred, target, mask, k=2)
    # the block = the two best ELIGIBLE candidates: the target and order[3]
    num_direct = g.num_relations // 2
    def rev(c):
        return float(model(g, torch.tensor([[[c, 0, 1 + num_direct]]]))[0, 0])
    r_t, r_3 = rev(int(target)), rev(int(order[3]))
    floor = min(r_t, r_3)
    assert abs(float(final[0, target]) - (float(pred[0, target]) + r_t)) < 1e-6
    assert abs(float(final[0, order[3]]) - (float(pred[0, order[3]]) + r_3)) < 1e-6
    # the filtered-out leaders were not re-scored: imputed with the floor
    for c in (int(order[0]), int(order[1])):
        assert abs(float(final[0, c]) - (float(pred[0, c]) + floor)) < 1e-6


def test_ensemble_is_mean_of_members():
    g = make_toy_graph()
    a = make_model(support=False, seed=1).eval()
    b = make_model(support=False, seed=2).eval()
    ens = ScoreEnsemble([a, b]).eval()
    batch = _all_negative_tail(g, 1, 0)
    assert torch.allclose(ens(g, batch), (a(g, batch) + b(g, batch)) / 2, atol=1e-6)
    rb = torch.tensor([[0, 1, 0]])
    assert torch.allclose(ens.forward_relation(g, rb),
                          (a.forward_relation(g, rb) + b.forward_relation(g, rb)) / 2,
                          atol=1e-6)


def test_normalization_matches_model():
    g = make_toy_graph()
    num_direct = g.num_relations // 2
    batch = _all_negative_head(g, 4, 1)
    h, t, r = negative_sample_to_tail(*batch.unbind(-1), num_direct)
    assert (h == 4).all() and (r == 1 + num_direct).all()
    assert torch.equal(t[0], torch.arange(g.num_nodes))
