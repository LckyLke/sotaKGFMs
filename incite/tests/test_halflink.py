"""Half-link masking: exactly the half is dropped, nothing else changes.

Toy graph, CPU. Checks:

* answer masking removes every (x, r, t) edge and every inverse copy
  (t, inv r, x) of a masked row, keeps every other edge;
* query masking removes every (h, r, x) edge and every inverse copy;
* unmasked rows are untouched; all-False masks return the graph itself;
* a head-direction positive (converted to tail form) masks the head's
  outgoing query-relation edges, the same fact seen from the other side;
* the training forward with all-False masks equals the plain forward
  bitwise, and a masked forward runs backward;
* the coin helper is a pure function of (seed, step, micro).
"""

import torch

from conftest import make_model, make_toy_graph
from incite.model import mask_halflinks, negative_sample_to_tail
from incite.train import halflink_masks


def _edges(g):
    return set(zip(g.edge_index[0].tolist(), g.edge_type.tolist(), g.edge_index[1].tolist()))


def _pick_positive(g, want_many=True):
    """A direct edge (h, r, t) whose t has at least two incoming r-edges."""
    nd = g.num_relations // 2
    e = _edges(g)
    for h, r, t in sorted(e):
        if r >= nd:
            continue
        others = [x for (x, rr, y) in e if rr == r and y == t and x != h]
        if others or not want_many:
            return h, r, t
    raise AssertionError("no suitable positive in the toy graph")


def test_answer_mask_drops_exactly_the_answer_half():
    g = make_toy_graph(num_edges=60)
    nd = g.num_relations // 2
    h, r, t = _pick_positive(g)
    before = _edges(g)
    out = mask_halflinks(g, torch.tensor([h]), torch.tensor([r]), torch.tensor([t]),
                         torch.tensor([True]), torch.tensor([False]), nd)
    after = _edges(out)
    dropped = before - after
    expect = {(x, rr, y) for (x, rr, y) in before if (rr == r and y == t) or (rr == r + nd and x == t)}
    assert dropped == expect and expect, (dropped, expect)
    assert after <= before and len(after) == len(before) - len(expect)


def test_query_mask_drops_exactly_the_query_half():
    g = make_toy_graph(num_edges=60)
    nd = g.num_relations // 2
    h, r, t = _pick_positive(g, want_many=False)
    before = _edges(g)
    out = mask_halflinks(g, torch.tensor([h]), torch.tensor([r]), torch.tensor([t]),
                         torch.tensor([False]), torch.tensor([True]), nd)
    dropped = before - _edges(out)
    expect = {(x, rr, y) for (x, rr, y) in before if (rr == r and x == h) or (rr == r + nd and y == h)}
    assert dropped == expect and expect


def test_unmasked_rows_untouched_and_all_false_is_identity():
    g = make_toy_graph(num_edges=60)
    nd = g.num_relations // 2
    h, r, t = _pick_positive(g)
    same = mask_halflinks(g, torch.tensor([h]), torch.tensor([r]), torch.tensor([t]),
                          torch.tensor([False]), torch.tensor([False]), nd)
    assert same is g
    # two rows, only the second masked: the first row's half survives
    h2, r2, t2 = _pick_positive(g, want_many=False)
    out = mask_halflinks(g, torch.tensor([h, h2]), torch.tensor([r, r2]), torch.tensor([t, t2]),
                         torch.tensor([False, True]), torch.tensor([False, False]), nd)
    after = _edges(out)
    if (r2, t2) != (r, t):
        assert any(rr == r and y == t for (x, rr, y) in after)


def test_head_query_masks_the_heads_outgoing_edges():
    g = make_toy_graph(num_edges=60)
    nd = g.num_relations // 2
    h, r, t = _pick_positive(g, want_many=False)
    # a head query for the same fact, as the training batch presents it:
    # h varies along the candidate axis, t fixed -> tail form (t, r+nd, h)
    batch = torch.tensor([[[h, t, r], [(h + 1) % g.num_nodes, t, r]]])
    hh, tt, rr = negative_sample_to_tail(*batch.unbind(-1), nd)
    assert int(hh[0, 0]) == t and int(rr[0, 0]) == r + nd and int(tt[0, 0]) == h
    before = _edges(g)
    out = mask_halflinks(g, hh[:, 0], rr[:, 0], tt[:, 0],
                         torch.tensor([True]), torch.tensor([False]), nd)
    dropped = before - _edges(out)
    # the "answer half" of the tail-form row is h's outgoing r-edges
    expect = {(x, r_, y) for (x, r_, y) in before if (r_ == r and x == h) or (r_ == r + nd and y == h)}
    assert dropped == expect and expect


def test_forward_identity_at_zero_and_backward_when_masked():
    g = make_toy_graph(num_edges=60)
    model = make_model(support=False).train()
    nd = g.num_relations // 2
    h, r, t = _pick_positive(g)
    neg = [(t + k) % g.num_nodes for k in range(1, 4)]
    batch = torch.tensor([[[h, t, r]] + [[h, x, r] for x in neg]])
    plain = model(g, batch)
    off = model(g, batch, halflink=(torch.tensor([False]), torch.tensor([False])))
    assert torch.equal(plain, off)
    on = model(g, batch, halflink=(torch.tensor([True]), torch.tensor([False])))
    assert on.shape == plain.shape
    on.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_masks_are_pure_functions_of_seed_step_micro():
    a = halflink_masks(16, 0.3, 0.1, seed=1024, step=5, micro=0)
    b = halflink_masks(16, 0.3, 0.1, seed=1024, step=5, micro=0)
    c = halflink_masks(16, 0.3, 0.1, seed=1024, step=6, micro=0)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not (torch.equal(a[0], c[0]) and torch.equal(a[1], c[1]))
    assert halflink_masks(16, 0.0, 0.0, 1024, 1) is None
