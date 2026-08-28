"""Support-set contracts: no leak, detached rows (the 14.85 GiB regression),
retrieval by head distance, PU-weighted labels.

The no-leak contract has two halves (incite/support.py docstring):
* build time -- the support pass for a pair (u, r, v) never sees that edge
  or its inverse in its message graph;
* query time -- rows whose support pair equals the query pair are excluded,
  because in training the query IS a graph edge.
"""

import torch

from conftest import (RecordingEncoder, ToySupportEncoder, make_model,
                      make_toy_graph, make_unique_hr_graph)

from incite import support as S
from incite.graphs import ans


def _build(graph, encoder, **kwargs):
    defaults = dict(seed=1024, per_relation_cap=4, neg_per_pos=2,
                    prototype_k=2, hops=3, ball_cap=64, class_prior=0.1,
                    build_batch_size=3)
    defaults.update(kwargs)
    return S.build_support(graph, encoder, **defaults)


def test_build_removes_each_pairs_own_edge():
    """On a graph with at most one edge per (head, relation), every pass
    that scored a support pair ran without that pair's edge or its inverse."""
    graph = make_unique_hr_graph()
    encoder = RecordingEncoder()
    store = _build(graph, encoder)
    num_direct = graph.num_relations // 2
    for rid in store.relation_ids():
        heads, tails = store._heads[rid], store._tails[rid]
        for u, v in zip(heads.tolist(), tails.tolist()):
            r_inv = rid + num_direct if rid < num_direct else rid - num_direct
            scored = [c for c in encoder.calls
                      if any(int(h) == u and int(r) == rid
                             for h, r in zip(c["heads"], c["rels"]))]
            assert scored, "pair (%d, %d, %d) was never scored" % (u, rid, v)
            for call in scored:
                ei, et = call["edge_index"], call["edge_type"]
                direct = ((ei[0] == u) & (ei[1] == v) & (et == rid)).any()
                inverse = ((ei[0] == v) & (ei[1] == u) & (et == r_inv)).any()
                assert not bool(direct), (u, rid, v)
                assert not bool(inverse), (u, rid, v)


def test_negatives_are_never_answers():
    """Hard negatives satisfy (u, r, x) not in G -- Ans over the inference
    graph only. Row identity comes from ToySupportEncoder's features: the
    row layout is [x_u ; x_cand ; s0 ; label], so feature[0] is the pass's
    head u and feature[d] is the candidate id."""
    graph = make_toy_graph()
    d = ToySupportEncoder().d
    store = _build(graph, ToySupportEncoder())
    for rid in store.relation_ids():
        feat = store._feat[rid]  # [m, 1 + neg, 2d + 2]
        for j in range(feat.shape[0]):
            u = int(feat[j, 0, 0])
            known = set(ans(graph, u, rid).tolist())
            positive = int(feat[j, 0, d])
            assert positive in known  # the positive IS an answer
            for k in range(1, feat.shape[1]):
                assert int(feat[j, k, d]) not in known


def test_retrieval_excludes_the_query_pair():
    graph = make_unique_hr_graph()
    store = _build(graph, ToySupportEncoder())
    rid = store.relation_ids()[0]
    u = int(store._heads[rid][0])
    v = int(store._tails[rid][0])
    with_pair = store.retrieve(rid, u, k=4)
    without = store.retrieve(rid, u, k=4, exclude_pair=(u, v))
    d = ToySupportEncoder().d
    # positive rows carry label +1 in the last feature; the row layout is
    # [x_u ; x_cand ; s0 ; label], so dim 0 names u and dim d the candidate
    def has_query_row(rows):
        if rows is None:
            return False
        pos = rows[:, -1] == 1.0
        return bool(((rows[:, 0] == u) & (rows[:, d] == v) & pos).any())
    assert has_query_row(with_pair)
    assert not has_query_row(without)


def test_prototypes_come_from_edge_removed_pair_states():
    """The prototype is the mean of the first K pair states, and those pair
    states were computed with the pair's own edge removed (covered by the
    recording test above); here: shape and determinism under the seed."""
    graph = make_toy_graph()
    store1 = _build(graph, ToySupportEncoder())
    store2 = _build(graph, ToySupportEncoder())
    for rid in store1.relation_ids():
        assert torch.equal(store1._proto[rid], store2._proto[rid])
        assert store1._proto[rid].shape == (2 * ToySupportEncoder().d,)


def test_pu_labels_downweight_negatives():
    graph = make_toy_graph()
    store = _build(graph, ToySupportEncoder(), class_prior=0.25)
    rid = store.relation_ids()[0]
    feat = store._feat[rid]
    assert float(feat[0, 0, -1]) == 1.0
    assert float(feat[0, 1, -1]) == -(1.0 - 0.25)


def test_support_rows_are_detached():
    """REGRESSION (docs/INCITE_PLAN.md lesson 3): support rows built through
    the real model must carry no autograd history. Rows with grad_fn pinned
    one encoder graph per row in CREST until the GPU filled (14.85 GiB)."""
    graph = make_toy_graph()
    model = make_model(dim=16, rounds=2, walks=False, support=True)
    model.train()  # even in train mode the builder must detach
    store = _build(graph, model, build_batch_size=2, per_relation_cap=2)
    tensors = store.tensors()
    assert tensors, "empty store proves nothing"
    for t in tensors:
        assert t.grad_fn is None, "support tensor carries autograd history"
        assert not t.requires_grad


def test_training_step_backward_works_with_support():
    """Gradients flow through the readout and trunk, not into the store."""
    from incite import train as T
    graph = make_toy_graph()
    model = make_model(dim=16, rounds=2, walks=False, support=True)
    model.eval()
    store = _build(graph, model, build_batch_size=2, per_relation_cap=2)
    model.train()
    generator = torch.Generator().manual_seed(1024)
    loss = T.entity_batch_loss(model, graph, batch_size=4, num_negative=3,
                               generator=generator, support=store)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.readout.parameters() if p.grad is not None]
    assert grads and any(bool((g != 0).any()) for g in grads)
    for t in store.tensors():
        assert t.grad_fn is None
