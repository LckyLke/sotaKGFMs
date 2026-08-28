"""The objective is TRIX's, pinned to the reference formula.

``incite/train.py`` reuses crest/train.py's loss verbatim (which restates
run_entity.py::train_and_validate). The hand computation below IS the
reference formula: BCE-with-logits over [positive | negatives] with
self-adversarial softmax weighting at the given temperature. torch.equal,
not allclose -- the code path must be the same arithmetic.
"""

import torch
from torch.nn import functional as F

from conftest import make_model, make_toy_graph

from incite import train as T


def test_self_adversarial_nll_matches_reference_formula():
    pred = torch.tensor([[1.5, -0.5, 0.25], [0.0, 2.0, -1.0]])
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    weight = torch.ones_like(pred)
    weight[:, 1:] = F.softmax(pred[:, 1:], dim=-1)  # temperature 1
    expected = ((bce * weight).sum(-1) / weight.sum(-1)).mean()
    got = T.self_adversarial_nll(pred, num_negative=2, adversarial_temperature=1.0)
    assert torch.equal(got, expected)
    # temperature 0 falls back to uniform 1/num_negative weights
    weight0 = torch.ones_like(pred)
    weight0[:, 1:] = 0.5
    expected0 = ((bce * weight0).sum(-1) / weight0.sum(-1)).mean()
    got0 = T.self_adversarial_nll(pred, num_negative=2, adversarial_temperature=0.0)
    assert torch.allclose(got0, expected0)


def test_strict_negatives_avoid_true_answers():
    graph = make_toy_graph()
    torch.manual_seed(1024)
    generator = torch.Generator().manual_seed(7)
    triples = T.sample_positive_triples(graph, 8, generator)
    batch = T.negative_sampling(graph, triples, 6, strict=True)
    h_index, t_index, r_index = batch.unbind(-1)
    bs = batch.shape[0]
    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_type.tolist(),
                    graph.edge_index[1].tolist()))
    for i in range(bs):
        h, r, t = int(h_index[i, 0]), int(r_index[i, 0]), int(t_index[i, 0])
        if i < bs // 2:  # tail negatives
            for neg in t_index[i, 1:].tolist():
                assert (h, r, neg) not in edges
        else:  # head negatives
            for neg in h_index[i, 1:].tolist():
                assert (neg, r, t) not in edges


def test_entity_batch_loss_runs_and_backprops():
    graph = make_toy_graph()
    model = make_model(dim=16, rounds=2, walks=False, support=False)
    model.train()
    generator = torch.Generator().manual_seed(1024)
    loss = T.entity_batch_loss(model, graph, batch_size=4, num_negative=3,
                               generator=generator,
                               adversarial_temperature=1.0, strict=True)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(bool((g != 0).any()) for g in grads)


def test_joint_loss_adds_relation_term():
    graph = make_toy_graph()
    model = make_model(dim=16, rounds=2, walks=False, support=False)
    model.train()
    generator = torch.Generator().manual_seed(1024)
    triples = T.sample_positive_triples(graph, 4, generator)
    l_ent = T.entity_loss_from_triples(model, graph, triples, num_negative=3)
    l_rel = T.relation_loss_from_triples(model, graph, triples)
    assert torch.isfinite(l_ent) and torch.isfinite(l_rel)
    total = l_ent + 0.5 * l_rel
    total.backward()  # both heads receive gradient through one trunk
    assert any(p.grad is not None and bool((p.grad != 0).any())
               for p in model.relation_mlp.parameters())
    assert any(p.grad is not None and bool((p.grad != 0).any())
               for p in model.score_mlp.parameters())
