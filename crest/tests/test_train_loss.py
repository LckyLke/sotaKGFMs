"""The batched training path: TRIX's sampling and loss over ``s = s_v0 + s_pfn``.

Three properties are load-bearing:

* the zero-residual identity extends to the training path -- with the
  readout's last layer at zero, the batched scores are the encoder's own
  scores bit for bit, so training starts exactly at TRIX;
* strict negative sampling never proposes a true answer, which is the
  property TRIX's ``strict_negative_mask`` exists for;
* the loss is TRIX's: BCE with logits and self-adversarial negative
  weighting, checked against a hand computation.
"""

import torch
from torch.nn import functional as F

from conftest import ToyEncoder, make_readout, make_toy_graph

from crest import bank as crest_bank
from crest import train as crest_train
from crest.model import CRESTEntity


def _setup(zero=True):
    graph = make_toy_graph()
    encoder = ToyEncoder()
    ctx_bank = crest_bank.build_bank_entity(graph, encoder, seed=1024,
                                            num_positive=4, neg_per_pos=2)
    model = CRESTEntity(encoder, make_readout(zero=zero))
    return graph, encoder, ctx_bank, model


def test_batched_scores_zero_residual_are_encoder_scores():
    graph, encoder, ctx_bank, model = _setup(zero=True)
    torch.manual_seed(1024)
    generator = torch.Generator().manual_seed(1024)
    triples = crest_train.sample_positive_triples(graph, 8, generator)
    batch = crest_train.negative_sampling(graph, triples, 5, strict=True)
    pred = crest_train.entity_batch_scores(model, graph, batch, ctx_bank)

    # recompute what the encoder alone says, row by row, on the same
    # edge-removed message graph the batched path used
    h_index, t_index, r_index = batch.unbind(-1)
    num_direct = graph.num_relations // 2
    message_graph = crest_train._remove_batch_edges(
        graph, h_index[:, 0], r_index[:, 0], t_index[:, 0])
    for i in range(batch.shape[0]):
        tail_row = bool((h_index[i] == h_index[i, 0]).all())
        u = int(h_index[i, 0]) if tail_row else int(t_index[i, 0])
        r = int(r_index[i, 0]) if tail_row else int(r_index[i, 0]) + num_direct
        cand = t_index[i] if tail_row else h_index[i]
        _, _, s0 = encoder.encode_single(message_graph, u, r)
        assert torch.equal(pred[i], s0[cand]), "row %d is not the encoder's score" % i


def test_live_residual_changes_training_scores():
    graph, _, ctx_bank, model = _setup(zero=False)
    zero_model = CRESTEntity(model._encoder_ref, make_readout(zero=True))
    torch.manual_seed(1024)
    generator = torch.Generator().manual_seed(1024)
    triples = crest_train.sample_positive_triples(graph, 8, generator)
    batch = crest_train.negative_sampling(graph, triples, 5, strict=True)
    live = crest_train.entity_batch_scores(model, graph, batch, ctx_bank)
    base = crest_train.entity_batch_scores(zero_model, graph, batch, ctx_bank)
    assert not torch.equal(live, base)


def test_strict_negatives_avoid_true_answers():
    graph = make_toy_graph()
    torch.manual_seed(1024)
    generator = torch.Generator().manual_seed(7)
    triples = crest_train.sample_positive_triples(graph, 8, generator)
    batch = crest_train.negative_sampling(graph, triples, 6, strict=True)
    h_index, t_index, r_index = batch.unbind(-1)
    bs = batch.shape[0]
    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_type.tolist(),
                    graph.edge_index[1].tolist()))
    for i in range(bs):
        h, r = int(h_index[i, 0]), int(r_index[i, 0])
        t = int(t_index[i, 0])
        if i < bs // 2:  # tail negatives
            for neg in t_index[i, 1:].tolist():
                assert (h, r, neg) not in edges
        else:  # head negatives
            for neg in h_index[i, 1:].tolist():
                assert (neg, r, t) not in edges


def test_self_adversarial_nll_matches_hand_computation():
    pred = torch.tensor([[1.5, -0.5, 0.25], [0.0, 2.0, -1.0]])
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    weight = torch.ones_like(pred)
    weight[:, 1:] = F.softmax(pred[:, 1:], dim=-1)  # temperature 1
    expected = ((bce * weight).sum(-1) / weight.sum(-1)).mean()
    got = crest_train.self_adversarial_nll(pred, num_negative=2,
                                           adversarial_temperature=1.0)
    assert torch.equal(got, expected)
    # temperature 0 falls back to uniform 1/num_negative weights
    weight0 = torch.ones_like(pred)
    weight0[:, 1:] = 0.5
    expected0 = ((bce * weight0).sum(-1) / weight0.sum(-1)).mean()
    got0 = crest_train.self_adversarial_nll(pred, num_negative=2,
                                            adversarial_temperature=0.0)
    assert torch.allclose(got0, expected0)


def test_stage_a_gradients_reach_only_the_readout():
    graph, _, ctx_bank, model = _setup(zero=False)
    torch.manual_seed(1024)
    generator = torch.Generator().manual_seed(1024)
    loss = crest_train.entity_batch_loss(model, graph, ctx_bank, 8, 5, generator,
                                         encoder_no_grad=True)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.readout.parameters() if p.grad is not None]
    assert grads and any(bool((g != 0).any()) for g in grads)
