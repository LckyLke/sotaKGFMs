"""Phase 0's gate in miniature: zero residual means CREST *is* the encoder.

Exactly, not approximately -- ``torch.equal``, no tolerance. Over real graphs
the same property is what ``scripts/verify_crest_identity.py`` checks row by
row against ``ranks/trix/``; a tolerance here would let two cancelling
defects pass (docs/CREST_PLAN.md section 2).
"""

import torch

from crest import bank as crest_bank
from crest.model import CRESTEntity, CRESTRelation

from conftest import ToyEncoder, make_readout, make_toy_graph


def test_entity_scores_equal_encoder_scores_exactly():
    graph = make_toy_graph()
    encoder = ToyEncoder()
    bank = crest_bank.build_bank_entity(graph, encoder, seed=1024, num_positive=4)
    model = CRESTEntity(encoder, make_readout(zero=True))
    candidates = torch.arange(graph.num_nodes)
    for (u, r) in [(0, 0), (3, 1), (7, 4), (2, 5)]:
        _, _, s0 = encoder.encode_single(graph, u, r)
        s = model(graph, u, r, candidates, bank)
        assert torch.equal(s, s0), "zero residual must reproduce s_v0 bit for bit"


def test_relation_scores_equal_encoder_scores_exactly():
    graph = make_toy_graph()
    encoder = ToyEncoder()
    bank = crest_bank.build_bank_relation(graph, encoder, seed=1024, num_positive=4)
    model = CRESTRelation(encoder, make_readout(zero=True))
    for (u, v) in [(0, 1), (4, 9), (11, 2)]:
        _, _, s0 = encoder.encode_relation_single(graph, u, v)
        assert torch.equal(model(graph, u, v, bank), s0)


def test_zero_residual_survives_everything_else_being_random():
    # only the last linear layer is zero; every other parameter is random,
    # which is exactly phase 0's configuration ("readout present")
    graph = make_toy_graph()
    encoder = ToyEncoder()
    bank = crest_bank.build_bank_entity(graph, encoder, seed=1024, num_positive=4)
    readout = make_readout(seed=99, zero=False)
    assert not readout.residual_is_zero()
    readout.zero_residual()
    assert readout.residual_is_zero()
    model = CRESTEntity(encoder, readout)
    candidates = torch.arange(graph.num_nodes)
    _, _, s0 = encoder.encode_single(graph, 1, 2)
    assert torch.equal(model(graph, 1, 2, candidates, bank), s0)


def test_a_nonzero_residual_actually_changes_scores():
    # the identity test above must be able to fail: with the last layer
    # non-zero the residual moves the scores, or the gate is vacuous
    graph = make_toy_graph()
    encoder = ToyEncoder()
    bank = crest_bank.build_bank_entity(graph, encoder, seed=1024, num_positive=4)
    readout = make_readout(seed=5, zero=False)
    with torch.no_grad():  # make sure the last layer is far from zero
        readout.reader.out.weight.fill_(0.5)
        readout.reader.out.bias.fill_(0.25)
    model = CRESTEntity(encoder, readout)
    candidates = torch.arange(graph.num_nodes)
    _, _, s0 = encoder.encode_single(graph, 1, 2)
    assert not torch.equal(model(graph, 1, 2, candidates, bank), s0)


def test_missing_bank_relation_falls_back_to_encoder_score():
    # a relation id with no inference edge has no bank entry; the documented
    # fallback is the raw encoder score, residual 0, even with a live readout
    graph = make_toy_graph()
    encoder = ToyEncoder()
    empty = crest_bank.ContextBank()
    readout = make_readout(seed=5, zero=False)
    model = CRESTEntity(encoder, readout)
    candidates = torch.arange(graph.num_nodes)
    _, _, s0 = encoder.encode_single(graph, 1, 2)
    assert torch.equal(model(graph, 1, 2, candidates, empty), s0)
