"""Entity-permutation equivariance of the deterministic model.

Scoped per docs/CREST_PLAN.md 4.3: track B's random channel exists precisely
to break relation symmetry, so when a channel is active this test SKIPS --
never fails -- and the deterministic configuration remains the one under test.

Two symmetries are asserted, both exactly (the toy encoder is
integer-featured, see conftest):

* relabelling entities relabels scores: ``s(pi(G), pi(u), r)[pi(v)] ==
  s(G, u, r)[v]``, with the *same* bank -- bank rows are feature vectors,
  entity identity must never enter the readout;
* permuting the candidate list permutes the scores: query rows attend the
  bank independently of one another (crest/pfn.py).
"""

import pytest
import torch

from crest import bank as crest_bank
from crest.model import CRESTEntity
from crest.randchan import NoiseChannel

from conftest import ToyEncoder, make_readout, make_toy_graph


def _permute(graph, perm):
    return crest_bank.Graph(
        edge_index=perm[graph.edge_index],
        edge_type=graph.edge_type.clone(),
        num_nodes=graph.num_nodes,
        num_relations=graph.num_relations,
    )


@pytest.mark.parametrize("channel", [None, NoiseChannel(feature_dim=4)])
def test_entity_permutation_equivariance(channel):
    if channel is not None and channel.active:
        pytest.skip("random channel active: it breaks this symmetry by design "
                    "(track B); the deterministic model is the one under test")
    graph = make_toy_graph()
    encoder = ToyEncoder()
    bank = crest_bank.build_bank_entity(graph, encoder, seed=1024, num_positive=4)
    model = CRESTEntity(encoder, make_readout(seed=11, zero=False), channel=channel)
    with torch.no_grad():  # a live residual, so the readout is actually exercised
        model.readout.reader.out.weight.fill_(0.125)

    perm = torch.randperm(graph.num_nodes, generator=torch.Generator().manual_seed(2))
    graph_p = _permute(graph, perm)
    candidates = torch.arange(graph.num_nodes)
    for (u, r) in [(0, 0), (3, 1), (7, 4)]:
        s = model(graph, u, r, candidates, bank)
        s_p = model(graph_p, int(perm[u]), r, candidates, bank)
        # candidate v of the original graph is candidate perm[v] afterwards
        assert torch.equal(s_p[perm], s)


def test_candidate_permutation_equivariance():
    graph = make_toy_graph()
    encoder = ToyEncoder()
    bank = crest_bank.build_bank_entity(graph, encoder, seed=1024, num_positive=4)
    model = CRESTEntity(encoder, make_readout(seed=11, zero=False))
    with torch.no_grad():
        model.readout.reader.out.weight.fill_(0.125)
    candidates = torch.arange(graph.num_nodes)
    shuffle = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(4))
    s = model(graph, 3, 1, candidates, bank)
    s_shuffled = model(graph, 3, 1, candidates[shuffle], bank)
    assert torch.equal(s_shuffled, s[shuffle])


def test_bank_row_order_does_not_matter():
    # attention over the bank is a weighted sum; inserting the rows in a
    # different order must not move a score. allclose rather than equal:
    # reordering the bank reorders a float32 softmax-weighted sum, which is
    # associativity, not semantics.
    graph = make_toy_graph()
    encoder = ToyEncoder()
    bank = crest_bank.build_bank_entity(graph, encoder, seed=1024, num_positive=4)
    model = CRESTEntity(encoder, make_readout(seed=11, zero=False))
    with torch.no_grad():
        model.readout.reader.out.weight.fill_(0.125)
    candidates = torch.arange(graph.num_nodes)
    s = model(graph, 3, 1, candidates, bank)

    reordered = crest_bank.ContextBank()
    perm = torch.randperm(bank.features(1).shape[0],
                          generator=torch.Generator().manual_seed(6))
    for rid in bank.relation_ids():
        reordered.put(rid, bank.features(rid)[perm], bank.labels(rid)[perm])
    s2 = model(graph, 3, 1, candidates, reordered)
    assert torch.allclose(s, s2, atol=1e-5)
