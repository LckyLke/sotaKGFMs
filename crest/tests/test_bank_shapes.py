"""Shape and layout contract of the bank (docs/CREST_PLAN.md and the row spec).

``f = [x_v ; z_r ; x_v * z_r ; cos(x_v, z_r) ; s_v0]`` of size 3d + 2 = 98
for d = 32; banks are [80, 98] per relation id; inverse relations are
separate ids with their own banks.
"""

import torch

from crest import bank as crest_bank

from conftest import D, ToyEncoder, make_toy_graph


def test_row_dim_is_98_for_d_32():
    assert D == 32
    assert crest_bank.row_dim(32) == 98
    assert crest_bank.ROWS_PER_RELATION == 80
    assert crest_bank.N_POSITIVE == 20
    assert crest_bank.NEG_PER_POS == 3


def test_row_feature_layout():
    d = 4
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -1.0, 2.0]])
    z = torch.tensor([2.0, 1.0, 0.0, -1.0])
    s0 = torch.tensor([7.0, -3.0])
    f = crest_bank.row_features(x, z, s0)
    assert f.shape == (2, crest_bank.row_dim(d))
    assert torch.equal(f[:, :d], x)
    assert torch.equal(f[:, d:2 * d], z.expand(2, d))
    assert torch.equal(f[:, 2 * d:3 * d], x * z)
    expected_cos = torch.nn.functional.cosine_similarity(x, z.expand(2, d), dim=-1)
    assert torch.equal(f[:, 3 * d], expected_cos)
    assert torch.equal(f[:, 3 * d + 1], s0)


def test_bank_tensors_are_80_by_98():
    graph = make_toy_graph(num_edges=80)  # enough edges that ids can fill 20 positives
    bank = crest_bank.build_bank_entity(graph, ToyEncoder(), seed=1024)
    assert len(bank) > 0
    for rid in bank.relation_ids():
        feats = bank.features(rid)
        labels = bank.labels(rid)
        assert feats.shape == (80, 98), (rid, feats.shape)
        assert labels.shape == (80,)
        assert int(labels.sum()) == 20  # 20 positives, 60 negatives
        # rows come in [positive, neg, neg, neg] blocks
        assert torch.equal(labels.reshape(20, 4)[:, 0], torch.ones(20, dtype=torch.long))
        assert int(labels.reshape(20, 4)[:, 1:].sum()) == 0


def test_sparse_relation_still_fills_80_rows():
    # a relation with fewer than 20 edges samples with replacement rather
    # than shrinking the tensor: the readout's context shape is uniform
    graph = make_toy_graph(num_edges=6)
    bank = crest_bank.build_bank_entity(graph, ToyEncoder(), seed=1024)
    for rid in bank.relation_ids():
        assert bank.features(rid).shape == (80, 98)


def test_inverse_relations_have_their_own_banks():
    graph = make_toy_graph()
    bank = crest_bank.build_bank_entity(graph, ToyEncoder(), seed=1024, num_positive=4)
    num_direct = graph.num_relations // 2
    present = set(bank.relation_ids())
    # the toy graph materialises every inverse, so each direct id with edges
    # is accompanied by its inverse id -- as a distinct bank
    for rid in list(present):
        assert crest_bank.inverse_relation(rid, graph.num_relations) in present
    assert any(rid >= num_direct for rid in present)
    direct = next(rid for rid in present if rid < num_direct)
    inverse = crest_bank.inverse_relation(direct, graph.num_relations)
    assert not torch.equal(bank.features(direct), bank.features(inverse))


def test_bank_rows_are_detached_data():
    # bank rows are data, never differentiated through: a builder that stores
    # grad-tracking rows pins one full encoder autograd graph per row for as
    # long as the bank lives, which is exactly the OOM that killed the first
    # live-readout evaluation sweep (38 of 41 graphs on a 15.6 GiB GPU)
    class GradEncoder(ToyEncoder):
        def __init__(self):
            super().__init__()
            self.scale = torch.ones(1, requires_grad=True)

        def encode_single(self, graph, u, r):
            x, z, s0 = super().encode_single(graph, u, r)
            return x * self.scale, z * self.scale, s0 * self.scale

        def encode_relation_single(self, graph, u, v):
            w, c, s0 = super().encode_relation_single(graph, u, v)
            return w * self.scale, c * self.scale, s0 * self.scale

    graph = make_toy_graph()
    bank = crest_bank.build_bank_entity(graph, GradEncoder(), seed=1024, num_positive=4)
    assert len(bank) > 0
    for rid in bank.relation_ids():
        feats = bank.features(rid)
        assert feats.grad_fn is None and not feats.requires_grad, rid
    rbank = crest_bank.build_bank_relation(graph, GradEncoder(), seed=1024, num_positive=4)
    for rid in rbank.relation_ids():
        feats = rbank.features(rid)
        assert feats.grad_fn is None and not feats.requires_grad, rid


def test_cache_roundtrip_and_key(tmp_path):
    graph = make_toy_graph()
    bank = crest_bank.build_bank_entity(graph, ToyEncoder(), seed=1024, num_positive=4)
    bank.graph_id = "FB15k237Inductive:v1"
    bank.checkpoint_hash = "deadbeef"
    path = bank.save(str(tmp_path))
    loaded = crest_bank.ContextBank.load(path)
    assert loaded.relation_ids() == bank.relation_ids()
    for rid in bank.relation_ids():
        assert torch.equal(loaded.features(rid), bank.features(rid))
    # a different seed is a different cache entry: identical rebuilds are the
    # thing the cache exists to prevent, different ones must not collide
    other = crest_bank.ContextBank("FB15k237Inductive:v1", "deadbeef", 2048)
    assert other.cache_path(str(tmp_path)) != bank.cache_path(str(tmp_path))
