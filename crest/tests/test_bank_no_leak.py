"""The leakage contract of docs/CREST_PLAN.md 4.1, asserted.

Three claims, each tested directly:

1. every bank row comes from an inference-graph edge, and the message graph
   the encoder sees has that edge *and its inverse* removed;
2. ``Ans(u, r)`` is computed over the inference graph only -- a triple that
   exists only in the test set is a legal negative, and filtering it out
   would itself be the leak;
3. negatives never collide with an inference-graph answer.
"""

import torch

from crest import bank as crest_bank

from conftest import RecordingEncoder, make_toy_graph


def _triples(edge_index, edge_type):
    return set(zip(edge_index[0].tolist(), edge_type.tolist(), edge_index[1].tolist()))


def test_message_graph_never_contains_the_sampled_edge():
    # a chunk removes its own edge plus at most one edge per *other* relation
    # (bank.py module docstring); the contract asserted per call is therefore:
    # every removed edge of the query's own relation belongs to this query,
    # the query's edge really is gone, and nothing non-inference ever appears
    graph = make_toy_graph()
    encoder = RecordingEncoder()
    crest_bank.build_bank_entity(graph, encoder, seed=1024, num_positive=4)
    assert encoder.calls, "the builder never called the encoder"
    inference = _triples(graph.edge_index, graph.edge_type)
    for call in encoder.calls:
        seen = _triples(call["edge_index"], call["edge_type"])
        u, r = call["u"], call["r"]
        r_inv = crest_bank.inverse_relation(r, graph.num_relations)
        removed = inference - seen
        assert removed, "encoder saw the full inference graph"
        # the chunk holds one edge per relation id, so the removed edges of
        # the query's own relation (and its inverse) are the query's edge
        own = {t for t in removed if t[1] in (r, r_inv)}
        assert own, "the query's own edge was not removed"
        assert all((t[0] == u and t[1] == r) or (t[1] == r_inv and t[2] == u)
                   for t in own), own
        assert seen <= inference, "message graph contains non-inference edges"


def test_bank_composition_ignores_the_chunk_size():
    # which edges and negatives make up the bank is drawn before any encoding,
    # so composition depends on (graph, seed) alone; build_batch_size=1 is
    # the per-edge build, and larger chunks may only perturb feature values
    # (the shared message graph), never rows or labels
    graph = make_toy_graph()
    b1 = crest_bank.build_bank_entity(graph, RecordingEncoder(), seed=1024,
                                      num_positive=4, build_batch_size=1)
    b32 = crest_bank.build_bank_entity(graph, RecordingEncoder(), seed=1024,
                                       num_positive=4, build_batch_size=32)
    assert b1.relation_ids() == b32.relation_ids()
    for rid in b1.relation_ids():
        assert torch.equal(b1.labels(rid), b32.labels(rid))
        assert b1.features(rid).shape == b32.features(rid).shape


def test_ans_ignores_test_edges_so_they_are_legal_negatives():
    # a graph where (u=0, r=0) has exactly the answers the inference graph
    # states; x=5 is an answer only in the (hypothetical) test set, which the
    # bank builder must never see
    graph = make_toy_graph()
    u, r, x = 0, 0, 5
    inference_answers = set(crest_bank.ans(graph, u, r).tolist())
    if x in inference_answers:  # make the point unconditional on the random toy
        x = next(v for v in range(graph.num_nodes) if v not in inference_answers)
    negatives = crest_bank._sample_negatives(
        graph, u, r, 512, torch.Generator().manual_seed(0))
    # (2) the test-only answer is in the negative pool: Ans saw only inference
    assert x in set(negatives.tolist())
    # (3) and no inference-graph answer ever is
    assert not (set(negatives.tolist()) & inference_answers)


def test_bank_depends_only_on_the_inference_graph():
    # the same inference graph must produce the same bank whatever else
    # exists in the world; test edges have no channel in and this pins the
    # builder to determinism in (graph, seed) alone
    graph = make_toy_graph()
    b1 = crest_bank.build_bank_entity(graph, RecordingEncoder(), seed=1024, num_positive=4)
    b2 = crest_bank.build_bank_entity(graph, RecordingEncoder(), seed=1024, num_positive=4)
    assert b1.relation_ids() == b2.relation_ids()
    for rid in b1.relation_ids():
        assert torch.equal(b1.features(rid), b2.features(rid))
        assert torch.equal(b1.labels(rid), b2.labels(rid))


def test_relation_bank_negatives_avoid_inference_relations_only():
    graph = make_toy_graph()
    encoder = RecordingEncoder()
    bank = crest_bank.build_bank_relation(graph, encoder, seed=1024, num_positive=4)
    num_direct = graph.num_relations // 2
    assert bank.relation_ids()
    assert all(rid < num_direct for rid in bank.relation_ids())
    # the message graph contract holds on the relation path too
    inference = _triples(graph.edge_index, graph.edge_type)
    for call in encoder.calls:
        assert _triples(call["edge_index"], call["edge_type"]) < inference
