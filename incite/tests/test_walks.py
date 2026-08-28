"""Walk seeding and the anonymization protocol (design C, PLAN amendment)."""

import torch

from conftest import make_toy_graph

from incite.walks import WalkModule, anonymize, sample_walks


def test_same_seed_gives_identical_features():
    """Two runs with the same seed produce bit-identical walk features."""
    graph = make_toy_graph()
    starts = torch.tensor([0, 3, 5])
    torch.manual_seed(4)
    module = WalkModule(dim=16, num_walks=8, walk_length=6, seed=1024)
    with torch.no_grad():
        a_ent, a_rel = module(graph, starts, seed_offset=0)
        b_ent, b_rel = module(graph, starts, seed_offset=0)
    assert torch.equal(a_ent, b_ent)
    assert torch.equal(a_rel, b_rel)


def test_different_offsets_give_different_walks():
    graph = make_toy_graph()
    starts = torch.tensor([0, 3, 5])
    e0, r0, _ = sample_walks(graph, starts, 8, 6, seed=1024, seed_offset=0)
    e1, r1, _ = sample_walks(graph, starts, 8, 6, seed=1024, seed_offset=1)
    assert not (torch.equal(e0, e1) and torch.equal(r0, r1))


def test_walks_follow_edges():
    graph = make_toy_graph()
    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_type.tolist(),
                    graph.edge_index[1].tolist()))
    starts = torch.tensor([0, 1, 2, 3])
    ents, rels, mask = sample_walks(graph, starts, 8, 6, seed=1024)
    assert torch.equal(ents[:, :, 0], starts.unsqueeze(-1).expand(-1, 8))
    for b in range(len(starts)):
        for w in range(8):
            for t in range(6):
                if mask[b, w, t]:
                    step = (int(ents[b, w, t]), int(rels[b, w, t]),
                            int(ents[b, w, t + 1]))
                    assert step in edges, step


def test_anonymization_is_first_visit_indexing():
    """FLOCK's record: first-visit index for entities, first-appearance for
    relations, keeping identity within the walk (the PETALS separator)."""
    ents = torch.tensor([[[7, 3, 7, 5, 3]]])
    rels = torch.tensor([[[4, 4, 9, 4]]])
    anon_e, anon_r = anonymize(ents, rels)
    assert anon_e.tolist() == [[[0, 1, 0, 2, 1]]]
    # alpha, alpha, beta, alpha -- relation identity is kept in-walk
    assert anon_r.tolist() == [[[0, 0, 1, 0]]]


def test_anonymization_is_label_invariant():
    """Relabeling global ids does not change the record (the reason walks
    transfer across relation vocabularies)."""
    ents = torch.tensor([[[7, 3, 7, 5, 3]]])
    rels = torch.tensor([[[4, 4, 9, 4]]])
    ents2 = torch.tensor([[[70, 30, 70, 50, 30]]])
    rels2 = torch.tensor([[[1, 1, 0, 1]]])
    a1 = anonymize(ents, rels)
    a2 = anonymize(ents2, rels2)
    assert torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1])
