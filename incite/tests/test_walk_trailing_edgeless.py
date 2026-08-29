"""Regression: walks from a trailing edgeless node must not gather OOB.

The phase-2.1 run died at ~step 9000 with a device-side assert: a walk
lane sat on a zero-degree node whose CSR offset equals the edge count
(the highest-indexed node had no outgoing edges after remove_easy_edges),
and the dead lane's masked-but-executed gather read dst[num_edges]. On
CPU the same bug raises IndexError deterministically, which is what this
test guards.
"""
import torch

from incite.walks import sample_walks


def _graph(edge_index, edge_type, num_nodes, num_relations):
    from torch_geometric.data import Data
    return Data(edge_index=edge_index, edge_type=edge_type,
                num_nodes=num_nodes, num_relations=num_relations)


def test_walk_from_trailing_edgeless_node():
    # Node 3 (the last) has no outgoing edges: its CSR offset == num_edges.
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_type = torch.tensor([0, 1, 0])
    g = _graph(edge_index, edge_type, num_nodes=4, num_relations=2)

    starts = torch.tensor([3, 0])
    ents, rels, masks = sample_walks(g, starts, num_walks=8, walk_length=5,
                                     seed=1024)
    # every lane of the edgeless start is dead from step one
    assert not masks[0].any()
    assert (ents[0] == 3).all()
    # the normal start keeps walking
    assert masks[1].any()


def test_walk_reaching_trailing_edgeless_node_mid_walk():
    # 0 -> 3 exists, 3 has no outgoing edges: lanes die AT node 3.
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 3]])
    edge_type = torch.tensor([0, 0, 1])
    g = _graph(edge_index, edge_type, num_nodes=4, num_relations=2)

    ents, rels, masks = sample_walks(g, torch.tensor([0]), num_walks=64,
                                     walk_length=6, seed=1024)
    reached = (ents[0] == 3).any()
    assert reached, "with 64 walks of length 6 some lane must reach node 3"
    # once at node 3, the lane is dead and pinned there
    at3 = ents[0, :, :-1] == 3
    assert (ents[0, :, 1:][at3] == 3).all()
