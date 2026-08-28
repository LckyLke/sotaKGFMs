"""Shared fixtures for the INCITE test suite.

Everything toy-sized runs on CPU in well under a second. The tests are
written to run INSIDE the container (scripts/test_incite.sh): the host has
no torch. Real-graph and pinned-TRIX tests skip themselves cleanly when the
data root or the TRIX tree is absent, so the suite also passes on a bare
checkout inside the container image alone.
"""

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "shared")):
    if p not in sys.path:
        sys.path.insert(0, p)

TRIX_ROOT = os.environ.get("TRIX_ROOT", "/kgfm/repos/trix")
_trix_src = os.path.join(TRIX_ROOT, "src")
if os.path.isdir(_trix_src) and _trix_src not in sys.path:
    sys.path.insert(0, _trix_src)

from incite import graphs as G  # noqa: E402

D = 16  # toy model width; the real config uses 32 (TRIX recipe)


def make_toy_graph(seed=7, num_nodes=12, num_direct=3, num_edges=28):
    """A random toy inference graph with inverses materialized, ULTRA-style."""
    generator = torch.Generator().manual_seed(seed)
    heads = torch.randint(num_nodes, (num_edges,), generator=generator)
    tails = torch.randint(num_nodes, (num_edges,), generator=generator)
    same = heads == tails
    tails[same] = (tails[same] + 1) % num_nodes
    rels = torch.randint(num_direct, (num_edges,), generator=generator)
    edge_index = torch.stack([torch.cat([heads, tails]), torch.cat([tails, heads])])
    edge_type = torch.cat([rels, rels + num_direct])
    return G.Graph(edge_index=edge_index, edge_type=edge_type,
                   num_nodes=num_nodes, num_relations=2 * num_direct)


def make_unique_hr_graph(num_nodes=10, num_direct=3):
    """A toy graph where every (head, relation) pair has AT MOST ONE edge,
    so 'the support pass removed its own edge' is checkable per pair."""
    triples = []
    seen = set()
    generator = torch.Generator().manual_seed(11)
    for _ in range(40):
        h = int(torch.randint(num_nodes, (1,), generator=generator))
        r = int(torch.randint(num_direct, (1,), generator=generator))
        t = int(torch.randint(num_nodes, (1,), generator=generator))
        if h == t or (h, r) in seen or (t, r + num_direct) in seen:
            continue
        seen.add((h, r))
        seen.add((t, r + num_direct))
        triples.append((h, r, t))
    heads = torch.tensor([x[0] for x in triples])
    rels = torch.tensor([x[1] for x in triples])
    tails = torch.tensor([x[2] for x in triples])
    edge_index = torch.stack([torch.cat([heads, tails]), torch.cat([tails, heads])])
    edge_type = torch.cat([rels, rels + num_direct])
    return G.Graph(edge_index=edge_index, edge_type=edge_type,
                   num_nodes=num_nodes, num_relations=2 * num_direct)


def make_model(dim=D, rounds=2, walks=False, support=True, seed=3, **kwargs):
    from incite.model import INCITE
    torch.manual_seed(seed)
    walk_cfg = dict(num_walks=4, walk_length=4, seed=1024) if walks else None
    return INCITE(dim=dim, rounds=rounds, layer_norm=True, short_cut=True,
                  count_channel=True, walks=walk_cfg, support_readout=support,
                  **kwargs)


class ToySupportEncoder:
    """Deterministic identity-revealing encoder for the support tests.

    ``x[j, v, 0] == v`` and ``x[j, v, 1] == heads[j]`` by construction, so a
    support row's feature vector names the pair it was built from -- what
    the no-leak test needs to identify excluded rows.
    """

    def __init__(self, d=8):
        self.d = d

    def encode_unlabeled(self, graph):
        x = torch.zeros(graph.num_nodes, self.d)
        x[:, 0] = torch.arange(graph.num_nodes).float()
        x[:, 2] = torch.bincount(graph.edge_index[0],
                                 minlength=graph.num_nodes).float()
        return x

    def encode_queries(self, graph, heads, rels):
        m, n = len(heads), graph.num_nodes
        x = torch.zeros(m, n, self.d)
        x[:, :, 0] = torch.arange(n).float()
        x[:, :, 1] = heads.float().unsqueeze(-1)
        x[:, :, 2] = torch.bincount(graph.edge_index[0], minlength=n
                                    ).float().unsqueeze(0)
        z = torch.zeros(m, self.d)
        z[:, 0] = rels.float()
        s0 = (torch.arange(n).float().unsqueeze(0) * 7
              + rels.float().unsqueeze(-1)) % 5
        return x, z, s0


class RecordingEncoder(ToySupportEncoder):
    """ToySupportEncoder that snapshots every message graph it encodes."""

    def __init__(self, d=8):
        super().__init__(d)
        self.calls = []

    def encode_queries(self, graph, heads, rels):
        self.calls.append({
            "edge_index": graph.edge_index.clone(),
            "edge_type": graph.edge_type.clone(),
            "heads": heads.clone(), "rels": rels.clone(),
        })
        return super().encode_queries(graph, heads, rels)


def load_real_graph():
    """Metafam's inference (test) graph via the pinned TRIX loaders, or skip."""
    root = os.environ.get("INCITE_TEST_DATA",
                          os.path.join(REPO, "data", "roots", "trix"))
    if not os.path.isdir(os.path.join(root, "mtdea")):
        pytest.skip("no processed data root at %s" % root)
    pytest.importorskip("trix")
    pytest.importorskip("torch_geometric")
    from easydict import EasyDict
    from trix import util
    dataset = util.build_dataset(EasyDict({
        "dataset": {"class": "Metafam", "root": root, "version": "Metafam"}}))
    graph = dataset[2]
    graph.num_relations = int(graph.num_relations)
    return graph


@pytest.fixture
def toy_graph():
    return make_toy_graph()
