"""Shared toy fixtures for the CREST test suite.

Everything here runs on the host CPU in well under a second: the toy graph has
12 entities and 3 direct relations (6 relation ids with inverses), far inside
the ~50-entity / ~5-relation budget these tests are allowed.

The ToyEncoder stands in for TRIX behind the same protocol
(``crest/model.py``'s docstring). Two properties are engineered in, because
tests lean on them:

* **Permutation equivariance**: every feature is a structural count
  (degrees, per-relation counts, query indicators), so relabelling entities
  relabels the feature rows and nothing else.
* **Exact arithmetic**: features and relation vectors are small integers, so
  sums are exact in float32 regardless of accumulation order, and the
  equivariance and identity tests can assert bitwise equality instead of
  allclose.
"""

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "shared")):
    if p not in sys.path:
        sys.path.insert(0, p)

from crest import bank as crest_bank  # noqa: E402
from crest import pfn  # noqa: E402

D = 32  # the TRIX hidden size; row_dim(D) == 98 is asserted in test_bank_shapes


def make_toy_graph(seed=7, num_nodes=12, num_direct=3, num_edges=28):
    """A random toy inference graph with inverses materialised, ULTRA-style."""
    generator = torch.Generator().manual_seed(seed)
    heads = torch.randint(num_nodes, (num_edges,), generator=generator)
    tails = torch.randint(num_nodes, (num_edges,), generator=generator)
    # no self-loops: they make "the edge and its inverse" degenerate
    same = heads == tails
    tails[same] = (tails[same] + 1) % num_nodes
    rels = torch.randint(num_direct, (num_edges,), generator=generator)
    edge_index = torch.stack([torch.cat([heads, tails]), torch.cat([tails, heads])])
    edge_type = torch.cat([rels, rels + num_direct])
    return crest_bank.Graph(edge_index=edge_index, edge_type=edge_type,
                            num_nodes=num_nodes, num_relations=2 * num_direct)


class ToyEncoder:
    """Deterministic, equivariant, integer-featured stand-in for TRIX."""

    def __init__(self, d=D):
        self.d = d

    def _z(self, r):
        return torch.tensor(
            [((int(r) * 31 + k * 7) % 5) - 2 for k in range(self.d)],
            dtype=torch.float32)

    def encode_single(self, graph, u, r):
        n, d = graph.num_nodes, self.d
        ei, et = graph.edge_index, graph.edge_type
        x = torch.zeros(n, d)
        x[:, 0] = torch.bincount(ei[0], minlength=n)
        x[:, 1] = torch.bincount(ei[1], minlength=n)
        for rid in range(min(graph.num_relations, d - 6)):
            x[:, 2 + rid] = torch.bincount(ei[0][et == rid], minlength=n)
        # query-dependent structure: edges out of u, edges of type r into v, u itself
        x[:, d - 3] = torch.bincount(ei[1][ei[0] == int(u)], minlength=n)
        x[:, d - 2] = torch.bincount(ei[1][et == int(r)], minlength=n)
        x[int(u), d - 1] = 1.0
        z = self._z(r)
        s0 = x @ z
        return x, z, s0

    def encode_relation_single(self, graph, u, v):
        num_direct = graph.num_relations // 2
        d = self.d
        ei, et = graph.edge_index, graph.edge_type
        w = torch.zeros(num_direct, d)
        for q in range(num_direct):
            w[q, 0] = int((et == q).sum())
            w[q, 1] = int(((ei[0] == int(u)) & (et == q)).sum())
            w[q, 2] = int(((ei[1] == int(v)) & (et == q)).sum())
            w[q, 3] = int(((ei[0] == int(u)) & (ei[1] == int(v)) & (et == q)).sum())
            for k in range(4, d):
                w[q, k] = ((q * 13 + k * 5) % 3) - 1
        c = torch.tensor([((k * 3) % 4) - 1 for k in range(d)], dtype=torch.float32)
        c[0] = float(torch.bincount(ei[0], minlength=graph.num_nodes)[int(u)])
        c[1] = float(torch.bincount(ei[1], minlength=graph.num_nodes)[int(v)])
        w_s0 = w @ c
        return w, c, w_s0


class RecordingEncoder(ToyEncoder):
    """ToyEncoder that snapshots every message graph it is asked to encode."""

    def __init__(self, d=D):
        super().__init__(d)
        self.calls = []

    def encode_single(self, graph, u, r):
        self.calls.append({
            "edge_index": graph.edge_index.clone(),
            "edge_type": graph.edge_type.clone(),
            "u": int(u), "r": int(r),
        })
        return super().encode_single(graph, u, r)

    def encode_relation_single(self, graph, u, v):
        self.calls.append({
            "edge_index": graph.edge_index.clone(),
            "edge_type": graph.edge_type.clone(),
            "u": int(u), "v": int(v),
        })
        return super().encode_relation_single(graph, u, v)


def make_readout(d=D, channel_dim=0, seed=3, zero=True):
    torch.manual_seed(seed)
    readout = pfn.Readout(crest_bank.row_dim(d) + channel_dim, width=32, depth=1, heads=2)
    if not zero:
        # QueryReader zero-initialises by design (the phase 0 identity), so a
        # live residual has to be asked for explicitly
        with torch.no_grad():
            gen = torch.Generator().manual_seed(seed)
            readout.reader.out.weight.copy_(
                torch.randn(readout.reader.out.weight.shape, generator=gen) * 0.1)
            readout.reader.out.bias.copy_(
                torch.randn(readout.reader.out.bias.shape, generator=gen) * 0.1)
    return readout


@pytest.fixture
def toy_graph():
    return make_toy_graph()


@pytest.fixture
def encoder():
    return ToyEncoder()
