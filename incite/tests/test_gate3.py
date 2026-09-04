"""PG3 (2026-09-04): the gate that can close. A lower start bias keeps the
exact identity; the two-sided proof loss pushes the gate product of
non-proof pairs shut; the default path is PG2's, draw for draw."""
import torch

from conftest import make_model, make_toy_graph

from incite import synth
from incite.model import EdgeGate
from test_gate import _batch, _v2_pool


def test_bias_two_is_still_the_exact_identity():
    g = make_toy_graph()
    plain = make_model(support=False, seed=3).eval()
    gated = make_model(support=False, seed=3, gate=True, gate_bias=2.0).eval()
    gated.load_state_dict(plain.state_dict(), strict=False)
    batch = _batch(g, 0, 1)
    with torch.no_grad():
        assert torch.equal(plain(g, batch), gated(g, batch))
    x = torch.randn(2, g.num_nodes, plain.dim)
    z = torch.randn(2, g.num_relations, plain.dim)
    q = torch.randn(2, plain.dim)
    gn, gr = gated.gates[0](x, z, q)
    assert torch.allclose(gn, torch.sigmoid(torch.tensor(2.0)))
    sn, sr = gated.gates[0].scales(gn, gr)
    assert torch.equal(sn, torch.ones_like(sn)) and torch.equal(sr, torch.ones_like(sr))
    assert gated.gates[0].bias_init == 2.0 and EdgeGate.BIAS == 6.0
    # forty times the slope: the gradient of a sigmoid at 2 versus at 6
    s2, s6 = torch.sigmoid(torch.tensor(2.0)), torch.sigmoid(torch.tensor(6.0))
    assert (s2 * (1 - s2)) / (s6 * (1 - s6)) > 40


def test_negatives_are_within_instance_non_proof_edges_and_default_is_pg2():
    insts, gen = _v2_pool(3)
    gen_a = torch.Generator().manual_seed(int(gen.initial_seed()))
    gen_b = torch.Generator().manual_seed(int(gen.initial_seed()))
    plain_union, plain_q = synth.union_batch(insts, 3, gen_a, isolate_relations=True)
    union, queries = synth.union_batch(insts, 3, gen_b, isolate_relations=True,
                                       proof_neg_per_pos=2)
    # the default path is PG2's: no negatives, identical pairs and queries
    assert not hasattr(plain_union, "proof_neg_rows")
    assert torch.equal(plain_union.proof_rows, union.proof_rows)
    assert torch.equal(plain_union.proof_edges, union.proof_edges)
    assert torch.equal(plain_q, queries)
    nrows, nedges = union.proof_neg_rows, union.proof_neg_edges
    assert nrows.numel() and nrows.numel() == nedges.numel()
    E = union.edge_index.shape[1] // 2
    proof_set = set(zip(union.proof_rows.tolist(), union.proof_edges.tolist()))
    # every negative is a non-proof edge with its inverse copy at + E, and
    # it lies in its own row's instance: the instances are contiguous
    # disjoint node blocks, so the node ranges spanned by a row's proof and
    # negative edges together must not overlap another row's range
    for row, e in zip(nrows.tolist(), nedges.tolist()):
        assert (row, e) not in proof_set
    ranges = []
    for row in range(3):
        es = torch.cat([union.proof_edges[union.proof_rows == row],
                        nedges[nrows == row]])
        nodes = union.edge_index[:, es].flatten()
        ranges.append((int(nodes.min()), int(nodes.max())))
    ranges.sort()
    assert all(ranges[i][1] < ranges[i + 1][0] for i in range(2)), ranges
    direct = nedges[nedges < E]
    assert torch.equal(torch.sort(direct + E).values,
                       torch.sort(nedges[nedges >= E]).values)
    # at most proof_neg_per_pos per proof edge, per row
    for row in range(3):
        n_pos = int((union.proof_rows == row).sum()) // 2
        n_neg = int((nrows == row).sum()) // 2
        assert n_neg <= 2 * n_pos


def test_two_sided_loss_separates_proof_from_non_proof_pairs():
    insts, gen = _v2_pool(3)
    union, queries = synth.union_batch(insts, 3, gen, isolate_relations=True,
                                       proof_neg_per_pos=2)
    model = make_model(support=False, gate=True, gate_bias=2.0, rounds=4)
    model.train()
    proof = (union.proof_rows, union.proof_edges, union.proof_neg_rows,
             union.proof_neg_edges, 0.5)
    one_sided = (union.proof_rows, union.proof_edges)
    _s, aux2 = model(union, queries, proof=proof, return_aux=True)
    _s, aux1 = model(union, queries, proof=one_sided, return_aux=True)
    assert torch.isfinite(aux2) and float(aux2) > float(aux1) > 0

    def products(m, rows, edges):
        with torch.no_grad():
            g = union
            x = torch.zeros(3, g.num_nodes, m.dim)
        # read the last round's raw gates on the pairs through a forward hook
        got = {}

        def hook(mod, inp, out):
            got["gn"], got["gr"] = out
        h = m.gates[-1].register_forward_hook(hook)
        with torch.no_grad():
            m(union, queries)
        h.remove()
        src, typ = g.edge_index[0, edges], g.edge_type[edges]
        return (got["gn"][rows, src] * got["gr"][rows, typ]).mean()

    before_pos = products(model, union.proof_rows, union.proof_edges)
    before_neg = products(model, union.proof_neg_rows, union.proof_neg_edges)
    assert abs(float(before_pos) - float(before_neg)) < 1e-6  # the identity start
    # Adam on the gate parameters alone: the gate must use the node and
    # relation states to tell proof sources from the rest (a shared bias
    # cannot; plain SGD at a large step drives everything to one end)
    opt = torch.optim.Adam(model.gates.parameters(), lr=0.05)
    for _ in range(100):
        opt.zero_grad()
        _s, a = model(union, queries, proof=proof, return_aux=True)
        a.backward()
        opt.step()
    after_pos = products(model, union.proof_rows, union.proof_edges)
    after_neg = products(model, union.proof_neg_rows, union.proof_neg_edges)
    # the non-proof products fall well below the proof products
    assert float(after_neg) < float(before_neg) - 0.3
    assert float(after_pos) > float(after_neg) + 0.3
    # synth_loss wires the negatives through when the union carries them
    loss2 = synth.synth_loss(model, union, queries, proof_weight=1.0, proof_neg_weight=0.5)
    loss0 = synth.synth_loss(model, union, queries, proof_weight=1.0, proof_neg_weight=0.0)
    assert float(loss2) > float(loss0)


def test_pg3_configs_parse():
    import os
    import yaml
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in ("incite_phase1_4g_synth30_gate3.yaml", "incite_recipe_mx1_gate3.yaml"):
        cfg = yaml.safe_load(open(os.path.join(repo, "configs", name)))
        assert cfg["model"]["gate"] and float(cfg["model"]["gate_bias"]) == 2.0
        scfg = synth.synth_config(cfg)
        assert scfg["proof_weight"] == 0.02 and scfg["proof_neg_per_pos"] == 2
        assert scfg["proof_neg_weight"] == 0.5
    cfg = yaml.safe_load(open(os.path.join(repo, "configs", "incite_phase1_4g_gate3.yaml")))
    assert cfg["model"]["gate"] and float(cfg["model"]["gate_bias"]) == 2.0
    assert synth.synth_config(cfg) is None  # R0: the gate without the mix
