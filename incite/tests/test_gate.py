"""The proof-guided propagation gate (incite/model.py::EdgeGate, PG1,
2026-09-03) and the proof recording of the rules prior.

  (a) a freshly attached gate is EXACTLY the identity: a plain model and
      a gated model with the same trunk weights score bitwise equal, and
      a plain checkpoint loads into a gated model with only ``gates.*``
      fresh;
  (b) proof supports are sound: every proof edge is an observed edge and
      the rules re-derive every real positive from the proof edges alone;
      the union's proof pairs point at the right edges, inverses included;
  (c) the proof loss is finite, reaches the gate parameters and the trunk,
      and a few steps on it alone open the gates on the proof pairs;
      proof_weight 0 is the plain synthetic loss;
  (d) ``prune_frac`` at evaluation changes the scores once the gates are
      non-uniform, keeps the requested share of edges, and is ignored in
      training mode;
  (e) ``distmult_sum`` with an edge weight equals the hand-computed masked
      sum on the python path, and the kernel path agrees on a GPU.

All CPU except the last kernel check, which skips without CUDA.
"""

import pytest
import torch

from conftest import make_model, make_toy_graph

from incite import synth
from incite.layers import distmult_sum
from incite.model import EdgeGate


def _batch(graph, h, r):
    v = graph.num_nodes
    return torch.tensor([[[h, t, r] for t in range(v)]])


def _v2_pool(n, seed=2048):
    gen = torch.Generator().manual_seed(seed)
    insts = [synth.create_rules_instance(gen, 64, num_positive=4,
                                         hard_neg_frac=0.5) for _ in range(n)]
    return insts, gen


def _facts(inst):
    return set(zip(inst.edge_index[0].tolist(), inst.edge_type.tolist(),
                   inst.edge_index[1].tolist()))


# ---------------------------------------------------------------------------
# (a) identity at init, warm start
# ---------------------------------------------------------------------------
def test_fresh_gate_is_exactly_the_identity():
    g = make_toy_graph()
    plain = make_model(support=False, seed=3).eval()
    gated = make_model(support=False, seed=3, gate=True).eval()
    missing, unexpected = gated.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing
    assert all(k.startswith("gates.") for k in missing), missing
    batch = _batch(g, 0, 1)
    with torch.no_grad():
        a = plain(g, batch)
        b = gated(g, batch)
    assert torch.equal(a, b)
    # train mode too (edge removal, same code path)
    plain.train()
    gated.train()
    assert torch.equal(plain(g, batch), gated(g, batch))
    # the raw gates sit at sigmoid(BIAS), the scales at exactly 1
    x = torch.randn(2, g.num_nodes, plain.dim)
    z = torch.randn(2, g.num_relations, plain.dim)
    q = torch.randn(2, plain.dim)
    gn, gr = gated.gates[0](x, z, q)
    assert torch.allclose(gn, torch.sigmoid(torch.tensor(EdgeGate.BIAS)))
    sn, sr = gated.gates[0].scales(gn, gr)
    assert torch.equal(sn, torch.ones_like(sn)) and torch.equal(sr, torch.ones_like(sr))


# ---------------------------------------------------------------------------
# (b) proof supports
# ---------------------------------------------------------------------------
def test_proof_supports_are_sound_and_the_union_points_at_them():
    insts, gen = _v2_pool(6)
    nonempty = 0
    for inst in insts:
        facts = _facts(inst)
        graph = sorted(facts)
        proof = {graph[i] for i in inst.proof_pos.tolist()}
        assert proof <= facts
        rows = inst.test_triplets.tolist()
        h, _t, r = rows[0]
        real = [row[1] for row, m in zip(rows[:inst.num_positive],
                                        inst.pos_mask.tolist()) if m]
        closure = synth.forward_chain(proof, inst.rules)
        for tt in real:
            assert (h, r, tt) in closure, "proof edges must re-derive the positive"
        nonempty += int(len(proof) > 0)
    assert nonempty == len(insts)
    union, queries = synth.union_batch(insts, 6, gen, isolate_relations=True)
    E = union.edge_index.shape[1] // 2
    rows, edges = union.proof_rows.tolist(), union.proof_edges.tolist()
    assert len(rows) == len(edges) == 2 * sum(int(i.proof_pos.numel()) for i in insts)
    # every proof pair names an edge whose endpoints lie in the row's own
    # instance (node offset range) and whose inverse partner is listed too
    for row, e in zip(rows, edges):
        src, dst = int(union.edge_index[0, e]), int(union.edge_index[1, e])
        q_head = int(queries[row, 0, 0])
        # the query head and the edge endpoints belong to one component:
        # the union's components are contiguous id ranges, so both ends
        # must be within the instance's size of the head
        assert abs(src - q_head) < 2000 and abs(dst - q_head) < 2000
        partner = e + E if e < E else e - E
        assert partner in edges
    # direct and inverse copies come in equal numbers
    assert sum(1 for e in edges if e < E) == sum(1 for e in edges if e >= E)


def test_proof_support_expands_derived_premises_to_observed_edges():
    # r2 <- r1 (hier), r3 <- r2 . r0 (comp): (0, r3, 2) rests on the observed
    # (0, r1, 1) through the derived (0, r2, 1), plus the observed (1, r0, 2)
    rules = [("hier", (1,), 2, 1.0), ("comp", (2, 0), 3, 1.0)]
    observed = {(0, 1, 1), (1, 0, 2)}
    deriv = {}
    closure = synth.forward_chain(observed, rules, derivations=deriv)
    assert (0, 3, 2) in closure and (0, 2, 1) in closure
    assert synth.proof_support((0, 3, 2), deriv, observed) == observed
    assert synth.proof_support((0, 2, 1), deriv, observed) == {(0, 1, 1)}
    assert synth.proof_support((0, 1, 1), deriv, observed) == {(0, 1, 1)}
    # the closure with recording equals the closure without it
    assert synth.forward_chain(observed, rules) == closure


# ---------------------------------------------------------------------------
# (c) the proof loss
# ---------------------------------------------------------------------------
def test_proof_loss_opens_the_gates_and_reaches_the_trunk():
    insts, gen = _v2_pool(3)
    union, queries = synth.union_batch(insts, 3, gen, isolate_relations=True)
    model = make_model(support=False, gate=True)
    model.train()
    proof = (union.proof_rows, union.proof_edges)
    _score, aux = model(union, queries, proof=proof, return_aux=True)
    assert aux.shape == () and torch.isfinite(aux) and float(aux) > 0
    model.zero_grad()
    loss = synth.synth_loss(model, union, queries, proof_weight=1.0)
    loss.backward()
    gate_grads = [p.grad for p in model.gates.parameters() if p.grad is not None]
    assert gate_grads and any(float(g.abs().sum()) > 0 for g in gate_grads)
    trunk = [p.grad for n, p in model.named_parameters()
             if not n.startswith("gates.") and p.grad is not None]
    assert any(float(g.abs().sum()) > 0 for g in trunk)
    # proof_weight 0: the plain synthetic loss of the same model
    plain = synth.synth_loss(model, union, queries, proof_weight=0.0)
    with_aux = synth.synth_loss(model, union, queries, proof_weight=1.0)
    assert float(with_aux) > float(plain)
    # a few steps on the aux term alone raise the raw gates on proof pairs
    opt = torch.optim.SGD(model.gates.parameters(), lr=1.0)
    before = float(aux)
    for _ in range(5):
        opt.zero_grad()
        _s, a = model(union, queries, proof=proof, return_aux=True)
        a.backward()
        opt.step()
    _s, after = model(union, queries, proof=proof, return_aux=True)
    assert float(after) < before


def test_forward_without_gates_ignores_proof_and_returns_zero_aux():
    insts, gen = _v2_pool(2)
    union, queries = synth.union_batch(insts, 2, gen, isolate_relations=True)
    model = make_model(support=False).eval()
    with torch.no_grad():
        s = model(union, queries)
        s2, aux = model(union, queries, proof=(union.proof_rows, union.proof_edges),
                        return_aux=True)
    assert torch.equal(s, s2) and float(aux) == 0.0
    # and a plain model gets the plain loss even at proof_weight 1
    a = synth.synth_loss(model, union, queries, proof_weight=1.0)
    b = synth.synth_loss(model, union, queries, proof_weight=0.0)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# (d) pruning at evaluation
# ---------------------------------------------------------------------------
def test_prune_frac_drops_edges_at_eval_only():
    g = make_toy_graph(num_edges=60)
    # four rounds: with two, the only message round can be shut entirely
    # by random gates and both scores collapse to the same constant
    model = make_model(support=False, gate=True, rounds=4)
    torch.manual_seed(11)
    for gate in model.gates:                     # non-uniform gates
        for lin in (gate.node, gate.rel, gate.node_q, gate.rel_q):
            torch.nn.init.normal_(lin.weight, std=0.5)
    model.eval()
    batch = _batch(g, 0, 1)
    with torch.no_grad():
        full = model(g, batch)
        model.prune_frac = 0.5
        half = model(g, batch)
        model.prune_frac = 0.0
        again = model(g, batch)
    assert torch.equal(full, again)
    assert not torch.equal(full, half)
    # the mask keeps the requested share: instrument one round by hand
    x = torch.randn(1, g.num_nodes, model.dim)
    z = torch.randn(1, g.num_relations, model.dim)
    q = torch.randn(1, model.dim)
    gn, gr = model.gates[0](x, z, q)
    prod = gn[:, g.edge_index[0]] * gr[:, g.edge_type]
    E = prod.shape[1]
    keep = max(1, int(round(0.5 * E)))
    thr = prod.kthvalue(E - keep + 1, dim=1, keepdim=True).values
    kept = int((prod >= thr).sum())
    assert keep <= kept <= keep + 2
    # training mode never prunes
    model.train()
    model.prune_frac = 0.5
    a = model(g, batch)
    model.prune_frac = 0.0
    b = model(g, batch)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# (e) the weighted message sum
# ---------------------------------------------------------------------------
def _manual(x, rel, edge_index, edge_type, boundary, w):
    out = boundary.clone()
    for e in range(edge_index.shape[1]):
        s, d, r = int(edge_index[0, e]), int(edge_index[1, e]), int(edge_type[e])
        for b in range(x.shape[0]):
            out[b, d] += float(w[b, e]) * rel[b, r] * x[b, s]
    return out


def test_weighted_distmult_matches_the_manual_sum():
    g = make_toy_graph(num_edges=40)
    torch.manual_seed(2)
    b, d = 3, 8
    x = torch.randn(b, g.num_nodes, d)
    rel = torch.randn(b, g.num_relations, d)
    boundary = torch.randn(b, g.num_nodes, d)
    w = (torch.rand(b, g.edge_index.shape[1]) > 0.5).float()
    got = distmult_sum(x, rel, g.edge_index, g.edge_type, boundary, w)
    want = _manual(x, rel, g.edge_index, g.edge_type, boundary, w)
    assert torch.allclose(got, want, atol=1e-5)
    ones = torch.ones(b, g.edge_index.shape[1])
    assert torch.allclose(distmult_sum(x, rel, g.edge_index, g.edge_type, boundary, ones),
                          distmult_sum(x, rel, g.edge_index, g.edge_type, boundary))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel path needs CUDA")
def test_weighted_distmult_kernel_matches_python_path():
    from incite.layers import _rspmm
    if _rspmm() is None:
        pytest.skip("no rspmm extension")
    g = make_toy_graph(num_edges=40)
    torch.manual_seed(4)
    b, d = 3, 8
    x = torch.randn(b, g.num_nodes, d)
    rel = torch.randn(b, g.num_relations, d)
    boundary = torch.randn(b, g.num_nodes, d)
    w = (torch.rand(b, g.edge_index.shape[1]) > 0.5).float()
    cpu = distmult_sum(x, rel, g.edge_index, g.edge_type, boundary, w)
    dev = torch.device("cuda")
    gpu = distmult_sum(x.to(dev), rel.to(dev), g.edge_index.to(dev),
                       g.edge_type.to(dev), boundary.to(dev), w.to(dev))
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-4)
    plain_cpu = distmult_sum(x, rel, g.edge_index, g.edge_type, boundary)
    plain_gpu = distmult_sum(x.to(dev), rel.to(dev), g.edge_index.to(dev),
                             g.edge_type.to(dev), boundary.to(dev))
    assert torch.allclose(plain_cpu, plain_gpu.cpu(), atol=1e-4)
