"""The scenario-conditioned readout (incite/model.py::scenario_features,
INCITE(scenario=True), SC1, 2026-09-03), synth.start_step, and the negative
mask against duplicate negatives.

  (a) a freshly attached scenario head is EXACTLY the identity, and a plain
      checkpoint loads with only ``scenario_mlp.*`` fresh;
  (b) the features equal a hand count on a toy graph, in eval on the full
      graph and in training on the graph WITHOUT the removed query edges;
  (c) gradients reach the head and the trunk, and the head changes scores
      once its last layer is non-zero;
  (d) start_step: no synthetic step before it, the same coins after it;
  (e) a pool shorter than the request is padded and masked, never
      resampled; the masked loss equals the loss on the unpadded row.
"""

import math

import torch

from conftest import make_model, make_toy_graph

from incite import synth
from incite.model import remove_easy_edges, scenario_features
from incite.train import multi_positive_nll


def _batch(graph, h, r):
    v = graph.num_nodes
    return torch.tensor([[[h, t, r] for t in range(v)]])


def _manual(graph, h, r, cands):
    ei, et = graph.edge_index.tolist(), graph.edge_type.tolist()
    rows = []
    out_h = sum(1 for (s, d), rr in zip(zip(*ei), et) if rr == r and s == h)
    for t in cands:
        cnt = sum(1 for (s, d), rr in zip(zip(*ei), et) if rr == r and d == t)
        rows.append([float(cnt > 0), math.log1p(cnt), float(out_h > 0), math.log1p(out_h)])
    return torch.tensor(rows)


# ---------------------------------------------------------------------------
# (a) identity and warm start
# ---------------------------------------------------------------------------
def test_fresh_scenario_head_is_the_identity():
    g = make_toy_graph()
    plain = make_model(support=False, seed=3).eval()
    sc = make_model(support=False, seed=3, scenario=True).eval()
    missing, unexpected = sc.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing
    assert all(k.startswith("scenario_mlp.") for k in missing), missing
    batch = _batch(g, 0, 1)
    with torch.no_grad():
        assert torch.equal(plain(g, batch), sc(g, batch))
    plain.train()
    sc.train()
    assert torch.equal(plain(g, batch), sc(g, batch))


# ---------------------------------------------------------------------------
# (b) the features
# ---------------------------------------------------------------------------
def test_scenario_features_match_a_hand_count():
    g = make_toy_graph(num_edges=60)
    h, r = 0, 1
    cands = torch.arange(g.num_nodes).unsqueeze(0)
    feat = scenario_features(g, torch.tensor([h]), torch.tensor([r]), cands, g.num_nodes)
    assert feat.shape == (1, g.num_nodes, 4)
    assert torch.allclose(feat[0], _manual(g, h, r, range(g.num_nodes)), atol=1e-6)
    # inverse relation ids (tail form of a head query) count the inverse edges
    r_inv = r + g.num_relations // 2
    feat_inv = scenario_features(g, torch.tensor([h]), torch.tensor([r_inv]), cands, g.num_nodes)
    assert torch.allclose(feat_inv[0], _manual(g, h, r_inv, range(g.num_nodes)), atol=1e-6)
    # two rows with different relations are handled independently
    two = scenario_features(g, torch.tensor([h, 2]), torch.tensor([r, 0]),
                            torch.stack([cands[0], cands[0]]), g.num_nodes)
    assert torch.allclose(two[1], _manual(g, 2, 0, range(g.num_nodes)), atol=1e-6)


def test_training_features_ignore_the_removed_query_edges():
    """In training the positive's own edge is removed from the message
    graph; its answer-half count must drop by one there."""
    g = make_toy_graph(num_edges=60)
    ei, et = g.edge_index, g.edge_type
    nd = g.num_relations // 2
    # a direct edge whose target has at least two incoming edges of its type
    pick = None
    for e in range(ei.shape[1]):
        r, t, h = int(et[e]), int(ei[1, e]), int(ei[0, e])
        if r < nd and int(((et == r) & (ei[1] == t)).sum()) >= 2:
            pick = (h, r, t)
            break
    assert pick is not None
    h, r, t = pick
    batch = torch.tensor([[[h, t, r], [h, (t + 1) % g.num_nodes, r]]])
    h_i, t_i, r_i = batch.unbind(-1)
    msg = remove_easy_edges(g, h_i, t_i, r_i)
    full = scenario_features(g, torch.tensor([h]), torch.tensor([r]), t_i, g.num_nodes)
    removed = scenario_features(msg, torch.tensor([h]), torch.tensor([r]), t_i, g.num_nodes)
    assert float(full[0, 0, 1]) > float(removed[0, 0, 1])
    assert abs(math.expm1(float(full[0, 0, 1])) - math.expm1(float(removed[0, 0, 1])) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# (c) gradients and effect
# ---------------------------------------------------------------------------
def test_scenario_head_trains_and_changes_scores():
    g = make_toy_graph(num_edges=60)
    model = make_model(support=False, scenario=True)
    model.train()
    batch = torch.tensor([[[0, 3, 1], [0, 4, 1], [0, 5, 1]]])
    score = model(g, batch)
    score[:, 0].sum().backward()
    # the zero last layer blocks nothing upstream of it: its own weights
    # get a gradient, the trunk gets the task gradient through score_mlp
    grads = [p.grad for p in model.scenario_mlp.parameters() if p.grad is not None]
    assert grads and any(float(x.abs().sum()) > 0 for x in grads)
    torch.manual_seed(0)
    torch.nn.init.normal_(model.scenario_mlp[-1].weight, std=0.5)
    model.eval()
    plain = make_model(support=False, seed=3).eval()
    plain.load_state_dict({k: v for k, v in model.state_dict().items()
                           if not k.startswith("scenario_mlp.")}, strict=False)
    with torch.no_grad():
        assert not torch.equal(plain(g, batch), model(g, batch))


# ---------------------------------------------------------------------------
# (d) start_step
# ---------------------------------------------------------------------------
def test_start_step_gates_the_synthetic_coin():
    base = dict(synth.SYNTH_DEFAULTS, enabled=True, prior="rules", fraction=0.5)
    late = dict(base, start_step=100)
    assert not any(synth.is_synth_step(s, late) for s in range(1, 100))
    coins_base = [synth.is_synth_step(s, base) for s in range(100, 400)]
    coins_late = [synth.is_synth_step(s, late) for s in range(100, 400)]
    assert coins_base == coins_late and any(coins_late)
    scfg = synth.synth_config({"synth": dict(enabled=True, prior="rules", start_step=20001)})
    assert scfg["start_step"] == 20001


# ---------------------------------------------------------------------------
# (e) padded negatives
# ---------------------------------------------------------------------------
def test_short_pools_are_padded_and_masked_not_resampled():
    gen = torch.Generator().manual_seed(2048)
    insts = [synth.create_rules_instance(gen, 500, num_positive=1, hard_neg_frac=0.5)
             for _ in range(4)]
    short = 0
    for inst in insts:
        rows = inst.test_triplets.tolist()
        negs = [row[1] for row in rows[1:]]
        mask = inst.neg_mask.tolist()
        real = [n for n, m in zip(negs, mask) if m]
        assert len(set(real)) == len(real), "no duplicates among real negatives"
        assert mask == sorted(mask, reverse=True), "padding is a suffix"
        if not all(mask):
            short += 1
            assert all(n == negs[0] for n, m in zip(negs, mask) if not m)
    assert short >= 1, "500 negatives must exceed at least one pool"
    union, queries = synth.union_batch(insts, 4, gen)
    assert union.query_neg_mask.shape == (4, 500) and union.query_pos_mask.shape == (4, 1)
    # the masked loss equals the loss on the unpadded rows
    torch.manual_seed(1)
    pred = torch.randn(4, 501)
    pm = union.query_pos_mask
    nm = union.query_neg_mask
    masked = multi_positive_nll(pred, pm, 1.0, neg_mask=nm)
    manual = []
    for i in range(4):
        keep = torch.cat([torch.tensor([True]), nm[i]])
        row = pred[i][keep].unsqueeze(0)
        manual.append(multi_positive_nll(row, torch.tensor([[True]]), 1.0))
    assert torch.allclose(masked, torch.stack(manual).mean(), atol=1e-6)
    # temperature 0: uniform over the real negatives only
    masked0 = multi_positive_nll(pred, pm, 0.0, neg_mask=nm)
    manual0 = torch.stack([multi_positive_nll(
        pred[i][torch.cat([torch.tensor([True]), nm[i]])].unsqueeze(0),
        torch.tensor([[True]]), 0.0) for i in range(4)]).mean()
    assert torch.allclose(masked0, manual0, atol=1e-6)
    # the defaults still carry no masks at all
    gen = torch.Generator().manual_seed(2048)
    plain = [synth.create_rules_instance(gen, 1) for _ in range(3)]
    u, _q = synth.union_batch(plain, 3, gen)
    assert getattr(u, "query_neg_mask", None) is None
    assert getattr(u, "query_pos_mask", None) is None
