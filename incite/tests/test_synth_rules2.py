"""The MX2 generator-side fixes of the rules prior (incite/synth.py,
2026-09-03): the synthetic steps use what only the generator knows.

What these tests pin:
  (a) the default code path draws exactly what it drew for MX1 (regression
      pins computed from the pre-MX2 module at synth.seed 2048);
  (b) isolate_relations: every instance owns a disjoint relation block,
      queries and inverses included; off, the shared vocabulary is kept;
  (c) neg_per_pos_rules 64 with hard_neg_frac 0.5: every negative is
      type-consistent, participating, absent from the graph and NOT
      derivable (the chainer is re-run), and the hard share sits within
      two hops of the head whenever the neighborhood can supply it;
  (d) num_positive_rules 4: the masked slots are distinct derivable-but-
      absent tails of the query (h, r); padding repeats the positive and
      is masked out; negatives start after the slots;
  (e) multi_positive_nll reduces to self_adversarial_nll for one positive
      and ignores padding slots;
  (f) unseen_answer_share steers the query draw both ways;
  (g) the MX2 config parses, the union has the expected shape and a
      training step runs end to end on the toy model;
  (h) bad knob values do not pass silently.

All CPU.
"""

import os

import torch

from conftest import REPO, make_model

from incite import synth
from incite.train import multi_positive_nll, self_adversarial_nll

SEED = 2048

# (num_nodes, num_edges, edge_index.sum, edge_type.sum, test_triplets,
#  num_relations) of the first three instances at seed 2048, neg 1,
# computed with the module as it was for MX1.
PINS = [
    (791, 1187, 914532, 33827, [[382, 199, 27], [382, 670, 27]], 64),
    (266, 512, 140840, 9082, [[84, 204, 20], [84, 37, 20]], 34),
    (281, 614, 159754, 7486, [[87, 180, 11], [87, 139, 11]], 26),
]

V2 = dict(neg_per_pos=64, num_positive=4, hard_neg_frac=0.5)


def _facts(inst):
    return set(zip(inst.edge_index[0].tolist(), inst.edge_type.tolist(),
                   inst.edge_index[1].tolist()))


def _near(inst, h):
    """Nodes within two undirected hops of ``h`` on the compacted graph."""
    adj = {}
    for a, b in inst.edge_index.t().tolist():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    hop1 = adj.get(h, set())
    near = set(hop1)
    for x in hop1:
        near |= adj.get(x, set())
    near.discard(h)
    return near


def _v2_pool(n, seed=SEED, **over):
    kw = dict(V2)
    kw.update(over)
    gen = torch.Generator().manual_seed(seed)
    return [synth.create_rules_instance(gen, **kw) for _ in range(n)], gen


# ---------------------------------------------------------------------------
# (a) the MX1 draws
# ---------------------------------------------------------------------------
def test_default_path_reproduces_the_mx1_draws():
    gen = torch.Generator().manual_seed(SEED)
    for pin in PINS:
        inst = synth.create_rules_instance(gen, 1)
        got = (int(inst.num_nodes), int(inst.edge_index.shape[1]),
               int(inst.edge_index.sum()), int(inst.edge_type.sum()),
               inst.test_triplets.tolist(), int(inst.num_relations))
        assert got == pin, (got, pin)
        assert inst.pos_mask.tolist() == [True] and inst.num_positive == 1
    # the defaults hand no mask to the union, so synth_loss keeps the
    # single-positive loss
    gen = torch.Generator().manual_seed(SEED)
    insts = [synth.create_rules_instance(gen, 1) for _ in range(3)]
    union, queries = synth.union_batch(insts, 3, gen)
    assert getattr(union, "query_pos_mask", None) is None
    assert queries.shape == (3, 2, 3)


# ---------------------------------------------------------------------------
# (b) relation blocks
# ---------------------------------------------------------------------------
def test_isolated_relation_blocks():
    insts, gen = _v2_pool(5)
    # union_batch draws its pick order from the generator first; replay it
    state = gen.get_state()
    union, queries = synth.union_batch(insts, 5, gen, isolate_relations=True)
    g2 = torch.Generator()
    g2.set_state(state)
    order = [insts[i] for i in torch.randperm(5, generator=g2).tolist()]
    num_direct = int(union.num_relations) // 2
    assert num_direct == sum(int(i.num_relations) for i in insts)
    E = union.edge_index.shape[1] // 2
    et_direct, et_inv = union.edge_type[:E], union.edge_type[E:]
    assert torch.equal(et_inv, et_direct + num_direct)
    # consecutive edge slices carry consecutive, disjoint relation blocks
    lo, e0, n0 = 0, 0, 0
    for inst, row in zip(order, queries):
        R, ne = int(inst.num_relations), inst.edge_index.shape[1]
        block = et_direct[e0:e0 + ne]
        assert int(block.min()) >= lo and int(block.max()) < lo + R
        assert torch.equal(block - lo, inst.edge_type)
        assert torch.equal(union.edge_index[:, e0:e0 + ne] - n0, inst.edge_index)
        # queries carry the offset relation and the offset nodes
        assert torch.equal(row[:, 2] - lo, inst.test_triplets[:, 2])
        assert torch.equal(row[:, :2] - n0, inst.test_triplets[:, :2])
        e0 += ne
        lo += R
        n0 += int(inst.num_nodes)
    assert e0 == E and lo == num_direct and n0 == int(union.num_nodes)
    # off: the shared vocabulary of the petals convention
    g3 = torch.Generator()
    g3.set_state(state)
    union2, queries2 = synth.union_batch(insts, 5, g3)
    assert int(union2.num_relations) == 2 * max(int(i.num_relations)
                                                for i in insts)
    assert torch.equal(union2.edge_index, union.edge_index)
    assert torch.equal(queries2[:, :, :2], queries[:, :, :2])


# ---------------------------------------------------------------------------
# (c) certified negatives, many and hard
# ---------------------------------------------------------------------------
def test_certified_negatives_many_and_hard():
    insts, _ = _v2_pool(8)
    hard_possible, hard_found = 0, 0
    for inst in insts:
        P = inst.num_positive
        rows = inst.test_triplets.tolist()
        assert len(rows) == P + 64
        h, _t, r = rows[0]
        facts = _facts(inst)
        closure = synth.forward_chain(facts, inst.rules)
        et = inst.entity_type.tolist()
        tail_types = set(inst.tail_types[r])
        negs = [row[1] for row in rows[P:]]
        for row in rows[P:]:
            assert row[0] == h and row[2] == r
        for n in negs:
            assert n != h
            assert et[n] in tail_types
            assert (h, r, n) not in facts
            assert (h, r, n) not in closure
        pool = [e for e in range(int(inst.num_nodes))
                if et[e] in tail_types and e != h and (h, r, e) not in closure]
        if len(pool) >= 64:
            assert len(set(negs)) == 64, "no replacement when the pool is long"
        near = _near(inst, h)
        can = min(32, len([e for e in pool if e in near]))
        got = sum(1 for n in negs[:can] if n in near)
        assert got == can, (got, can)
        hard_possible += can
        hard_found += got
    assert hard_possible > 0 and hard_found == hard_possible


# ---------------------------------------------------------------------------
# (d) full-closure positives
# ---------------------------------------------------------------------------
def test_full_closure_positives_and_mask():
    insts, gen = _v2_pool(12)
    multi = 0
    for inst in insts:
        P = inst.num_positive
        assert P == 4 and inst.pos_mask.shape == (4,)
        rows = inst.test_triplets.tolist()
        h, t, r = rows[0]
        facts = _facts(inst)
        closure = synth.forward_chain(facts, inst.rules)
        mask = inst.pos_mask.tolist()
        assert mask[0] is True
        real = [row[1] for row, m in zip(rows[:P], mask) if m]
        assert len(set(real)) == len(real)
        for p in real:
            assert (h, r, p) in closure and (h, r, p) not in facts
        for row, m in zip(rows[:P], mask):
            assert row[0] == h and row[2] == r
            if not m:
                assert row[1] == t, "padding repeats the positive"
        # the mask is a prefix of Trues
        assert mask == sorted(mask, reverse=True)
        # every derivable-but-absent tail is either a slot or not a negative
        others = {tt for hh, rr, tt in closure - facts if hh == h and rr == r}
        assert set(real) <= others
        if len(others) >= 4:
            assert len(real) == 4
        else:
            assert len(real) == len(others)
        multi += int(len(real) > 1)
    assert multi > 0, "the pool must contain multi-positive queries"
    union, queries = synth.union_batch(insts, 12, gen, isolate_relations=True)
    assert union.query_pos_mask.shape == (12, 4)
    assert queries.shape == (12, 68, 3)


# ---------------------------------------------------------------------------
# (e) the loss
# ---------------------------------------------------------------------------
def test_multi_positive_loss_reduces_to_the_single_positive_loss():
    gen = torch.Generator().manual_seed(3)
    pred = torch.randn(5, 65, generator=gen)
    ones = torch.ones(5, 1, dtype=torch.bool)
    for temp in (1.0, 0.5, 0.0):
        a = self_adversarial_nll(pred, 64, temp)
        b = multi_positive_nll(pred, ones, temp)
        assert torch.allclose(a, b, atol=1e-6), (temp, float(a), float(b))
    # padding slots change nothing: [p, p, p | negs] with mask [1, 0, 0]
    padded = torch.cat([pred[:, :1].expand(5, 3), pred[:, 1:]], dim=1)
    mask = torch.tensor([[True, False, False]] * 5)
    c = multi_positive_nll(padded, mask, 1.0)
    assert torch.allclose(c, self_adversarial_nll(pred, 64, 1.0), atol=1e-6)
    # a second real positive with a low score raises the loss, a high one
    # lowers it
    two = torch.tensor([[True, True, False]] * 5)
    low = padded.clone()
    low[:, 1] = -5.0
    high = padded.clone()
    high[:, 1] = 5.0
    assert float(multi_positive_nll(low, two, 1.0)) > float(c)
    assert float(multi_positive_nll(high, two, 1.0)) < float(c) or \
        bool((pred[:, 0] > 5.0).all())
    # gradient flows to the real slots only
    x = padded.clone().requires_grad_(True)
    multi_positive_nll(x, mask, 1.0).backward()
    assert float(x.grad[:, 1:3].abs().sum()) == 0.0
    assert float(x.grad[:, 0].abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# (f) scenario targeting
# ---------------------------------------------------------------------------
def _unseen_share(insts):
    hits = 0
    for inst in insts:
        h, t, r = inst.test_triplets[0].tolist()
        has_in = any(int(rr) == r and int(tt) == t for rr, tt in zip(
            inst.edge_type, inst.edge_index[1]))
        hits += int(not has_in)
    return hits / len(insts)


def test_unseen_answer_share_targeting():
    all_unseen, _ = _v2_pool(16, seed=11, unseen_answer_share=1.0)
    none_unseen, _ = _v2_pool(16, seed=11, unseen_answer_share=0.0)
    natural, _ = _v2_pool(16, seed=11)
    assert _unseen_share(all_unseen) >= 0.9
    assert _unseen_share(none_unseen) <= 0.1
    nat = _unseen_share(natural)
    assert 0.15 <= nat <= 0.85, nat
    # the natural draw is the MX1 draw: the knob at -1 adds no coin
    a, _ = _v2_pool(4, seed=5)
    b, _ = _v2_pool(4, seed=5, unseen_answer_share=-1.0)
    for x, y in zip(a, b):
        assert torch.equal(x.test_triplets, y.test_triplets)


# ---------------------------------------------------------------------------
# (g) the MX2 config and a step end to end
# ---------------------------------------------------------------------------
def test_v2_config_parses_and_the_step_runs_end_to_end():
    yaml = __import__("yaml")
    path = os.path.join(REPO, "configs", "incite_phase1_4g_synth30_v2.yaml")
    scfg = synth.synth_config(yaml.safe_load(open(path)))
    assert scfg is not None and scfg["prior"] == "rules"
    assert scfg["neg_per_pos_rules"] == 64 and scfg["hard_neg_frac"] == 0.5
    assert scfg["num_positive_rules"] == 4 and scfg["isolate_relations"]
    assert scfg["unseen_answer_share"] < 0
    assert scfg["fraction"] == 0.3 and scfg["seed"] == 2048
    small = dict(scfg, instances_per_step=3, fraction=1.0)
    model = make_model(walks=False, support=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in (1, 2):
        optimizer.zero_grad()
        loss, k = synth.synth_step_loss(model, small, step, walk_offset=step)
        assert k == 3 and torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        a, _ = synth.synth_step_loss(model, small, 9, walk_offset=9)
        b, _ = synth.synth_step_loss(model, small, 9, walk_offset=9)
    assert float(a) == float(b), "a resumed run must rebuild the same step"
    # the union the step built: isolated blocks and the mask are in place
    gen = torch.Generator().manual_seed(small["seed"] + 9)
    insts = synth.generate_instances(small, gen, 3)
    union, queries = synth.union_batch(insts, 3, gen, isolate_relations=True)
    assert int(union.num_relations) == 2 * sum(int(i.num_relations)
                                               for i in insts)
    assert union.query_pos_mask.shape == (3, 4) and queries.shape == (3, 68, 3)


# ---------------------------------------------------------------------------
# (h) bad knobs
# ---------------------------------------------------------------------------
def test_synth_config_rejects_bad_knobs():
    for bad in ({"num_positive_rules": 0}, {"hard_neg_frac": 1.5},
                {"unseen_answer_share": 2.0}, {"neg_per_pos_rules": 0}):
        try:
            synth.synth_config({"synth": dict(enabled=True, prior="rules",
                                              **bad)})
        except AssertionError:
            pass
        else:
            raise AssertionError("%r must not pass" % bad)
    ok = synth.synth_config({"synth": dict(enabled=True, prior="rules",
                                           isolate_relations="yes")})
    assert ok["isolate_relations"] is True
