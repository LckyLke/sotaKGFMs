"""The rules prior (synth.prior == "rules", incite/synth.py).

What these tests pin, per the synthetic-prior sweep's requirements:
  (a) rules instances are well-formed: typed (every edge respects the
      relation signatures), ids in range, the query edge absent from the
      graph, the positive derivable and the negatives NOT derivable --
      verified by re-running the chainer;
  (b) determinism: the pool is a pure function of the generator seed;
  (c) rule diversity: across 20 instances every rule family appears;
  (d) gradient flows through synth_loss on a rules union batch, walks on;
  (e) the sweep configs parse and dispatch to the rules family, and the
      phase-2.1b config keeps dispatching to petals (the live queue);
  (f) forward-chaining respects the iteration cap and the confidences
      (statistical sanity, seeded).

All CPU. Instance sizes are the real ranges (hundreds of nodes); the model
forwards keep k small so the suite stays fast.
"""

import os

import torch

from conftest import REPO, make_model

from incite import synth
from incite.model import remove_easy_edges

SEED = 2048


def _rules_pool(n, seed=SEED, neg=1):
    gen = torch.Generator().manual_seed(seed)
    return [synth.create_rules_instance(gen, neg) for _ in range(n)], gen


def _graph_facts(inst):
    return set(zip(inst.edge_index[0].tolist(), inst.edge_type.tolist(),
                   inst.edge_index[1].tolist()))


# ---------------------------------------------------------------------------
# (a) well-formedness
# ---------------------------------------------------------------------------
def test_rules_instances_are_wellformed():
    instances, _ = _rules_pool(6)
    for inst in instances:
        assert getattr(inst, "family", None) == "rules"
        assert int(inst.edge_index.min()) >= 0
        assert int(inst.edge_index.max()) < int(inst.num_nodes)
        assert inst.edge_index.shape[1] == inst.edge_type.shape[0]
        assert int(inst.edge_type.min()) >= 0
        assert int(inst.edge_type.max()) < int(inst.num_relations)
        assert 8 <= int(inst.num_relations) <= 64
        assert inst.entity_type.shape[0] == int(inst.num_nodes)
        # every entity participates: compaction leaves no isolated ids
        used = set(inst.edge_index.reshape(-1).tolist())
        assert used == set(range(int(inst.num_nodes)))
        # every edge respects the (propagated) type signatures
        et = inst.entity_type.tolist()
        for (h, t), r in zip(inst.edge_index.t().tolist(),
                             inst.edge_type.tolist()):
            assert et[h] in inst.head_types[r], (h, r)
            assert et[t] in inst.tail_types[r], (t, r)
        # the query block: one positive plus negatives, shared head/relation
        assert inst.test_triplets.shape == (2, 3)
        rows = inst.test_triplets.tolist()
        h, t_true, r = rows[0]
        assert all(row[0] == h and row[2] == r for row in rows)
        assert len({row[1] for row in rows}) == len(rows)
        assert 0 <= r < int(inst.num_relations)
        # bookkeeping the design note reports
        assert inst.num_base >= 8 and inst.num_query_pool >= 1
        assert inst.num_dropped >= 0 and inst.num_derived_kept >= 0


def test_rules_labels_certain_and_negatives_underivable():
    """Re-run the bounded chainer on the observed graph: the positive must
    be derived (its label is certain), the query edge must be absent, and
    no negative may be derivable."""
    instances, _ = _rules_pool(6)
    for inst in instances:
        facts = _graph_facts(inst)
        closure = synth.forward_chain(facts, inst.rules)
        rows = inst.test_triplets.tolist()
        h, t_true, r = rows[0]
        assert (h, r, t_true) in closure
        assert (h, r, t_true) not in facts
        for _h, t_neg, _r in rows[1:]:
            assert (h, r, t_neg) not in closure
            assert (h, r, t_neg) not in facts


# ---------------------------------------------------------------------------
# (b) determinism
# ---------------------------------------------------------------------------
def test_rules_instances_are_deterministic():
    a, _ = _rules_pool(3)
    b, _ = _rules_pool(3)
    for x, y in zip(a, b):
        assert torch.equal(x.edge_index, y.edge_index)
        assert torch.equal(x.edge_type, y.edge_type)
        assert torch.equal(x.test_triplets, y.test_triplets)
        assert torch.equal(x.entity_type, y.entity_type)
        assert x.rules == y.rules
        assert int(x.num_nodes) == int(y.num_nodes)
    c, _ = _rules_pool(1, seed=SEED + 1)
    x, y = a[0], c[0]
    assert not (x.edge_index.shape == y.edge_index.shape
                and torch.equal(x.edge_index, y.edge_index)
                and torch.equal(x.test_triplets, y.test_triplets))


# ---------------------------------------------------------------------------
# (c) rule diversity
# ---------------------------------------------------------------------------
def test_rule_families_all_appear_across_instances():
    gen = torch.Generator().manual_seed(7)
    kinds, comp_lengths = set(), set()
    for _ in range(20):
        inst = synth.create_rules_instance(gen)
        for kind, body, head, conf in inst.rules:
            kinds.add(kind)
            assert 0.6 <= conf <= 0.95
            if kind == "comp":
                comp_lengths.add(len(body))
                assert 2 <= len(body) <= 3
            elif kind == "sym":
                assert body == (head,)
            else:
                assert len(body) == 1 and body[0] != head
    assert kinds == {"comp", "hier", "inv", "sym"}
    assert comp_lengths == {2, 3}, "both chain lengths must occur"


# ---------------------------------------------------------------------------
# (f) the chainer: iteration cap and confidences
# ---------------------------------------------------------------------------
def test_forward_chain_respects_the_iteration_cap():
    """A hierarchy cascade r0 -> r1 -> ... needs one iteration per hop:
    with the cap at 3 (the prior's setting), depth 3 is reached and depth 4
    is not."""
    rules = [("hier", (i,), i + 1, 1.0) for i in range(5)]
    closure = synth.forward_chain({(0, 0, 1)}, rules, max_iters=3)
    assert (0, 3, 1) in closure
    assert (0, 4, 1) not in closure
    deeper = synth.forward_chain({(0, 0, 1)}, rules, max_iters=4)
    assert (0, 4, 1) in deeper


def test_forward_chain_confidence_statistics():
    """Each candidate derivation is decided once at the rule's confidence,
    so the firing rate over many independent candidates is binomial around
    it (500 trials at 0.7: five sigma is under 0.06)."""
    facts = {(2 * i, 0, 2 * i + 1) for i in range(500)}
    rules = [("hier", (0,), 1, 0.7)]
    gen = torch.Generator().manual_seed(SEED)
    noisy = synth.forward_chain(facts, rules, gen)
    fired = sum(1 for h, r, t in noisy if r == 1)
    assert 300 <= fired <= 400, fired
    # confidence 1.0 with a generator equals the deterministic closure
    sure = [("hier", (0,), 1, 1.0)]
    gen = torch.Generator().manual_seed(SEED)
    assert synth.forward_chain(facts, sure, gen) == synth.forward_chain(
        facts, sure)
    # and the noisy closure is a pure function of the seed
    a = synth.forward_chain(facts, rules,
                            torch.Generator().manual_seed(SEED))
    b = synth.forward_chain(facts, rules,
                            torch.Generator().manual_seed(SEED))
    assert a == b


# ---------------------------------------------------------------------------
# the union: budget, interface, remove_easy_edges
# ---------------------------------------------------------------------------
def test_rules_union_of_16_stays_under_the_edge_budget():
    instances, gen = _rules_pool(16)
    union, queries = synth.union_batch(instances, 16, gen)
    assert union.edge_index.shape[1] < 50000, "training-friendliness budget"
    assert queries.shape == (16, 2, 3)
    assert int(union.num_relations) == 2 * max(int(i.num_relations)
                                               for i in instances)
    assert int(union.num_nodes) == sum(int(i.num_nodes) for i in instances)


def test_rules_query_edges_absent_and_removal_is_a_noop():
    instances, gen = _rules_pool(4)
    union, queries = synth.union_batch(instances, 4, gen)
    edges = set(zip(union.edge_index[0].tolist(), union.edge_type.tolist(),
                    union.edge_index[1].tolist()))
    num_direct = int(union.num_relations) // 2
    for h, t, r in queries.reshape(-1, 3).tolist():
        assert (h, r, t) not in edges
        assert (t, r + num_direct, h) not in edges
    h_index, t_index, r_index = queries.unbind(-1)
    msg = remove_easy_edges(union, h_index, t_index, r_index)
    assert torch.equal(msg.edge_index, union.edge_index)
    assert torch.equal(msg.edge_type, union.edge_type)


# ---------------------------------------------------------------------------
# (d) gradient flow, walks on
# ---------------------------------------------------------------------------
def test_rules_synth_loss_gradient_reaches_the_walk_module():
    instances, gen = _rules_pool(4)
    union, queries = synth.union_batch(instances, 4, gen)
    model = make_model(walks=True, support=False)
    model.train()
    model.zero_grad()
    loss = synth.synth_loss(model, union, queries)
    assert loss.shape == () and torch.isfinite(loss)
    loss.backward()
    walk_grads = [p.grad for p in model.walk_module.parameters()
                  if p.grad is not None]
    assert walk_grads
    assert any(float(g.abs().sum()) > 0 for g in walk_grads), \
        "walk_module gradients are all exactly zero on the rules prior"
    trunk = [p for n, p in model.named_parameters()
             if not n.startswith("walk_module") and p.grad is not None]
    assert any(float(p.grad.abs().sum()) > 0 for p in trunk)


# ---------------------------------------------------------------------------
# neg_per_pos_rules: the [positive | negatives] layout
# ---------------------------------------------------------------------------
def test_rules_neg_per_pos_widens_the_query_block():
    instances, gen = _rules_pool(3, neg=3)
    for inst in instances:
        assert inst.test_triplets.shape == (4, 3)
    union, queries = synth.union_batch(instances, 3, gen)
    assert queries.shape == (3, 4, 3)

    class Fixed(torch.nn.Module):
        def __init__(self, margin):
            super().__init__()
            self.margin = margin

        def forward(self, data, batch, support=None, walk_offset=0):
            out = torch.zeros(batch.shape[0], batch.shape[1])
            out[:, 0] = self.margin
            return out

    good = synth.synth_loss(Fixed(2.0), union, queries)
    bad = synth.synth_loss(Fixed(-2.0), union, queries)
    assert float(good) < float(bad)
    # temperature 0 exercises the uniform 1/num_negative branch, which is
    # where the negative count wiring actually matters
    uniform = synth.synth_loss(Fixed(2.0), union, queries,
                               adversarial_temperature=0.0)
    assert torch.isfinite(uniform)


def test_rules_generate_instances_reads_the_config_knob():
    cfg = dict(synth.SYNTH_DEFAULTS, enabled=True, prior="rules",
               neg_per_pos_rules=2)
    gen = torch.Generator().manual_seed(SEED)
    out = synth.generate_instances(cfg, gen, 2)
    assert len(out) == 2
    for inst in out:
        assert inst.test_triplets.shape == (3, 3)


# ---------------------------------------------------------------------------
# (e) config gating: the sweep dispatches to rules, phase 2.1b to petals
# ---------------------------------------------------------------------------
def test_sweep_configs_parse_and_dispatch_to_the_rules_family():
    yaml = __import__("yaml")
    fractions = {}
    for point in (25, 75, 100):
        path = os.path.join(REPO, "configs",
                            "incite_synthsweep_%d.yaml" % point)
        cfg = yaml.safe_load(open(path))
        scfg = synth.synth_config(cfg)
        assert scfg is not None, path
        assert scfg["prior"] == "rules"
        assert scfg["seed"] == 2048 and scfg["instances_per_step"] == 16
        assert scfg["neg_per_pos_rules"] == 1          # the default
        fractions[point] = scfg["fraction"]
        gen = torch.Generator().manual_seed(scfg["seed"])
        inst = synth.generate_instances(scfg, gen, 1)[0]
        assert getattr(inst, "family", None) == "rules"
    assert fractions == {25: 0.25, 75: 0.75, 100: 1.0}
    # the live phase-2.1b queue keeps the petals family
    cfg = yaml.safe_load(
        open(os.path.join(REPO, "configs", "incite_phase21b_walksynth.yaml")))
    scfg = synth.synth_config(cfg)
    assert scfg["prior"] == "petals"
    gen = torch.Generator().manual_seed(scfg["seed"])
    inst = synth.generate_instances(scfg, gen, 1)[0]
    assert getattr(inst, "family", "petals") == "petals"
    # an unknown family name must not pass silently
    try:
        synth.synth_config({"synth": {"enabled": True, "prior": "petunias"}})
    except AssertionError as exc:
        assert "petunias" in str(exc)
    else:
        raise AssertionError("bad synth.prior must not pass")


# ---------------------------------------------------------------------------
# the step function end to end on the rules prior
# ---------------------------------------------------------------------------
def test_rules_synth_step_loss_end_to_end():
    scfg = dict(synth.SYNTH_DEFAULTS, enabled=True, prior="rules",
                fraction=1.0, instances_per_step=2)
    model = make_model(walks=True, support=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.walk_module.gru.weight_ih_l0.detach().clone()
    for step in range(1, 3):
        assert synth.is_synth_step(step, scfg)
        optimizer.zero_grad()
        loss, k = synth.synth_step_loss(model, scfg, step, walk_offset=step)
        assert k == 2 and torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, model.walk_module.gru.weight_ih_l0)

    model.eval()
    with torch.no_grad():
        a, _ = synth.synth_step_loss(model, scfg, 17, walk_offset=17)
        b, _ = synth.synth_step_loss(model, scfg, 17, walk_offset=17)
        c, _ = synth.synth_step_loss(model, scfg, 18, walk_offset=18)
    assert float(a) == float(b), "a resumed run must rebuild the same step"
    assert float(a) != float(c)
