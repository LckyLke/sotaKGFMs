"""Rule recovery from relation states (incite/model.py::RuleHead, RR1,
2026-09-03) and the rule hypotheses the rules prior attaches.

  (a) targets: every positive is a rule of the instance that the observed
      facts evidence (its body pattern occurs at least twice); no negative
      is a rule of the system; counts follow rule_neg_per_pos; the draw is
      deterministic; the defaults attach nothing;
  (b) the union shifts the relation ids into each row's block and marks
      isolation;
  (c) the head scores every kind, the loss is finite, reaches the head and
      the trunk, and a few steps on a fixed batch lower it; rule_weight 0
      is the plain loss; a model without the head ignores the targets;
  (d) the RR1 config parses and a step runs end to end.
"""

import os

import torch

from conftest import REPO, make_model

from incite import synth
from incite.model import RuleHead, rule_recovery_loss


def _pool(n, seed=2048, **over):
    kw = dict(neg_per_pos=8, num_positive=2, hard_neg_frac=0.5,
              rule_targets=True, rule_neg_per_pos=4)
    kw.update(over)
    gen = torch.Generator().manual_seed(seed)
    return [synth.create_rules_instance(gen, **kw) for _ in range(n)], gen


def _facts(inst):
    return set(zip(inst.edge_index[0].tolist(), inst.edge_type.tolist(),
                   inst.edge_index[1].tolist()))


# ---------------------------------------------------------------------------
# (a) targets
# ---------------------------------------------------------------------------
def test_rule_targets_are_certain_and_evidenced():
    insts, _ = _pool(8)
    total_pos = 0
    for inst in insts:
        idx, lab = inst.rule_idx, inst.rule_label
        assert idx.shape[0] == lab.shape[0] and idx.shape[1] == 4
        keys = {}
        for kind, body, head, _c in inst.rules:
            key = synth._rule_key(kind, body, head)
            if key is not None:
                keys.setdefault(synth.RULE_KIND[kind], set()).add(key)
        facts = _facts(inst)
        by_rel = {}
        for h, r, t in facts:
            by_rel.setdefault(r, []).append((h, t))
        for r in by_rel:
            by_rel[r].sort()
        pos_by_kind = {k: 0 for k in range(4)}
        neg_by_kind = {k: 0 for k in range(4)}
        for row, y in zip(idx.tolist(), lab.tolist()):
            k, key = row[0], tuple(row[1:])
            if y == 1.0:
                assert key in keys.get(k, set()), (k, key)
                kind = [n for n, v in synth.RULE_KIND.items() if v == k][0]
                body = (key[0],) if k in (0, 1, 2) else (key[0], key[1])
                head = key[1] if k in (0, 1) else (key[0] if k == 2 else key[2])
                assert len(synth._rule_candidates(kind, body, head, by_rel, 2)) >= 2
                pos_by_kind[k] += 1
            else:
                assert key not in keys.get(k, set())
                if k in (0, 1):
                    assert key[0] != key[1]
                neg_by_kind[k] += 1
            for r in key:
                assert -1 <= r < int(inst.num_relations)
        for k in range(4):
            assert neg_by_kind[k] <= 4 * max(1, pos_by_kind[k])
            assert neg_by_kind[k] >= 1
        total_pos += sum(pos_by_kind.values())
    assert total_pos > 0
    # deterministic, and absent by default
    a, _ = _pool(2, seed=9)
    b, _ = _pool(2, seed=9)
    for x, y in zip(a, b):
        assert torch.equal(x.rule_idx, y.rule_idx) and torch.equal(x.rule_label, y.rule_label)
    plain, _ = _pool(1, seed=9, rule_targets=False)
    assert getattr(plain[0], "rule_idx", None) is None


# ---------------------------------------------------------------------------
# (b) the union
# ---------------------------------------------------------------------------
def test_union_shifts_rule_ids_into_the_row_block():
    insts, gen = _pool(4)
    state = gen.get_state()
    union, _q = synth.union_batch(insts, 4, gen, isolate_relations=True)
    g2 = torch.Generator()
    g2.set_state(state)
    order = [insts[i] for i in torch.randperm(4, generator=g2).tolist()]
    assert union.rule_isolated
    lo = {}
    acc = 0
    for i, inst in enumerate(order):
        lo[i] = acc
        acc += int(inst.num_relations)
    idx = union.rule_idx
    assert idx.shape[1] == 5 and idx.shape[0] == union.rule_label.shape[0]
    for row in idx.tolist():
        i = row[0]
        R = int(order[i].num_relations)
        for r in row[2:]:
            assert r == -1 or lo[i] <= r < lo[i] + R
    g3 = torch.Generator()
    g3.set_state(state)
    shared, _ = synth.union_batch(insts, 4, g3)
    assert not shared.rule_isolated


# ---------------------------------------------------------------------------
# (c) head and loss
# ---------------------------------------------------------------------------
def test_rule_head_scores_every_kind_and_the_loss_trains():
    insts, gen = _pool(3)
    union, queries = synth.union_batch(insts, 3, gen, isolate_relations=True)
    kinds = set(union.rule_idx[:, 1].tolist())
    assert kinds == {0, 1, 2, 3}, kinds
    model = make_model(support=False, rule_head=True)
    model.train()
    assert isinstance(model.rule_head, RuleHead)
    _s, _a, z = model(union, queries, return_states=True)
    logits = model.rule_head(z, union.rule_idx)
    assert logits.shape == (union.rule_idx.shape[0],) and torch.isfinite(logits).all()
    model.zero_grad()
    loss = synth.synth_loss(model, union, queries, rule_weight=1.0)
    loss.backward()
    head_grads = [p.grad for p in model.rule_head.parameters() if p.grad is not None]
    assert head_grads and any(float(g.abs().sum()) > 0 for g in head_grads)
    trunk = [p.grad for n, p in model.named_parameters()
             if not n.startswith("rule_head.") and p.grad is not None]
    assert any(float(g.abs().sum()) > 0 for g in trunk)
    plain = synth.synth_loss(model, union, queries, rule_weight=0.0)
    assert float(loss) != float(plain)
    # steps on the rule loss alone lower it
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = float(rule_recovery_loss(model.rule_head, z.detach(), union.rule_idx,
                                      union.rule_label))
    for _ in range(8):
        opt.zero_grad()
        _s, _a, z = model(union, queries, return_states=True)
        rl = rule_recovery_loss(model.rule_head, z, union.rule_idx, union.rule_label)
        rl.backward()
        opt.step()
    _s, _a, z = model(union, queries, return_states=True)
    after = float(rule_recovery_loss(model.rule_head, z, union.rule_idx, union.rule_label))
    assert after < before
    # without the head, the targets are ignored
    bare = make_model(support=False)
    a = synth.synth_loss(bare, union, queries, rule_weight=1.0)
    b = synth.synth_loss(bare, union, queries, rule_weight=0.0)
    assert torch.equal(a, b)
    # without isolation the loss refuses
    shared, q2 = synth.union_batch(insts, 3, torch.Generator().manual_seed(1))
    try:
        synth.synth_loss(model, shared, q2, rule_weight=1.0)
    except AssertionError:
        pass
    else:
        raise AssertionError("rule recovery on a shared vocabulary must not pass")


# ---------------------------------------------------------------------------
# (d) config and step
# ---------------------------------------------------------------------------
def test_rr1_config_parses_and_the_step_runs():
    yaml = __import__("yaml")
    path = os.path.join(REPO, "configs", "incite_phase1_4g_synth30_v2_rules.yaml")
    cfg = yaml.safe_load(open(path))
    assert cfg["model"]["rule_head"] is True
    scfg = synth.synth_config(cfg)
    assert scfg["rule_targets"] and scfg["rule_weight"] > 0 and scfg["isolate_relations"]
    small = dict(scfg, instances_per_step=2, fraction=1.0)
    model = make_model(support=False, rule_head=True)
    model.train()
    loss, k = synth.synth_step_loss(model, small, 3, walk_offset=3)
    assert k == 2 and torch.isfinite(loss)
    loss.backward()
