"""synth.step_offset (2026-09-03): a warm-started lever (steps 1..10000)
paired against a resumed continuation (steps 20001..30000) must see the
same coin sequence and the same instances at the same position. PG2 ran
without it and its comparison with MX1 carried data-order noise."""
import os

import torch
import yaml

from incite import synth

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cfg(**over):
    base = dict(enabled=True, prior="rules", fraction=0.3, seed=2048,
                instances_per_step=2)
    base.update(over)
    return synth.synth_config({"synth": base})


def test_the_offset_reproduces_the_resumed_coin_sequence():
    warm, resumed = _cfg(step_offset=20000), _cfg()
    coins_warm = [synth.is_synth_step(s, warm) for s in range(1, 3001)]
    coins_resumed = [synth.is_synth_step(s, resumed) for s in range(20001, 23001)]
    assert coins_warm == coins_resumed
    # and they are not the coins of the un-offset warm start
    plain = [synth.is_synth_step(s, _cfg()) for s in range(1, 3001)]
    assert plain != coins_warm
    assert 0.2 < sum(coins_warm) / len(coins_warm) < 0.4


def test_the_offset_shifts_start_step_and_the_instance_seed_alike():
    # start_step is compared with the offset step: a warm start with offset
    # 20000 and start_step 20001 mixes from its first step on
    late = _cfg(step_offset=20000, start_step=20001)
    assert any(synth.is_synth_step(s, late) for s in range(1, 200))
    assert not synth.is_synth_step(0, late)
    # the instances of warm step s equal the instances of resumed step
    # s + 20000: the same generator seed
    warm, resumed = _cfg(step_offset=20000), _cfg()
    for s in (1, 7, 9999):
        a = synth.generate_instances(
            warm, torch.Generator().manual_seed(warm["seed"] + s + 20000), 2)
        b = synth.generate_instances(
            resumed, torch.Generator().manual_seed(resumed["seed"] + s + 20000), 2)
        assert all(torch.equal(x.edge_index, y.edge_index) for x, y in zip(a, b))


def test_the_default_is_zero_and_negative_offsets_are_rejected():
    assert synth.SYNTH_DEFAULTS["step_offset"] == 0
    assert _cfg()["step_offset"] == 0
    try:
        _cfg(step_offset=-1)
    except AssertionError:
        pass
    else:
        raise AssertionError("a negative step_offset must not pass")


def test_the_warm_started_levers_carry_the_offset_and_mx1s_synth_block():
    """SC1 and RR2 warm-start from the 4-graph checkpoint (new parameters
    need --init_from); every other knob of their synth block is MX1's,
    except the documented lever knobs."""
    mx1 = synth.synth_config(yaml.safe_load(open(os.path.join(
        REPO, "configs", "incite_phase1_4g_synth30.yaml"))))
    assert mx1["step_offset"] == 0
    for name, levers in (("incite_phase1_4g_synth30_scenario.yaml", set()),
                         ("incite_phase1_4g_synth30_iso_rules.yaml",
                          {"isolate_relations", "rule_targets",
                           "rule_weight"})):  # rule_neg_per_pos 4 = default
        got = synth.synth_config(yaml.safe_load(open(os.path.join(
            REPO, "configs", name))))
        assert got["step_offset"] == 20000, name
        diff = {k for k in mx1 if k != "step_offset" and got[k] != mx1[k]}
        assert diff == levers, (name, diff)
