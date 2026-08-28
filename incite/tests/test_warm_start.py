"""Warm start (--init_from): a walks-on model inherits the phase-1 trunk.

The lever run loads a walks-off checkpoint with strict=False. The contract:
every checkpoint tensor lands in the new model (no unexpected keys), the
trunk tensors are equal after the load, and the only missing keys belong to
the newly enabled walk module.
"""
import torch

from conftest import make_model


def test_walks_on_model_inherits_walks_off_trunk():
    base = make_model(walks=False, support=False, seed=7)
    state = {"model": base.state_dict()}

    lever = make_model(walks=True, support=False, seed=8)
    missing, unexpected = lever.load_state_dict(state["model"], strict=False)

    assert not unexpected, unexpected
    assert missing, "walks-on model must add new tensors"
    assert all("walk" in k for k in missing), missing
    for k, v in state["model"].items():
        assert torch.equal(lever.state_dict()[k], v), k
