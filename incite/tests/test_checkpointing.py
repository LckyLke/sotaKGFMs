"""Per-round activation checkpointing (train.checkpoint_activations).

Gradient correctness is the whole contract: torch.utils.checkpoint with
use_reentrant=False recomputes each round during backward, so every grad
must equal the uncheckpointed run's. Two things could break that silently:

* RNG-sensitive ops inside a round -- the trunk holds none (no dropout;
  asserted below), so recompute replays the identical arithmetic;
* nondeterministic walk features -- walks are sampled ONCE before round 1
  from a fresh ``torch.Generator`` seeded with seed+offset per call, so they
  are round inputs, not recomputed state. The grad test still runs walks ON
  (they are off in phase 1) so the walk parameters' grads flow through the
  checkpointed rounds and are compared too, and a dedicated test checkpoints
  the walk module itself to prove the seeded sampler is recompute-safe.
"""

import torch
from torch.utils.checkpoint import checkpoint

from conftest import make_model, make_toy_graph

from incite import train as T
from incite.walks import WalkModule

ATOL = 1e-6


def _joint_loss_grads(model, graph, ckpt):
    """One joint (entity + relation) loss and every named grad, with fixed
    seeds for the two RNG consumers outside the model: positive sampling
    (explicit generator) and negative sampling (global RNG)."""
    model.checkpoint_activations = ckpt
    model.train()
    model.zero_grad(set_to_none=True)
    generator = torch.Generator().manual_seed(1024)
    triples = T.sample_positive_triples(graph, 4, generator)
    torch.manual_seed(11)  # negative_sampling draws from the global RNG
    loss = T.entity_loss_from_triples(model, graph, triples, num_negative=3,
                                      walk_offset=5)
    loss = loss + 0.5 * T.relation_loss_from_triples(model, graph, triples,
                                                     walk_offset=5)
    loss.backward()
    grads = {n: (None if p.grad is None else p.grad.clone())
             for n, p in model.named_parameters()}
    return loss.detach(), grads


def test_checkpointed_grads_match_uncheckpointed_walks_on():
    """Forward and every parameter grad agree at 1e-6 with checkpointing on
    vs off, walks ON, both trunk tasks (entity + relation) exercised."""
    graph = make_toy_graph()
    model = make_model(dim=16, rounds=2, walks=True, support=False)
    loss_off, grads_off = _joint_loss_grads(model, graph, ckpt=False)
    loss_on, grads_on = _joint_loss_grads(model, graph, ckpt=True)
    assert torch.allclose(loss_off, loss_on, atol=ATOL), (loss_off, loss_on)
    assert grads_off.keys() == grads_on.keys()
    compared = 0
    for name in grads_off:
        a, b = grads_off[name], grads_on[name]
        if a is None and b is None:  # heads unused by this loss, both runs
            continue
        assert a is not None and b is not None, name
        assert torch.allclose(a, b, atol=ATOL), (name, (a - b).abs().max())
        compared += 1
    assert compared > 0
    # the walk parameters received (equal) gradient THROUGH the checkpointed
    # rounds -- the recompute saw identical walk features
    walk_grads = [grads_on[n] for n in grads_on if n.startswith("walk_module.")]
    assert walk_grads and any(bool((g != 0).any()) for g in walk_grads
                              if g is not None)


def test_walk_module_is_recompute_deterministic():
    """checkpoint(walk_module) == walk_module: sampling draws from a fresh
    Generator seeded with seed+offset per call, so a recompute during
    backward replays identical walks; grads match at 1e-6."""
    graph = make_toy_graph()
    starts = torch.tensor([0, 3, 5])

    def run(use_ckpt):
        torch.manual_seed(4)
        module = WalkModule(dim=16, num_walks=8, walk_length=6, seed=1024)
        if use_ckpt:
            w_ent, w_rel = checkpoint(module, graph, starts, 2,
                                      use_reentrant=False)
        else:
            w_ent, w_rel = module(graph, starts, 2)
        (w_ent.sum() + w_rel.sum()).backward()
        return (w_ent.detach(), w_rel.detach(),
                {n: p.grad.clone() for n, p in module.named_parameters()})

    e_off, r_off, g_off = run(False)
    e_on, r_on, g_on = run(True)
    assert torch.allclose(e_off, e_on, atol=ATOL)
    assert torch.allclose(r_off, r_on, atol=ATOL)
    for name in g_off:
        assert torch.allclose(g_off[name], g_on[name], atol=ATOL), name


def test_trunk_has_no_dropout():
    """No dropout anywhere in the model, so checkpoint recompute cannot
    silently diverge on RNG (the GRU's functional dropout is 0 too)."""
    model = make_model(dim=16, rounds=2, walks=True, support=True)
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert model.walk_module.gru.dropout == 0


def test_checkpointing_defaults_off():
    assert make_model(dim=16, rounds=2).checkpoint_activations is False
