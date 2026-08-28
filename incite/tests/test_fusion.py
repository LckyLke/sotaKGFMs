"""Fused vs unfused FactorizedRelationStep (incite/layers.py).

The fused forward performs the same sums in 2 pair_sum launches instead of
~12; per output slot the contributing pair list, its order, and the final
summation order are identical, so outputs and grads must agree at 1e-6
(bitwise on the CPU fallback). The checkpoint contract is stronger than a
key mapping: fusion touches no parameters, so an unfused state_dict loads
STRICTLY into the fused module -- the live phase-1 checkpoint stays usable.
"""

import pytest
import torch

from conftest import make_toy_graph

from incite.graphs import incidence_pairs
from incite.layers import FactorizedRelationStep

ATOL = 1e-6


def _random_inputs(graph, b=3, d=16, seed=5, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    num_relations = int(graph.num_relations)
    z = torch.randn(b, num_relations, d, generator=generator,
                    requires_grad=requires_grad)
    node_repr = torch.randn(b, graph.num_nodes, d, generator=generator,
                            requires_grad=requires_grad)
    boundary = torch.randn(b, num_relations, d, generator=generator,
                           requires_grad=requires_grad)
    return z, node_repr, boundary


@pytest.mark.parametrize("count_channel", [False, True])
def test_fused_loads_unfused_state_dict_and_matches(count_channel):
    """Construct unfused, save; load into a differently-initialized fused
    module STRICTLY; outputs agree at 1e-6 on random input."""
    graph = make_toy_graph()
    torch.manual_seed(2)
    unfused = FactorizedRelationStep(16, layer_norm=True,
                                     count_channel=count_channel, fused=False)
    state = unfused.state_dict()
    torch.manual_seed(77)  # different init -- the load must overwrite it
    fused = FactorizedRelationStep(16, layer_norm=True,
                                   count_channel=count_channel, fused=True)
    fused.load_state_dict(state)  # strict=True: exact key compatibility
    z, node_repr, boundary = _random_inputs(graph)
    pairs = incidence_pairs(graph)
    with torch.no_grad():
        out_u = unfused(z, node_repr, pairs, boundary)
        out_f = fused(z, node_repr, pairs, boundary)
    assert torch.allclose(out_f, out_u, atol=ATOL), (out_f - out_u).abs().max()


def test_state_dict_keys_are_identical():
    fused = FactorizedRelationStep(16, fused=True)
    unfused = FactorizedRelationStep(16, fused=False)
    assert set(fused.state_dict()) == set(unfused.state_dict())


@pytest.mark.parametrize("count_channel", [False, True])
def test_fused_grads_match_unfused(count_channel):
    """Same weights, same leaf inputs: input grads and parameter grads from
    the fused path equal the unfused path at 1e-6."""
    graph = make_toy_graph()
    pairs = incidence_pairs(graph)
    torch.manual_seed(2)
    step = FactorizedRelationStep(16, layer_norm=True,
                                  count_channel=count_channel)

    def run(fused):
        step.fused = fused
        step.zero_grad(set_to_none=True)
        z, node_repr, boundary = _random_inputs(graph, requires_grad=True)
        out = step(z, node_repr, pairs, boundary)
        out.square().sum().backward()
        param_grads = {n: p.grad.clone() for n, p in step.named_parameters()}
        return (out.detach(), z.grad.clone(), node_repr.grad.clone(),
                boundary.grad.clone(), param_grads)

    out_u, gz_u, gn_u, gb_u, pg_u = run(fused=False)
    out_f, gz_f, gn_f, gb_f, pg_f = run(fused=True)
    assert torch.allclose(out_f, out_u, atol=ATOL)
    assert torch.allclose(gz_f, gz_u, atol=ATOL)
    assert torch.allclose(gn_f, gn_u, atol=ATOL)
    assert torch.allclose(gb_f, gb_u, atol=ATOL)
    for name in pg_u:
        assert torch.allclose(pg_f[name], pg_u[name], atol=ATOL), name


def test_fused_is_the_default():
    """Models built anywhere in the tree (build_model included) run fused."""
    assert FactorizedRelationStep(16).fused is True
