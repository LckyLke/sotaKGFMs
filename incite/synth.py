"""Synthetic automorphic-instance supervision (phase 2.1b: the walks revival).

Why this module exists
------------------------------------------------------------------------------
results/incite/PHASE21_RESULT.md: the walks lever injected the CAPABILITY to
break PETALS ties (0/220 exact ties, mean margin 0.26) but broke them at
chance accuracy (0.47 against a 0.90 bar). Diagnosis: FB15k237 / WN18RR /
CoDExMedium contain no automorphic candidate pairs, so nothing in the
pretraining mix ever told the GRU WHICH way to break a tie. The walk pathway
received capability but no aligning gradient.

This module manufactures that gradient: small PETALS-family instances with a
LABELED true tail, mixed into pretraining as a small fraction of steps.

The structural family here overlaps the PETALS diagnostic BY DESIGN. This is
not a held-out generalization claim and must never be reported as one: the
experiment asks only whether symmetry-breaking supervision can align the
walk module's tie-breaking with truth AT ALL. A PETALS score after training
on this family is a train-distribution score; the honest reading is
"supervision aligns / does not align the pathway", and the transfer question
(DEV10, the 41-graph benchmark) is what carries the claim. Instances are
seeded from ``synth.seed`` (default 2048), deliberately different from the
eval set's 1024, so the exact instances differ even though the family does not.

The instance family
------------------------------------------------------------------------------
Restated from diagnostics/generate_petals.py::create_instance (the FLOCK
generator convention), with the fixed config list replaced by sampling:

  * a hub (node 0) carrying ``n_petals`` petals, 2..6;
  * each petal is a two-branch cycle of ``cycle_size`` 2..6 hops per branch
    meeting at an apex; branch ODD is entirely colour ``a``, branch EVEN
    starts with colour ``b != a`` and continues in ``a``;
  * a stem of ``tail_len`` 1..4 edges of relation 0 hanging off the hub; the
    query head is the hub or a stem node;
  * the query is (head, relation 0, ?) with the two candidates at equal depth
    in one petal: the odd-branch node (TRUE, first row, generator convention)
    and the even-branch node (FALSE).

Colourings keep the symmetric structure of the diagnostic's configs -- the
cyclic family ``[[c1,c2],[c2,c3],...,[cm,c1]]``, the complete family (all
ordered pairs of a colour set) and the swapped-pair family
``[[c1,c2],[c2,c1],...]`` -- which is what makes the two candidates collapse
for a relation-permutation-equivariant invariant, and colours are always
>= 1 so relation 0 stays the stem/query relation and the QUERY EDGE IS NOT
IN THE GRAPH.

remove_easy_edges on synthetic queries
------------------------------------------------------------------------------
``INCITE.forward`` runs ``remove_easy_edges`` in training mode. The synthetic
query edges are absent from the graph by construction (relation 0 only ever
runs hub->stem; candidates are petal nodes), and the removal helper is
no-op-safe for absent edges: ``_edge_match`` returns zero matches, the mask
stays all-True and the returned graph carries the same edges. Hashing is
exact here -- after inverse augmentation every node appears as both source
and target, so query node ids never exceed the per-row maxima the hash radix
is built from, and no spurious match is possible. So the synthetic loss calls
``model(union, batch)`` exactly as ``entity_loss_from_triples`` does, in
training mode, with no bypass and no restructuring. A test asserts both facts
(query edges absent; removal is a value no-op).

The union batch and its one approximation
------------------------------------------------------------------------------
``k`` instances are scored in ONE forward over their disjoint union. Entity
message passing cannot cross components, so each query's entity states are
exactly what it would see alone. The RELATION states are not: the relation
vocabulary is shared across the union (that is what makes one call possible)
and the factorized relation step sums node features over every incidence pair
in the union, so instances contribute a bulk term to each other's relation
states. That is a deliberate batching approximation, recorded in
results/incite/config_diff.md; the alternative is k separate forwards per
step. The within-instance true/false asymmetry -- the only thing the loss
scores -- is untouched by it, since both candidates of a query sit in the
same component and see the same relation states.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - the container always has PyG
    from torch_geometric.data import Data as _Data
except Exception:  # pragma: no cover
    _Data = None

try:
    from .train import self_adversarial_nll
except ImportError:  # pragma: no cover - flat invocation
    from train import self_adversarial_nll  # type: ignore

__all__ = ["generate_instances", "union_batch", "synth_loss", "synth_config",
           "is_synth_step", "synth_step_loss", "SYNTH_DEFAULTS"]

#: Defaults for the ``synth:`` config block. ``enabled`` absent or false is
#: the zero-behavior-change path: nothing in this module runs.
SYNTH_DEFAULTS = {
    "enabled": False,
    "fraction": 0.05,        # share of steps that become synthetic steps
    "instances_per_step": 16,
    "seed": 2048,            # NOT 1024: the eval instances' seed
    "pool_size": 32,         # instances drawn per synthetic step, k sampled
    "palette": 8,            # colour ids are sampled from 1..palette
}

#: Salt keeping the per-step coin stream disjoint from the per-step instance
#: stream (both are pure functions of (synth.seed, step)).
_COIN_SALT = 1_000_003


class _Instance:
    """Fallback graph container when torch_geometric is unavailable.

    Duck-typed exactly like the PyG ``Data`` objects the container builds
    (incite/graphs.py's convention): ``edge_index``, ``edge_type``,
    ``num_nodes``, ``num_relations``, plus ``test_triplets``.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make(**kwargs):
    return _Data(**kwargs) if _Data is not None else _Instance(**kwargs)


def _rand_int(high: int, generator: torch.Generator) -> int:
    return int(torch.randint(int(high), (1,), generator=generator))


# ---------------------------------------------------------------------------
# The instance family
# ---------------------------------------------------------------------------
def sample_colouring(generator: torch.Generator, palette: int = 8) -> List[List[int]]:
    """One randomized colouring with the symmetric structure of the
    diagnostic's configs. 2..6 petals, every pair has ``a != b``, colour ids
    are >= 1 (relation 0 belongs to the stem and the query)."""
    scheme = _rand_int(3, generator)
    perm = (torch.randperm(int(palette), generator=generator) + 1).tolist()
    if scheme == 0:                       # cyclic: [[c1,c2],[c2,c3],...,[cm,c1]]
        m = 2 + _rand_int(5, generator)                      # 2..6 petals
        c = perm[:m]
        return [[c[i], c[(i + 1) % m]] for i in range(m)]
    if scheme == 1:                       # complete: every ordered pair, i != j
        m = 2 + _rand_int(2, generator)                      # 2 or 6 petals
        c = perm[:m]
        return [[c[i], c[j]] for i in range(m) for j in range(m) if i != j]
    p = 1 + _rand_int(3, generator)       # swapped pairs: 2, 4 or 6 petals
    c = perm[:2 * p]
    out: List[List[int]] = []
    for i in range(p):
        out += [[c[2 * i], c[2 * i + 1]], [c[2 * i + 1], c[2 * i]]]
    return out


def create_instance(colourings: Sequence[Sequence[int]], cycle_size: int,
                    tail_len: int, generator: torch.Generator):
    """One petal instance. Structure restated from
    diagnostics/generate_petals.py::create_instance; the only changes are the
    passed-in ``generator`` (instead of the global RNG) and the explicit
    ``true_tail`` / ``false_tail`` fields. ``test_triplets`` keeps the
    generator's convention: row 0 is the TRUE candidate."""
    n_petals = len(colourings)
    max_color = 0
    for pair in colourings:
        max_color = max(max_color, int(pair[0]), int(pair[1]))

    edge_index, edge_type = [], []
    curr_min = 0
    for a, b in colourings:
        a, b = int(a), int(b)
        edge_index.append([0, curr_min + 1])
        edge_type.append(a)
        edge_index.append([0, curr_min + 2])
        edge_type.append(b)
        for i in range(0, cycle_size - 2):
            edge_index.append([curr_min + 2 * i + 1, curr_min + 2 * i + 3])
            edge_type.append(a)
            edge_index.append([curr_min + 2 * i + 2, curr_min + 2 * i + 4])
            edge_type.append(a)
        edge_index.append([curr_min + 2 * cycle_size - 2, curr_min + 2 * cycle_size - 1])
        edge_type.append(a)
        edge_index.append([curr_min + 2 * cycle_size - 3, curr_min + 2 * cycle_size - 1])
        edge_type.append(a)
        curr_min += 2 * cycle_size - 1

    # the stem: relation 0, the query relation
    edge_index.append([0, curr_min + 1])
    edge_type.append(0)
    curr_min += 1
    for _ in range(1, tail_len):
        edge_index.append([curr_min, curr_min + 1])
        edge_type.append(0)
        curr_min += 1

    cycle_id = _rand_int(n_petals, generator)
    x = _rand_int(cycle_size - 1, generator)
    tail_pos = _rand_int(tail_len, generator)
    head_id = 0 if tail_pos == 0 else curr_min - tail_pos

    base = cycle_id * (2 * cycle_size - 1)
    true_tail = base + 2 * x + 1     # odd branch: entirely colour a
    false_tail = base + 2 * x + 2    # even branch: colour b, then a

    return _make(
        edge_index=torch.tensor(edge_index, dtype=torch.long).T,
        edge_type=torch.tensor(edge_type, dtype=torch.long),
        num_nodes=1 + n_petals * (2 * cycle_size - 1) + tail_len,
        num_relations=1 + max_color,
        test_triplets=torch.tensor([[head_id, true_tail, 0],
                                    [head_id, false_tail, 0]], dtype=torch.long),
        head_id=head_id, true_tail=true_tail, false_tail=false_tail,
        cycle_id=cycle_id, cycle_size=int(cycle_size), tail_len=int(tail_len),
    )


def generate_instances(cfg, generator: torch.Generator,
                       count: Optional[int] = None) -> list:
    """A pool of randomized PETALS-family instances.

    ``cfg`` is either the whole run config (a ``synth:`` block is read out of
    it) or the normalized dict ``synth_config`` returns. Everything random --
    scheme, colours, petal count, cycle size, tail length, queried petal,
    depth and head position -- is drawn from ``generator``, so the pool is a
    pure function of the generator's seed.
    """
    scfg = cfg if isinstance(cfg, dict) and "palette" in cfg else synth_config(
        cfg, force=True)
    n = int(count if count is not None else scfg["pool_size"])
    palette = int(scfg["palette"])
    out = []
    for _ in range(n):
        colourings = sample_colouring(generator, palette)
        cycle_size = 2 + _rand_int(5, generator)   # 2..6
        tail_len = 1 + _rand_int(4, generator)     # 1..4
        out.append(create_instance(colourings, cycle_size, tail_len, generator))
    return out


# ---------------------------------------------------------------------------
# The union batch
# ---------------------------------------------------------------------------
def union_batch(instances: Sequence, k: int, generator: torch.Generator
                ) -> Tuple[object, torch.Tensor]:
    """Disjoint union of ``k`` sampled instances plus their query batch.

    Returns ``(union, queries)`` where ``union`` is one graph with node ids
    offset per instance and inverse edges materialized over a SHARED relation
    vocabulary (``num_direct = max colour id + 1`` over the sampled
    instances, inverses at ``+ num_direct`` -- the TRIX convention that
    diagnostics/petals_eval.py::augment applies per instance), and ``queries``
    is ``[k, 2, 3]``: row ``(head, true_tail, 0)`` then ``(head, false_tail,
    0)`` per instance, node ids offset to match.
    """
    k = int(k)
    assert k > 0 and len(instances) > 0
    if k <= len(instances):
        pick = torch.randperm(len(instances), generator=generator)[:k].tolist()
    else:  # more queries than pool entries: draw with replacement
        pick = torch.randint(len(instances), (k,), generator=generator).tolist()
    chosen = [instances[i] for i in pick]

    num_direct = max(int(inst.num_relations) for inst in chosen)
    eis, ets, queries, offset = [], [], [], 0
    for inst in chosen:
        eis.append(inst.edge_index + offset)
        ets.append(inst.edge_type)
        trip = inst.test_triplets.clone()
        trip[:, 0] += offset
        trip[:, 1] += offset
        queries.append(trip)
        offset += int(inst.num_nodes)

    ei = torch.cat(eis, dim=1)
    et = torch.cat(ets)
    union = _make(
        edge_index=torch.cat([ei, ei.flip(0)], dim=1),
        edge_type=torch.cat([et, et + num_direct]),
        num_nodes=offset,
        num_relations=2 * num_direct,
    )
    return union, torch.stack(queries)


def to_device(union, queries: torch.Tensor, device):
    """Move a union batch onto ``device`` (works for PyG Data and the
    fallback container alike)."""
    if device is None:
        return union, queries
    union.edge_index = union.edge_index.to(device)
    union.edge_type = union.edge_type.to(device)
    return union, queries.to(device)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------
def synth_loss(model, union_graph, queries: torch.Tensor, walk_offset: int = 0,
               adversarial_temperature: float = 1.0) -> torch.Tensor:
    """Scalar supervision loss on one union batch.

    ``model(union_graph, queries)`` with ``queries [k, 2, 3]`` scores both
    candidates of every instance in one call -- the same entry point
    ``entity_loss_from_triples`` uses, training mode included (see the module
    docstring on ``remove_easy_edges``). The scores land as ``[k, 2]`` with
    the TRUE candidate in column 0, which is exactly the ``[positive |
    negatives]`` layout ``self_adversarial_nll`` expects with
    ``num_negative=1``.

    With a single negative the self-adversarial weighting is a softmax over
    one element (weight 1.0), so it coincides with uniform weighting and the
    temperature is immaterial; the recipe's 1.0 is kept for consistency.

    Gradients reach ``model.walk_module``: the walk features are added to the
    entity and relation states before round 1, so they are upstream of every
    score. That is the entire point of the exercise.
    """
    pred = model(union_graph, queries, support=None, walk_offset=walk_offset)
    return self_adversarial_nll(pred, 1, adversarial_temperature)


# ---------------------------------------------------------------------------
# Step branching (factored out of pretrain.py so it is unit-testable)
# ---------------------------------------------------------------------------
def synth_config(cfg, force: bool = False) -> Optional[dict]:
    """Normalized ``synth`` block, or ``None`` when the block is absent or
    disabled. ``force`` returns the block even when disabled (tests, and the
    generator helpers that only need the shape parameters)."""
    block = {}
    if cfg is not None:
        got = cfg.get("synth", None) if hasattr(cfg, "get") else getattr(
            cfg, "synth", None)
        if got:
            block = dict(got)
    out = dict(SYNTH_DEFAULTS)
    out.update({k: v for k, v in block.items() if k in SYNTH_DEFAULTS})
    unknown = set(block) - set(SYNTH_DEFAULTS)
    assert not unknown, "unknown synth config keys: %s" % sorted(unknown)
    out["enabled"] = bool(out["enabled"])
    out["fraction"] = float(out["fraction"])
    for key in ("instances_per_step", "seed", "pool_size", "palette"):
        out[key] = int(out[key])
    assert 0.0 <= out["fraction"] <= 1.0, "synth.fraction must be in [0, 1]"
    if not out["enabled"] and not force:
        return None
    return out


def is_synth_step(step: int, scfg: dict) -> bool:
    """Whether ``step`` is a synthetic step.

    A pure function of ``(synth.seed, step)``: the coin comes from a fresh
    generator, so it neither consumes nor perturbs the real-graph generators
    (``pick_gen``/``pos_gen``) and a resumed run makes the same decision for
    the same step number. What a resume does NOT reproduce is the real-graph
    draw sequence -- that already restarts fresh on resume (pretrain.py's
    documented behavior), and skipping a real step also means ``pick_gen``
    advances once less than it would in a synth-off run.
    """
    fraction = float(scfg["fraction"])
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    gen = torch.Generator().manual_seed(int(scfg["seed"]) * _COIN_SALT + int(step))
    return bool(float(torch.rand(1, generator=gen)) < fraction)


def synth_step_loss(model, scfg: dict, step: int, device=None,
                    walk_offset: Optional[int] = None,
                    adversarial_temperature: float = 1.0
                    ) -> Tuple[torch.Tensor, int]:
    """One synthetic step's loss: fresh pool, fresh union batch, one forward.

    Instances are drawn from a ``torch.Generator`` seeded ``synth.seed +
    step``, so a synthetic step's data is a pure function of the step number
    (resume-stable) and independent of the real-graph generators.

    Synthetic steps IGNORE ``accum``: one union batch of
    ``instances_per_step`` instances is one optimizer step, deliberately --
    the union already batches k instances into a single forward, and gradient
    accumulation over several unions would only re-average the same
    distribution at k times the cost. Returns ``(loss, k)``.
    """
    gen = torch.Generator().manual_seed(int(scfg["seed"]) + int(step))
    instances = generate_instances(scfg, gen)
    k = int(scfg["instances_per_step"])
    union, queries = union_batch(instances, k, gen)
    union, queries = to_device(union, queries, device)
    offset = step if walk_offset is None else int(walk_offset)
    return synth_loss(model, union, queries, walk_offset=offset,
                      adversarial_temperature=adversarial_temperature), k
