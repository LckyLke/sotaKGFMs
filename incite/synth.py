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

The rules prior (``synth.prior: rules``, the synthetic-prior sweep)
------------------------------------------------------------------------------
The fraction sweep (configs/incite_synthsweep_{25,75,100}.yaml) asks a
TabPFN-shaped question: can the model learn its inductive bias from a
SYNTHETIC prior alone, with DEV10 (real graphs) measuring transfer? The
petals family is far too narrow a prior for that -- it exercises exactly one
capability. ``synth.prior: "rules"`` selects a second instance family: each
instance samples a latent RULE SYSTEM (typed relations; composition,
hierarchy, inversion and symmetry rules with per-rule confidences), generates
degree-skewed base facts respecting the type signatures, forward-chains the
rules to a capped, noisy closure, and drops a fraction of derived facts. The
query is one fact the rules derive from the observed graph but which is
absent from it (label certain by construction), paired with type-consistent
negatives the rules do NOT derive. Rows are ``(head, tail, relation)`` with
row 0 the positive -- the petals interface -- so ``union_batch`` and
``synth_loss`` work unchanged; ``synth.neg_per_pos_rules`` (default 1) adds
extra negatives per positive, and ``synth_loss`` reads the negative count
off the query tensor. ``prior`` absent or "petals" is the old family,
byte-identical. Full design note: results/incite/RULES_PRIOR.md.

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

import math

from typing import List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - the container always has PyG
    from torch_geometric.data import Data as _Data
except Exception:  # pragma: no cover
    _Data = None

try:
    from .train import self_adversarial_nll, multi_positive_nll
except ImportError:  # pragma: no cover - flat invocation
    from train import self_adversarial_nll, multi_positive_nll  # type: ignore

__all__ = ["generate_instances", "union_batch", "synth_loss", "synth_config",
           "is_synth_step", "synth_step_loss", "SYNTH_DEFAULTS",
           "RULES_RANGES", "sample_rule_system", "forward_chain",
           "create_rules_instance"]

#: Defaults for the ``synth:`` config block. ``enabled`` absent or false is
#: the zero-behavior-change path: nothing in this module runs.
SYNTH_DEFAULTS = {
    "enabled": False,
    "fraction": 0.05,        # share of steps that become synthetic steps
    "instances_per_step": 16,
    "seed": 2048,            # NOT 1024: the eval instances' seed
    "pool_size": 32,         # instances drawn per synthetic step, k sampled
    "palette": 8,            # colour ids are sampled from 1..palette
    "prior": "petals",       # instance family: "petals" (2.1b) or "rules"
    "neg_per_pos_rules": 1,  # negatives per positive, rules family only
    # ---- MX2 generator-side knobs (2026-09-03), rules family only. The
    # defaults reproduce the MX1 code path draw for draw. ----
    "num_positive_rules": 1,      # positive slots per query row (masked)
    "hard_neg_frac": 0.0,         # share of negatives from the head's 1-2 hop
                                  # neighborhood; the rest are uniform
    "unseen_answer_share": -1.0,  # target share of unseen-answer queries;
                                  # negative = the natural draw (about 0.47)
    "isolate_relations": False,   # per-instance relation blocks in the union
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


# ---------------------------------------------------------------------------
# The rules family (synth.prior == "rules")
# ---------------------------------------------------------------------------
#: Parameter ranges of the rules prior, in one place (results/incite/
#: config_diff.md records why). Sizes are tuned so a union of 16 instances
#: stays under ~50k edges INCLUDING the materialized inverses: max_facts
#: caps an instance's observed facts, so 16 * max_facts * 2 = 44.8k is the
#: hard worst case. Diversity across steps matters more than size per step.
RULES_RANGES = {
    "entities": (100, 2000),      # latent entity pool, log-uniform
    "relations": (8, 64),         # uniform
    "types": (4, 16),             # uniform
    "type_sig": (1, 3),           # head/tail types per relation, uniform
    "zipf": (0.7, 1.1),           # entity-popularity exponent, uniform
    "rel_zipf": (0.5, 1.0),       # relation-frequency exponent, uniform
    "density": (0.8, 1.6),        # base facts per entity, uniform
    "confidence": (0.6, 0.95),    # per-rule application probability, uniform
    "drop": (0.1, 0.4),           # incompleteness on derived facts, uniform
    "hier": (1, 4),               # rules per family, uniform inclusive
    "inv": (1, 3),
    "sym": (1, 3),
    "comp": (1, 4),               # chain length 2 or 3, coin per rule
    "max_base": 900,              # caps below: the union edge budget
    "max_facts": 1400,
    "chain_iters": 3,
}


def _rand_float(generator: torch.Generator) -> float:
    return float(torch.rand(1, generator=generator))


def _rand_range(lo: float, hi: float, generator: torch.Generator) -> float:
    return lo + (hi - lo) * _rand_float(generator)


def _rand_between(lo: int, hi: int, generator: torch.Generator) -> int:
    """Uniform integer in [lo, hi], both ends inclusive."""
    return int(lo) + _rand_int(int(hi) - int(lo) + 1, generator)


def _sample_type_set(num_types: int, lo: int, hi: int,
                     generator: torch.Generator) -> set:
    k = min(_rand_between(lo, hi, generator), int(num_types))
    return set(torch.randperm(int(num_types),
                              generator=generator)[:k].tolist())


def _propagate_signatures(rules: Sequence[tuple], head_types: List[set],
                          tail_types: List[set]) -> None:
    """Expand relation signatures to a fixpoint under the rules, in place.

    Afterwards every rule's conclusion satisfies its head relation's
    signature whenever the premises satisfy theirs, by induction over the
    chaining -- so EVERY fact in a closure of signature-respecting base
    facts respects the (expanded) signatures. Monotone and bounded (sets
    only grow, at most num_relations * num_types * 2 element insertions),
    so the loop terminates.
    """
    changed = True
    while changed:
        changed = False
        for kind, body, head, _conf in rules:
            if kind == "hier":
                new_h, new_t = head_types[body[0]], tail_types[body[0]]
            elif kind in ("inv", "sym"):
                new_h, new_t = tail_types[body[0]], head_types[body[0]]
            else:                                    # comp
                new_h, new_t = head_types[body[0]], tail_types[body[-1]]
            for dst, src in ((head_types[head], new_h),
                             (tail_types[head], new_t)):
                if not src <= dst:
                    dst |= src
                    changed = True


def _weighted_pick(weights: Optional[torch.Tensor],
                   generator: torch.Generator, subset: list = None) -> int:
    """One relation id, drawn by ``weights`` (uniform when None), optionally
    restricted to ``subset``."""
    if weights is None:
        if subset is not None:
            return subset[_rand_int(len(subset), generator)]
        raise AssertionError("uniform pick needs a subset")
    if subset is not None:
        i = int(torch.multinomial(weights[torch.tensor(subset)], 1,
                                  generator=generator))
        return int(subset[i])
    return int(torch.multinomial(weights, 1, generator=generator))


def sample_rule_system(num_relations: int, num_types: int,
                       generator: torch.Generator, ranges: dict = None,
                       body_weights: Optional[torch.Tensor] = None
                       ) -> Tuple[list, List[set], List[set]]:
    """One latent rule system: per-relation type signatures plus a mix of
    the rule families real ontologies exhibit.

    Returns ``(rules, head_types, tail_types)``. Each rule is a tuple
    ``(kind, body, head, confidence)`` with ``kind`` in {"hier", "inv",
    "sym", "comp"}, ``body`` a tuple of body relation ids (2 or 3 for
    composition, 1 otherwise), ``head`` the derived relation id:

      * hier: head(x, y) <- body[0](x, y)
      * inv:  head(y, x) <- body[0](x, y)
      * sym:  head == body[0], head(y, x) <- head(x, y)
      * comp: head(x, z) <- body[0](x, y) AND body[1](y, z) [AND body[2]]

    Composition bodies are sampled join-compatible (consecutive relations
    share an intermediate type; if none exists one is grafted on), and
    ``_propagate_signatures`` then guarantees closure facts stay typed.
    At least one rule of every family is drawn per system.

    ``body_weights`` (the base-fact relation frequencies) biases BODY
    relation picks: rules whose premises sit on fact-poor relations derive
    almost nothing, so an unweighted draw over a Zipf-skewed fact
    distribution yields a mostly rule-free graph. Head relations stay
    uniform -- a derived relation being rare among base facts is realistic.
    """
    rng = dict(RULES_RANGES)
    if ranges:
        rng.update(ranges)
    R, T = int(num_relations), int(num_types)
    lo_s, hi_s = rng["type_sig"]
    head_types = [_sample_type_set(T, lo_s, hi_s, generator) for _ in range(R)]
    tail_types = [_sample_type_set(T, lo_s, hi_s, generator) for _ in range(R)]

    rules, seen = [], set()

    def add(kind, body, head):
        key = (kind, tuple(body), head)
        if key in seen:
            return False
        seen.add(key)
        conf = _rand_range(rng["confidence"][0], rng["confidence"][1],
                           generator)
        rules.append((kind, tuple(body), int(head), conf))
        return True

    everything = list(range(R))
    for _ in range(_rand_between(*rng["hier"], generator=generator)):
        for _try in range(4):
            r1 = _weighted_pick(body_weights, generator, everything)
            r2 = _rand_int(R, generator)
            if r1 != r2 and add("hier", (r1,), r2):
                break
    for _ in range(_rand_between(*rng["inv"], generator=generator)):
        for _try in range(4):
            r1 = _weighted_pick(body_weights, generator, everything)
            r2 = _rand_int(R, generator)
            if r1 != r2 and add("inv", (r1,), r2):
                break
    for _ in range(_rand_between(*rng["sym"], generator=generator)):
        for _try in range(4):
            r = _weighted_pick(body_weights, generator, everything)
            if add("sym", (r,), r):
                break
    for _ in range(_rand_between(*rng["comp"], generator=generator)):
        length = 2 + _rand_int(2, generator)             # 2 or 3
        body = [_weighted_pick(body_weights, generator, everything)]
        for _hop in range(length - 1):
            prev = body[-1]
            eligible = [r for r in range(R)
                        if head_types[r] & tail_types[prev]]
            if eligible:
                nxt = _weighted_pick(body_weights, generator, eligible)
            else:                    # graft a shared intermediate type on
                nxt = _weighted_pick(body_weights, generator, everything)
                shared = sorted(tail_types[prev])
                head_types[nxt].add(
                    shared[_rand_int(len(shared), generator)])
            body.append(nxt)
        add("comp", tuple(body), _rand_int(R, generator))

    _propagate_signatures(rules, head_types, tail_types)
    return rules, head_types, tail_types


def _rule_candidates(kind: str, body: tuple, head: int, by_rel: dict,
                     cap: int) -> list:
    """Facts one rule derives from the indexed fact set, enumerated in a
    deterministic (sorted-premise) order and truncated at ``cap``."""
    out = []
    if kind == "hier":
        for h, t in by_rel.get(body[0], ()):
            out.append((h, head, t))
            if len(out) >= cap:
                break
    elif kind in ("inv", "sym"):
        for h, t in by_rel.get(body[0], ()):
            out.append((t, head, h))
            if len(out) >= cap:
                break
    else:                                                # comp
        pairs = by_rel.get(body[0], ())
        for rel in body[1:]:
            nxt = {}
            for h, t in by_rel.get(rel, ()):
                nxt.setdefault(h, []).append(t)
            joined = []
            for h, t in pairs:
                for z in nxt.get(t, ()):
                    joined.append((h, z))
                    if len(joined) >= cap:
                        break
                if len(joined) >= cap:
                    break
            pairs = joined
        out = [(h, head, t) for h, t in pairs]
    return out


def forward_chain(facts, rules: Sequence[tuple],
                  generator: Optional[torch.Generator] = None,
                  max_iters: int = 3, cap: Optional[int] = 12000,
                  per_rule_cap: int = 4000) -> set:
    """Forward-chain ``rules`` over ``facts`` -- (h, r, t) triples -- for at
    most ``max_iters`` iterations. Returns the closed fact SET (inputs
    included).

    ``generator`` None is the deterministic semantics: every applicable
    rule fires. With a generator, each candidate derivation (rule, fact)
    fires with probability = the rule's confidence, and the coin is rolled
    ONCE per (rule, fact) -- a rejected derivation stays rejected, so the
    observed firing rate matches the confidence instead of inflating over
    iterations. A fact proposed by several rules lands if any of them
    fires.

    ``cap`` bounds the total fact count and ``per_rule_cap`` the facts one
    rule may propose per iteration (hub-heavy composition joins can go
    quadratic); both truncations take a sorted prefix, so the closure is a
    pure function of (facts, rules, seed). The caps are part of the prior's
    generative semantics: "derivable" means derivable by THIS bounded
    chainer, which is also what the verification tests re-run.
    """
    known = set(facts)
    rejected = set()
    for _ in range(int(max_iters)):
        if cap is not None and len(known) >= cap:
            break
        by_rel = {}
        for h, r, t in known:
            by_rel.setdefault(r, []).append((h, t))
        for r in by_rel:
            by_rel[r].sort()
        cands = set()
        for idx, rule in enumerate(rules):
            for fact in _rule_candidates(rule[0], rule[1], rule[2], by_rel,
                                         int(per_rule_cap)):
                if fact not in known and (idx, fact) not in rejected:
                    cands.add((idx, fact))
        if not cands:
            break
        ordered = sorted(cands)
        if generator is None:
            fired = {fact for _idx, fact in ordered}
        else:
            coins = torch.rand(len(ordered), generator=generator).tolist()
            fired = set()
            for (idx, fact), coin in zip(ordered, coins):
                if coin < rules[idx][3]:
                    fired.add(fact)
                else:
                    rejected.add((idx, fact))
        new = sorted(fired)
        if cap is not None:
            new = new[:max(0, int(cap) - len(known))]
        if not new:
            continue
        known.update(new)
    return known


def _draw_from(pool: list, n: int, generator: torch.Generator) -> list:
    """``n`` entries of ``pool``: without replacement when it is long
    enough, with replacement otherwise (a short pool must not bias the
    instance draw toward hub-heavy graphs by retries)."""
    if n <= 0:
        return []
    if len(pool) >= n:
        idx = torch.randperm(len(pool), generator=generator)[:n]
    else:
        idx = torch.randint(len(pool), (n,), generator=generator)
    return [pool[i] for i in idx.tolist()]


def _draw_negatives(pool: list, n: int, hard_frac: float, h: int, graph_set,
                    generator: torch.Generator):
    """``n`` certified negatives from ``pool`` (type-consistent participating
    tails that no rule derives, see the caller). With ``hard_frac`` > 0 that
    share is drawn from the head's 1-2 hop neighborhood first -- the
    structurally close candidates a path model confuses -- and the rest
    from the remainder of the pool. Real KGs cannot offer this: a close
    candidate there may be a true fact that is merely missing. An empty
    pool returns None (the caller redraws the query). With ``hard_frac`` 0
    and ``n`` 1 the draw sequence is the MX1 one."""
    if not pool:
        return None
    if hard_frac <= 0.0 or n <= 1:
        return _draw_from(pool, n, generator)
    n_hard = int(round(hard_frac * n))
    adj = {}
    for hh, _rr, tt in graph_set:
        adj.setdefault(hh, set()).add(tt)
        adj.setdefault(tt, set()).add(hh)
    hop1 = adj.get(h, set())
    near = set(hop1)
    for x in hop1:
        near |= adj.get(x, set())
    near.discard(h)
    hard_pool = [e for e in pool if e in near]
    hard = _draw_from(hard_pool, min(n_hard, len(hard_pool)), generator)
    taken = set(hard)
    rest_pool = [e for e in pool if e not in taken] or pool
    return hard + _draw_from(rest_pool, n - len(hard), generator)


def _try_rules_instance(generator: torch.Generator, neg_per_pos: int,
                        rng: dict, num_positive: int = 1,
                        hard_neg_frac: float = 0.0,
                        unseen_answer_share: float = -1.0):
    """One attempt at a rules instance; None when the draw yields no usable
    query (retried by ``create_rules_instance``, generator advancing).

    The three keyword knobs are the MX2 generator-side levers; at their
    defaults the function draws exactly what it drew for MX1."""
    lo_e, hi_e = rng["entities"]
    E = int(round(math.exp(_rand_range(math.log(lo_e), math.log(hi_e),
                                       generator))))
    R = _rand_between(*rng["relations"], generator=generator)
    T = _rand_between(*rng["types"], generator=generator)
    # relation frequencies first: they bias the rule bodies (see
    # sample_rule_system on why rules must sit where the facts are)
    rel_w = (torch.randperm(R, generator=generator).double() + 1.0) ** (
        -_rand_range(*rng["rel_zipf"], generator=generator))
    rules, head_types, tail_types = sample_rule_system(
        R, T, generator, rng, body_weights=rel_w)

    type_w = 0.5 + 1.5 * torch.rand(T, generator=generator)
    entity_type = torch.multinomial(type_w, E, replacement=True,
                                    generator=generator)
    alpha = _rand_range(*rng["zipf"], generator=generator)
    pop = (torch.randperm(E, generator=generator).double() + 1.0) ** (-alpha)
    type_members = [(entity_type == t).nonzero(as_tuple=True)[0]
                    for t in range(T)]

    # base facts: skewed over relations AND entities, signature-respecting
    n_target = min(int(E * _rand_range(*rng["density"], generator=generator)),
                   int(rng["max_base"]))
    counts = torch.bincount(
        torch.multinomial(rel_w, n_target, replacement=True,
                          generator=generator), minlength=R)
    base = set()
    for r in range(R):
        n_r = int(counts[r])
        if n_r == 0:
            continue
        h_pool = torch.cat([type_members[t] for t in sorted(head_types[r])])
        t_pool = torch.cat([type_members[t] for t in sorted(tail_types[r])])
        if h_pool.numel() == 0 or t_pool.numel() == 0:
            continue
        hs = h_pool[torch.multinomial(pop[h_pool], n_r, replacement=True,
                                      generator=generator)]
        ts = t_pool[torch.multinomial(pop[t_pool], n_r, replacement=True,
                                      generator=generator)]
        for h, t in zip(hs.tolist(), ts.tolist()):
            if h != t:
                base.add((h, r, t))
    if len(base) < 8:
        return None

    # noisy closure -> incompleteness -> the observed graph
    noisy = forward_chain(base, rules, generator, rng["chain_iters"],
                          cap=int(rng["max_facts"]))
    derived = sorted(noisy - base)
    delta = _rand_range(*rng["drop"], generator=generator)
    if derived:
        coins = torch.rand(len(derived), generator=generator).tolist()
        kept = [f for f, c in zip(derived, coins) if c >= delta]
    else:
        kept = []
    graph_set = set(base) | set(kept)

    # labels: what the bounded chainer still derives from the OBSERVED graph
    label_closure = forward_chain(graph_set, rules)
    full_closure = forward_chain(base, rules)
    query_pool = sorted(label_closure - graph_set)
    if not query_pool:
        return None
    forbidden = label_closure | full_closure

    participating = sorted({e for h, r, t in graph_set for e in (h, t)})
    part_by_type = {}
    for e in participating:
        part_by_type.setdefault(int(entity_type[e]), []).append(e)

    # scenario targeting: with a non-negative target, the query comes from
    # the unseen-answer part of the pool (the answer has no incoming
    # query-relation edge in the observed graph) with that probability and
    # from the seen-answer part otherwise; an empty part falls back to the
    # whole pool. The natural draw (target < 0) is about 47 percent unseen
    # at these ranges, already above the benchmark's 37 percent.
    draw_pool = query_pool
    if unseen_answer_share >= 0.0:
        has_in = {(rr, tt) for _hh, rr, tt in graph_set}
        unseen = [f for f in query_pool if (f[1], f[2]) not in has_in]
        seen = [f for f in query_pool if (f[1], f[2]) in has_in]
        part = unseen if _rand_float(generator) < unseen_answer_share else seen
        draw_pool = part if part else query_pool

    picked = None
    for _try in range(12):
        h, r, t = draw_pool[_rand_int(len(draw_pool), generator)]
        pool = [e for ty in sorted(tail_types[r])
                for e in part_by_type.get(ty, ())
                if e != h and (h, r, e) not in forbidden]
        negs = _draw_negatives(pool, neg_per_pos, hard_neg_frac, h, graph_set,
                               generator)
        if negs is not None:
            picked = (h, r, t, negs)
            break
    if picked is None:
        return None
    h, r, t, negs = picked

    # full-closure positives: the other derivable-but-absent tails of the
    # query (h, r), up to num_positive - 1 of them, drawn only when asked.
    # They are certain labels the single-positive row left unscored, and
    # scoring them costs no extra propagation (same head, same relation).
    extra_pos: List[int] = []
    if num_positive > 1:
        others = sorted(tt for hh, rr, tt in query_pool
                        if hh == h and rr == r and tt != t)
        if others:
            perm = torch.randperm(len(others), generator=generator)
            extra_pos = [others[i] for i in perm.tolist()[:num_positive - 1]]

    # compact ids to the participating entities (h, t and negs are among
    # them: closure facts only mention graph entities, negatives are drawn
    # from them -- degree-0 negatives would be trivially rejectable)
    old2new = {e: i for i, e in enumerate(participating)}
    graph = sorted(graph_set)
    edge_index = torch.tensor(
        [[old2new[hh] for hh, _r, _t in graph],
         [old2new[tt_] for _h, _r, tt_ in graph]], dtype=torch.long)
    edge_type = torch.tensor([rr for _h, rr, _t in graph], dtype=torch.long)
    # row layout: [positive | extra positives | padding | negatives]; the
    # padding repeats the positive and is masked out of the loss, so every
    # instance of a pool has the same row count and union_batch can stack
    rows = [[old2new[h], old2new[t], r]]
    rows += [[old2new[h], old2new[p], r] for p in extra_pos]
    while len(rows) < num_positive:
        rows.append(list(rows[0]))
    rows += [[old2new[h], old2new[n], r] for n in negs]
    pos_mask = torch.tensor([True] * (1 + len(extra_pos))
                            + [False] * (num_positive - 1 - len(extra_pos)))
    return _make(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=len(participating),
        num_relations=R,
        test_triplets=torch.tensor(rows, dtype=torch.long),
        pos_mask=pos_mask,
        num_positive=int(num_positive),
        family="rules",
        entity_type=entity_type[torch.tensor(participating,
                                             dtype=torch.long)],
        rules=rules,
        head_types=[sorted(s) for s in head_types],
        tail_types=[sorted(s) for s in tail_types],
        num_base=len(base), num_derived_kept=len(kept),
        num_dropped=len(derived) - len(kept),
        num_query_pool=len(query_pool),
    )


def create_rules_instance(generator: torch.Generator, neg_per_pos: int = 1,
                          ranges: dict = None, num_positive: int = 1,
                          hard_neg_frac: float = 0.0,
                          unseen_answer_share: float = -1.0):
    """One rules-family instance (see the module docstring).

    Duck-typed like the petals instances -- ``edge_index`` (direct edges
    only; the union materializes inverses), ``edge_type``, ``num_nodes``,
    ``num_relations``, ``test_triplets`` -- with ``test_triplets`` shaped
    ``[1 + neg_per_pos, 3]``, rows ``(head, tail, relation)``, row 0 the
    positive. Everything is drawn from ``generator``; the rare unusable
    draw (no derivable-but-absent fact, or no negative pool) is retried
    with the generator advancing, so the result is still a pure function
    of the generator state.
    """
    rng = dict(RULES_RANGES)
    if ranges:
        rng.update(ranges)
    for _attempt in range(8):
        inst = _try_rules_instance(
            generator, int(neg_per_pos), rng, num_positive=int(num_positive),
            hard_neg_frac=float(hard_neg_frac),
            unseen_answer_share=float(unseen_answer_share))
        if inst is not None:
            return inst
    raise AssertionError("rules-prior instance generation failed 8 draws "
                         "in a row; the ranges are mis-tuned")


def generate_instances(cfg, generator: torch.Generator,
                       count: Optional[int] = None) -> list:
    """A pool of randomized synthetic instances.

    ``cfg`` is either the whole run config (a ``synth:`` block is read out of
    it) or the normalized dict ``synth_config`` returns. ``synth.prior``
    dispatches the family: "petals" (default, the phase-2.1b generator,
    unchanged) or "rules" (the latent-rule-system prior). Everything random
    is drawn from ``generator``, so the pool is a pure function of the
    generator's seed either way.
    """
    scfg = cfg if isinstance(cfg, dict) and "palette" in cfg else synth_config(
        cfg, force=True)
    n = int(count if count is not None else scfg["pool_size"])
    prior = str(scfg.get("prior", SYNTH_DEFAULTS["prior"]))
    if prior == "rules":
        neg = int(scfg.get("neg_per_pos_rules",
                           SYNTH_DEFAULTS["neg_per_pos_rules"]))
        return [create_rules_instance(
            generator, neg,
            num_positive=int(scfg.get("num_positive_rules", 1)),
            hard_neg_frac=float(scfg.get("hard_neg_frac", 0.0)),
            unseen_answer_share=float(scfg.get("unseen_answer_share", -1.0)))
            for _ in range(n)]
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
def union_batch(instances: Sequence, k: int, generator: torch.Generator,
                isolate_relations: bool = False
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

    # isolate_relations (MX2): every instance gets its own block of
    # relation ids, so its relation states are computed from its own facts
    # only -- as they are for a real graph -- instead of being mixed with
    # the bulk terms of k - 1 unrelated rule systems (the crosstalk risk in
    # RULES_PRIOR.md). Off: one shared vocabulary, the petals convention.
    if isolate_relations:
        rel_offsets, acc = [], 0
        for inst in chosen:
            rel_offsets.append(acc)
            acc += int(inst.num_relations)
        num_direct = acc
    else:
        rel_offsets = [0] * len(chosen)
        num_direct = max(int(inst.num_relations) for inst in chosen)
    eis, ets, queries, masks, offset = [], [], [], [], 0
    for inst, roff in zip(chosen, rel_offsets):
        eis.append(inst.edge_index + offset)
        ets.append(inst.edge_type + roff if roff else inst.edge_type)
        trip = inst.test_triplets.clone()
        trip[:, 0] += offset
        trip[:, 1] += offset
        if roff:
            trip[:, 2] += roff
        queries.append(trip)
        masks.append(getattr(inst, "pos_mask", None))
        offset += int(inst.num_nodes)

    ei = torch.cat(eis, dim=1)
    et = torch.cat(ets)
    union = _make(
        edge_index=torch.cat([ei, ei.flip(0)], dim=1),
        edge_type=torch.cat([et, et + num_direct]),
        num_nodes=offset,
        num_relations=2 * num_direct,
    )
    # several positive slots per row: hand the mask to synth_loss. A pool
    # of single-positive rows carries no mask, so the loss stays the old one.
    if all(m is not None for m in masks) and any(int(m.shape[0]) > 1
                                                  for m in masks):
        union.query_pos_mask = torch.stack(masks)
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

    ``model(union_graph, queries)`` with ``queries [k, 1 + neg, 3]`` scores
    every candidate of every instance in one call -- the same entry point
    ``entity_loss_from_triples`` uses, training mode included (see the module
    docstring on ``remove_easy_edges``). The scores land as ``[k, 1 + neg]``
    with the TRUE candidate in column 0, which is exactly the ``[positive |
    negatives]`` layout ``self_adversarial_nll`` expects; the negative count
    is read off the query tensor (petals: 1, rules: neg_per_pos_rules).

    With a single negative the self-adversarial weighting is a softmax over
    one element (weight 1.0), so it coincides with uniform weighting and the
    temperature is immaterial; the recipe's 1.0 is kept for consistency.
    With several negatives the recipe's self-adversarial weighting applies
    as in real training.

    Gradients reach ``model.walk_module``: the walk features are added to the
    entity and relation states before round 1, so they are upstream of every
    score. That is the entire point of the exercise.
    """
    pred = model(union_graph, queries, support=None, walk_offset=walk_offset)
    pos_mask = getattr(union_graph, "query_pos_mask", None)
    if pos_mask is not None:
        # [positive slots (masked) | negatives], the full-closure layout
        return multi_positive_nll(pred, pos_mask.to(pred.device),
                                  adversarial_temperature)
    return self_adversarial_nll(pred, int(queries.shape[1]) - 1,
                                adversarial_temperature)


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
    out["isolate_relations"] = bool(out["isolate_relations"])
    out["fraction"] = float(out["fraction"])
    out["hard_neg_frac"] = float(out["hard_neg_frac"])
    out["unseen_answer_share"] = float(out["unseen_answer_share"])
    for key in ("instances_per_step", "seed", "pool_size", "palette",
                "neg_per_pos_rules", "num_positive_rules"):
        out[key] = int(out[key])
    out["prior"] = str(out["prior"])
    assert out["prior"] in ("petals", "rules"), \
        "synth.prior must be 'petals' or 'rules', got %r" % out["prior"]
    assert out["neg_per_pos_rules"] >= 1, \
        "synth.neg_per_pos_rules must be >= 1"
    assert out["num_positive_rules"] >= 1, \
        "synth.num_positive_rules must be >= 1"
    assert 0.0 <= out["hard_neg_frac"] <= 1.0, \
        "synth.hard_neg_frac must be in [0, 1]"
    assert out["unseen_answer_share"] <= 1.0, \
        "synth.unseen_answer_share must be <= 1 (negative = natural)"
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

    For the RULES prior the pool is exactly ``instances_per_step``: the pool
    is regenerated per step anyway, so ``pool_size`` oversampling would only
    double the (non-trivial) generation cost for the same distribution. The
    petals path keeps ``pool_size`` so its draw stream stays byte-identical.
    """
    gen = torch.Generator().manual_seed(int(scfg["seed"]) + int(step))
    k = int(scfg["instances_per_step"])
    count = k if str(scfg.get("prior", "petals")) == "rules" else None
    instances = generate_instances(scfg, gen, count)
    union, queries = union_batch(
        instances, k, gen,
        isolate_relations=bool(scfg.get("isolate_relations", False)))
    union, queries = to_device(union, queries, device)
    offset = step if walk_offset is None else int(walk_offset)
    return synth_loss(model, union, queries, walk_offset=offset,
                      adversarial_temperature=adversarial_temperature), k
