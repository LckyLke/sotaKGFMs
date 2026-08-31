"""Synthetic automorphic-instance supervision (phase 2.1b, incite/synth.py).

What these tests pin:
  * the union batch is a genuine disjoint union -- offsets, components,
    queries and the shared doubled relation vocabulary;
  * the synthetic query edges are ABSENT from the graph and TRIX's
    ``remove_easy_edges`` is therefore a value no-op on them, which is what
    lets the synthetic loss call the model exactly as
    ``entity_loss_from_triples`` does, in training mode, with no bypass;
  * ``synth_loss.backward()`` puts gradient on the WALK MODULE (the entire
    point: PHASE21_RESULT.md's walk pathway had capability and no gradient)
    and on the trunk;
  * everything is a pure function of the seeds, so a resumed run classifies
    and rebuilds each step identically;
  * step branching is off unless a config turns it on.

All CPU, all sub-second: instances are ~30-90 nodes.
"""

import os

import torch

from conftest import REPO, make_model

from incite import synth
from incite.model import remove_easy_edges

SEED = 2048


def _pool(n=6, seed=SEED, palette=8):
    gen = torch.Generator().manual_seed(seed)
    cfg = dict(synth.SYNTH_DEFAULTS, pool_size=n, palette=palette)
    return synth.generate_instances(cfg, gen), gen


def _components(union):
    """Connected components of the union as a list of node-id sets."""
    adj = {v: set() for v in range(int(union.num_nodes))}
    for u, v in union.edge_index.t().tolist():
        adj[u].add(v)
        adj[v].add(u)
    seen, comps = set(), []
    for start in range(int(union.num_nodes)):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(adj[node] - comp)
        seen |= comp
        comps.append(comp)
    return comps


# ---------------------------------------------------------------------------
# The instance family
# ---------------------------------------------------------------------------
def test_instances_are_wellformed_petals():
    instances, _ = _pool(12)
    assert len(instances) == 12
    for inst in instances:
        assert int(inst.edge_index.max()) < int(inst.num_nodes)
        assert int(inst.edge_index.min()) >= 0
        assert inst.edge_index.shape[1] == inst.edge_type.shape[0]
        # colours are >= 1 so relation 0 is the stem/query relation alone
        petal_edges = inst.edge_type[inst.edge_type != 0]
        assert int(petal_edges.min()) >= 1
        assert int(inst.num_relations) == int(inst.edge_type.max()) + 1
        assert 2 <= inst.cycle_size <= 6 and 1 <= inst.tail_len <= 4
        assert inst.true_tail != inst.false_tail
        assert inst.test_triplets.tolist() == [
            [inst.head_id, inst.true_tail, 0],
            [inst.head_id, inst.false_tail, 0]]


def test_true_tail_is_the_uniform_colour_branch():
    """The label is a consistent structural rule, not a coin flip: the TRUE
    candidate sits on the branch whose first edge repeats the colour of the
    rest of the petal, the FALSE one on the branch that starts with a
    different colour (diagnostics/generate_petals.py's convention, kept)."""
    instances, _ = _pool(12)
    for inst in instances:
        base = inst.cycle_id * (2 * inst.cycle_size - 1)
        edges = {(int(u), int(v)): int(r) for (u, v), r in
                 zip(inst.edge_index.t().tolist(), inst.edge_type.tolist())}
        a = edges[(0, base + 1)]      # odd branch: the petal's colour a
        b = edges[(0, base + 2)]      # even branch: colour b
        assert a != b
        assert (inst.true_tail - base) % 2 == 1     # odd branch
        assert (inst.false_tail - base) % 2 == 0    # even branch
        # every petal edge other than the b-edge carries colour a
        petal = range(base + 1, base + 2 * inst.cycle_size)
        for (u, v), r in edges.items():
            if v in petal and (u, v) != (0, base + 2):
                assert r == a, ((u, v), r, a)


def test_colourings_keep_the_diagnostic_families():
    """2..6 petals; each colouring is one of the symmetric configs' shapes,
    so a colour-permutation-equivariant invariant still collapses the pair."""
    gen = torch.Generator().manual_seed(SEED)
    seen = set()
    for _ in range(200):
        colouring = synth.sample_colouring(gen, palette=8)
        assert 2 <= len(colouring) <= 6
        for a, b in colouring:
            assert a != b and a >= 1 and b >= 1
        firsts = sorted(c[0] for c in colouring)
        seconds = sorted(c[1] for c in colouring)
        # the symmetry that makes the families automorphic: every colour is
        # used as often first as second
        assert firsts == seconds
        seen.add(len(colouring))
    assert len(seen) > 1, "petal counts must actually vary"


# ---------------------------------------------------------------------------
# The union batch
# ---------------------------------------------------------------------------
def test_union_offsets_and_disjoint_components():
    instances, gen = _pool(5)
    union, queries = synth.union_batch(instances, 5, gen)
    assert int(union.num_nodes) == sum(int(i.num_nodes) for i in instances)
    assert int(union.edge_index.max()) < int(union.num_nodes)
    assert int(union.edge_index.min()) >= 0
    comps = _components(union)
    assert len(comps) == 5, "instances must not be wired together"
    assert sorted(len(c) for c in comps) == sorted(
        int(i.num_nodes) for i in instances)
    # each component is a contiguous block: offsets, not interleaving
    for comp in comps:
        assert max(comp) - min(comp) + 1 == len(comp)


def test_union_queries_point_at_the_right_nodes():
    instances, gen = _pool(5)
    union, queries = synth.union_batch(instances, 5, gen)
    assert queries.shape == (5, 2, 3)
    offsets = {min(c): c for c in _components(union)}
    local = {(int(i.num_nodes), i.head_id, i.true_tail, i.false_tail)
             for i in instances}
    for row in queries:
        (h1, t_true, r1), (h2, t_false, r2) = row.tolist()
        assert h1 == h2 and r1 == r2 == 0        # same head, query relation 0
        assert t_true != t_false
        comp = next(c for c in offsets.values() if h1 in c)
        off = min(comp)
        assert t_true in comp and t_false in comp
        assert (len(comp), h1 - off, t_true - off, t_false - off) in local


def test_relation_vocabulary_is_shared_and_doubled():
    instances, gen = _pool(5)
    union, _ = synth.union_batch(instances, 5, gen)
    num_direct = max(int(i.num_relations) for i in instances)
    assert int(union.num_relations) == 2 * num_direct
    half = union.edge_index.shape[1] // 2
    direct_ei, direct_et = union.edge_index[:, :half], union.edge_type[:half]
    inv_ei, inv_et = union.edge_index[:, half:], union.edge_type[half:]
    assert int(direct_et.max()) < num_direct
    assert torch.equal(inv_ei, direct_ei.flip(0))       # swapped ends
    assert torch.equal(inv_et, direct_et + num_direct)  # TRIX's +num_direct


def test_same_seed_gives_identical_union_batches():
    def build(seed):
        gen = torch.Generator().manual_seed(seed)
        cfg = dict(synth.SYNTH_DEFAULTS, pool_size=8)
        instances = synth.generate_instances(cfg, gen)
        return synth.union_batch(instances, 4, gen)

    a_graph, a_q = build(SEED)
    b_graph, b_q = build(SEED)
    assert torch.equal(a_graph.edge_index, b_graph.edge_index)
    assert torch.equal(a_graph.edge_type, b_graph.edge_type)
    assert torch.equal(a_q, b_q)
    assert int(a_graph.num_nodes) == int(b_graph.num_nodes)
    c_graph, c_q = build(SEED + 1)
    assert not (a_graph.edge_index.shape == c_graph.edge_index.shape
                and torch.equal(a_graph.edge_index, c_graph.edge_index)
                and torch.equal(a_q, c_q))


# ---------------------------------------------------------------------------
# remove_easy_edges on synthetic queries
# ---------------------------------------------------------------------------
def test_query_edges_are_absent_and_removal_is_a_noop():
    """The synthetic queries are not graph edges, and TRIX's removal helper
    leaves the message graph untouched -- so training-mode forward needs no
    bypass (incite/synth.py's contract)."""
    instances, gen = _pool(6)
    union, queries = synth.union_batch(instances, 6, gen)
    edges = set(zip(union.edge_index[0].tolist(), union.edge_type.tolist(),
                    union.edge_index[1].tolist()))
    num_direct = int(union.num_relations) // 2
    for row in queries.reshape(-1, 3).tolist():
        h, t, r = row
        assert (h, r, t) not in edges
        assert (t, r + num_direct, h) not in edges
    h_index, t_index, r_index = queries.unbind(-1)
    msg = remove_easy_edges(union, h_index, t_index, r_index)
    assert torch.equal(msg.edge_index, union.edge_index)
    assert torch.equal(msg.edge_type, union.edge_type)


# ---------------------------------------------------------------------------
# The loss and its gradients
# ---------------------------------------------------------------------------
def test_synth_loss_shape_and_value():
    instances, gen = _pool(4)
    union, queries = synth.union_batch(instances, 4, gen)
    model = make_model(walks=True, support=False)
    model.train()
    loss = synth.synth_loss(model, union, queries)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    # sanity: the same call scores exactly the two candidates per instance
    pred = model(union, queries, support=None, walk_offset=0)
    assert pred.shape == (4, 2)


def test_synth_loss_gradient_reaches_the_walk_module():
    """THE point of phase 2.1b: the supervision must move the walk GRU."""
    instances, gen = _pool(4)
    union, queries = synth.union_batch(instances, 4, gen)
    model = make_model(walks=True, support=False)
    model.train()
    model.zero_grad()
    synth.synth_loss(model, union, queries).backward()

    walk_grads = {name: p.grad for name, p in model.walk_module.named_parameters()
                  if p.grad is not None}
    assert walk_grads, "no walk_module parameter received a gradient at all"
    assert any(float(g.abs().sum()) > 0 for g in walk_grads.values()), \
        "walk_module gradients are all exactly zero: the pathway is dead"
    # and the trunk trains too
    trunk = [p for n, p in model.named_parameters()
             if not n.startswith("walk_module") and p.grad is not None]
    assert any(float(p.grad.abs().sum()) > 0 for p in trunk)


def test_synth_loss_is_lower_when_the_true_candidate_scores_higher():
    """The loss orders the pair the way the label says (a fixed-model check,
    no training involved)."""
    class Fixed(torch.nn.Module):
        def __init__(self, margin):
            super().__init__()
            self.margin = margin

        def forward(self, data, batch, support=None, walk_offset=0):
            out = torch.zeros(batch.shape[0], batch.shape[1])
            out[:, 0] = self.margin
            return out

    instances, gen = _pool(3)
    union, queries = synth.union_batch(instances, 3, gen)
    good = synth.synth_loss(Fixed(2.0), union, queries)
    bad = synth.synth_loss(Fixed(-2.0), union, queries)
    assert float(good) < float(bad)


# ---------------------------------------------------------------------------
# Step branching (what pretrain.py calls)
# ---------------------------------------------------------------------------
def test_synth_config_is_off_unless_a_config_turns_it_on():
    assert synth.synth_config({}) is None
    assert synth.synth_config({"synth": {"enabled": False}}) is None
    scfg = synth.synth_config({"synth": {"enabled": True}})
    assert scfg["fraction"] == 0.05 and scfg["instances_per_step"] == 16
    assert scfg["seed"] == 2048          # deliberately not the eval set's 1024
    try:
        synth.synth_config({"synth": {"enabled": True, "typo": 1}})
    except AssertionError as exc:
        assert "typo" in str(exc)
    else:
        raise AssertionError("unknown synth keys must not pass silently")


def test_shipped_configs_keep_synth_off_except_the_deliberate_ones():
    """Only the configs that deliberately carry synthetic supervision --
    phase 2.1b, the v1 composite it graduated into, and the synthetic-prior
    sweep -- may enable it; every other config must be untouched, because
    the live run reads some of them. Globbed, not listed, so a config added
    later cannot quietly turn synthetic supervision on. The sweep configs'
    own assertions live in test_synth_rules.py."""
    import glob

    yaml = __import__("yaml")
    synth_on = ("phase21b", "synthsweep", "v1_full")
    others = [p for p in sorted(glob.glob(os.path.join(REPO, "configs",
                                                       "incite_*.yaml")))
              if not any(tag in os.path.basename(p) for tag in synth_on)]
    assert len(others) >= 4, others
    for path in others:
        cfg = yaml.safe_load(open(path))
        assert synth.synth_config(cfg) is None, os.path.basename(path)
    cfg = yaml.safe_load(
        open(os.path.join(REPO, "configs", "incite_phase21b_walksynth.yaml")))
    scfg = synth.synth_config(cfg)
    assert scfg is not None
    assert scfg["fraction"] == 0.05 and scfg["instances_per_step"] == 16
    assert scfg["seed"] == 2048
    # the live phase-2.1b queue must keep dispatching to the petals family
    assert scfg["prior"] == "petals"
    assert cfg["walks"]["enabled"] and not cfg["support"]["enabled"]
    assert float(cfg["relation"]["lambda"]) == 0.0
    # ... and so must the queued v1 composite (its block carries no prior key)
    v1 = yaml.safe_load(
        open(os.path.join(REPO, "configs", "incite_v1_full.yaml")))
    v1_scfg = synth.synth_config(v1)
    assert v1_scfg is not None
    assert v1_scfg["prior"] == "petals" and v1_scfg["fraction"] == 0.05


def test_is_synth_step_is_deterministic_and_hits_the_fraction():
    scfg = dict(synth.SYNTH_DEFAULTS, enabled=True, fraction=0.05)
    flags = [synth.is_synth_step(s, scfg) for s in range(1, 4001)]
    again = [synth.is_synth_step(s, scfg) for s in range(1, 4001)]
    assert flags == again                      # pure function of (seed, step)
    rate = sum(flags) / len(flags)
    assert 0.035 < rate < 0.065, rate
    assert not all(flags) and any(flags)
    # the seed moves the schedule
    other = dict(scfg, seed=4096)
    assert [synth.is_synth_step(s, other) for s in range(1, 4001)] != flags
    # the degenerate ends
    assert not any(synth.is_synth_step(s, dict(scfg, fraction=0.0))
                   for s in range(1, 50))
    assert all(synth.is_synth_step(s, dict(scfg, fraction=1.0))
               for s in range(1, 50))


def test_synth_step_loss_trains_the_model_end_to_end():
    """The pretrain integration in miniature: branch, build, backward, step.

    The container driver's real-dataset path cannot run in a unit test (it
    needs the FB15k237/WN18RR/CoDExMedium roots and a TRIX tree), so the
    branching + step function pretrain.py calls is exercised directly.
    """
    scfg = dict(synth.SYNTH_DEFAULTS, enabled=True, fraction=1.0,
                instances_per_step=4, pool_size=6)
    model = make_model(walks=True, support=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.walk_module.gru.weight_ih_l0.detach().clone()
    losses = []
    for step in range(1, 4):
        assert synth.is_synth_step(step, scfg)
        optimizer.zero_grad()
        loss, k = synth.synth_step_loss(model, scfg, step,
                                        walk_offset=step)
        assert k == 4
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert len(losses) == 3
    assert not torch.equal(before, model.walk_module.gru.weight_ih_l0), \
        "three synthetic steps left the walk GRU untouched"


def test_synth_step_loss_is_a_pure_function_of_the_step():
    """Resume-stability: the same step number rebuilds the same batch."""
    scfg = dict(synth.SYNTH_DEFAULTS, enabled=True, instances_per_step=4,
                pool_size=6)
    model = make_model(walks=True, support=False)
    model.eval()  # eval mode only to make the two calls comparable
    with torch.no_grad():
        a, _ = synth.synth_step_loss(model, scfg, 17, walk_offset=17)
        b, _ = synth.synth_step_loss(model, scfg, 17, walk_offset=17)
        c, _ = synth.synth_step_loss(model, scfg, 18, walk_offset=18)
    assert float(a) == float(b)
    assert float(a) != float(c)
