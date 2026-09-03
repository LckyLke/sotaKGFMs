# Two hypotheses on the generator's latent structure: PG1 and RR1

Date: 2026-09-03. Both are queued in plan v10 (baseline branch) as 10k
continuations from the 4-graph last checkpoint, each paired against MX2
(the same continuation without the lever). Code: `incite/model.py`
(`EdgeGate`, `RuleHead`, `rule_recovery_loss`), `incite/layers.py`
(`distmult_sum` edge weights), `incite/synth.py` (proof recording in
`forward_chain`, `proof_support`, `rule_targets_for`, the union's proof
pairs and rule hypotheses), `diagnostics/gate_prune_dev.py`. Tests:
`incite/tests/test_gate.py`, `test_rule_recovery.py` (105 pass).

## PG1: the proof-guided propagation gate

The idea (the user's, 2026-09-03): the generator knows which edges matter
for a synthetic query, so the propagation of the NBFNet-style trunk can be
taught where to look, and later pruned for speed.

What the generator knows. The bounded chainer now records, for every fact
it adds, the rule and the premises that landed it (the lowest-indexed
firing rule, so the record is a pure function of the same inputs and the
closure is byte-identical). `proof_support` expands the premises of a
query's positives down to OBSERVED facts: the proof edges. Tests re-run
the chainer on the proof edges alone and require every real positive to
be re-derived.

The gate. One `EdgeGate` per round: a per-(query, source node) scale
times a per-(query, relation) scale on every message. The factorization
is what makes it free: the node scale folds into the source states, the
relation scale into the projected relation features, and the fused
rspmm kernel runs unchanged; no `[b, E, d]` tensor is materialized (on
FB15k237 that would be 1.1 GB per round). Logits are linear in the state
plus a query term; weights start at zero and every scale is divided by
sigmoid(bias) on the same device, so a freshly attached gate is EXACTLY
the identity (bitwise-equal scores, tested) and the warm start from the
4-graph checkpoint changes nothing at step 0.

The loss. On synthetic steps, one-sided: `-log` of the raw node gate at
the proof edge's source and of the raw relation gate at its relation,
averaged over proof pairs and rounds, weight 1 (`synth.proof_weight`).
Proof edges are pushed open; nothing is pushed shut, because type
context flows through non-proof edges (the unary result says global
context matters). The task loss trains the gate everywhere, on real and
synthetic steps. Closing is therefore learned only where it helps the
task; the proof term is a prior on what must stay open.

The measurement (PG1P). After training, `diagnostics/gate_prune_dev.py`
zeroes the lowest-gated share x of every query's edges (product of the
two raw gates, per-row threshold, a per-edge weight through the kernel
one row at a time) and reports DEV10 valid MRR for x in 0, 0.2, 0.4,
0.6, 0.8, 0.9, 0.95, plus the share of gate products below 0.5 at x = 0.
A curve flat to 0.6 or 0.8 says sparse propagation (per-query edge
subsets with their own kernels, A*Net-style, about a week) is worth
building for larger batches or more rounds on 16 GB. Gates that never
leave their open start say the task loss found nothing to close, and
the next dose is a sparsity pressure.

Risks. Proofs exist only on the 30 percent synthetic steps. Real-graph
"proofs" are fuzzy, so the gate may stay open on real graphs and the
curve may say "random pruning". The gate adds a small per-round cost
(two linear maps over nodes and relations).

## RR1: rule recovery from the relation states

Every rules-prior instance carries its latent rule system. `RuleHead`
scores rule hypotheses on the trunk's final relation states: hierarchy
and inversion as bilinear forms on two states, symmetry as a linear form
on one, composition of length 2 as an MLP on three. Labels are certain:
positives are the instance's rules that the observed facts evidence
(the body pattern occurs at least twice, so the rule is readable in
principle), negatives are uniformly drawn hypotheses that are not rules
of the system, four per positive and at least four per kind. BCE per
kind, averaged over the kinds present, weight 1 (`synth.rule_weight`).
Needs the isolated relation blocks (asserted), so a row's relation
states come from its own facts only.

Why it might transfer: the relation encoder is what an unseen-relation
graph relies on, and the ind_er levers so far moved little. A head that
forces hierarchy, inversion, symmetry and composition to be linearly
readable from the states is a direct pressure on that encoder. Why it
might not: the states are query-conditioned and the rule structure is
not; the head reads the row's own query-conditioned states.

## The reading rule for both

Paired bootstrap against MX2 on the 41 graphs, per-scenario tables, and
for PG1 the pruning curve. A lever joins the recipe only if its interval
against MX2 excludes zero on at least one group and the other group does
not lose. The winner among all continuation levers becomes the
from-scratch recipe run R1 (plan v10 picks it by mean group MRR).
