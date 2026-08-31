# The rules prior (synth.prior: "rules")

Date: 2026-08-31. Consumer: the synthetic-prior fraction sweep
(configs/incite_synthsweep_{25,75,100}.yaml). Code: incite/synth.py
(`create_rules_instance`, `forward_chain`, `sample_rule_system`), behind
the `synth.prior` dispatch; `prior` absent or "petals" is the phase-2.1b
family, byte-identical. PHASE21B_RESULT.md is the causal ground this
stands on: synthetic supervision with certain labels demonstrably aligns
a pathway that real graphs cannot. The sweep asks the TabPFN-shaped
follow-up -- how much of pretraining can a synthetic prior REPLACE, with
DEV10 (real graphs, untouched) measuring transfer.

## The generative process

Every instance is a fresh latent rule system and a KG sampled from it.
All draws come from the passed torch.Generator, so an instance pool is a
pure function of (synth.seed + step); a resumed run rebuilds each step.

1. **Sizes.** Entities log-uniform 100-2000 (the latent pool; only
   entities that end up in a fact are materialized, so num_nodes is the
   participating count). Relations uniform 8-64. Types uniform 4-16;
   every entity gets one type, type masses drawn uniform 0.5-2.0.
   Each relation gets a head-type set and a tail-type set (1-3 types).
2. **Rules.** Per system, uniform counts of: hierarchy r2(x,y) <- r1(x,y)
   (1-4), inversion r2(y,x) <- r1(x,y) (1-3), symmetry r1(y,x) <- r1(x,y)
   (1-3), composition rh(x,z) <- r1(x,y) AND r2(y,z) [AND r3(z,w)] with
   chain length 2 or 3 (1-4). Each rule carries a confidence uniform
   0.6-0.95. Body relations are drawn by the base-fact relation
   frequencies (a rule whose premises sit on a fact-poor relation derives
   nothing; unweighted drawing left the graphs ~3% derived, weighted they
   run ~15-35%). Composition bodies are join-compatible by construction,
   and signatures are propagated to a fixpoint so every derivable fact
   provably respects its relation's (expanded) signature.
3. **Base facts.** Count = entities x uniform(0.8, 1.6), capped at 900.
   Relation frequencies Zipf (exponent 0.5-1.0 over a random relation
   order); entity popularity Zipf (exponent 0.7-1.1 over a random entity
   order), applied within the type-eligible pools for each relation's
   head and tail -- a few hubs, a long tail, no signature violations.
4. **Forward chaining.** At most 3 iterations. Each candidate derivation
   (rule, fact) is decided ONCE at the rule's confidence -- rejected
   derivations stay rejected, so firing rates match confidences instead
   of inflating over iterations. Total facts cap at 1400 per instance and
   4000 candidates per rule per iteration (hub-heavy composition joins go
   quadratic); truncations take sorted prefixes, so the closure stays a
   pure function of the seed. The caps ARE the semantics: "derivable"
   means derivable by this bounded chainer, which is what the
   verification tests re-run.
5. **Incompleteness and the query.** A uniform 10-40% of the noisily
   derived facts is dropped. The query positive is drawn from the facts
   the deterministic bounded chainer derives from the OBSERVED graph but
   which are absent from it -- dropped facts whose premises survived,
   plus confidence-suppressed derivations. Either way the label is
   certain: the rules derive it from what the model sees. Negatives are
   type-consistent tails (same head, same relation) that participate in
   the graph (degree-0 negatives would be trivially rejectable), are not
   derivable from the observed graph, and additionally are not derivable
   from the full base closure -- so no negative is a true fact whose
   evidence happened to be dropped. `synth.neg_per_pos_rules` (default 1)
   sets the negatives per positive; rows are (head, tail, relation) with
   row 0 positive, so union_batch and synth_loss run unchanged and the
   loss reads the negative count off the query tensor.

Sizes at synth.seed 2048: first instance 791 nodes, 64 relations, 1187
direct edges (base 884, derived kept 303, dropped 174, query pool 347);
the union of 16 instances is 20,842 edges (inverses included), 4,890
nodes. Hard worst case 16 x 1400 x 2 = 44.8k, under the ~50k training
budget. Generation runs ~8 ms/instance on CPU (~0.13 s per synthetic
step), noise next to a training step.

## What it models, and what it does not

Models: relational rules of the families real ontologies exhibit
(composition, hierarchy, inversion, symmetry), typed relation signatures,
degree skew on entities AND relations, rule noise (confidences), and
incompleteness -- with the training queries exactly the completion task
KGFMs are evaluated on: infer a missing fact the observed graph entails.

Does not model: textual semantics (no names, no descriptions), attribute
literals, temporal structure, or cross-instance vocabulary consistency
(relation 3 in two instances are unrelated latent relations). Textual
semantics is fine to omit for THIS model class: INCITE, like
TRIX/ULTRA, sees only (edge_index, edge_type) -- its entire inductive
bias is structural, so a structural prior spans its whole input space.
Not modeling negation/disjointness means negatives are "not derivable"
rather than "provably false"; with type-consistency and the base-closure
screen this is the standard local-closed-world assumption real KG
training also makes.

## Known risks

* **Prior mis-specification.** The sweep measures transfer from THIS
  prior, not from "synthetic data" in general. If DEV10 transfer is
  poor, the honest conclusions divide: the prior's rule mix/ranges are
  wrong, or no rule-system prior at this scale teaches what real graphs
  teach. The sweep's 25/75 points exist to separate "synthetic replaces
  real" from "synthetic adds to real".
* **Relation-state crosstalk in the union.** The union shares one
  relation vocabulary across k instances (the documented petals
  approximation). It is heavier here: relation ids collide across
  UNRELATED rule systems, so the factorized relation step mixes their
  bulk terms. Both candidates of a query still sit in one component and
  see the same relation states. If the prior underperforms, the first
  mitigation to try is per-instance relation-id offsets in the union
  (disjoint blocks; costs relation-state width, max 16 x 64 x 2).
* **Bounded-chainer semantics.** Facts needing >3 iterations or beyond
  the caps count as non-derivable; a negative could be "true" under
  unbounded chaining. Consistent within the prior (tests re-run the same
  chainer), but a modeling choice, not a theorem about logic.
* **One query per instance.** Each instance supervises one (head,
  relation) completion; the remaining dropped facts are unexploited
  incompleteness. More queries per instance would need per-row
  head/relation variation, which the model's entity interface forbids
  within one row-set -- more supervision per step means more instances,
  not wider rows.
