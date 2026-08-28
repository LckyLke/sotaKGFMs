# INCITE — original design (author: Luke Friedrichs, 2026-08-28)

This is the design text as written, kept verbatim as the source document.
The harness adjustments live in `docs/INCITE_PLAN.md`; where the two
disagree, the PLAN wins. Sections A–E below are the normative model spec.

---

**Answer in one line.** Build one message-passing model on the triple
incidence graph, feed it a retrieved support set with hard negatives, add
short anonymized walks for symmetry breaks, and train it on entity and
relation prediction at the same time.

## 1. What the numbers say

The paper averages come from different graph sets. Only KGPFN's Table 1 puts
all families on one set of 57 graphs. Zero-shot MRR: ULTRA 0.367, MOTIF
0.370, KG-ICL 0.382, TRIX 0.388. Fine-tuned MRR: TRIX 0.418, KG-ICL 0.430.
KGPFN with in-context inference only: 0.432 MRR and 0.628 Hits@10.

Two facts follow. In-context learning alone does not win: zero-shot KG-ICL
(0.382) loses to zero-shot TRIX (0.388). Labeled triples of the query
relation, read from the target graph with negatives, do win: KGPFN beats
zero-shot TRIX by 0.044 MRR and 0.077 Hits@10. FLOCK holds the second
lever, relation prediction: 0.881 MRR against 0.792 for TRIX on 54 graphs.

## 2. Where each family stops

- Relation-graph models (ULTRA, MOTIF, ULTRA+, GAMMA, TRIX) fix the rule
  weights per structural signature. A 2026 analysis shows the cost: the
  known r-tails of the head outscore the true target on up to 91% of
  seen-query cases on WDsinger. The weights must come from the target graph.
- KGPFN uses an ULTRA-level relation encoder, collects 20 positives with no
  similarity rule, does no relation prediction, and reports no cost.
- TRIX costs O(|E| + |V|α²) per query and is about 10 times slower than
  ULTRA. It trains two models.
- FLOCK needs 27.9 GB per training batch, loses on dense graphs, and also
  ships one checkpoint per task (`flock_entity.pth`, `flock_relation.pth`).
- Every deterministic model scores 50% on PETALS.

## 3. The design (working name INCITE)

**A. One network on the incidence graph.** Treat each triple as a node with
three ports: head, relation, tail. One round has two steps. Step 1 updates
each entity from its triples. Step 2 updates each relation from its
triples, with the current entity states as edge features. This is the TRIX
alternation without the tensor `A_R`.

Why it is exact and cheap. TRIX's relation layer, head-head role, computes
`m(r_j) = Σ_i Σ_v E_h[v,r_i]·E_h[v,r_j]·(z_{r_i} ⊙ W x_v)`. The message is
bilinear and the aggregation is a sum, so the double sum factorizes:
`m(r_j) = Σ_v E_h[v,r_j]·(W x_v ⊙ s_v)` with
`s_v = Σ_i E_h[v,r_i]·z_{r_i}`. Each sum touches every triple once. The
cost is O(|E|), not O(|V|α²), and the identity is exact. Set `W x_v = 1`
in a second channel and the same formula gives ULTRA's count message. Keep
both channels: the 2026 analysis shows entity-tagged edges hurt the
all-unseen case and entity-agnostic motifs help it.

**B. Support set with retrieval and hard negatives.** A support set is a
set of labeled triples of the query relation r. Build it in three steps:

1. Encode all entities once per graph with the unlabeled network.
2. Retrieve the K = 16 positives (u_i, r, v_i) whose head is closest to h
   in that space.
3. Take hard negatives x from the 3-hop ball of u_i with (u_i, r, x) not
   in G. Rank them by the current score.

Why retrieval. KGPFN's Theorem 1 shows the score is a kernel average over
the support set. The bias of such an estimator is at most L·δ, where L is
the Lipschitz constant of the rule confidence and δ is the distance to the
support heads. Random positives set δ to the width of the whole relation.
Retrieval sets δ to the nearest neighbors. KGPFN's Hassabis and Li example
is a case with L > 0.

Why hard negatives. The confidence of a rule ρ → r is P(r | ρ). Positives
estimate the numerator. Only negatives that satisfy ρ estimate the
denominator. On WN18RR (40,943 entities, mean degree 4.2 with inverses), a
3-hop ball holds about 100 entities. A uniform negative lands in it with
probability 0.0024, so 60 uniform negatives contain about 0.15 informative
ones. Negatives from the ball of u_i cost nothing extra, because the pass
from u_i already scores them. Some hard negatives are true facts absent
from the graph. Down-weight them with the class prior, as in
positive-unlabeled learning.

**C. Walk-equality features.** Before round 1, sample n = 32 anonymized
walks of length 8 from h (and from t for relation prediction), with FLOCK's
recording protocol. Encode each walk with a small GRU. Pool the outputs
into the states of the visited entities and relations.

Why this solves PETALS without the coverage cost. The two candidates a₁ and
a₂ sit in one petal. Deterministic relation invariants force z(r₁) = z(r₂).
With that equality, the swap of a₁ and a₂ along their chains is an
automorphism of the colored graph, so message passing gives them equal
states in every round. A walk record keeps relation identity inside the
walk: b₀→a₁→a₃ records (α, α), b₀→a₂→a₄ records (α, β). A length-2 walk
separates them. Message passing still covers the full neighborhood, so
FLOCK's coverage limit does not apply. FLOCK's own control shows
unstructured noise is not enough: TRIX plus noise gets 52% on PETALS, and
its relation MRR falls from 0.792 to 0.739.

**D. One model for both tasks.** Add a task flag to the label vector. Label
(h, r) as in TRIX for entity prediction. Label h with +1, t with −1 and all
relations with 1 for relation prediction, so one pass scores all relations.
Score relation r with two terms: the TRIX head on (p(h,t), z_r), plus the
dot product of p(h,t) with a prototype c_r. The prototype is the mean pair
state of K support pairs of r, computed once per graph with the direct edge
removed. Train with L = L_entity + λ·L_relation.

Why. A rule ρ → r serves both query types, so a shared encoder receives two
gradient signals per pattern. Chen et al. (2021) showed relation prediction
as an auxiliary loss improves link prediction. A softmax over relations
penalizes the collapse that MOTIF observed on WN-v2 (0.296 against 0.684).
A nearest-prototype classifier is Bayes-optimal for class densities in a
regular exponential family with shared dispersion (Snell et al., 2017). No
model in the set trains one network for both tasks.

**E. Training recipe.** Pretrain on FB15k-237, WN18RR and CoDEx-M for a
fair comparison. Then scale to 8 graphs, because TRIX, KG-ICL and FLOCK all
gain from more pretraining graphs. Use self-adversarial negatives in the
loss. A 2026 study (KMAS) reports that stronger negative sampling lifts
zero-shot MRR for ULTRA, TRIX, MOTIF, SEMMA and FLOCK.

## 4. Cost per query, in edge messages

- ULTRA: 6 layers × |E|, plus a small relation graph. About 6|E|.
- TRIX: 5 rounds × (|E| + 4 to 10 |E|). About 25|E| to 55|E|.
- INCITE: 6 rounds × 3|E| = 18|E|, plus 256 walk steps, plus attention over
  at most 80 support rows.

So INCITE costs about 3 times ULTRA and 1.4 to 3 times less than TRIX.
Support heads are encoded once per graph, capped at 64 per relation. On
FB15k-237 that is about 15,000 passes, or 0.75 of the test set.

## 5. Test plan and targets

1. Run the 57-graph set with the three standard pretraining graphs and
   five seeds.
2. Run FLOCK's 54-graph relation task and report Hits@1.
3. Run PETALS and Metafam.
4. Report the four half-link scenarios beside the average.
5. Ablate each part: `A_R` instead of the incidence graph, random instead
   of retrieved positives, uniform instead of hard negatives, no walks, no
   relation loss.

Targets, as hypotheses: zero-shot entity MRR ≥ 0.45 on 57 graphs, relation
MRR ≥ 0.88 on 54 graphs, cost ≤ 3× ULTRA. The entity target assumes the
TRIX-over-ULTRA gain (0.021 on this set) adds to KGPFN's 0.432. That
additivity is unproven. The comparison against zero-shot baselines means no
gradient update, not no target labels; state this as KGPFN did.

## 6. Risks

- The PFN attention may already absorb part of the structural gain. Then
  the entity target drops toward 0.44.
- Walk features add variance. Average 4 passes at test time.
- Relation text (Semma) is an orthogonal lever. It breaks the pure
  structural setting, so keep it out of the core model.
