# Synthetic rules-prior pilot (P1): a KGFM trained on no real KG

Date: 2026-09-03. Setup: configs/incite_synthsweep_100.yaml -- every
training step is a union batch of 16 instances from the latent
rule-system prior (RULES_PRIOR.md: hierarchy, inversion, symmetry,
composition rules with confidences, typed signatures, Zipf degrees,
incompleteness; queries are derivable-but-missing facts). 10,000 steps,
constant lr 5e-4, walks on, batch 32. Training took 41 minutes (4.3
steps per second; real-graph steps run at 0.6). DEV10 best at step 6k.

## Benchmark (41 real graphs, test splits; ranks/incite-synth100-pilot)

| model | ind_e MRR | ind_er MRR |
| --- | --- | --- |
| synthetic only, 10k steps | 0.3593 | 0.2795 |
| ULTRA 3g (released) | 0.4158 | 0.3421 |
| INCITE 4g decay (reference) | 0.4560 | 0.3852 |

86 percent of ULTRA-3g on ind_e and 82 percent on ind_er, from a model
that never saw a real knowledge graph. The DEV10 curve was still rising
at 10k (0.2535 at 1k to 0.2871 at 10k, noisy).

## Per-scenario: the complement of the real-data profile

| ind_e | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| synthetic only | 0.3728 | 0.2434 | 0.4532 | 0.3078 |
| ULTRA 3g | 0.4581 | 0.2205 | 0.5871 | 0.2896 |
| INCITE 4g decay | 0.4929 | 0.2856 | 0.6221 | 0.3187 |

| ind_er | SQSA | SQUA | UQSA | UQUA |
| --- | --- | --- | --- | --- |
| synthetic only | 0.3004 | 0.1072 | 0.4262 | 0.2186 |
| ULTRA 3g | 0.3798 | 0.1141 | 0.5447 | 0.1914 |
| INCITE 4g decay | 0.4079 | 0.1724 | 0.5896 | 0.2032 |

On the seen-answer cells the synthetic model trails ULTRA by 0.08 to
0.13. On the unseen-answer cells it matches or beats ULTRA (SQUA ind_e
+0.023, UQUA +0.018 / +0.027) and sits near the reference on UQUA. The
prior's queries are rule-derived missing facts, so unseen-answer
positives are common and their targets are ordinary entities -- the
realistic unseen-answer supervision that half-link masking could not
manufacture from real graphs (MASKING_RESULT.md).

## Standing

Two results. (1) Synthetic-only pretraining transfers to real KGs at a
level no prior work reports, and at one seventh of the per-step cost.
(2) Its scenario profile complements real data. The mixing run (MX1:
the 4-graph continuation with 30 percent synthetic steps, paired against
L1) is queued next in scripts/research_plan_v7.sh. Risks unchanged
(RULES_PRIOR.md): prior mis-specification, relation-state crosstalk in
union batches.
