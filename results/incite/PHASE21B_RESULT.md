# Phase 2.1b result: supervision REVIVES the walk mechanic

Date: 2026-08-30. Setup: walks + ~5% synthetic automorphic instances
with labeled true tails in the pretraining mix (incite/synth.py), warm
start from the phase-1 floor, 20k steps, DEV10 best at step 5000.

## PETALS (the question this run existed to answer)

| model | 1-pass | 8-pass avg | ties | mean margin |
| --- | --- | --- | --- | --- |
| floor (deterministic) | 0.4545 | 0.4545 | 130/220 | 0.000000 |
| walks, unsupervised (2.1) | 0.4727 | 0.4136 | 0 | 0.2555 |
| walks + synth supervision | **0.8182** | **0.9364** | 0 | 2.4179 |

The 8-pass average clears the plan's 90% bar; the primary 1-pass
protocol reaches 82% from 47%. Margin grew tenfold. Conclusion: the
mechanic was never dead -- it lacked a gradient pointing at truth, and
~1,000 synthetic steps supplied it (diagnosis in PHASE21_RESULT.md
confirmed by intervention).

## Benchmark (ranks/incite-walksynth/)

ind_e 0.4512 (floor 0.4553), ind_er 0.3724 (floor 0.3740) -- within
noise, as the kill switch requires. Real graphs lack automorphic pairs,
so the capability cannot lift the average; it was never supposed to.

## Standing

The lever now does exactly what design C promised: solves the
symmetry-breaking diagnostic at negligible benchmark cost. Whether v1
carries it is a scope decision: it buys a capability class, not MRR.
Open observation for the ledger: this run trained at 1.02 it/s vs the
2.1 run's ~0.47 on identical configs -- unexplained, flagged.
