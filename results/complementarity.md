# Mechanism complementarity (trix, incite, incite-4g, flock)

Queries joined 1:1 on (dataset, query id, direction): 174438 over 40 graphs (graphs any model lacks are dropped)

## Group MRR (per-graph unweighted mean)
- ind_e: trix 0.4562 | incite 0.4553 | incite-4g 0.4542 | flock 0.4558
- ind_er: trix 0.3667 | incite 0.3730 | incite-4g 0.3786 | flock 0.3674

## Pairwise complementarity (hit@10 level)
- incite vs trix: only-incite 0.031, only-trix 0.035, both 0.496
- incite vs incite-4g: only-incite 0.025, only-incite-4g 0.029, both 0.502
- incite-4g vs trix: only-incite-4g 0.032, only-trix 0.033, both 0.499
- flock vs trix: only-flock 0.040, only-trix 0.033, both 0.498
- flock vs incite: only-flock 0.043, only-incite 0.031, both 0.495
- flock vs incite-4g: only-flock 0.043, only-incite-4g 0.036, both 0.495

## Oracle and trivial fusions (ENSEMBLES -- label them so)
- ind_e: oracle(min-rank) 0.5227, mean-RR fusion 0.4553
- ind_er: oracle(min-rank) 0.4521, mean-RR fusion 0.3714

## Where does KGPFN win vs TRIX (per graph, MRR delta)
