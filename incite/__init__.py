"""INCITE: one factorized message-passing network on the triple incidence
graph, with retrieved support sets, anonymized walks, and joint task heads.

Spec: docs/INCITE_DESIGN.md (sections A-E), adjusted by docs/INCITE_PLAN.md
(the PLAN wins where they disagree). Layout mirrors crest/ on the crest
branch: the package is self-contained; only ``run.py`` and ``pretrain.py``
import the patched TRIX tree and therefore run in the container alone.
"""
