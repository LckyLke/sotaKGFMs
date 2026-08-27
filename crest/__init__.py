"""CREST: TRIX plus an in-context readout over banks of example triples.

The encoder is TRIX, imported unchanged from the patched tree that
``scripts/prepare_trix_workdir.sh`` produces. The readout reads a per-relation
bank of scored example triples and emits an additive residual on top of the
encoder's score, so that with the residual's last linear layer at zero the
model *is* TRIX -- that identity is phase 0's gate and this package's most
important invariant (see ``crest/tests/test_residual_zero.py``).

Nothing in this package computes a reported metric. Evaluation is
``shared/metrics.py`` over rank dumps in the shared schema, like every other
model in this harness; there is deliberately no ``crest/eval.py``.

Import discipline: every module here except ``crest/run.py`` (evaluation) and
``crest/pretrain.py`` (training driver) depends only on torch, so the test
suite runs on the host CPU without TRIX, PyG or a GPU. TRIX enters through
the adapter in ``crest/run.py``; both drivers are container-only.
"""

__all__ = ["bank", "pfn", "model", "messages", "randchan", "train"]
