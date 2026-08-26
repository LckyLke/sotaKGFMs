"""Cross-container artifacts: the frozen suite, the one metric implementation, analysis.

Importable either as a package (``from shared import metrics``) or flat, with
``shared/`` on ``sys.path`` (``import suite``) -- containers use the flat form,
since each one pins its own interpreter and cannot depend on this repo's layout.
"""
