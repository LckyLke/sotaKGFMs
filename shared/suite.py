"""Frozen definition of the 54-graph zero-shot evaluation suite.

Single source of truth. No dataset list may be hardcoded anywhere else in the
project -- containers, patches, metric code and analysis all read from here.

Deliberately dependency-free (stdlib only, no torch, no numpy, no pandas) so
that it can be imported unchanged inside every model container regardless of
which framework stack that container pins.

------------------------------------------------------------------------------
Where the number 54 comes from
------------------------------------------------------------------------------
The ULTRA repository packs 57 knowledge graphs: 16 transductive, 18 inductive
(entity), 23 inductive (entity, relation).  The reference checkpoint
``ultra_3g.pth`` is *pre-trained* on three of the transductive graphs --
FB15k237, WN18RR and CoDExMedium -- so those three are not zero-shot and are
excluded.  57 - 3 = 54.

    group          n    contents
    -------------  ---  --------------------------------------------------
    ind_e          18   GraIL x3 families x v1-v4, ILPC 2022 x2, HM x4
    ind_er         23   Ingram x13, MTDEA x10
    transductive   13   16 packed transductive graphs minus the 3 pre-training
                        graphs (FB15k237, WN18RR, CoDExMedium)

------------------------------------------------------------------------------
Family assignment
------------------------------------------------------------------------------
``family`` is the *source knowledge base* the inference graph is drawn from,
used only for grouping in analysis.  The rule, applied uniformly:

    FB      Freebase-derived   (GraIL FB, Ingram FB, sparse FB15k237_*,
                                Hamaguchi/INDIGO HM -- all built from FB15k-237)
    WN      WordNet            (GraIL WN)
    NELL    NELL               (GraIL NELL, Ingram NL, NELL995, NELL23k)
    CoDEx   CoDEx              (its own Wikidata-derived benchmark family)
    WK      Wikidata           (Ingram WK, ILPC 2022, MTDEA WikiTopics, WDsinger)
    other   everything else, plus any graph that spans two source KBs
            (FBNELL is Freebase-train -> NELL-inference, so it is neither;
             Metafam is synthetic; YAGO310/Hetionet/ConceptNet100k/DBpedia100k/
             AristoV4 are each their own source)

------------------------------------------------------------------------------
Tail-only graphs
------------------------------------------------------------------------------
Five transductive graphs are evaluated on tail prediction only, per the ULTRA
README ("only tail evaluation"): WDsinger, NELL23k, FB15k237_10, FB15k237_20,
FB15k237_50.  On these, containers emit ``direction == "tail"`` rows only and
metrics are computed over those rows alone.

------------------------------------------------------------------------------
``id`` vs ``run_id``
------------------------------------------------------------------------------
``id`` is the canonical name of the graph in this project; it is what appears in
``ranks/<model>/<id>.parquet`` and in every report.

``run_id`` is the string to hand to ULTRA's ``run_many.py -d``.  They differ for
exactly two graphs.  ``run_many.py`` only sets the ``version`` template variable
when the ``-d`` entry contains a colon; otherwise jinja renders the unset
variable as the *string* ``"None"``, and ``MTDEAInductive.__init__`` asserts
``version in self.versions`` before anything can normalise it.  So bare
``-d Metafam`` and ``-d FBNELL`` raise AssertionError upstream and must be
spelled ``Metafam:Metafam`` and ``FBNELL:FBNELL_v1``.  See patches/ultra.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, Optional, Tuple

__all__ = [
    "Graph",
    "GRAPHS",
    "GROUPS",
    "FAMILIES",
    "RANK_COLUMNS",
    "RANK_COLUMN_TYPES",
    "by_id",
    "by_run_id",
    "ids",
    "of_group",
    "of_family",
    "tail_only_ids",
    "run_arg",
]

GROUPS: Tuple[str, ...] = ("ind_e", "ind_er", "transductive")
FAMILIES: Tuple[str, ...] = ("FB", "WN", "NELL", "CoDEx", "WK", "other")

#: Column order of every ``ranks/<model>/<dataset>.parquet`` file.  One row per
#: scored query.  Every container writes exactly this schema and nothing else --
#: no container computes a reported metric.
RANK_COLUMNS: Tuple[str, ...] = (
    "dataset",       # suite id, e.g. "FB15k237Inductive:v1"
    "model",         # container/model name, e.g. "ultra"
    "seed",          # RNG seed of the run
    "direction",     # "head" or "tail"
    "query_id",      # index of the query triple within the split, 0-based, stable
    "h",             # head entity id   (model-internal integer id)
    "r",             # relation id      (model-internal integer id)
    "t",             # tail entity id   (model-internal integer id)
    "rank",          # filtered rank of the true answer, 1-based (see metrics.py)
    "n_candidates",  # number of surviving filtered candidates, target excluded
)

#: Logical types for RANK_COLUMNS, as plain strings so that this module stays
#: free of any array/dataframe dependency.  Writers must match these exactly.
RANK_COLUMN_TYPES: Dict[str, str] = {
    "dataset": "string",
    "model": "string",
    "seed": "int64",
    "direction": "string",
    "query_id": "int64",
    "h": "int64",
    "r": "int64",
    "t": "int64",
    "rank": "int64",
    "n_candidates": "int64",
}


@dataclasses.dataclass(frozen=True)
class Graph:
    """One evaluation graph."""

    id: str
    dataset: str                 # ULTRA dataset class name
    version: Optional[str]       # ULTRA version string, None for transductive
    group: str                   # one of GROUPS
    family: str                  # one of FAMILIES
    tail_only: bool
    config: str                  # "inductive" | "transductive" -- ULTRA yaml
    run_id: str                  # what to pass to run_many.py -d

    @property
    def task_name(self) -> str:
        return "InductiveInference" if self.config == "inductive" else "TransductiveInference"


def _g(dataset, version, group, family, tail_only=False, run_version=None):
    config = "transductive" if group == "transductive" else "inductive"
    gid = dataset if version is None else "{}:{}".format(dataset, version)
    rv = run_version if run_version is not None else version
    run_id = dataset if rv is None else "{}:{}".format(dataset, rv)
    return Graph(
        id=gid,
        dataset=dataset,
        version=version,
        group=group,
        family=family,
        tail_only=tail_only,
        config=config,
        run_id=run_id,
    )


GRAPHS: Tuple[Graph, ...] = (
    # ---------------------------------------------------------------- ind_e (18)
    # 12 GraIL datasets: new entities at inference, relation vocabulary unchanged
    _g("FB15k237Inductive", "v1", "ind_e", "FB"),
    _g("FB15k237Inductive", "v2", "ind_e", "FB"),
    _g("FB15k237Inductive", "v3", "ind_e", "FB"),
    _g("FB15k237Inductive", "v4", "ind_e", "FB"),
    _g("WN18RRInductive", "v1", "ind_e", "WN"),
    _g("WN18RRInductive", "v2", "ind_e", "WN"),
    _g("WN18RRInductive", "v3", "ind_e", "WN"),
    _g("WN18RRInductive", "v4", "ind_e", "WN"),
    _g("NELLInductive", "v1", "ind_e", "NELL"),
    _g("NELLInductive", "v2", "ind_e", "NELL"),
    _g("NELLInductive", "v3", "ind_e", "NELL"),
    _g("NELLInductive", "v4", "ind_e", "NELL"),
    # 2 ILPC 2022 (Wikidata)
    _g("ILPC2022", "small", "ind_e", "WK"),
    _g("ILPC2022", "large", "ind_e", "WK"),
    # 4 Hamaguchi / INDIGO benchmarks, built from FB15k-237
    _g("HM", "1k", "ind_e", "FB"),
    _g("HM", "3k", "ind_e", "FB"),
    _g("HM", "5k", "ind_e", "FB"),
    _g("HM", "indigo", "ind_e", "FB"),

    # --------------------------------------------------------------- ind_er (23)
    # 13 Ingram datasets: new entities *and* new relations at inference
    _g("FBIngram", "25", "ind_er", "FB"),
    _g("FBIngram", "50", "ind_er", "FB"),
    _g("FBIngram", "75", "ind_er", "FB"),
    _g("FBIngram", "100", "ind_er", "FB"),
    _g("WKIngram", "25", "ind_er", "WK"),
    _g("WKIngram", "50", "ind_er", "WK"),
    _g("WKIngram", "75", "ind_er", "WK"),
    _g("WKIngram", "100", "ind_er", "WK"),
    _g("NLIngram", "0", "ind_er", "NELL"),
    _g("NLIngram", "25", "ind_er", "NELL"),
    _g("NLIngram", "50", "ind_er", "NELL"),
    _g("NLIngram", "75", "ind_er", "NELL"),
    _g("NLIngram", "100", "ind_er", "NELL"),
    # 10 MTDEA datasets
    _g("WikiTopicsMT1", "tax", "ind_er", "WK"),
    _g("WikiTopicsMT1", "health", "ind_er", "WK"),
    _g("WikiTopicsMT2", "org", "ind_er", "WK"),
    _g("WikiTopicsMT2", "sci", "ind_er", "WK"),
    _g("WikiTopicsMT3", "art", "ind_er", "WK"),
    _g("WikiTopicsMT3", "infra", "ind_er", "WK"),
    _g("WikiTopicsMT4", "sci", "ind_er", "WK"),
    _g("WikiTopicsMT4", "health", "ind_er", "WK"),
    # Metafam and FBNELL are single-version; upstream cannot parse them bare,
    # hence the explicit run_version (see module docstring).
    _g("Metafam", None, "ind_er", "other", run_version="Metafam"),
    _g("FBNELL", None, "ind_er", "other", run_version="FBNELL_v1"),

    # --------------------------------------------------- transductive (13)
    # the 16 packed transductive graphs minus FB15k237, WN18RR, CoDExMedium,
    # which ultra_3g.pth is pre-trained on
    _g("NELL995", None, "transductive", "NELL"),
    _g("YAGO310", None, "transductive", "other"),
    _g("CoDExSmall", None, "transductive", "CoDEx"),
    _g("CoDExLarge", None, "transductive", "CoDEx"),
    _g("Hetionet", None, "transductive", "other"),
    _g("ConceptNet100k", None, "transductive", "other"),
    _g("DBpedia100k", None, "transductive", "other"),
    _g("AristoV4", None, "transductive", "other"),
    # 5 sparse graphs, tail prediction only
    _g("WDsinger", None, "transductive", "WK", tail_only=True),
    _g("NELL23k", None, "transductive", "NELL", tail_only=True),
    _g("FB15k237_10", None, "transductive", "FB", tail_only=True),
    _g("FB15k237_20", None, "transductive", "FB", tail_only=True),
    _g("FB15k237_50", None, "transductive", "FB", tail_only=True),
)

_BY_ID: Dict[str, Graph] = {g.id: g for g in GRAPHS}
_BY_RUN_ID: Dict[str, Graph] = {g.run_id: g for g in GRAPHS}


def by_id(graph_id: str) -> Graph:
    """Look a graph up by its canonical suite id. Raises KeyError if unknown."""
    try:
        return _BY_ID[graph_id]
    except KeyError:
        raise KeyError(
            "{!r} is not in the suite; known ids: {}".format(graph_id, ", ".join(sorted(_BY_ID)))
        ) from None


def by_run_id(run_id: str) -> Graph:
    """Look a graph up by the string passed to ``run_many.py -d``.

    Runners know their own ``-d`` argument, not the canonical suite id, and the
    two differ for Metafam and FBNELL. Rank dumps must be keyed by ``Graph.id``,
    so every runner-side lookup goes through here.
    """
    try:
        return _BY_RUN_ID[run_id]
    except KeyError:
        return by_id(run_id)


def of_group(group: str) -> Tuple[Graph, ...]:
    if group not in GROUPS:
        raise ValueError("unknown group {!r}, expected one of {}".format(group, GROUPS))
    return tuple(g for g in GRAPHS if g.group == group)


def of_family(family: str) -> Tuple[Graph, ...]:
    if family not in FAMILIES:
        raise ValueError("unknown family {!r}, expected one of {}".format(family, FAMILIES))
    return tuple(g for g in GRAPHS if g.family == family)


def ids(group: Optional[str] = None) -> Tuple[str, ...]:
    """Canonical ids, optionally restricted to one group, in suite order."""
    source: Iterable[Graph] = GRAPHS if group is None else of_group(group)
    return tuple(g.id for g in source)


def tail_only_ids() -> Tuple[str, ...]:
    return tuple(g.id for g in GRAPHS if g.tail_only)


def is_tail_only(graph_id: str) -> bool:
    return by_id(graph_id).tail_only


def run_arg(group: str) -> str:
    """The comma-joined string for ``run_many.py -d`` for a whole group."""
    return ",".join(g.run_id for g in of_group(group))


# ---------------------------------------------------------------------------
# Invariants. These run at import so a bad edit fails loudly and immediately.
# ---------------------------------------------------------------------------
def _check() -> None:
    assert len(GRAPHS) == 54, "suite must hold 54 graphs, found {}".format(len(GRAPHS))
    assert len(_BY_ID) == 54, "duplicate graph id in suite"
    assert len(_BY_RUN_ID) == 54, "duplicate run_id in suite"
    counts = {g: len(of_group(g)) for g in GROUPS}
    assert counts == {"ind_e": 18, "ind_er": 23, "transductive": 13}, counts
    assert len(tail_only_ids()) == 5, tail_only_ids()
    assert set(tail_only_ids()) == {
        "WDsinger", "NELL23k", "FB15k237_10", "FB15k237_20", "FB15k237_50"
    }, tail_only_ids()
    for g in GRAPHS:
        assert g.group in GROUPS, g
        assert g.family in FAMILIES, g
        assert g.config in ("inductive", "transductive"), g
        # only transductive graphs may be tail-only, and only they are versionless
        if g.tail_only:
            assert g.group == "transductive", g
    assert len(RANK_COLUMNS) == len(RANK_COLUMN_TYPES) == 10
    assert set(RANK_COLUMNS) == set(RANK_COLUMN_TYPES)


_check()


if __name__ == "__main__":  # pragma: no cover - convenience for shell use
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in GROUPS:
        print(run_arg(sys.argv[1]))
    else:
        for _g_ in GRAPHS:
            print("{:<28} {:<13} {:<6} {}".format(
                _g_.id, _g_.group, _g_.family, "tail-only" if _g_.tail_only else ""))
