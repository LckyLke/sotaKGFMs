"""The one metric implementation in this project.

Reads ``ranks/<model>/<dataset>.parquet`` and returns MRR and Hits@1/3/10.
No container computes a reported metric; containers emit ranks and stop here.

Dependencies: pyarrow (or pandas) for parquet I/O, numpy for arithmetic.
Deliberately no torch -- this module runs outside every container.

==============================================================================
RANK DEFINITION -- what a ``rank`` column value means
==============================================================================
Reproduced from ULTRA at pin 427966ad (``ultra/tasks.py::compute_ranking``,
``ultra/tasks.py::strict_negative_mask``, ``script/run.py::test``)::

    def compute_ranking(pred, target, mask=None):
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        if mask is not None:
            ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        else:
            ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
        return ranking

and, inside ``strict_negative_mask``, for the tail direction::

    t_mask = torch.ones(batch, num_nodes, dtype=torch.bool)
    t_mask[sample_id, t_truth_index] = 0        # every true tail for this (h, r)
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)   # and the target itself

------------------------------------------------------------------------------
Rank offset:  1-based.
------------------------------------------------------------------------------
The ``+ 1`` makes a perfect prediction rank 1, never 0.  A rank of ``k`` means
``k - 1`` filtered-in candidates scored at least as high as the true answer.
Consequently ``1 <= rank <= n_candidates + 1``.

------------------------------------------------------------------------------
Tie rule:  PESSIMISTIC (worst case).
------------------------------------------------------------------------------
Two facts combine:

  1. The comparison is ``pos_pred <= pred``, i.e. non-strict.  A candidate whose
     score exactly equals the true answer's score counts *against* it.
  2. The target itself is masked out (the ``scatter_`` line above), so the
     target's own trivially-equal comparison contributes 0.

So if the true answer ties with ``k`` other surviving candidates and nothing
outscores it, the rank is ``k + 1`` -- the worst position it could occupy.  This
is neither the optimistic rule (rank 1) nor the average/"realistic" rule
(``(k + 2) / 2``).  Any implementation using those will disagree with ULTRA,
and disagreement grows with tie mass, which on sparse graphs is not negligible.

------------------------------------------------------------------------------
Filtering:  strict (a.k.a. "filtered" ranking).
------------------------------------------------------------------------------
``n_candidates`` is ``mask.sum(dim=-1)``: the number of entities that are
neither a known true answer for this (h, r) / (t, r) query in the filtering
graph, nor the target itself.  The target is therefore NOT counted in
``n_candidates``, which is why the rank can reach ``n_candidates + 1``.

------------------------------------------------------------------------------
Directions
------------------------------------------------------------------------------
ULTRA scores every test triple twice -- once predicting the tail from (h, r, ?)
and once predicting the head from (?, r, t) -- and takes metrics over the
concatenation of both.  On the five tail-only graphs (see shared/suite.py) only
the tail rows exist and metrics run over those alone.

==============================================================================
FLOATING POINT -- why float32 is the default
==============================================================================
ULTRA reduces in float32: ``_ranking`` is int64, ``.float()`` casts to float32,
and ``.mean()`` accumulates in float32 on the GPU.  This module mirrors that by
default (``dtype="float32"``) so that its output is comparable with ULTRA's
stock ``ultra_results_*.csv`` at printed precision.

What that buys, split by whether the metric's reduction is order-independent:

  * ``hits@k`` is bit-exact **by construction**.  The summands are exactly 0.0 or
    1.0, so the sum is an exact integer for any test set below 2**24 rows and
    the division is a single correctly-rounded operation.  Reduction order
    cannot change the answer, on any device, in any framework.
  * ``mrr``, ``mr`` and ``hits@k_M`` are **not** order-independent.  ``1 / rank``
    is inexact in binary, so a float32 sum depends on the order it is taken in,
    and numpy's pairwise summation is neither torch's CPU cascade nor CUDA's
    block-tree reduction.  They agree exactly on most datasets and differ by one
    float32 ulp (~6e-8 relative) on some.

    Where they differ, this module is not the one that is wrong.  Measured on
    FB15k237Inductive:v2 (1894 queries): ULTRA reports 0.5005503296852112, this
    module returns 0.5005502700805664, and the *correctly rounded* float32 value
    of the exact sum -- via ``math.fsum`` over the same float32 summands -- is
    0.5005502700805664.  ULTRA's own accumulation is the one that lands an ulp
    off.  Matching it would mean reimplementing ``torch.Tensor.mean``'s internal
    blocking, which is device- and version-specific and would not transfer to
    the GPU numbers anyway.  So the residual stands, documented and bounded, at
    five orders of magnitude inside the +/-0.002 acceptance band.

One trap, since it cost a wrong answer once already: numpy promotes
``float32 / int64`` to **float64** while torch keeps float32.  Any expression
mixing a cast rank with a raw integer column must cast both operands explicitly,
or the whole chain silently runs at a precision ULTRA never used.  See
``hits_at_k_unbiased``.

Pass ``dtype="float64"`` for downstream analysis where reproducing ULTRA's
float32 rounding is not the point.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

try:  # keep suite.py authoritative for the schema
    from . import suite as _suite  # type: ignore
except ImportError:  # pragma: no cover - flat sys.path use, e.g. inside a container
    import suite as _suite  # type: ignore

__all__ = [
    "HITS_KS",
    "METRIC_NAMES",
    "read_ranks",
    "compute",
    "hits_at_k_unbiased",
    "compute_file",
    "compute_dir",
    "group_mean",
]

#: Hits@k thresholds this project reports.
HITS_KS: Sequence[int] = (1, 3, 10)

#: Reported metric names, in report order.
METRIC_NAMES: Sequence[str] = ("mrr",) + tuple("hits@{}".format(k) for k in HITS_KS)

_DTYPES = {"float32": np.float32, "float64": np.float64}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_ranks(path: str, validate: bool = True) -> Dict[str, np.ndarray]:
    """Read one rank parquet into a dict of column name -> numpy array.

    Validates the file against ``suite.RANK_COLUMNS`` unless ``validate=False``.
    """
    columns = list(_suite.RANK_COLUMNS)
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        missing = [c for c in columns if c not in table.column_names]
        if missing:
            raise ValueError("{}: missing rank columns {}".format(path, missing))
        data = {c: table.column(c).to_numpy(zero_copy_only=False) for c in columns}
    except ImportError:  # pragma: no cover - pandas fallback
        import pandas as pd

        frame = pd.read_parquet(path)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError("{}: missing rank columns {}".format(path, missing))
        data = {c: frame[c].to_numpy() for c in columns}

    if validate:
        _validate(path, data)
    return data


def _validate(path: str, data: Mapping[str, np.ndarray]) -> None:
    rank = data["rank"]
    n_cand = data["n_candidates"]
    if len(rank) == 0:
        raise ValueError("{}: no rows".format(path))
    if rank.min() < 1:
        raise ValueError("{}: rank {} < 1 -- ranks are 1-based".format(path, rank.min()))
    bad = int(np.count_nonzero(rank > n_cand + 1))
    if bad:
        raise ValueError(
            "{}: {} rows with rank > n_candidates + 1; the target must be excluded "
            "from n_candidates".format(path, bad)
        )

    directions = set(np.unique(data["direction"]).tolist())
    if not directions <= {"head", "tail"}:
        raise ValueError("{}: unexpected direction values {}".format(path, sorted(directions)))

    names = set(np.unique(data["dataset"]).tolist())
    if len(names) != 1:
        raise ValueError("{}: expected exactly one dataset, found {}".format(path, sorted(names)))
    name = names.pop()
    graph = _suite.by_id(name)
    if graph.tail_only:
        if directions != {"tail"}:
            raise ValueError(
                "{}: {} is tail-only, expected tail rows only, found {}".format(
                    path, name, sorted(directions)
                )
            )
    elif directions != {"head", "tail"}:
        raise ValueError(
            "{}: {} expects both head and tail rows, found {}".format(
                path, name, sorted(directions)
            )
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute(
    ranks: Iterable[int],
    dtype: str = "float32",
    hits_ks: Sequence[int] = HITS_KS,
) -> Dict[str, float]:
    """MRR, MR and Hits@k over one array of 1-based ranks.

    ``dtype="float32"`` reproduces ULTRA's own reduction precision; see the
    module docstring for what that does and does not guarantee.
    """
    if dtype not in _DTYPES:
        raise ValueError("dtype must be one of {}, got {!r}".format(sorted(_DTYPES), dtype))
    np_dtype = _DTYPES[dtype]

    rank = np.asarray(ranks)
    if rank.size == 0:
        raise ValueError("cannot compute metrics over zero ranks")
    if rank.min() < 1:
        raise ValueError("ranks are 1-based; found {}".format(rank.min()))

    out: Dict[str, float] = {}
    # mirrors: (1 / _ranking.float()).mean()
    out["mrr"] = float(np.mean((1.0 / rank.astype(np_dtype)), dtype=np_dtype))
    # mirrors: _ranking.float().mean()
    out["mr"] = float(np.mean(rank.astype(np_dtype), dtype=np_dtype))
    for k in hits_ks:
        # mirrors: (_ranking <= threshold).float().mean()
        out["hits@{}".format(k)] = float(
            np.mean((rank <= k).astype(np_dtype), dtype=np_dtype)
        )
    out["n"] = int(rank.size)
    return out


def hits_at_k_unbiased(
    ranks: Iterable[int],
    n_candidates: Iterable[int],
    k: int = 10,
    num_sample: int = 50,
    dtype: str = "float32",
) -> float:
    """ULTRA's ``hits@k_M`` -- Hits@k estimated against a sample of M candidates.

    Not one of this project's four reported metrics.  It is implemented here for
    one reason: it is the only quantity that consumes ``n_candidates``, so
    reproducing it is what proves that column of the rank schema is correct.
    Without it, a dump could get ``n_candidates`` wrong and criterion A would
    still pass.

    Mirrors ``script/run.py::test`` exactly, including the accumulation order of
    the ``i`` loop, which is what makes bitwise comparison meaningful::

        fp_rate = (_ranking - 1).float() / _num_neg
        score = 0
        for i in range(threshold):
            num_comb = factorial(M - 1) / factorial(i) / factorial(M - i - 1)
            score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (M - i - 1))
        score = score.mean()
    """
    np_dtype = _DTYPES[dtype]
    rank = np.asarray(ranks)
    # Both operands are cast explicitly. numpy promotes float32/int64 to float64
    # while torch keeps float32, so dividing by the raw int64 column would run
    # the whole chain below at a precision ULTRA never used.
    n_neg = np.asarray(n_candidates).astype(np_dtype)
    fp_rate = (rank - 1).astype(np_dtype) / n_neg
    score = np.zeros_like(fp_rate)
    for i in range(k):
        num_comb = (
            math.factorial(num_sample - 1)
            / math.factorial(i)
            / math.factorial(num_sample - i - 1)
        )
        score = score + np_dtype(num_comb) * (fp_rate ** i) * (
            (1 - fp_rate) ** (num_sample - i - 1)
        )
    return float(np.mean(score, dtype=np_dtype))


def compute_file(path: str, dtype: str = "float32",
                 rank_column: str = "rank") -> Dict[str, float]:
    """Metrics for one ``ranks/<model>/<dataset>.parquet``.

    ``rank_column`` exists for KG-ICL, which dumps two. ``rank`` is the shared
    definition every cross-model table reads; ``rank_native`` is what its own
    ``cal_ranks`` returned, and criterion A has to use that one, because the
    CSV it is compared against was computed from it.
    """
    data = read_ranks(path)
    if rank_column != "rank":
        import pyarrow.parquet as pq

        column = pq.read_table(path, columns=[rank_column]).column(rank_column)
        data = dict(data, rank=column.to_numpy(zero_copy_only=False))
    result = compute(data["rank"], dtype=dtype)
    result["hits@10_50"] = hits_at_k_unbiased(
        data["rank"], data["n_candidates"], k=10, num_sample=50, dtype=dtype
    )
    result["dataset"] = str(data["dataset"][0])
    result["model"] = str(data["model"][0])
    return result


def compute_dir(
    rank_dir: str,
    graph_ids: Optional[Iterable[str]] = None,
    dtype: str = "float32",
    rank_column: str = "rank",
) -> Dict[str, Dict[str, float]]:
    """Metrics for every graph in ``graph_ids`` found under ``rank_dir``.

    Missing files are omitted from the result rather than raising, so that a
    partial run still reports; the caller is responsible for noticing the gap.
    """
    wanted = list(graph_ids) if graph_ids is not None else list(_suite.ids())
    out: Dict[str, Dict[str, float]] = {}
    for gid in wanted:
        path = os.path.join(rank_dir, "{}.parquet".format(gid.replace(":", "_")))
        if not os.path.exists(path):
            legacy = os.path.join(rank_dir, "{}.parquet".format(gid))
            if not os.path.exists(legacy):
                continue
            path = legacy
        out[gid] = compute_file(path, dtype=dtype, rank_column=rank_column)
    return out


def group_mean(
    per_dataset: Mapping[str, Mapping[str, float]],
    metric: str,
) -> float:
    """Unweighted mean of ``metric`` across datasets.

    Unweighted is the point: every graph counts once, regardless of how many
    test queries it has.  This is how ULTRA's published group numbers are
    formed, and weighting by query count would let the few large graphs decide
    the headline.
    """
    values: List[float] = []
    for name in sorted(per_dataset):
        row = per_dataset[name]
        if metric not in row:
            raise KeyError("{}: no metric {!r}".format(name, metric))
        value = float(row[metric])
        if math.isnan(value):
            raise ValueError("{}: metric {!r} is NaN".format(name, metric))
        values.append(value)
    if not values:
        raise ValueError("no datasets to average over")
    return float(sum(values) / len(values))
