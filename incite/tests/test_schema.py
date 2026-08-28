"""Parquet schema equality against the existing shared-schema dumps.

An INCITE entity dump goes through the SAME writer TRIX used
(``trix.rank_dump.Dumper``, patches/trix/0001), so schema equality is
inherited -- but inherited claims rot, so this test writes a dump through
that writer and compares it FIELD BY FIELD against ranks/trix/Metafam.parquet
(name order, arrow types), and validates it with shared/metrics.py. The
relation-dump path of incite/run.py is exercised the same way.
"""

import os

import pyarrow
import pyarrow.parquet
import pytest
import torch

import metrics
import suite
from conftest import REPO

TRIX_PARQUET = os.path.join(REPO, "ranks", "trix", "Metafam.parquet")
SEED = 1024


def _reference_schema():
    if not os.path.exists(TRIX_PARQUET):
        pytest.skip("no reference dump at %s" % TRIX_PARQUET)
    return pyarrow.parquet.read_table(TRIX_PARQUET).schema


def _write_entity_dump(tmp_path):
    """A tiny dump through the shared Dumper (the writer incite/run.py uses)."""
    pytest.importorskip("trix.rank_dump")
    from trix.rank_dump import Dumper
    triples = torch.tensor([[0, 1, 0], [2, 3, 1], [4, 5, 0], [6, 7, 2]])
    spec = {"dir": str(tmp_path), "dataset": "Metafam:Metafam",
            "model": "incite", "seed": SEED}
    dumper = Dumper(spec, triples, 1, 0, batch_size=2)
    n = 9  # pretend candidate count
    for start in (0, 2):
        batch = triples[dumper.cursor:dumper.cursor + 2]
        t_rank = torch.tensor([1, 3])
        h_rank = torch.tensor([2, 1])
        negs = torch.tensor([n, n])
        dumper.add(batch, t_rank, h_rank, negs, negs)
    return dumper.write()


def test_entity_dump_schema_equals_trix_reference(tmp_path):
    reference = _reference_schema()
    path = _write_entity_dump(tmp_path)
    ours = pyarrow.parquet.read_table(path).schema
    assert [f.name for f in ours] == [f.name for f in reference]
    for name in suite.RANK_COLUMNS:
        assert ours.field(name).type == reference.field(name).type, name
    # and the shared metrics module consumes it whole
    data = metrics.read_ranks(path)
    assert list(data) == list(suite.RANK_COLUMNS)
    assert data["rank"].min() >= 1
    assert (data["rank"] <= data["n_candidates"] + 1).all()


def test_reference_dump_matches_declared_types():
    """The reference itself matches suite.RANK_COLUMN_TYPES -- if this fails
    the schema moved under us and BOTH writers need looking at."""
    reference = _reference_schema()
    arrow_types = {"string": ("string", "large_string"), "int64": ("int64",)}
    assert [f.name for f in reference] == list(suite.RANK_COLUMNS)
    for name in suite.RANK_COLUMNS:
        expected = arrow_types[suite.RANK_COLUMN_TYPES[name]]
        assert str(reference.field(name).type) in expected, name


def test_relation_dump_schema(tmp_path):
    """incite/run.py's relation writer emits the same schema with
    direction == 'relation' and the unfiltered n_candidates."""
    reference = _reference_schema()
    num_direct = 4
    n = 5
    arrow_types = {"string": pyarrow.string(), "int64": pyarrow.int64()}
    table = {
        "dataset": ["Metafam"] * n, "model": ["incite"] * n,
        "seed": [SEED] * n, "direction": ["relation"] * n,
        "query_id": list(range(n)), "h": [0, 1, 2, 3, 4],
        "r": [0, 1, 2, 3, 0], "t": [5, 6, 7, 8, 9],
        "rank": [1, 2, 1, 4, 3], "n_candidates": [num_direct - 1] * n,
    }
    schema = pyarrow.schema([(name, arrow_types[suite.RANK_COLUMN_TYPES[name]])
                             for name in suite.RANK_COLUMNS])
    path = str(tmp_path / "Metafam.parquet")
    pyarrow.parquet.write_table(
        pyarrow.table({k: table[k] for k in suite.RANK_COLUMNS}, schema=schema), path)
    ours = pyarrow.parquet.read_table(path).schema
    for name in suite.RANK_COLUMNS:
        assert ours.field(name).type == reference.field(name).type, name
    data = metrics.read_ranks(path, task="relation")
    assert set(data["direction"]) == {"relation"}
    with pytest.raises(ValueError):
        metrics.read_ranks(path)  # the entity default must refuse it
