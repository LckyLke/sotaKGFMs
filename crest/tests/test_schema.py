"""A CREST dump satisfies the shared rank schema, end to end on a toy graph.

What ``scripts/verify_rank_dump.py`` checks for the other models, checked
here at build time: ``suite.RANK_COLUMNS`` with the declared arrow types,
1-based ranks, ``rank <= n_candidates + 1`` on every row, and validation by
``shared/metrics.py`` itself -- including the new ``task="relation"`` path.

The ranking used is ``crest.model.compute_ranking``, the verbatim copy of
TRIX's: 1-based, pessimistic ties, strict filtering with the target excluded
from ``n_candidates``.
"""

import pyarrow
import pyarrow.parquet
import pytest
import torch

import metrics
import suite
from crest import bank as crest_bank
from crest.model import CRESTEntity, compute_ranking

from conftest import ToyEncoder, make_readout, make_toy_graph

SEED = 1024


def _strict_mask(filter_graph, u, r, target, direction, num_nodes):
    """Brute-force strict filtering: drop known answers and the target."""
    mask = torch.ones(num_nodes, dtype=torch.bool)
    ei, et = filter_graph.edge_index, filter_graph.edge_type
    if direction == "tail":
        mask[ei[1][(ei[0] == u) & (et == r)]] = False
    else:
        mask[ei[0][(ei[1] == u) & (et == r)]] = False
    mask[target] = False
    return mask


def _toy_dump(tmp_path, dataset_id, direction_of, model_name="crest"):
    """Score toy test triples both ways and write a schema-true parquet."""
    graph = make_toy_graph()
    encoder = ToyEncoder()
    bank = crest_bank.build_bank_entity(graph, encoder, seed=SEED, num_positive=4)
    model = CRESTEntity(encoder, make_readout(seed=11, zero=False))
    num_direct = graph.num_relations // 2

    # "test" triples: existing inference edges reused as queries (the schema,
    # not generalisation, is under test); the filter graph is the inference
    # graph itself, which makes filtering non-trivial
    test = [(int(graph.edge_index[0, e]), int(graph.edge_type[e]), int(graph.edge_index[1, e]))
            for e in range(0, 8) if int(graph.edge_type[e]) < num_direct]
    columns = {name: [] for name in suite.RANK_COLUMNS}
    candidates = torch.arange(graph.num_nodes)
    for qid, (u, r, v) in enumerate(test):
        for direction in direction_of:
            if direction == "tail":
                scores = model(graph, u, r, candidates, bank)
                target = v
                mask = _strict_mask(graph, u, r, v, "tail", graph.num_nodes)
            else:
                # head query through the inverse relation id, TRIX-style
                scores = model(graph, v, r + num_direct, candidates, bank)
                target = u
                mask = _strict_mask(graph, v, r, u, "head", graph.num_nodes)
            rank = compute_ranking(scores.unsqueeze(0),
                                   torch.tensor([target]), mask.unsqueeze(0))
            columns["dataset"].append(dataset_id)
            columns["model"].append(model_name)
            columns["seed"].append(SEED)
            columns["direction"].append(direction)
            columns["query_id"].append(qid)
            columns["h"].append(u)
            columns["r"].append(r)
            columns["t"].append(v)
            columns["rank"].append(int(rank))
            columns["n_candidates"].append(int(mask.sum()))

    arrow_types = {"string": pyarrow.string(), "int64": pyarrow.int64()}
    schema = pyarrow.schema(
        [(name, arrow_types[suite.RANK_COLUMN_TYPES[name]]) for name in suite.RANK_COLUMNS])
    table = pyarrow.table({n: columns[n] for n in suite.RANK_COLUMNS}, schema=schema)
    path = str(tmp_path / (dataset_id.replace(":", "_") + ".parquet"))
    pyarrow.parquet.write_table(table, path)
    return path, columns


def test_entity_dump_passes_shared_validation(tmp_path):
    path, columns = _toy_dump(tmp_path, "FB15k237Inductive:v1", ("tail", "head"))
    data = metrics.read_ranks(path)  # raises on any schema violation
    assert list(data) == list(suite.RANK_COLUMNS)
    assert data["rank"].min() >= 1
    assert (data["rank"] <= data["n_candidates"] + 1).all()
    # arrow types exactly as declared, not merely convertible
    table = pyarrow.parquet.read_table(path)
    for name in suite.RANK_COLUMNS:
        expected = suite.RANK_COLUMN_TYPES[name]
        actual = str(table.schema.field(name).type)
        assert (expected == "string" and actual in ("string", "large_string")) or \
            expected == actual, (name, expected, actual)
    # and the metrics module consumes it whole
    result = metrics.compute_file(path)
    assert result["n"] == len(columns["rank"])


def test_ranks_are_pessimistic_and_one_based():
    # a perfect prediction ranks 1; k equal-scoring survivors rank k + 1
    scores = torch.tensor([[5.0, 5.0, 3.0, 5.0, 1.0]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    mask[0, 0] = False  # the target itself is always masked
    assert int(compute_ranking(scores, torch.tensor([0]), mask)) == 3
    lone = torch.tensor([[9.0, 1.0, 2.0, 3.0, 4.0]])
    assert int(compute_ranking(lone, torch.tensor([0]), mask)) == 1


def test_relation_dump_validates_under_the_task_argument(tmp_path):
    # same schema, direction == "relation" on every row; accepted by
    # task="relation" and rejected by the entity default
    graph = make_toy_graph()
    num_direct = graph.num_relations // 2
    columns = {name: [] for name in suite.RANK_COLUMNS}
    for qid in range(4):
        u = int(graph.edge_index[0, qid])
        v = int(graph.edge_index[1, qid])
        r = int(graph.edge_type[qid]) % num_direct
        columns["dataset"].append("FB15k237Inductive:v1")
        columns["model"].append("crest")
        columns["seed"].append(SEED)
        columns["direction"].append("relation")
        columns["query_id"].append(qid)
        columns["h"].append(u)
        columns["r"].append(r)
        columns["t"].append(v)
        columns["rank"].append(qid % num_direct + 1)
        columns["n_candidates"].append(num_direct - 1)
    arrow_types = {"string": pyarrow.string(), "int64": pyarrow.int64()}
    schema = pyarrow.schema(
        [(name, arrow_types[suite.RANK_COLUMN_TYPES[name]]) for name in suite.RANK_COLUMNS])
    path = str(tmp_path / "relation.parquet")
    pyarrow.parquet.write_table(
        pyarrow.table({n: columns[n] for n in suite.RANK_COLUMNS}, schema=schema), path)

    data = metrics.read_ranks(path, task="relation")
    assert set(data["direction"]) == {"relation"}
    with pytest.raises(ValueError, match="direction"):
        metrics.read_ranks(path)  # the entity default must refuse it
