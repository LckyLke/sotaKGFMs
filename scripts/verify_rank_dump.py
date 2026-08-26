#!/usr/bin/env python
"""Prove the rank dump's query_id actually indexes the test split.

The dump attaches a stable ``query_id`` by rebuilding the loader's
``DistributedSampler`` order in a second, independent sampler (see
``ultra/rank_dump.py``). That is only sound if the two samplers agree. This
checks it the direct way: reload the dataset, index ``test_triplets`` by each
dumped ``query_id``, and require the triple there to be exactly the dumped
(h, r, t). Any disagreement means every dumped row is mislabelled.

Also re-derives the rank bounds: 1 <= rank <= n_candidates + 1, and the
head/tail row counts implied by the tail-only flag.

    usage: verify_rank_dump.py --ultra <patched tree> --root <processed root> \
                               --ranks ranks/ultra [--dataset FB15k237Inductive:v1 ...]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))

import suite  # noqa: E402


def check(graph, root, ranks_dir, datasets_module, torch):
    path = os.path.join(ranks_dir, graph.id.replace(":", "_") + ".parquet")
    if not os.path.exists(path):
        return None, "no rank file"

    import pyarrow.parquet as pq
    table = pq.read_table(path).to_pydict()

    cls = getattr(datasets_module, graph.dataset)
    # Construct the dataset the way run_many.py does: with the version out of
    # run_id, not out of Graph.version. The two differ for Metafam and FBNELL,
    # which suite.py records as version=None because their rank file is keyed by
    # the bare id. Their __init__ pops "version" unconditionally before choosing
    # versions[0], so cls(root=root) raises KeyError: 'version' instead of
    # defaulting. Every other graph has run_id version == Graph.version.
    run_version = graph.run_id.split(":", 1)[1] if ":" in graph.run_id else None
    dataset = cls(root=root, version=run_version) if run_version is not None else cls(root=root)
    test_data = dataset[2]
    test_triplets = torch.cat(
        [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]
    ).t()

    problems = []
    n = len(table["rank"])
    for i in range(n):
        qid = table["query_id"][i]
        if not (0 <= qid < len(test_triplets)):
            problems.append("row {}: query_id {} out of range".format(i, qid))
            break
        h, t, r = (int(x) for x in test_triplets[qid])
        if (h, r, t) != (table["h"][i], table["r"][i], table["t"][i]):
            problems.append(
                "row {}: query_id {} points at ({}, {}, {}) but dump says ({}, {}, {})".format(
                    i, qid, h, r, t, table["h"][i], table["r"][i], table["t"][i]))
            break
        if not (1 <= table["rank"][i] <= table["n_candidates"][i] + 1):
            problems.append("row {}: rank {} outside [1, n_candidates+1={}]".format(
                i, table["rank"][i], table["n_candidates"][i] + 1))
            break

    directions = set(table["direction"])
    expected = {"tail"} if graph.tail_only else {"head", "tail"}
    if directions != expected:
        problems.append("directions {} != expected {}".format(sorted(directions), sorted(expected)))
    expected_rows = len(test_triplets) * (1 if graph.tail_only else 2)
    if n != expected_rows:
        problems.append("{} rows, expected {} ({} test triples)".format(
            n, expected_rows, len(test_triplets)))
    if len(set(table["query_id"])) != len(test_triplets):
        problems.append("{} distinct query_ids, expected {}".format(
            len(set(table["query_id"])), len(test_triplets)))

    return (n, problems)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultra", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ranks", default=os.path.join(WORKSPACE, "ranks", "ultra"))
    parser.add_argument("--dataset", action="append", default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, args.ultra)
    import torch
    from ultra import datasets as ultra_datasets

    wanted = args.dataset or [
        g.id for g in suite.GRAPHS
        if os.path.exists(os.path.join(args.ranks, g.id.replace(":", "_") + ".parquet"))
    ]
    failures = 0
    for gid in wanted:
        graph = suite.by_id(gid)
        result, problems = check(graph, args.root, args.ranks, ultra_datasets, torch), None
        if result[0] is None:
            print("{:<28} SKIP  {}".format(gid, result[1]))
            continue
        n, problems = result
        if problems:
            failures += 1
            print("{:<28} FAIL  {}".format(gid, "; ".join(problems)))
        else:
            print("{:<28} ok    {} rows, query_id -> (h,r,t) verified".format(gid, n))
    print("\n{} checked, {} failed".format(len(wanted), failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
