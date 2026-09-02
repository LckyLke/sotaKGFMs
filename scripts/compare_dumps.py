#!/usr/bin/env python3
"""Compare two rank dumps of the same model on their common graphs.

Written for the stack-identity check (a cu118 dump against a cu128 dump of
one checkpoint): per graph, the MRR and Hits@10 of both, the delta, the share
of queries whose rank is identical, and the largest rank move. Joined on
(dataset, query_id, direction). Prints markdown.

    python3 scripts/compare_dumps.py ranks/incite-4g-last ranks/incite-4g-last-bw
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "shared"))
import metrics  # noqa: E402
import suite  # noqa: E402

a_dir, b_dir = sys.argv[1], sys.argv[2]
ma = metrics.compute_dir(a_dir, dtype="float64")
mb = metrics.compute_dir(b_dir, dtype="float64")
common = [g for g in suite.ids() if g in ma and g in mb]


def prov(d):
    p = os.path.join(d, "PROVENANCE.json")
    return json.load(open(p)) if os.path.exists(p) else {}


pa, pb = prov(a_dir), prov(b_dir)
print("A = %s (%s, torch %s, cuda %s)" % (a_dir, pa.get("gpu"), pa.get("torch"), pa.get("cuda")))
print("B = %s (%s, torch %s, cuda %s)\n" % (b_dir, pb.get("gpu"), pb.get("torch"), pb.get("cuda")))
print("| graph | MRR A | MRR B | delta | H@10 A | H@10 B | identical ranks | max |rank move| | rows |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for g in common:
    f = g.replace(":", "_") + ".parquet"
    ta = pq.read_table(os.path.join(a_dir, f), columns=["query_id", "direction", "rank"])
    tb = pq.read_table(os.path.join(b_dir, f), columns=["query_id", "direction", "rank"])
    ka = {(int(q), str(d)): int(r) for q, d, r in zip(ta.column("query_id").to_pylist(), ta.column("direction").to_pylist(), ta.column("rank").to_pylist())}
    kb = {(int(q), str(d)): int(r) for q, d, r in zip(tb.column("query_id").to_pylist(), tb.column("direction").to_pylist(), tb.column("rank").to_pylist())}
    keys = sorted(set(ka) & set(kb))
    ra = np.array([ka[k] for k in keys]); rb = np.array([kb[k] for k in keys])
    same = float((ra == rb).mean()) if len(keys) else float("nan")
    print("| %s | %.4f | %.4f | %+.4f | %.4f | %.4f | %.1f%% | %d | %d |" % (
        g, ma[g]["mrr"], mb[g]["mrr"], mb[g]["mrr"] - ma[g]["mrr"], ma[g]["hits@10"], mb[g]["hits@10"],
        100 * same, int(np.abs(ra - rb).max()) if len(keys) else 0, len(keys)))
if not common:
    print("no common graphs")
