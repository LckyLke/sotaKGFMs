#!/usr/bin/env python
"""Phase 0's stop rule: CREST with a zero residual must BE TRIX, row by row.

Compares ranks/crest/ against ranks/trix/ per graph: same row count, and the
``rank`` column identical on every row after aligning rows by (direction,
query_id) -- the stable key both dumps carry (see the Dumper in
patches/trix/0001). This replaces the old tolerance gate on a group mean
deliberately: a zeroed residual is arithmetically TRIX, so any differing rank
is a defect in the wrapper, the data root or the dump, and a tolerance could
not distinguish a correct wrapper from two errors that cancel.

Exit status 0 only if every compared graph is identical AND every graph
present on the TRIX side was compared. A graph missing from ranks/crest/ is
reported as incomplete rather than failed, so a partial phase 0 run can be
inspected -- but it cannot pass.

    usage: verify_crest_identity.py [--crest ranks/crest] [--trix ranks/trix]
                                    [--dataset FB15k237Inductive:v1 ...]
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))

import suite  # noqa: E402


def compare(gid, crest_dir, trix_dir):
    """Returns (status, detail): status in {"ok", "missing", "fail"}."""
    import pyarrow.parquet as pq

    name = gid.replace(":", "_") + ".parquet"
    trix_path = os.path.join(trix_dir, name)
    crest_path = os.path.join(crest_dir, name)
    if not os.path.exists(trix_path):
        return "skip", "no TRIX dump to compare against"
    if not os.path.exists(crest_path):
        return "missing", "no CREST dump yet"

    t = pq.read_table(trix_path, columns=["direction", "query_id", "rank", "h", "r", "t"]).to_pydict()
    c = pq.read_table(crest_path, columns=["direction", "query_id", "rank", "h", "r", "t"]).to_pydict()

    n_t, n_c = len(t["rank"]), len(c["rank"])
    if n_t != n_c:
        return "fail", "row count differs: trix {} vs crest {}".format(n_t, n_c)

    def keyed(d):
        out = {}
        for i in range(len(d["rank"])):
            key = (d["direction"][i], d["query_id"][i])
            if key in out:
                return None, "duplicate row for {}".format(key)
            out[key] = i
        return out, None

    t_idx, err = keyed(t)
    if err:
        return "fail", "trix dump: " + err
    c_idx, err = keyed(c)
    if err:
        return "fail", "crest dump: " + err
    if set(t_idx) != set(c_idx):
        only_t = len(set(t_idx) - set(c_idx))
        only_c = len(set(c_idx) - set(t_idx))
        return "fail", "row keys differ: {} only in trix, {} only in crest".format(only_t, only_c)

    diffs = []
    for key, i in t_idx.items():
        j = c_idx[key]
        # the triple identity must agree before a rank comparison means anything
        if (t["h"][i], t["r"][i], t["t"][i]) != (c["h"][j], c["r"][j], c["t"][j]):
            diffs.append("{}: triple mismatch trix ({},{},{}) vs crest ({},{},{})".format(
                key, t["h"][i], t["r"][i], t["t"][i], c["h"][j], c["r"][j], c["t"][j]))
        elif t["rank"][i] != c["rank"][j]:
            diffs.append("{}: rank {} vs {}".format(key, t["rank"][i], c["rank"][j]))
        if len(diffs) >= 5:
            break
    if diffs:
        total = sum(
            1 for key, i in t_idx.items()
            if t["rank"][i] != c["rank"][c_idx[key]])
        return "fail", "{} of {} ranks differ; first: {}".format(total, n_t, "; ".join(diffs))
    return "ok", "{} rows identical".format(n_t)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crest", default=os.path.join(WORKSPACE, "ranks", "crest"))
    parser.add_argument("--trix", default=os.path.join(WORKSPACE, "ranks", "trix"))
    parser.add_argument("--dataset", action="append", default=None,
                        help="restrict to these suite ids (default: the 41 inductive graphs)")
    args = parser.parse_args(argv)

    wanted = args.dataset or [g.id for g in suite.GRAPHS if g.group in ("ind_e", "ind_er")]
    counts = {"ok": 0, "fail": 0, "missing": 0, "skip": 0}
    for gid in wanted:
        status, detail = compare(gid, args.crest, args.trix)
        counts[status] += 1
        label = {"ok": "ok   ", "fail": "FAIL ", "missing": "MISS ", "skip": "SKIP "}[status]
        print("{:<28} {} {}".format(gid, label, detail))

    print("\n{} ok, {} failed, {} missing, {} skipped (no TRIX dump)".format(
        counts["ok"], counts["fail"], counts["missing"], counts["skip"]))
    if counts["fail"]:
        print("STOP (docs/CREST_PLAN.md phase 0): a zeroed residual is "
              "arithmetically TRIX; a differing rank is a wrapper/data/dump "
              "defect and no later number can be trusted until it is found.")
        return 1
    if counts["missing"]:
        print("phase 0 incomplete: every TRIX graph must have a CREST dump to pass.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
