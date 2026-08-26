"""Turn dumped ranks into the baseline report: criterion A and criterion B.

Criterion A -- metric equivalence.  Recompute every metric from the dumped
ranks with ``shared/metrics.py`` and compare against ULTRA's own
``ultra_results_<timestamp>.csv``, per dataset, at the precision the CSV prints.
A mismatch means either the rank dump is wrong or the tie rule here differs from
ULTRA's; the fix is always to this module's side, never to ULTRA.

Criterion B -- published numbers.  Unweighted group means against the ULTRA
repository's own PyG figures.

Nothing here tunes anything.  It reports and stops.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import struct
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from . import metrics as _metrics, suite as _suite  # type: ignore
except ImportError:  # pragma: no cover - flat sys.path use
    import metrics as _metrics  # type: ignore
    import suite as _suite  # type: ignore

#: ULTRA repository PyG zero-shot figures (README, pin 427966ad).  These are the
#: acceptance targets -- NOT the paper's 0.430/0.566 and 0.345/0.512.  Landing on
#: the paper numbers instead would mean something is wrong and is reported as an
#: anomaly.
PUBLISHED = {
    "ind_e": {"n": 18, "mrr": 0.420, "hits@10": 0.562},
    "ind_er": {"n": 23, "mrr": 0.344, "hits@10": 0.511},
}
PAPER = {
    "ind_e": {"mrr": 0.430, "hits@10": 0.566},
    "ind_er": {"mrr": 0.345, "hits@10": 0.512},
}
TOLERANCE = 0.002


# ---------------------------------------------------------------------------
# float comparison helpers
# ---------------------------------------------------------------------------
def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def ulps_f32(a: float, b: float) -> int:
    """Distance between two values in float32 representable steps."""
    if a == b:
        return 0
    if math.isnan(a) or math.isnan(b):
        return -1
    ia, ib = _f32_bits(a), _f32_bits(b)
    if ia & 0x80000000:
        ia = 0x80000000 - (ia & 0x7FFFFFFF)
    if ib & 0x80000000:
        ib = 0x80000000 - (ib & 0x7FFFFFFF)
    return abs(ia - ib)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def read_ultra_csv(path: str) -> Dict[str, Dict[str, float]]:
    """Parse a stock ``ultra_results_*.csv`` into {suite id: {metric: value}}.

    The CSV is keyed by ULTRA's ``-d`` string, which differs from the canonical
    suite id for Metafam and FBNELL; ``suite.by_run_id`` normalises it.
    """
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.pop("dataset")
            graph = _suite.by_run_id(name)
            out[graph.id] = {k: float(v) for k, v in row.items() if v not in (None, "")}
    return out


def find_latest_ultra_csv(where: str) -> Optional[str]:
    found = sorted(glob.glob(os.path.join(where, "**", "ultra_results_*.csv"), recursive=True))
    return found[-1] if found else None


# ---------------------------------------------------------------------------
# criterion A
# ---------------------------------------------------------------------------
def criterion_a(
    ours: Mapping[str, Mapping[str, float]],
    theirs: Mapping[str, Mapping[str, float]],
    compare_metrics: Sequence[str] = ("mrr", "hits@1", "hits@3", "hits@10", "mr"),
) -> Tuple[List[dict], bool]:
    """Per-dataset, per-metric diff between our recomputation and ULTRA's CSV."""
    rows: List[dict] = []
    passed = True
    for gid in sorted(set(ours) & set(theirs)):
        for metric in compare_metrics:
            if metric not in ours[gid] or metric not in theirs[gid]:
                continue
            a, b = float(ours[gid][metric]), float(theirs[gid][metric])
            exact = repr(a) == repr(b)
            rows.append({
                "dataset": gid,
                "metric": metric,
                "ours": a,
                "ultra": b,
                "exact": exact,
                "abs_diff": abs(a - b),
                "ulps": ulps_f32(a, b),
            })
            if not exact:
                passed = False
    return rows, passed


# ---------------------------------------------------------------------------
# criterion B
# ---------------------------------------------------------------------------
def criterion_b(per_dataset: Mapping[str, Mapping[str, float]]) -> Tuple[List[dict], bool]:
    rows: List[dict] = []
    passed = True
    for group, target in PUBLISHED.items():
        wanted = set(_suite.ids(group))
        present = {k: v for k, v in per_dataset.items() if k in wanted}
        complete = len(present) == target["n"]
        for metric in ("mrr", "hits@10"):
            if not present:
                rows.append({
                    "group": group, "metric": metric, "n_present": 0, "n_expected": target["n"],
                    "ours": float("nan"), "target": target[metric], "delta": float("nan"),
                    "within": False, "complete": False,
                    "paper": PAPER[group][metric], "paper_delta": float("nan"),
                })
                passed = False
                continue
            value = _metrics.group_mean(present, metric)
            delta = value - target[metric]
            within = abs(delta) <= TOLERANCE
            rows.append({
                "group": group, "metric": metric,
                "n_present": len(present), "n_expected": target["n"],
                "ours": value, "target": target[metric], "delta": delta,
                "within": within, "complete": complete,
                "paper": PAPER[group][metric],
                "paper_delta": value - PAPER[group][metric],
            })
            if not (within and complete):
                passed = False
    return rows, passed


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------
def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_criterion_a(rows: Sequence[dict], passed: bool) -> str:
    if not rows:
        return "_No overlap between dumped ranks and ULTRA's CSV -- criterion A not evaluated._"
    body = [
        [r["dataset"], r["metric"], repr(r["ours"]), repr(r["ultra"]),
         "yes" if r["exact"] else "**no**", "{:.3e}".format(r["abs_diff"]), str(r["ulps"])]
        for r in rows
    ]
    head = _table(["dataset", "metric", "metrics.py", "ULTRA csv", "exact", "|diff|", "ulps(f32)"], body)
    mism = [r for r in rows if not r["exact"]]
    summary = "**Criterion A: {}** -- {} comparisons, {} exact, {} mismatched.".format(
        "PASS" if passed else "FAIL", len(rows), len(rows) - len(mism), len(mism))
    return summary + "\n\n" + head


def render_criterion_b(rows: Sequence[dict], passed: bool) -> str:
    body = [
        [r["group"], r["metric"], "{}/{}".format(r["n_present"], r["n_expected"]),
         "{:.4f}".format(r["ours"]) if not math.isnan(r["ours"]) else "n/a",
         "{:.3f}".format(r["target"]),
         "{:+.4f}".format(r["delta"]) if not math.isnan(r["delta"]) else "n/a",
         "yes" if r["within"] else "**no**",
         "{:.3f}".format(r["paper"]),
         "{:+.4f}".format(r["paper_delta"]) if not math.isnan(r["paper_delta"]) else "n/a"]
        for r in rows
    ]
    head = _table(["group", "metric", "datasets", "ours", "repo target", "delta",
                   "within +/-0.002", "paper", "delta vs paper"], body)
    return "**Criterion B: {}**\n\n".format("PASS" if passed else "FAIL") + head


def render_per_dataset(per_dataset: Mapping[str, Mapping[str, float]]) -> str:
    out = []
    for group in ("ind_e", "ind_er", "transductive"):
        gids = [g for g in _suite.ids(group) if g in per_dataset]
        if not gids:
            continue
        body = []
        for gid in gids:
            row = per_dataset[gid]
            body.append([gid, _suite.by_id(gid).family, str(int(row["n"])),
                         "{:.4f}".format(row["mrr"]), "{:.4f}".format(row["hits@10"])])
        out.append("### {} ({} of {} graphs)\n\n".format(
            group, len(gids), len(_suite.ids(group))) +
            _table(["dataset", "family", "queries", "MRR", "Hits@10"], body))
    return "\n\n".join(out) if out else "_No rank files found._"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", default="ranks/ultra", help="directory of rank parquets")
    parser.add_argument("--ultra-csv", default=None, help="stock ultra_results_*.csv")
    parser.add_argument("--search", default="output", help="where to look for the CSV if not given")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    args = parser.parse_args(argv)

    per_dataset = _metrics.compute_dir(args.ranks, dtype=args.dtype)
    print("## Per-dataset results\n")
    print(render_per_dataset(per_dataset))

    csv_path = args.ultra_csv or find_latest_ultra_csv(args.search)
    print("\n\n## Criterion A -- metric equivalence\n")
    if csv_path and per_dataset:
        rows, ok = criterion_a(per_dataset, read_ultra_csv(csv_path))
        print("Source CSV: `{}`\n".format(csv_path))
        print(render_criterion_a(rows, ok))
    else:
        print("_Not evaluated: no ultra_results_*.csv found._")

    print("\n\n## Criterion B -- published numbers\n")
    rows_b, ok_b = criterion_b(per_dataset)
    print(render_criterion_b(rows_b, ok_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
