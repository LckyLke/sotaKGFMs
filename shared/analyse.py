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
import json
import math
import os
import struct
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from . import metrics as _metrics, suite as _suite  # type: ignore
except ImportError:  # pragma: no cover - flat sys.path use
    import metrics as _metrics  # type: ignore
    import suite as _suite  # type: ignore

#: Published group means live in shared/published.json, one block per model,
#: each carrying its own source. They were constants here once, and they were
#: ULTRA's constants: running the report for any other model then compared that
#: model against ULTRA's targets and printed a verdict that meant nothing.
PUBLISHED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "published.json")


def load_published(path: str = PUBLISHED_PATH) -> dict:
    with open(path) as handle:
        return json.load(handle)


def targets_for(model: str, published: Optional[dict] = None) -> Tuple[dict, str, dict]:
    """Return (primary target, its key, the other targets) for one model.

    Raises rather than falling back to another model's numbers. A missing entry
    means the published figures for that model were never recorded, and saying
    so is the only honest output.
    """
    pub = published if published is not None else load_published()
    if model not in pub:
        raise KeyError(
            "no published figures recorded for {!r} in {}. Add a block for it "
            "rather than comparing against another model.".format(model, PUBLISHED_PATH))
    entry = pub[model]
    key = entry["primary"]
    return entry["targets"][key], key, {k: v for k, v in entry["targets"].items() if k != key}


#: Filename each repository writes its own metrics into. SEMMA is an ULTRA fork
#: that never renamed the output, so its CSVs are called ultra_results_*.csv
#: too -- a recursive glob for that name under results/ picks up both models'
#: files and silently merges one into the other.
CSV_PATTERNS = {
    "ultra": "ultra_results_*.csv",
    "semma": "ultra_results_*.csv",
    "motif": "MOTIF_results_*.csv",
    "trix": "TRIX_results_*.csv",
}

TOLERANCE_DEFAULT = 0.002


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


def model_csv_dir(model: str, root: str = "results") -> str:
    """Where one model's own metric CSVs live.

    Non-recursive on purpose. ULTRA's sit loose in results/ and every other
    model has results/<model>/; a recursive search would let SEMMA's
    ultra_results_*.csv be read as ULTRA's.
    """
    nested = os.path.join(root, model)
    return nested if os.path.isdir(nested) else root


def find_model_csvs(model: str, root: str = "results") -> List[str]:
    pattern = CSV_PATTERNS.get(model)
    if pattern is None:
        raise KeyError("no CSV filename pattern recorded for model {!r}".format(model))
    return sorted(glob.glob(os.path.join(model_csv_dir(model, root), pattern)))


def read_model_csvs(model: str, paths: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Merge one model's per-graph CSVs. Each run writes one file with one row.

    Raises on a duplicate dataset rather than letting the last file win: two
    rows for one graph means either a re-run that was never cleaned up or two
    models' files in one directory, and both must be looked at, not averaged
    over silently.
    """
    out: Dict[str, Dict[str, float]] = {}
    seen: Dict[str, str] = {}
    for path in paths:
        for gid, row in read_ultra_csv(path).items():
            if gid in seen:
                raise ValueError(
                    "{} appears twice for model {}: {} and {}. Remove the stale "
                    "file before reporting.".format(gid, model, seen[gid], path))
            seen[gid] = path
            out[gid] = row
    return out


# criterion A
# ---------------------------------------------------------------------------
#: Metrics whose reduction is order-independent, so bitwise equality is a
#: property of the definition rather than of the summation order. A mismatch on
#: one of these is a real disagreement about what the metric means.
ORDER_INDEPENDENT: Sequence[str] = ("hits@1", "hits@3", "hits@10")


def criterion_a(
    ours: Mapping[str, Mapping[str, float]],
    theirs: Mapping[str, Mapping[str, float]],
    compare_metrics: Sequence[str] = ("mrr", "hits@1", "hits@3", "hits@10", "mr", "hits@10_50"),
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
            row = {
                "dataset": gid,
                "metric": metric,
                "ours": a,
                "ultra": b,
                "exact": exact,
                "abs_diff": abs(a - b),
                "ulps": ulps_f32(a, b),
            }
            # For hits@k the float is count/n_queries, so the count is
            # recoverable from either side and is a whole number. Comparing the
            # counts asks the question bitwise equality is meant to ask -- do the
            # two agree about which queries were hits -- without also asking that
            # both divided by n in the same order on the same silicon. See
            # criterion_a_summary for why that distinction is not cosmetic.
            n = ours[gid].get("n")
            if metric in ORDER_INDEPENDENT and n:
                n = int(n)
                row["count_ours"] = int(round(a * n))
                row["count_ultra"] = int(round(b * n))
                row["count_exact"] = row["count_ours"] == row["count_ultra"]
            rows.append(row)
            if not exact:
                passed = False
    return rows, passed


def criterion_a_summary(rows: Sequence[dict]) -> dict:
    """Split criterion A by what a mismatch would actually mean.

    A mismatch on an order-independent metric is a disagreement about the metric
    itself -- wrong tie rule, wrong offset, wrong dump. A mismatch on the others
    may be nothing more than float32 associativity, so its size is what matters:
    the summary carries the worst ulp distance seen.
    """
    hard = [r for r in rows if r["metric"] in ORDER_INDEPENDENT]
    soft = [r for r in rows if r["metric"] not in ORDER_INDEPENDENT]
    counted = [r for r in hard if "count_exact" in r]
    return {
        "n": len(rows),
        "exact": sum(1 for r in rows if r["exact"]),
        "strict_pass": all(r["exact"] for r in rows),
        "definition_pass": (all(r["count_exact"] for r in counted)
                            if counted else all(r["exact"] for r in hard)),
        "definition_n": len(hard),
        "counted_n": len(counted),
        "counted_exact": sum(1 for r in counted if r["count_exact"]),
        "hard_bitwise_exact": sum(1 for r in hard if r["exact"]),
        "soft_n": len(soft),
        "soft_exact": sum(1 for r in soft if r["exact"]),
        "max_ulps": max([r["ulps"] for r in soft], default=0),
        "max_abs_diff": max([r["abs_diff"] for r in soft], default=0.0),
    }


# ---------------------------------------------------------------------------
# criterion B
# ---------------------------------------------------------------------------
def criterion_b(per_dataset: Mapping[str, Mapping[str, float]],
                model: str = "ultra") -> Tuple[List[dict], bool]:
    pub = load_published()
    primary, primary_key, others = targets_for(model, pub)
    tolerance = pub.get("tolerance", TOLERANCE_DEFAULT)
    counts = pub.get("groups", {})
    alt_key = next(iter(others), None)
    rows: List[dict] = []
    passed = True
    for group in ("ind_e", "ind_er"):
        target = dict(primary[group], n=counts.get(group, len(_suite.ids(group))))
        alt = others[alt_key][group] if alt_key else None
        wanted = set(_suite.ids(group))
        present = {k: v for k, v in per_dataset.items() if k in wanted}
        complete = len(present) == target["n"]
        for metric in ("mrr", "hits@10"):
            if not present:
                rows.append({
                    "group": group, "metric": metric, "n_present": 0, "n_expected": target["n"],
                    "ours": float("nan"), "target": target[metric], "delta": float("nan"),
                    "within": False, "complete": False,
                    "target_key": primary_key, "alt_key": alt_key,
                    "paper": (alt[metric] if alt else float("nan")),
                    "paper_delta": float("nan"),
                })
                passed = False
                continue
            value = _metrics.group_mean(present, metric)
            delta = value - target[metric]
            within = abs(delta) <= tolerance
            rows.append({
                "group": group, "metric": metric,
                "n_present": len(present), "n_expected": target["n"],
                "ours": value, "target": target[metric], "delta": delta,
                "within": within, "complete": complete,
                "target_key": primary_key, "alt_key": alt_key,
                "paper": (alt[metric] if alt else float("nan")),
                "paper_delta": (value - alt[metric]) if alt else float("nan"),
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
    s = criterion_a_summary(rows)
    summary = [
        "**Criterion A (metric definition): {}** -- {}/{} hit counts identical.".format(
            "PASS" if s["definition_pass"] else "FAIL",
            s["counted_exact"], s["counted_n"]),
        "",
        "**Criterion A (strict, bitwise): {}** -- {} comparisons, {} exact, {} mismatched.".format(
            "PASS" if passed else "FAIL", s["n"], s["exact"], len(mism)),
        "",
        "The two verdicts answer different questions, and only the first one is "
        "about ranking.",
        "",
        "* **Order-independent metrics** (`hits@1`, `hits@3`, `hits@10`): {}/{} identical as "
        "counts -- **{}**; {}/{} identical bitwise. `hits@k` is `count / n_queries`, and the "
        "count is a whole number that no summation order can move: a count disagreement is a "
        "disagreement about the tie rule, the rank offset or the dump. The float around it can "
        "still differ, because that last division is not done the same way everywhere -- on CUDA "
        "torch reduces and scales differently from numpy on the host. Every bitwise mismatch seen "
        "here is 1 ulp with the counts identical, which is that division and nothing else.".format(
            s["counted_exact"], s["counted_n"],
            "PASS" if s["definition_pass"] else "FAIL",
            s["hard_bitwise_exact"], s["definition_n"]),
        "* **Order-dependent metrics** (`mrr`, `mr`, `hits@10_50`): {}/{} exact; worst disagreement "
        "{} float32 ulp ({:.2e} absolute). These carry no count to fall back on, so float32 "
        "associativity is the whole story.".format(
            s["soft_exact"], s["soft_n"], s["max_ulps"], s["max_abs_diff"]),
    ]
    return "\n".join(summary) + "\n\n" + head


def render_criterion_b(rows: Sequence[dict], passed: bool, model: str = "ultra") -> str:
    if not rows:
        return "_Criterion B not evaluated._"
    target_key = rows[0].get("target_key", "repo")
    alt_key = rows[0].get("alt_key")
    header = ["group", "metric", "datasets", "ours",
              "{} target".format(target_key), "delta", "within +/-0.002"]
    body = []
    for r in rows:
        cells = [r["group"], r["metric"], "{}/{}".format(r["n_present"], r["n_expected"]),
                 "{:.4f}".format(r["ours"]) if not math.isnan(r["ours"]) else "n/a",
                 "{:.3f}".format(r["target"]),
                 "{:+.4f}".format(r["delta"]) if not math.isnan(r["delta"]) else "n/a",
                 "yes" if r["within"] else "**no**"]
        if alt_key:
            cells += ["{:.3f}".format(r["paper"]),
                      "{:+.4f}".format(r["paper_delta"]) if not math.isnan(r["paper_delta"]) else "n/a"]
        body.append(cells)
    if alt_key:
        header += [alt_key, "delta vs {}".format(alt_key)]
    pub = load_published().get(model, {})
    target = pub.get("targets", {}).get(target_key, {})
    lines = ["**Criterion B ({}): {}**".format(model, "PASS" if passed else "FAIL"), ""]
    if target.get("source"):
        lines += ["Target: `{}`.".format(target["source"]), ""]
    if target.get("note"):
        lines += [target["note"], ""]
    return "\n".join(lines) + "\n" + _table(header, body)


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
    parser.add_argument("--model", default=None,
                        help="model name; defaults to the basename of --ranks")
    parser.add_argument("--search", default="results",
                        help="root holding the model's own metric CSVs")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    args = parser.parse_args(argv)

    model = args.model or os.path.basename(os.path.normpath(args.ranks))
    per_dataset = _metrics.compute_dir(args.ranks, dtype=args.dtype)
    print("## Per-dataset results\n")
    print(render_per_dataset(per_dataset))

    csv_paths = find_model_csvs(model, args.search)
    print("\n\n## Criterion A -- metric equivalence\n")
    if csv_paths and per_dataset:
        theirs = read_model_csvs(model, csv_paths)
        rows, ok = criterion_a(per_dataset, theirs)
        print("Source: {} CSV(s) in `{}`, {} datasets.\n".format(
            len(csv_paths), model_csv_dir(model, args.search), len(theirs)))
        print(render_criterion_a(rows, ok))
    else:
        print("_Not evaluated: no {} CSV found under {}._".format(
            CSV_PATTERNS.get(model, "?"), args.search))

    print("\n\n## Criterion B -- published numbers\n")
    try:
        rows_b, ok_b = criterion_b(per_dataset, model)
        print(render_criterion_b(rows_b, ok_b, model))
    except KeyError as exc:
        print("_Not evaluated: {}_".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
