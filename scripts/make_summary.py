#!/usr/bin/env python
"""Generate baseline_report.md: every model on one suite, side by side.

Per-model detail lives in reports/<model>.md, written by make_report.py. This
file is the cross-model view -- the thing the project exists to produce -- and
like every other report here it is computed, never transcribed.

A model appears only if it has rank dumps. A group mean appears only if every
graph in that group is present, because an unweighted mean over a subset is not
comparable with one over the whole group and would read as if it were.

    usage: make_summary.py [--out baseline_report.md]
"""

import argparse
import datetime
import json
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))

import analyse  # noqa: E402
import metrics  # noqa: E402
import suite  # noqa: E402

MODELS = ["ultra", "motif", "trix", "semma", "flock", "kg-icl", "kgpfn"]
GROUPS = ["ind_e", "ind_er"]
SHOWN = ["mrr", "hits@1", "hits@3", "hits@10"]


def timings(model):
    path = os.path.join(WORKSPACE, "ranks", model, "TIMINGS.jsonl")
    if not os.path.exists(path):
        return None
    secs = []
    for line in open(path):
        row = json.loads(line)
        # index 1 absorbs the one-time JIT compile of rspmm and is not
        # comparable with the rest of the run.
        if row.get("status") == "ok" and row.get("index") != 1:
            secs.append(row["seconds"])
    return (len(secs), sum(secs)) if secs else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(WORKSPACE, "baseline_report.md"))
    args = parser.parse_args(argv)

    published = analyse.load_published()
    present = {}
    for model in MODELS:
        rank_dir = os.path.join(WORKSPACE, "ranks", model)
        if not os.path.isdir(rank_dir):
            continue
        per = metrics.compute_dir(rank_dir)
        if per:
            present[model] = per

    out = ["# KGFM baselines on one suite\n",
           "Generated {} by `scripts/make_summary.py`. Every number is computed "
           "from the rank dumps in `ranks/`; nothing is transcribed.\n".format(
               datetime.date.today().isoformat()),
           "All runs are zero-shot entity prediction, one seed (1024), on a single GPU. "
           "Ranks are 1-based with pessimistic ties under strict filtering, identically "
           "for every model -- see `shared/metrics.py` for the definition and "
           "`docs/report_notes.md` for the evidence that each repository computes it "
           "the same way.\n"]

    # ---- coverage -----------------------------------------------------------
    out.append("## Coverage\n")
    rows = []
    for model in MODELS:
        per = present.get(model)
        if per is None:
            rows.append([model, "-", "-", "_not run_"])
            continue
        counts = ["{}/{}".format(sum(1 for i in suite.ids(g) if i in per), len(suite.ids(g)))
                  for g in GROUPS]
        t = timings(model)
        cost = "{:.0f} min".format(t[1] / 60) if t else "-"
        rows.append([model] + counts + [cost])
    out.append(analyse._table(["model"] + GROUPS + ["suite wall clock"], rows))

    # ---- results ------------------------------------------------------------
    for group in GROUPS:
        gids = suite.ids(group)
        out.append("\n## {} ({} graphs)\n".format(group, len(gids)))
        rows = []
        for model in MODELS:
            per = present.get(model)
            if not per or sum(1 for i in gids if i in per) != len(gids):
                continue
            sub = {k: v for k, v in per.items() if k in set(gids)}
            cells = ["{:.4f}".format(metrics.group_mean(sub, m)) for m in SHOWN]
            try:
                target, key, _ = analyse.targets_for(model, published)
                delta = metrics.group_mean(sub, "mrr") - target[group]["mrr"]
                ref = "{:.3f} ({})".format(target[group]["mrr"], key)
                verdict = "{:+.4f}".format(delta)
            except KeyError:
                ref, verdict = "-", "-"
            rows.append([model] + cells + [ref, verdict])
        if rows:
            out.append(analyse._table(
                ["model"] + SHOWN + ["published MRR", "delta"], rows))
        else:
            out.append("_No model has a complete {} run._".format(group))

    out.append("\n## Per-model detail\n")
    out.append("\n".join(
        "* [{0}](reports/{0}.md)".format(m) for m in MODELS if m in present))

    with open(args.out, "w") as handle:
        handle.write("\n".join(out) + "\n")
    print("wrote {} ({} models)".format(args.out, len(present)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
