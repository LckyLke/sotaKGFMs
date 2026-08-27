#!/usr/bin/env python
"""Generate baseline_report.md from the dumped ranks and ULTRA's own CSVs.

Every number in the report comes from this script reading data. The prose states
what was measured; the tables are computed. Nothing is transcribed by hand, so
re-running after a longer run regenerates a report that is still true.
"""

import argparse
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))

import analyse  # noqa: E402
import metrics  # noqa: E402
import suite  # noqa: E402

PREAMBLE = """# ULTRA baseline report

ULTRA is the reference this project measures every other repo against, so this
report is about one question before it is about any number: **does
`shared/metrics.py`, reading only the dumped ranks, reproduce what ULTRA itself
reports?** If it does not, no later comparison means anything.

"""


def table(header, rows):
    return "\n".join(
        ["| " + " | ".join(header) + " |",
         "| " + " | ".join("---" for _ in header) + " |"]
        + ["| " + " | ".join(r) + " |" for r in rows])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", default=os.path.join(WORKSPACE, "ranks", "ultra"))
    parser.add_argument("--rank-column", default="rank",
                        help="which rank column to score; KG-ICL dumps rank and rank_native")
    parser.add_argument("--model", default=None,
                        help="model name; defaults to the basename of --ranks")
    parser.add_argument("--results", default=os.path.join(WORKSPACE, "results"),
                        help="root holding the model's own metric CSVs")
    parser.add_argument("--out", default=os.path.join(WORKSPACE, "baseline_report.md"))
    parser.add_argument("--notes", default=os.path.join(WORKSPACE, "docs", "report_notes.md"),
                        help="hand-written sections appended verbatim")
    args = parser.parse_args(argv)

    model = args.model or os.path.basename(os.path.normpath(args.ranks))
    per_dataset = metrics.compute_dir(args.ranks, rank_column=args.rank_column)

    # Merge this model's own CSVs -- one per graph, since each runner
    # invocation writes its own file. Discovery is model-aware: SEMMA is an
    # ULTRA fork that still names its output ultra_results_*.csv, so a plain
    # recursive glob would read its rows as ULTRA's.
    csv_files = analyse.find_model_csvs(model, args.results)
    theirs = analyse.read_model_csvs(model, csv_files)

    rows_a, ok_a = analyse.criterion_a(per_dataset, theirs)
    sum_a = analyse.criterion_a_summary(rows_a)
    rows_b, ok_b = analyse.criterion_b(per_dataset, model)

    have = {g: sum(1 for i in suite.ids(g) if i in per_dataset) for g in suite.GROUPS}
    want = {g: len(suite.ids(g)) for g in suite.GROUPS}

    out = [PREAMBLE]
    out.append("Generated {} by `scripts/make_report.py`.\n".format(
        datetime.date.today().isoformat()))

    out.append("## Verdict\n")
    out.append(table(
        ["criterion", "result", "coverage"],
        [["A (strict) — every value bitwise identical to ULTRA's CSV",
          "**PASS**" if ok_a else "**FAIL**",
          "{}/{} exact over {} graphs".format(
              sum_a["exact"], sum_a["n"], len({r['dataset'] for r in rows_a}))],
         ["A (metric definition) — order-independent metrics bitwise identical",
          "**PASS**" if sum_a["definition_pass"] else "**FAIL**",
          "{} comparisons; order-dependent ones within {} float32 ulp".format(
              sum_a["definition_n"], sum_a["max_ulps"])],
         ["B — group means within ±0.002 of the repository figures",
          "**PASS**" if ok_b else "**FAIL**",
          "inductive (e) {}/{}, inductive (e,r) {}/{}".format(
              have["ind_e"], want["ind_e"], have["ind_er"], want["ind_er"])]]))
    out.append("")
    if not (ok_a and ok_b):
        out.append(
            "Both criteria must pass. They do not. What is missing and why is in "
            "*Deviations* below, and the criterion A residual is dissected in *The "
            "resolved tie rule* — it is float32 associativity inside ULTRA's own "
            "reduction, not a disagreement about what a rank is. Nothing was tuned "
            "to close any gap.\n")

    out.append("\n## Criterion A — metric equivalence\n")
    out.append(
        "`shared/metrics.py` recomputes each metric from `ranks/ultra/*.parquet` and is "
        "compared against the raw `ultra_results_*.csv` ULTRA wrote during the same run, "
        "value by value, at the full 17-digit precision `csv.DictWriter` prints. "
        "`ulps(f32)` is the distance in float32 representable steps; 0 means the two "
        "floats are the same bit pattern.\n")
    if csv_files:
        out.append("Source CSVs (unmodified, kept in `results/`):\n")
        for path in csv_files:
            out.append("* `{}`".format(os.path.basename(path)))
        out.append("")
    out.append(analyse.render_criterion_a(rows_a, ok_a))

    out.append("\n\n## Per-dataset results\n")
    out.append(analyse.render_per_dataset(per_dataset))

    out.append("\n\n## Criterion B — published numbers\n")
    out.append(
        "Targets are the ULTRA **repository's** PyG figures (README at the pinned SHA), "
        "not the paper's. Group means are unweighted over datasets: every graph counts "
        "once regardless of how many test queries it has. The last two columns show the "
        "distance to the paper numbers as well — landing on those instead of the "
        "repository ones would be an anomaly worth reporting.\n")
    out.append(analyse.render_criterion_b(rows_b, ok_b, model))

    if os.path.exists(args.notes):
        out.append("\n\n")
        out.append(open(args.notes).read())

    text = "\n".join(out).rstrip() + "\n"
    with open(args.out, "w") as handle:
        handle.write(text)
    print("wrote {} ({} graphs, criterion A {}, criterion B {})".format(
        args.out, len(per_dataset), "pass" if ok_a else "fail", "pass" if ok_b else "fail"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
