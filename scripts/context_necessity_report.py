#!/usr/bin/env python3
"""Table of the context-necessity runs (diagnostics/context_necessity.py).

Reads every output/context-necessity/<name>/result.json and prints markdown:
one row per (mode, withhold, variant) with the mean and, when several seeds
exist, the spread of the final held-out MRR under the three conditions, and
the K3 verdict per run. Nothing is transcribed; a missing run is a missing
row.

    python3 scripts/context_necessity_report.py [output/context-necessity]
"""
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "output", "context-necessity")
COND = ("full", "shuffled", "none")

runs = []
for path in sorted(glob.glob(os.path.join(base, "*", "result.json"))):
    r = json.load(open(path))
    p = r["provenance"]
    if not r.get("final"):
        continue
    variant = "detached rows" if p.get("scorer", {}).get("detach_rows") else ""
    runs.append(dict(name=os.path.basename(os.path.dirname(path)), mode=p["mode"],
                     withhold=float(p["withhold"]), seed=int(p["seed"]), variant=variant,
                     steps=int(p["steps"]), device=p.get("device"), final=r["final"],
                     k3=r.get("k3"), n=r["final"]["full"]["n"],
                     held=p.get("held_out", {})))

groups = {}
for run in runs:
    groups.setdefault((run["mode"], run["withhold"], run["variant"]), []).append(run)


def cell(vals):
    if len(vals) == 1:
        return "%.4f" % vals[0]
    return "%.4f ± %.4f" % (statistics.mean(vals), statistics.pstdev(vals))


print("## Context-necessity diagnostic (held-out synthetic instances, final step)\n")
if runs:
    h = runs[0]["held"]
    print("Held-out set: %d instances per run, seed %s; mean %.0f nodes, %.0f edges, "
          "%.1f eval candidates of which %.1f inside the head's 3-hop ball. "
          "Ranks: 1-based, pessimistic ties, over the type-consistent non-derivable pool.\n"
          % (runs[0]["n"], h.get("seed"), h.get("mean_nodes", 0), h.get("mean_edges", 0),
             h.get("mean_eval_pool", 0), h.get("mean_eval_hard", 0)))
print("| mode | withhold | variant | seeds | MRR full | MRR shuffled | MRR none | H@1 full | K3 ordered | full − none |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for key in sorted(groups, key=lambda k: (-k[1], k[0], k[2])):
    rs = groups[key]
    mode, w, variant = key
    seeds = ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
    mrr = {c: [r["final"][c]["mrr"] for r in rs] for c in COND}
    h1 = [r["final"]["full"]["hits@1"] for r in rs]
    if mode == "floor":
        k3, gap = "n/a", "n/a"
    else:
        k3 = "/".join("yes" if r["k3"]["ordered"] else "no" for r in rs)
        gap = cell([r["k3"]["full_minus_none"] for r in rs])
    print("| %s | %.1f | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        mode, w, variant or "-", seeds, cell(mrr["full"]), cell(mrr["shuffled"]),
        cell(mrr["none"]), cell(h1), k3, gap))
print("\nRuns: %d. Per-run files: output/context-necessity/<name>/result.json (curve, provenance)." % len(runs))
