#!/usr/bin/env python
"""Prove that KG-ICL's bundled graphs are the same graphs the other repos download.

KG-ICL ships its own datasets.zip rather than downloading from the GraIL, Ingram
and ILPC sources the ULTRA-derived repos use. Its numbers are only comparable
with theirs if the underlying triples are identical, and that has to be shown,
not assumed: its directory names differ (`fb237_v1_ind` for
`FB15k237Inductive:v1`), its split filenames differ, and nothing would complain
if a graph had been regenerated.

Each split is compared as a SET of (head, relation, tail) strings. Order and
whitespace are ignored; identity of the triples is not.

    usage: verify_kgicl_data.py --kgicl <unpacked datasets dir> [--ultra data/roots/ultra]
"""

import argparse
import os
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))
import suite  # noqa: E402

# suite id -> (KG-ICL directory, ULTRA raw directory, [(ultra file, kgicl file)])
GRAIL_SPLITS = [("train_ind.txt", "train.txt"),
                ("valid_ind.txt", "valid.txt"),
                ("test_ind.txt", "test.txt")]
# The Ingram and ILPC families carry four files: a transductive training graph,
# a separate inference graph the model is given at test time, and the validation
# and test queries over that inference graph. KG-ICL keeps all four under its
# own names, and the names are the trap: its `train.txt` is ULTRA's
# transductive_train.txt, while ULTRA's inference_graph.txt is its `msg.txt`.
# Pairing train with train compares two different graphs and both are real.
INGRAM_SPLITS = [("transductive_train.txt", "train.txt"),
                 ("inference_graph.txt", "msg.txt"),
                 ("inf_valid.txt", "valid.txt"),
                 ("inf_test.txt", "test.txt")]
ILPC_SPLITS = [("transductive_train.txt", "train.txt"),
               ("inference_graph.txt", "inference.txt"),
               ("inf_valid.txt", "inference_validation.txt"),
               ("inf_test.txt", "inference_test.txt")]

MAP = {}
for n in (1, 2, 3, 4):
    MAP["FB15k237Inductive:v%d" % n] = (
        "inductive/fb237_v%d_ind" % n, "grail/IndFB15k237/v%d/raw" % n, GRAIL_SPLITS)
    MAP["WN18RRInductive:v%d" % n] = (
        "inductive/WN18RR_v%d_ind" % n, "grail/IndWN18RR/v%d/raw" % n, GRAIL_SPLITS)
    MAP["NELLInductive:v%d" % n] = (
        "inductive/nell_v%d_ind" % n, "grail/IndNELL/v%d/raw" % n, GRAIL_SPLITS)
for size in ("small", "large"):
    MAP["ILPC2022:%s" % size] = (
        "ILPC/ILPC-%s" % size, "ilpc2022/%s/raw" % size, ILPC_SPLITS)
for pct in (25, 50, 75, 100):
    MAP["FBIngram:%d" % pct] = ("fully-inductive/FB-%d" % pct, "ingram/fb/%d/raw" % pct, INGRAM_SPLITS)
    MAP["WKIngram:%d" % pct] = ("fully-inductive/WK-%d" % pct, "ingram/wk/%d/raw" % pct, INGRAM_SPLITS)
for pct in (0, 25, 50, 75, 100):
    MAP["NLIngram:%d" % pct] = ("fully-inductive/NL-%d" % pct, "ingram/nl/%d/raw" % pct, INGRAM_SPLITS)


def triples(path):
    out = set()
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3:
                out.add((parts[0], parts[1], parts[2]))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kgicl", required=True, help="directory datasets.zip was unpacked into")
    parser.add_argument("--ultra", default=os.path.join(WORKSPACE, "data", "roots", "ultra"))
    args = parser.parse_args(argv)

    covered = [g for g in suite.ids() if g in MAP]
    absent = [g for g in suite.ids("ind_e") + suite.ids("ind_er") if g not in MAP]
    same = diff = skipped = 0
    for gid in covered:
        kdir, udir, splits = MAP[gid]
        kpath = os.path.join(args.kgicl, kdir)
        upath = os.path.join(args.ultra, udir)
        if not os.path.isdir(kpath) or not os.path.isdir(upath):
            print("  {:26} SKIP (kgicl={} ultra={})".format(
                gid, os.path.isdir(kpath), os.path.isdir(upath)))
            skipped += 1
            continue
        rows, ok = [], True
        for ufile, kfile in splits:
            uf, kf = os.path.join(upath, ufile), os.path.join(kpath, kfile)
            if not (os.path.exists(uf) and os.path.exists(kf)):
                rows.append("{}: MISSING".format(kfile))
                ok = False
                continue
            a, b = triples(uf), triples(kf)
            if a == b:
                rows.append("{}={}".format(kfile.split(".")[0], len(a)))
            else:
                ok = False
                rows.append("{}: ultra {} vs kgicl {}, {} only-ultra, {} only-kgicl".format(
                    kfile.split(".")[0], len(a), len(b), len(a - b), len(b - a)))
        print("  {:26} {}  {}".format(gid, "SAME" if ok else "**DIFFERENT**", "  ".join(rows)))
        same += ok
        diff += (not ok)

    print("\n{} identical, {} different, {} skipped, of {} suite graphs KG-ICL ships".format(
        same, diff, skipped, len(covered)))
    print("{} of the 41 are not in datasets.zip at all: {}".format(len(absent), ", ".join(absent)))
    return 1 if diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
