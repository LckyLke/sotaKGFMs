#!/usr/bin/env python
"""Build KG-ICL's processed format for all 41 graphs, under ULTRA's conventions.

KG-ICL ships datasets.zip with 27 of the 41, and running it from that archive is
not comparable with the other six models, for three reasons that have nothing to
do with the model:

  1. Test message graph. For the Ingram and ILPC families KG-ICL's preprocessor
     writes background.txt as `inference + valid`, while ULTRA's test_data is
     the inference graph alone. Those extra edges are input to message passing,
     not a scoring convention -- KG-ICL simply sees more of the graph.
  2. Test filter. On the 12 GraIL graphs KG-ICL filters with
     `train + valid + test`; ULTRA filters with `inference + test`.
  3. Coverage. HM, WikiTopics, Metafam and FBNELL are not in the archive at all.

All three are properties of the *data preparation*, so they are fixed here
rather than in the model. Every graph is built from the same raw triples the
other six repos read, which scripts/verify_kgicl_data.py has already shown to be
identical to KG-ICL's own copies where both exist.

The case sampling -- the prompt graphs that are KG-ICL's whole method -- is left
to the authors' own `KG.build_cases_for_large_graph`, imported from the
`utils_all.py` inside datasets.zip. Nothing about the model is reimplemented.

    usage: build_kgicl_datasets.py --out <dir> [--only <suite id,...>] [--jobs N]
"""

import argparse
import os
import shutil
import sys
import zipfile

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))
import suite  # noqa: E402

ULTRA_ROOT = os.path.join(WORKSPACE, "data", "roots", "ultra")

# suite id -> (raw dir under data/roots/ultra, filenames, filter rule)
#
# filter rule mirrors ULTRA's run_many.py: the ILPC and Ingram classes put
# validation targets in the test filter, every other inductive class does not.
GRAIL = dict(train="train.txt", inference="train_ind.txt",
             valid="valid_ind.txt", test="test_ind.txt", filter_valid=False)
INF = dict(train="transductive_train.txt", inference="inference_graph.txt",
           valid="inf_valid.txt", test="inf_test.txt", filter_valid=True)
HMF = dict(INF, filter_valid=False)
# WikiTopics, Metafam and FBNELL have no inference-side validation set at all;
# ULTRA sets valid_on_inf = False for them and validates on the transductive
# graph, which is what `valid_transductive` reproduces here.
MTDEA = dict(train="transductive_train.txt", inference="inference_graph.txt",
             valid="transductive_valid.txt", test="inf_test.txt",
             filter_valid=False, valid_transductive=True)

SOURCES = {}
for fam, cls in (("IndFB15k237", "FB15k237Inductive"), ("IndWN18RR", "WN18RRInductive"),
                 ("IndNELL", "NELLInductive")):
    for n in (1, 2, 3, 4):
        SOURCES["%s:v%d" % (cls, n)] = ("grail/%s/v%d/raw" % (fam, n), GRAIL)
for size in ("small", "large"):
    SOURCES["ILPC2022:%s" % size] = ("ilpc2022/%s/raw" % size, INF)
for short, pcts in (("fb", (25, 50, 75, 100)), ("wk", (25, 50, 75, 100)),
                    ("nl", (0, 25, 50, 75, 100))):
    cls = {"fb": "FBIngram", "wk": "WKIngram", "nl": "NLIngram"}[short]
    for p in pcts:
        SOURCES["%s:%d" % (cls, p)] = ("ingram/%s/%d/raw" % (short, p), INF)
for gid, sub in (("HM:1k", "Hamaguchi-BM_both-1000"), ("HM:3k", "Hamaguchi-BM_both-3000"),
                 ("HM:5k", "Hamaguchi-BM_both-5000"), ("HM:indigo", "INDIGO-BM")):
    SOURCES[gid] = ("hm/%s/raw" % sub, HMF)
for mt, subs in (("MT1", ("tax", "health")), ("MT2", ("org", "sci")),
                 ("MT3", ("art", "infra")), ("MT4", ("sci", "health"))):
    for sub in subs:
        SOURCES["WikiTopics%s:%s" % (mt, sub)] = ("mtdea/WikiTopics-%s/%s/raw" % (mt, sub), MTDEA)
SOURCES["Metafam"] = ("mtdea/Metafam/Metafam/raw", MTDEA)
SOURCES["FBNELL"] = ("mtdea/FBNELL/FBNELL_v1/raw", MTDEA)


def read_triples(path):
    """Raw files are whitespace separated; some families use spaces, some tabs."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3:
                # KG-ICL's own reader replaces '/' in a relation, because the
                # relation becomes a directory name under cases/.
                out.append((parts[0], parts[1].replace("/", "#"), parts[2]))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="processed_data directory to write")
    parser.add_argument("--only", default=None, help="comma-separated suite ids")
    parser.add_argument("--utils", default=None,
                        help="utils_all.py from datasets.zip (default: extract it)")
    args = parser.parse_args(argv)

    # utils_all.py lives inside datasets.zip. Extract rather than vendor it, so
    # the case-sampling code always matches the pinned repository.
    utils_dir = args.utils or os.path.join(args.out, "_kgicl_utils")
    if not args.utils:
        os.makedirs(utils_dir, exist_ok=True)
        with zipfile.ZipFile(os.path.join(WORKSPACE, "repos", "kg-icl", "datasets.zip")) as zf:
            with zf.open("utils_all.py") as src, open(os.path.join(utils_dir, "utils_all.py"), "wb") as dst:
                shutil.copyfileobj(src, dst)
    sys.path.insert(0, utils_dir)
    from utils_all import (KG, set2dict, get_entities_relations_from_triples,  # noqa: E402
                           triple2ids, write_triple, write_dict, write_cases)

    wanted = args.only.split(",") if args.only else [g for g in suite.ids("ind_e") + suite.ids("ind_er")]
    for gid in wanted:
        rel_dir, spec = SOURCES[gid]
        raw = os.path.join(ULTRA_ROOT, rel_dir)
        train = read_triples(os.path.join(raw, spec["train"]))
        inference = read_triples(os.path.join(raw, spec["inference"]))
        valid = read_triples(os.path.join(raw, spec["valid"]))
        test = read_triples(os.path.join(raw, spec["test"]))
        name = gid.replace(":", "_")

        # Two vocabularies: the transductive graph the model was never shown,
        # and the inference graph it is evaluated on. They share relations and
        # share no entities, which is the whole point of these datasets.
        ents_tr, rels_tr = get_entities_relations_from_triples(list(set(train + valid))
                                                              if spec.get("valid_transductive") else list(set(train)))
        ents_te, rels_te = get_entities_relations_from_triples(list(set(inference + valid + test))
                                                              if not spec.get("valid_transductive")
                                                              else list(set(inference + test)))
        e2i_tr, _ = set2dict(ents_tr)
        r2i_tr, _ = set2dict(rels_tr)
        e2i_te, _ = set2dict(ents_te)
        r2i_te, _ = set2dict(rels_te)

        train_i = triple2ids(train, e2i_tr, r2i_tr)
        inference_i = triple2ids(inference, e2i_te, r2i_te)
        test_i = triple2ids(test, e2i_te, r2i_te)
        if spec.get("valid_transductive"):
            valid_i = triple2ids(valid, e2i_tr, r2i_tr)
        else:
            valid_i = triple2ids(valid, e2i_te, r2i_te)

        # ULTRA's conventions, and the reason this script exists:
        #   test message graph = the inference graph, alone.
        #   test filter        = inference + test, plus valid only for ILPC/Ingram.
        test_background = inference_i
        test_filter = inference_i + test_i + (valid_i if spec["filter_valid"] else [])

        kg_te = KG(test_background, len(e2i_te), len(r2i_te))
        cases_te = kg_te.build_cases_for_large_graph(case_num=25, enclosing=False, hop=3)

        out_test = os.path.join(args.out, name, "test")
        shutil.rmtree(out_test, ignore_errors=True)
        os.makedirs(os.path.join(out_test, "cases"), exist_ok=True)
        write_triple(os.path.join(out_test, "background.txt"), test_background)
        write_triple(os.path.join(out_test, "facts.txt"), test_i)
        write_triple(os.path.join(out_test, "filter.txt"), test_filter)
        write_dict(os.path.join(out_test, "entity2id.txt"), e2i_te)
        write_dict(os.path.join(out_test, "relation2id.txt"), r2i_te)
        for relation, rid in r2i_te.items():
            os.makedirs(os.path.join(out_test, "cases", relation), exist_ok=True)
            block = cases_te.get(rid)
            if not block:
                continue
            for i in range(len(block)):
                write_cases(os.path.join(out_test, "cases", relation, str(i)), block[i])

        # The loader reads the sibling valid/ directory for answer_distance even
        # when only test is being evaluated, so it has to exist.
        valid_background = train_i if spec.get("valid_transductive") else inference_i
        out_valid = os.path.join(args.out, name, "valid")
        shutil.rmtree(out_valid, ignore_errors=True)
        os.makedirs(os.path.join(out_valid, "cases"), exist_ok=True)
        write_triple(os.path.join(out_valid, "background.txt"), valid_background)
        write_triple(os.path.join(out_valid, "facts.txt"), valid_i)
        write_triple(os.path.join(out_valid, "filter.txt"), valid_background + valid_i)
        v_e2i, v_r2i = (e2i_tr, r2i_tr) if spec.get("valid_transductive") else (e2i_te, r2i_te)
        write_dict(os.path.join(out_valid, "entity2id.txt"), v_e2i)
        write_dict(os.path.join(out_valid, "relation2id.txt"), v_r2i)
        for relation in v_r2i:
            os.makedirs(os.path.join(out_valid, "cases", relation), exist_ok=True)

        print("  {:24} train={:7} inference={:7} valid={:6} test={:6} filter={:7} rels={}".format(
            gid, len(train), len(inference), len(valid), len(test), len(test_filter), len(r2i_te)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
