#!/usr/bin/env python3
"""Generate patches/kg-icl/*.diff by editing a scratch copy and diffing it."""
import os, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-kgicl-tree")
OUT = os.path.join(ROOT, "patches", "kg-icl")


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout


def edit(path, pairs):
    full = os.path.join(SCRATCH, path)
    text = open(full).read()
    for old, new in pairs:
        assert text.count(old) == 1, "anchor {!r} x{} in {}".format(old[:70], text.count(old), path)
        text = text.replace(old, new)
    open(full, "w").write(text)


def commit(msg):
    run("git", "add", "-A", cwd=SCRATCH)
    run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", msg, cwd=SCRATCH)


def diff_out(name, reason, paths):
    text = run("git", "diff", "HEAD", "--", *paths, cwd=SCRATCH)
    assert text.strip(), "no diff for " + name
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write("Reason: " + reason + "\n\n" + text)
    print("wrote patches/kg-icl/{}  ({} lines)".format(name, len(text.splitlines())))


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos/kg-icl"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "datasets.zip",
                                              "*.pdf", "*.jpg", "*.png"))
# KG-ICL's sources are CRLF, and `git diff` emits LF context regardless, so a
# patch generated from them can never apply: `patch` refuses with "different
# line endings". Both sides are normalised to LF instead -- here before the
# baseline commit, and in containers/kg-icl/Dockerfile before the patches are
# applied. Line endings are the only thing this changes, and Python does not
# care about them; it keeps every patch a readable diff of the lines it means
# to change rather than a whole-file rewrite.
for _root, _dirs, _files in os.walk(SCRATCH):
    for _name in _files:
        if not _name.endswith(".py"):
            continue
        _path = os.path.join(_root, _name)
        with open(_path, "rb") as _h:
            _raw = _h.read()
        if b"\r\n" in _raw:
            with open(_path, "wb") as _h:
                _h.write(_raw.replace(b"\r\n", b"\n"))

run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "upstream", cwd=SCRATCH)

RANK_DUMP = '''"""Per-query rank dump for the shared ``ranks/`` schema, for KG-ICL.

KG-ICL is not an ULTRA fork and its evaluation loop has a different shape, so
this cannot be the same Dumper the other repos use. Two differences matter.

Grouping. KG-ICL scores one row per *query* ``(h, r)`` and carries a list of
true answers for it, emitting one rank per answer. The ULTRA-derived repos score
one row per test triple. The two agree in total, and this module emits one row
per (query, answer) pair, which is the same set of scored facts.

Ranking. ``utils.cal_ranks`` breaks ties with ``rankdata(method='ordinal')``,
which resolves a tie by entity id. That is neither the pessimistic rule ULTRA
uses -- where every tied candidate counts against the answer -- nor the
optimistic one, and it cannot be compared with either. So both are recorded:

  ``rank``         the shared definition, 1-based, pessimistic ties, strict
                   filtering. This is the column every cross-model table reads.
  ``rank_native``  exactly what ``cal_ranks`` returned, so criterion A can still
                   check this project's metric code against KG-ICL's own.

The two are computed from the same scores in the same call. Neither is derived
from the other.
"""

import os

import numpy as np

import suite  # shared/suite.py, on PYTHONPATH


class Dumper:
    """Accumulates one row per (query, answer) and writes a single parquet."""

    EXTRA = ["rank_native"]

    def __init__(self, spec):
        self.spec = spec
        self.graph = suite.by_run_id(spec["dataset"])
        self.columns = {name: [] for name in list(suite.RANK_COLUMNS) + self.EXTRA}
        self.query_id = 0

    def add(self, subs, rels, answers, scores, filters, native_ranks, n_relations):
        """One batch. ``filters[i]`` is 1 for a known true answer of query i."""
        native = list(native_ranks)
        cursor = 0
        for i in range(len(subs)):
            keep = filters[i] == 0                      # candidates ULTRA would score
            n_candidates = int(keep.sum())
            row_scores = scores[i]
            for target in answers[i]:
                target = int(target)
                # The shared definition: 1-based, pessimistic ties, strict
                # filtering. Every surviving candidate scoring at least as high
                # as the answer counts against it, which is what
                # ``torch.sum((pos_pred <= pred) & mask) + 1`` does in ULTRA.
                rank = int(np.count_nonzero(keep & (row_scores >= row_scores[target]))) + 1
                # A relation id at or above half the vocabulary is KG-ICL's
                # inverse of a base relation, which is how it asks a head
                # question. The stored triple is put back in base orientation so
                # (h, r, t) means the same thing as in every other model's dump.
                inverse = int(rels[i]) >= n_relations // 2
                base_r = int(rels[i]) - n_relations // 2 if inverse else int(rels[i])
                head, tail = (target, int(subs[i])) if inverse else (int(subs[i]), target)
                self.columns["dataset"].append(self.graph.id)
                self.columns["model"].append(self.spec["model"])
                self.columns["seed"].append(self.spec["seed"])
                self.columns["direction"].append("head" if inverse else "tail")
                self.columns["query_id"].append(self.query_id)
                self.columns["h"].append(head)
                self.columns["r"].append(base_r)
                self.columns["t"].append(tail)
                self.columns["rank"].append(rank)
                self.columns["n_candidates"].append(n_candidates)
                self.columns["rank_native"].append(int(native[cursor]) if cursor < len(native) else -1)
                cursor += 1
                self.query_id += 1

    def write(self):
        import pyarrow
        import pyarrow.parquet

        arrow_types = {"string": pyarrow.string(), "int64": pyarrow.int64()}
        fields = [(name, arrow_types[suite.RANK_COLUMN_TYPES[name]]) for name in suite.RANK_COLUMNS]
        fields += [(name, pyarrow.int64()) for name in self.EXTRA]
        schema = pyarrow.schema(fields)
        names = list(suite.RANK_COLUMNS) + self.EXTRA
        table = pyarrow.table({n: self.columns[n] for n in names}, schema=schema)

        os.makedirs(self.spec["dir"], exist_ok=True)
        path = os.path.join(self.spec["dir"], "%s.parquet" % self.graph.id.replace(":", "_"))
        pyarrow.parquet.write_table(table, path)
        return path
'''

open(os.path.join(SCRATCH, "src", "rank_dump.py"), "w").write(RANK_DUMP)
run("git", "add", "src/rank_dump.py", cwd=SCRATCH)

edit("src/experiment.py", [
    ("from utils import cal_ranks, cal_performance\n",
     "from utils import cal_ranks, cal_performance\nfrom rank_dump import Dumper\n"),
    ("                        ranking = []\n",
     "                        ranking = []\n"
     "                        # rank dump only; the ranking below is untouched\n"
     "                        dump_dir = getattr(self.args, 'rank_dump_dir', None)\n"
     "                        dumper = None\n"
     "                        if dump_dir and mode == 'test':\n"
     "                            dumper = Dumper({'dir': dump_dir, 'dataset': loader.name,\n"
     "                                             'model': 'kg-icl', 'seed': self.args.seed})\n"),
    ("                            ranks = cal_ranks(scores, objs, filters)\n"
     "                            ranking += list(ranks)\n",
     "                            ranks = cal_ranks(scores.copy(), objs, filters)\n"
     "                            if dumper is not None:\n"
     "                                dumper.add(subs, rels, [np.nonzero(o)[0] for o in objs],\n"
     "                                           scores, filters, ranks,\n"
     "                                           loader.kg.relation_num)\n"
     "                            ranking += list(ranks)\n"),
    ("                        ranking = np.array(ranking)\n",
     "                        if dumper is not None:\n"
     "                            print('Rank dump written to', dumper.write())\n"
     "                        ranking = np.array(ranking)\n"),
])
diff_out("0001-rank-dump.diff",
         "emit one parquet row per scored fact into the shared ranks/ schema, carrying BOTH rank "
         "definitions. KG-ICL's utils.cal_ranks breaks ties with rankdata(method='ordinal'), which "
         "resolves a tie by entity id -- neither the pessimistic rule ULTRA uses nor the optimistic "
         "one, so it is not comparable with either. `rank` is recomputed from the same scores under "
         "the shared definition and is what cross-model tables read; `rank_native` is what "
         "cal_ranks returned, so criterion A can still compare this project's metric code against "
         "KG-ICL's own numbers. The ranking KG-ICL reports is unchanged: cal_ranks now gets a copy "
         "of the scores because it mutates its argument in place (filter_scores = scores, then "
         "filter_scores[...] = -1e7), which would otherwise corrupt the shared rank computed after "
         "it. Head questions are stored in base orientation so (h, r, t) means the same thing here "
         "as in every other model's dump.",
         ["src/experiment.py", "src/rank_dump.py"])
commit("0001")

edit("src/evaluation.py", [
    ("parser.add_argument('--use_rspmm', type=bool, default=False)",
     "parser.add_argument('--use_rspmm', type=bool, default=False)\n"
     "parser.add_argument('--rank_dump_dir', type=str, default=None,\n"
     "                    help='write one parquet of per-query test ranks per dataset here')\n"
     "parser.add_argument('--results_csv', type=str, default=None,\n"
     "                    help=\"write KG-ICL's own metric values here, for criterion A\")"),
])
diff_out("0002-cli-flags.diff",
         "add --rank_dump_dir and --results_csv. Upstream logs its metrics and discards them, so "
         "there is nothing for criterion A to compare against; ULTRA, MOTIF and TRIX all write a "
         "CSV and this makes KG-ICL comparable the same way. Applies on top of 0001.",
         ["src/evaluation.py"])
commit("0002")

# ------------------------------------------------------------------ 0003
edit("src/experiment.py", [
    ("""                        self.args.logger.info(loader.name+' MRR:%.4f H@1:%.4f H@3:%.4f H@5:%.4f H@10:%.4f[TIME] train:%.4f inference:%.4f\\n' % (
                            v_mrr, v_h1, v_h3, v_h5, v_h10, self.t_time, i_time))""",
     """                        self.args.logger.info(loader.name+' MRR:%.4f H@1:%.4f H@3:%.4f H@5:%.4f H@10:%.4f[TIME] train:%.4f inference:%.4f\\n' % (
                            v_mrr, v_h1, v_h3, v_h5, v_h10, self.t_time, i_time))

                        # Record the metric values KG-ICL computed, so criterion
                        # A can be run against them. Upstream logs them and
                        # discards them. These come from cal_performance over
                        # cal_ranks' own output, so they describe rank_native in
                        # the dump, not the shared `rank` column.
                        csv_path = getattr(self.args, 'results_csv', None)
                        if csv_path and mode == 'test':
                            import csv as _csv
                            _new = not os.path.exists(csv_path)
                            _dir = os.path.dirname(csv_path)
                            if _dir:
                                os.makedirs(_dir, exist_ok=True)
                            with open(csv_path, 'a', newline='') as _f:
                                _w = _csv.writer(_f)
                                if _new:
                                    _w.writerow(['dataset', 'mrr', 'hits@1', 'hits@3', 'hits@5', 'hits@10'])
                                _w.writerow([loader.name, v_mrr, v_h1, v_h3, v_h5, v_h10])"""),
])
diff_out("0003-results-csv.diff",
         "record the metric values KG-ICL computes. Upstream logs them and discards them, so there "
         "is nothing for criterion A to compare this project's metric code against. The values "
         "come from cal_performance over cal_ranks' output, so they correspond to the rank_native "
         "column of the dump, not to the shared rank -- the two are different definitions and "
         "comparing our metrics over `rank` against these would fail by construction and mean "
         "nothing. Applies on top of 0002.",
         ["src/experiment.py"])

commit("0003")

# ------------------------------------------------------------------ 0004
edit("src/data_loader.py", [
    ("            query, answer = np.array(self.kg.query), np.array(self.kg.answer, dtype=object)",
     "            # np.array(list_of_lists, dtype=object) only produces the ragged\n"
     "            # 1-D array this code expects when the lists have *different*\n"
     "            # lengths. When every query has exactly one answer numpy builds a\n"
     "            # 2-D (n, 1) object array instead, answer[i] is then an ndarray\n"
     "            # rather than a list, and the line below fails with\n"
     "            #   IndexError: arrays used as indices must be of integer type\n"
     "            # Of the 41 graphs in this suite exactly one, Metafam, has a\n"
     "            # single answer for every query, which is why the shipped\n"
     "            # datasets never trip it.\n"
     "            query = np.array(self.kg.query)\n"
     "            answer = np.empty(len(self.kg.answer), dtype=object)\n"
     "            answer[:] = self.kg.answer"),
])
diff_out("0004-ragged-answers.diff",
         "build the answer array as a genuine ragged 1-D object array. get_batch relies on "
         "np.array(self.kg.answer, dtype=object) giving one entry per query, each a list of true "
         "answers, but numpy only does that when the lists differ in length. Where every query has "
         "exactly one answer it builds a 2-D (n, 1) array, answer[i] becomes an ndarray, and using "
         "it as an index raises IndexError. Metafam is the one graph in this suite with a single "
         "answer per query, and it fails outright without this. Nothing changes for any dataset "
         "that was already working: the object array holds the same lists. Applies on top of 0003.",
         ["src/data_loader.py"])

print("\ndone")
