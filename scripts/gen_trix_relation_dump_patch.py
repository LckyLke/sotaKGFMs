#!/usr/bin/env python3
"""Generate patches/trix/0004-relation-rank-dump.diff (stacks on 0001-0003).

This harness never measured relation prediction before: this baseline runs
TRIX's own run_relation.py with relation_prediction.pth. (The baseline was
first built for the CREST project, which lives on the `crest` branch.)
This patch gives that script the
same two outputs every other run here has -- a per-query rank dump and a CSV of
the model's own metric values for criterion A. The ranking itself is untouched;
in particular compute_ranking_relation's unfiltered branch is left exactly as
shipped, because it is correct as shipped (docs/report_notes.md, "The unfiltered
rank offset is correct, and looks like a bug").
"""
import os, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-trix-reldump")
OUT = os.path.join(ROOT, "patches", "trix")


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout


def edit(path, pairs):
    full = os.path.join(SCRATCH, path)
    text = open(full).read()
    for old, new in pairs:
        assert text.count(old) == 1, "anchor {!r} x{} in {}".format(old[:70], text.count(old), path)
        text = text.replace(old, new)
    open(full, "w").write(text)


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos/trix"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pth"))
for name in sorted(os.listdir(OUT)):
    if name.endswith(".diff") and not name.startswith("0004"):
        with open(os.path.join(OUT, name)) as h:
            subprocess.run(["patch", "-p1", "--batch", "--forward"],
                           cwd=SCRATCH, stdin=h, check=True, capture_output=True)
run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "base", cwd=SCRATCH)

DUMPER = '''"""Per-query rank dump for relation prediction, in the shared ranks schema.

One row per test triple, ``direction = "relation"``. The semantics mirror what
``run_relation.py::test`` computes and nothing more:

* the candidate set is the direct relations only -- ``all_negative_relation``
  builds ``arange(num_relations // 2)`` -- so ``n_candidates`` is the direct
  count minus the target, and ``rank <= n_candidates + 1`` holds by
  construction;
* the ranking is UNFILTERED: other true relations between the same pair stay in
  the candidate set. That is TRIX's own protocol for this task, stated in
  docs/report_notes.md, and belongs beside any table built from these dumps;
* the rank is 1-based because the unfiltered comparison counts the target
  against itself; there is no ``+ 1`` and none is needed.

Query order is recovered the same way the entity Dumper recovers it: a second
``DistributedSampler`` with the same (length, world_size, rank) yields the same
permutation, so a stable ``query_id`` can be attached without touching the real
loader.
"""

import os

from torch.utils import data as torch_data

import suite  # shared/suite.py, on PYTHONPATH


class RelationDumper:

    def __init__(self, spec, test_triplets, world_size, rank, batch_size):
        if world_size != 1:
            raise RuntimeError(
                "rank dumping is single-process only (run with --gpus [0]); got "
                "world_size=%d" % world_size)
        self.spec = spec
        self.graph = suite.by_run_id(spec["dataset"])
        self.order = [int(i) for i in torch_data.DistributedSampler(test_triplets, world_size, rank)]
        self.cursor = 0
        self.columns = {name: [] for name in suite.RANK_COLUMNS}

    def add(self, batch, r_ranking, num_negative):
        size = len(batch)
        query_ids = self.order[self.cursor:self.cursor + size]
        self.cursor += size
        assert len(query_ids) == size, "sampler order desynchronised from the loader"

        pos_h, pos_t, pos_r = [column.tolist() for column in batch.t()]
        ranking = r_ranking.tolist()
        for i in range(size):
            self.columns["dataset"].append(self.graph.id)
            self.columns["model"].append(self.spec["model"])
            self.columns["seed"].append(self.spec["seed"])
            self.columns["direction"].append("relation")
            self.columns["query_id"].append(query_ids[i])
            self.columns["h"].append(pos_h[i])
            self.columns["r"].append(pos_r[i])
            self.columns["t"].append(pos_t[i])
            self.columns["rank"].append(ranking[i])
            self.columns["n_candidates"].append(int(num_negative))

    def write(self):
        import pyarrow
        import pyarrow.parquet

        arrow_types = {"string": pyarrow.string(), "int64": pyarrow.int64()}
        schema = pyarrow.schema([
            (name, arrow_types[suite.RANK_COLUMN_TYPES[name]]) for name in suite.RANK_COLUMNS
        ])
        table = pyarrow.table({name: self.columns[name] for name in suite.RANK_COLUMNS}, schema=schema)

        os.makedirs(self.spec["dir"], exist_ok=True)
        path = os.path.join(self.spec["dir"], "%s.parquet" % self.graph.id.replace(":", "_"))
        pyarrow.parquet.write_table(table, path)
        return path
'''
open(os.path.join(SCRATCH, "src", "trix", "rank_dump_relation.py"), "w").write(DUMPER)
run("git", "add", "src/trix/rank_dump_relation.py", cwd=SCRATCH)

edit("src/run_relation.py", [
    ("from trix.models_relation import TRIX\n",
     "from trix.models_relation import TRIX\nfrom trix.rank_dump_relation import RelationDumper\n"),
    ("def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False):",
     "def test(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False, dump=None):"),
    ("    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)\n",
     "    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)\n"
     "\n    # rank dump only; everything below is untouched upstream code\n"
     "    dumper = RelationDumper(dump, test_triplets, world_size, rank, cfg.train.batch_size) if dump else None\n"),
    ("        rankings += [r_ranking]\n",
     "        if dumper is not None:\n"
     "            dumper.add(batch, r_ranking, test_graph.num_relations.item() // 2 - 1)\n"
     "        rankings += [r_ranking]\n"),
    ("    ranking = torch.cat(rankings)\n",
     "    if dumper is not None:\n"
     "        logger.warning(\"Rank dump written to %s\" % dumper.write())\n\n"
     "    ranking = torch.cat(rankings)\n"),
    ("    test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger)",
     "    dump = None if not os.environ.get(\"TRIX_RANK_DUMP_DIR\") else {\n"
     "        \"dir\": os.environ[\"TRIX_RANK_DUMP_DIR\"],\n"
     "        \"dataset\": vars[\"dataset\"] + (\":\" + str(vars[\"version\"]) if vars.get(\"version\") not in (None, \"\") else \"\"),\n"
     "        \"model\": \"trix\", \"seed\": args.seed,\n"
     "    }\n"
     "    metrics = test(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, dump=dump, return_metrics=True)\n"
     "\n"
     "    # The model's own metric values, in the shape criterion A reads. Mirrors\n"
     "    # patches/trix/0003 for the entity task; upstream only logs them.\n"
     "    if util.get_rank() == 0:\n"
     "        import csv as _csv, time as _time\n"
     "        _row = {k: v.item() if hasattr(v, \"item\") else v for k, v in metrics.items()}\n"
     "        _row[\"dataset\"] = dump[\"dataset\"] if dump else vars[\"dataset\"]\n"
     "        _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), \"TRIX_relation_results\")\n"
     "        os.makedirs(_dir, exist_ok=True)\n"
     "        _path = os.path.join(_dir, \"TRIX_relation_results_%s.csv\" % _time.strftime(\"%Y-%m-%d-%H-%M-%S\"))\n"
     "        with open(_path, \"a\", newline=\"\") as _f:\n"
     "            _w = _csv.DictWriter(_f, fieldnames=[\"dataset\"] + [k for k in _row if k != \"dataset\"])\n"
     "            if _f.tell() == 0:\n"
     "                _w.writeheader()\n"
     "            _w.writerow(_row)\n"
     "        logger.warning(\"Results written to %s\" % _path)"),
])

# Patch 0002 templated only the entity configs; the relation configs still
# hardcode /output and /kg-datasets/, which exist in no container here. Same
# treatment, same reasoning as 0002: no jinja default on purpose, parse_args
# marks every template variable required and the caller must state both paths.
for cfg in ("config/run_relation_inductive.yaml", "config/run_relation_transductive.yaml"):
    edit(cfg, [
        ("output_dir: /output", "output_dir: {{ output_dir }}"),
        ("root: /kg-datasets/", "root: {{ data_root }}"),
    ])

diff = run("git", "diff", "HEAD", "--",
           "src/run_relation.py", "src/trix/rank_dump_relation.py",
           "config/run_relation_inductive.yaml", "config/run_relation_transductive.yaml",
           cwd=SCRATCH)
assert diff.strip()
reason = (
    "give run_relation.py the two outputs every other run here has: a per-query rank dump in the "
    "shared schema (direction 'relation', one row per test triple) and a CSV of the model's own "
    "metric values for criterion A, which upstream logs and discards. The ranking is untouched. "
    "n_candidates is the direct-relation count minus the target, matching all_negative_relation's "
    "candidate set of arange(num_relations // 2), so rank <= n_candidates + 1 holds by "
    "construction. The evaluation is unfiltered -- other true relations between the same pair "
    "stay candidates -- which is TRIX's own protocol for this task and is recorded in "
    "docs/report_notes.md; compute_ranking_relation is deliberately not modified. The dump "
    "directory arrives by the same TRIX_RANK_DUMP_DIR environment variable as patch 0001, for "
    "the same reason: parse_args marks every config template variable required. Also templates the relation configs, whose output_dir and data root patch 0002 only fixed for the entity configs, leaving /output and /kg-datasets/ hardcoded on every relation run.")
open(os.path.join(OUT, "0004-relation-rank-dump.diff"), "w").write("Reason: " + reason + "\n\n" + diff)
print("wrote patches/trix/0004-relation-rank-dump.diff ({} lines)".format(len(diff.splitlines())))
