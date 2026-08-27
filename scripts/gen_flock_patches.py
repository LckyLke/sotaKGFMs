#!/usr/bin/env python3
"""Generate patches/flock/*.diff by editing a scratch copy and diffing it.

Lives in the repository, not in a scratch directory. The generators for the
other repos did not, and when that directory was wiped the patches could no
longer be regenerated against a new upstream pin.
"""
import os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-flock-tree")
OUT = os.path.join(ROOT, "patches", "flock")


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout


def rank_dump_source():
    """Lift ultra/rank_dump.py out of the ULTRA patch: one Dumper, all repos."""
    text = open(os.path.join(ROOT, "patches/ultra/0001-rank-dump.diff")).read()
    start = text.index("+++ b/ultra/rank_dump.py")
    body = text[text.index("@@", start):]
    src = "\n".join(l[1:] for l in body.splitlines()[1:] if l.startswith("+")) + "\n"
    src = src.replace("``ultra/tasks.py``", "``flock/tasks.py``")
    src = src.replace("``script/run.py::test``", "``src_entity/run.py::test``")
    assert "class Dumper" in src and "import suite" in src
    return src


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
    print("wrote patches/flock/{}  ({} lines)".format(name, len(text.splitlines())))


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos/flock"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__"))
run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "upstream", cwd=SCRATCH)

# ------------------------------------------------------------------ 0001
open(os.path.join(SCRATCH, "src_entity", "flock", "rank_dump.py"), "w").write(rank_dump_source())
run("git", "add", "src_entity/flock/rank_dump.py", cwd=SCRATCH)

edit("src_entity/run.py", [
    ("from flock.models import Flock\n",
     "from flock.models import Flock\nfrom flock.rank_dump import Dumper\n"),
    ("def test(\n    cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False\n):",
     "def test(\n    cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False,\n    dump=None\n):"),
    ("    test_loader = torch_data.DataLoader(test_triplets, test_batch_size, sampler=sampler)\n",
     "    test_loader = torch_data.DataLoader(test_triplets, test_batch_size, sampler=sampler)\n"
     "\n    # rank dump only; everything below is untouched upstream code\n"
     "    dumper = Dumper(dump, test_triplets, world_size, rank, test_batch_size) if dump else None\n"),
    ("        rankings += [t_ranking, h_ranking]\n",
     "        if dumper is not None:\n"
     "            dumper.add(batch, t_ranking, h_ranking, num_t_negative, num_h_negative)\n\n"
     "        rankings += [t_ranking, h_ranking]\n"),
    ("    ranking = torch.cat(rankings)\n",
     "    if dumper is not None:\n"
     "        logger.warning(\"Rank dump written to %s\" % dumper.write())\n\n"
     "    ranking = torch.cat(rankings)\n"),
])

edit("src_entity/run_many.py", [
    ('    parser.add_argument("-s", "--subdir", help="result subdirectory")\n',
     '    parser.add_argument("-s", "--subdir", help="result subdirectory")\n'
     '    parser.add_argument("--rank_dump_dir",\n'
     '                        help="write one parquet of per-query test ranks per dataset here",\n'
     '                        default=None, type=str)\n'),
    ("            metrics = test(\n"
     "                cfg,\n"
     "                model,\n"
     "                test_data,\n"
     "                filtered_data=test_filtered_data,\n"
     "                return_metrics=True,\n"
     "                device=device,\n"
     "                logger=logger,\n"
     "            )",
     "            # `graph` is upstream's -d spelling, which suite.by_run_id\n"
     "            # normalises. FLOCK writes Metafam and FBNELL without a version,\n"
     "            # where ULTRA writes Metafam:Metafam and FBNELL:FBNELL_v1.\n"
     "            dump = None if args.rank_dump_dir is None else {\n"
     "                \"dir\": args.rank_dump_dir, \"dataset\": graph,\n"
     "                \"model\": \"flock\", \"seed\": seed,\n"
     "            }\n"
     "            metrics = test(\n"
     "                cfg,\n"
     "                model,\n"
     "                test_data,\n"
     "                filtered_data=test_filtered_data,\n"
     "                return_metrics=True,\n"
     "                device=device,\n"
     "                logger=logger,\n"
     "                dump=dump,\n"
     "            )"),
])
diff_out("0001-rank-dump.diff",
         "emit one parquet row per scored query into the shared ranks/ schema; dump only, the "
         "ranking itself is untouched. FLOCK's compute_ranking is byte-identical to ULTRA's and "
         "its test() differs only by an optional test_batch_size, so the rank definition carries "
         "over unchanged. The seed is recorded because FLOCK is the one model here whose inference "
         "is stochastic: it scores by sampling random walks and averaging test_samples of them, so "
         "a rank dump without its seed cannot be reproduced.",
         ["src_entity/run.py", "src_entity/run_many.py", "src_entity/flock/rank_dump.py"])
commit("0001")

# ------------------------------------------------------------------ 0002
configs = []
for kind in ("zeroshot_inductive", "zeroshot_transductive"):
    d = os.path.join(SCRATCH, "src_entity", "config", kind)
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        if not name.endswith(".yaml"):
            continue
        rel = os.path.join("src_entity", "config", kind, name)
        edit(rel, [
            ("output_dir: ~/flock/output-entity",
             "output_dir: {{ output_dir | default('/kgfm/output', true) }}"),
            ("root: ~/flock/kg-datasets-entity/",
             "root: {{ data_root | default('/kgfm/data/roots/flock/', true) }}"),
        ])
        configs.append(rel)
diff_out("0002-data-root.diff",
         "give FLOCK its own processed root (data/roots/flock/) so the cache in processed/data.pt "
         "cannot be shared with another repo's pre_transform. Upstream points both paths at ~/flock, "
         "which exists in no container here. Defaults are supplied so an unset variable is still a "
         "valid path rather than a jinja error.",
         configs)
print("\ndone: {} config files".format(len(configs)))
