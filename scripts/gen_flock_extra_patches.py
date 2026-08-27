#!/usr/bin/env python3
"""Generate patches/flock/0003-test-batch-size.diff and 0004-seed-numpy.diff."""
import os, re, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-flock-extra")
OUT = os.path.join(ROOT, "patches", "flock")


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout


def commit(msg):
    run("git", "add", "-A", cwd=SCRATCH)
    run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", msg, cwd=SCRATCH)


def diff_out(name, reason, paths):
    text = run("git", "diff", "HEAD", "--", *paths, cwd=SCRATCH)
    assert text.strip(), "no diff for " + name
    open(os.path.join(OUT, name), "w").write("Reason: " + reason + "\n\n" + text)
    print("wrote patches/flock/{}  ({} lines)".format(name, len(text.splitlines())))


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos/flock"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__"))
for name in sorted(os.listdir(OUT)):
    if name.endswith(".diff") and name[:4] in ("0001", "0002"):
        with open(os.path.join(OUT, name)) as h:
            subprocess.run(["patch", "-p1", "--batch", "--forward"],
                           cwd=SCRATCH, stdin=h, check=True, capture_output=True)
run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "0001-0002", cwd=SCRATCH)

# ------------------------------------------------------------------ 0003
configs = []
for kind in ("zeroshot_inductive", "zeroshot_transductive"):
    d = os.path.join(SCRATCH, "src_entity", "config", kind)
    if not os.path.isdir(d):
        continue
    for name in sorted(os.listdir(d)):
        if not name.endswith(".yaml"):
            continue
        rel = os.path.join("src_entity", "config", kind, name)
        full = os.path.join(SCRATCH, rel)
        text = open(full).read()
        found = re.search(r"^  batch_size: (\d+)$", text, re.M)
        assert found, rel
        size = found.group(1)
        # Default is this config's own batch_size, so leaving the variable unset
        # reproduces upstream behaviour exactly.
        text = text.replace(
            "  batch_size: {}\n".format(size),
            "  batch_size: {}\n  test_batch_size: {{{{ test_batch_size | default({}, true) }}}}\n".format(size, size),
            1)
        open(full, "w").write(text)
        configs.append(rel)
diff_out("0003-test-batch-size.diff",
         "let the evaluation batch size be set without touching any value that changes a result. "
         "FLOCK's README states it was tested on H100s. Every zero-shot config pairs its batch_size "
         "with its walk_num so the product is 512 -- 32x16, 16x32, 8x64, 4x128 -- and scoring one "
         "batch materialises a GRU input of batch x candidates x walk_num x walk_len x hidden. On a "
         "16 GB card that asks for 18.8 GiB on the first graph and the run dies. run.py already "
         "reads cfg.train.test_batch_size when it is present and falls back to batch_size, so this "
         "only exposes the knob it already honours. The default is each config's own batch_size, so "
         "an unset variable is upstream behaviour unchanged. Nothing here alters walk_num, "
         "test_samples, walk_len or any model hyper-parameter.",
         configs)
commit("0003")

# ------------------------------------------------------------------ 0004
path = os.path.join(SCRATCH, "src_entity", "run_many.py")
text = open(path).read()
OLD = """def set_seed(seed):
    random.seed(seed + util.get_rank())
    # np.random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed(seed + util.get_rank())"""
NEW = '''def set_seed(seed):
    random.seed(seed + util.get_rank())
    # Upstream ships this line commented out. It is the only one that reaches
    # the random walks, and the walks are how FLOCK scores: models.py calls
    # graph_walker.random_walks_fast without a seed and graph_walker._seed(None),
    # both of which draw from numpy's global generator. With numpy unseeded,
    # every run of the same graph with the same checkpoint returns different
    # numbers, and no result can be reproduced or compared with another model's
    # to a stated precision.
    #
    # Seeding it does not make this run match the published one -- the published
    # one was not reproducible either -- it makes this run reproducible at all.
    # Set FLOCK_UNSEEDED_WALKS=1 to restore upstream behaviour, which is what to
    # do when measuring how much of a FLOCK number is walk-sampling noise.
    if not os.environ.get("FLOCK_UNSEEDED_WALKS"):
        np.random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed(seed + util.get_rank())'''
assert text.count(OLD) == 1
text = text.replace(OLD, NEW)
if "\nimport numpy as np\n" not in text:
    text = text.replace("import torch\n", "import torch\nimport numpy as np\n", 1)
open(path, "w").write(text)
diff_out("0004-seed-numpy.diff",
         "make FLOCK's inference reproducible. It is the only stochastic model in this suite: it "
         "scores by sampling random walks and averaging an ensemble of them. run_many.py::set_seed "
         "seeds python's random and torch, but ships the numpy line commented out -- and numpy's "
         "generator is the one the walk sampler actually uses, through "
         "graph_walker.random_walks_fast(seed=None) and graph_walker._seed(None). As shipped, two "
         "runs of one graph from one checkpoint give different numbers, so nothing can be "
         "reproduced or compared to a stated precision. The seed value is unchanged: run_many.py "
         "already picks 1024 for the first repeat, the seed every other model here runs under. "
         "FLOCK_UNSEEDED_WALKS=1 restores upstream behaviour for measuring the sampling spread.",
         ["src_entity/run_many.py"])
print("\ndone")
