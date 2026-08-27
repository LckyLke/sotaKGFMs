#!/usr/bin/env python3
"""Generate patches/trix/0004-relation-rank-offset.diff.

Lives in the repository like the other patch generators (see
scripts/gen_flock_patches.py's docstring for why): copy repos/trix to a
scratch tree, apply the existing patches, edit, `git diff`. repos/trix itself
is never touched.
"""
import os, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-trix-relation")
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


def diff_out(name, reason, paths):
    text = run("git", "diff", "HEAD", "--", *paths, cwd=SCRATCH)
    assert text.strip(), "no diff for " + name
    open(os.path.join(OUT, name), "w").write("Reason: " + reason + "\n\n" + text)
    print("wrote patches/trix/{}  ({} lines)".format(name, len(text.splitlines())))


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos/trix"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__"))
# layer on top of the existing patch set, like gen_flock_extra_patches.py does
for name in sorted(os.listdir(OUT)):
    if name.endswith(".diff") and name[:4] in ("0001", "0002", "0003"):
        with open(os.path.join(OUT, name)) as h:
            subprocess.run(["patch", "-p1", "--batch", "--forward"],
                           cwd=SCRATCH, stdin=h, check=True, capture_output=True)
run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "0001-0003", cwd=SCRATCH)

# ------------------------------------------------------------------ 0004
edit("src/trix/tasks.py", [
    ("def compute_ranking_relation(pred, target, mask=None):\n"
     "    pos_pred = pred.gather(-1, target.unsqueeze(-1))\n"
     "    if mask is not None:\n"
     "        # filtered ranking\n"
     "        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1\n"
     "    else:\n"
     "        # unfiltered ranking\n"
     "        ranking = torch.sum(pos_pred <= pred, dim=-1)\n"
     "    return ranking",
     "def compute_ranking_relation(pred, target, mask=None):\n"
     "    pos_pred = pred.gather(-1, target.unsqueeze(-1))\n"
     "    if mask is not None:\n"
     "        # filtered ranking\n"
     "        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1\n"
     "    else:\n"
     "        # unfiltered ranking\n"
     "        # + 1 restored: without it a perfect prediction ranks 0 and its\n"
     "        # reciprocal is infinite; every other ranking in this repo (the\n"
     "        # filtered branch above, both branches of compute_ranking) is\n"
     "        # 1-based, and src/run_relation.py::test calls this with\n"
     "        # mask=None, so relation MRR was computed over 0-based ranks.\n"
     "        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1\n"
     "    return ranking"),
])
diff_out("0004-relation-rank-offset.diff",
         "make compute_ranking_relation's unfiltered branch 1-based like every other ranking here. "
         "Upstream omits the + 1 that its own filtered branch and both branches of compute_ranking "
         "have, so a perfect prediction ranks 0 and 1/rank is infinite; src/run_relation.py::test "
         "always calls it with mask=None (line 149), so the unfiltered path is not dead code and "
         "the omission reaches every relation metric. No entity number moves: nothing on the "
         "entity-prediction path calls this function. docs/CREST_PLAN.md phase 1 requires this "
         "settled before any relation number is recorded.",
         ["src/trix/tasks.py"])
print("done")
