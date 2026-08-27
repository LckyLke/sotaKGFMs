#!/usr/bin/env python3
"""Generate patches/semma/0006-skip-entity-labels.diff (stacks on 0001-0005)."""
import os, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-semma-ent")
OUT = os.path.join(ROOT, "patches", "semma")


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos/semma"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__"))
for name in sorted(os.listdir(OUT)):
    if name.endswith(".diff") and not name.startswith("0006"):
        with open(os.path.join(OUT, name)) as h:
            subprocess.run(["patch", "-p1", "--batch", "--forward"],
                           cwd=SCRATCH, stdin=h, check=True, capture_output=True)
run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "0001-0005", cwd=SCRATCH)

path = os.path.join(SCRATCH, "ultra", "datasets.py")
text = open(path).read()

OLD = "def get_entities(entity_ids):\n    _requested = list(entity_ids)\n"
NEW = '''def get_entities(entity_ids):
    # Entity labels are never read. They are fetched from Wikidata one HTTP
    # request per 50 ids, copied onto train_/valid_/test_ prefixed attributes,
    # and then the originals are deleted by attrs_to_remove before the dataset
    # is saved. Nothing in ultra/, script/ or anywhere else in this repository
    # reads id2entity or any of its prefixed copies -- the semantic half of
    # SEMMA is built from *relation* descriptions, not entity ones.
    #
    # Across the 14 Wikidata graphs in this suite that is 225816 entities, about
    # 4500 API requests, for a value that is discarded. Skipping it changes no
    # result and takes hours off a run. The identity mapping is returned rather
    # than an empty dict so that any future reader sees an id instead of a
    # KeyError. Set SEMMA_FETCH_ENTITY_LABELS=1 to restore upstream behaviour.
    if not os.environ.get("SEMMA_FETCH_ENTITY_LABELS"):
        return {i: i for i in entity_ids}
    _requested = list(entity_ids)
'''
assert text.count(OLD) == 1
text = text.replace(OLD, NEW)

# The label cache is read once and held, not re-parsed per batch. With entity
# labels skipped it stays small, but the quadratic cost was real: get_entities
# ran once per 50 ids and parsed the whole file each time.
OLD_READ = '''def _wikidata_cache_read():
    try:
        with open(WIKIDATA_CACHE) as handle:
            return json.load(handle)
    except Exception:
        return {}'''
NEW_READ = '''_WIKIDATA_CACHE_MEM = None


def _wikidata_cache_read():
    # Held in memory. This used to re-parse the whole file on every call, and
    # every call is one 50-id batch, so filling a large cache cost O(n^2) in
    # JSON parsing before a single label was used.
    global _WIKIDATA_CACHE_MEM
    if _WIKIDATA_CACHE_MEM is None:
        try:
            with open(WIKIDATA_CACHE) as handle:
                _WIKIDATA_CACHE_MEM = json.load(handle)
        except Exception:
            _WIKIDATA_CACHE_MEM = {}
    return _WIKIDATA_CACHE_MEM'''
assert text.count(OLD_READ) == 1
text = text.replace(OLD_READ, NEW_READ)

# Make the cache readable outside the container. mkstemp creates 0600, and the
# file is written by root through a bind mount.
text = text.replace("        os.replace(tmp, WIKIDATA_CACHE)",
                    "        os.replace(tmp, WIKIDATA_CACHE)\n"
                    "        os.chmod(WIKIDATA_CACHE, 0o644)")
text = text.replace("        os.replace(tmp, _PROPS_CACHE)",
                    "        os.replace(tmp, _PROPS_CACHE)\n"
                    "        os.chmod(_PROPS_CACHE, 0o644)")

open(path, "w").write(text)
diff = run("git", "diff", "HEAD", "--", "ultra/datasets.py", cwd=SCRATCH)
assert diff.strip()
reason = (
    "stop fetching 225816 Wikidata entity labels that nothing reads. For the 14 Wikidata graphs "
    "in this suite, datasets.py resolves every entity id to an English label, one request per 50 "
    "ids, roughly 4500 requests -- and then attrs_to_remove deletes id2entity before the dataset "
    "is saved. id2entity and its train_/valid_/test_ copies appear nowhere in this repository "
    "outside datasets.py, and never on the right-hand side: SEMMA's semantic stream is built from "
    "relation descriptions, not entity ones. Relation labels are still fetched, because those are "
    "what edge2id and the description lookup are keyed by. This cannot change a result and takes "
    "hours off a run; SEMMA_FETCH_ENTITY_LABELS=1 restores upstream behaviour. Also holds the "
    "label cache in memory instead of re-parsing the file once per batch, which was quadratic, and "
    "chmods it readable so it can be inspected outside the container.")
open(os.path.join(OUT, "0006-skip-entity-labels.diff"), "w").write("Reason: " + reason + "\n\n" + diff)
print("wrote patches/semma/0006-skip-entity-labels.diff ({} lines)".format(len(diff.splitlines())))
