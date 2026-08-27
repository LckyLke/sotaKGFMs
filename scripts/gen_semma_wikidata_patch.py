#!/usr/bin/env python3
"""Generate patches/semma/0004-wikidata-labels.diff.

Applies the three existing SEMMA patches to a scratch copy first, so this one
stacks on them exactly as the container applies them in order.
"""
import os, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-semma-tree")
OUT = os.path.join(ROOT, "patches", "semma")


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos/semma"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__"))
for name in sorted(os.listdir(OUT)):
    if name.endswith(".diff") and not name.startswith("0004"):
        with open(os.path.join(OUT, name)) as handle:
            subprocess.run(["patch", "-p1", "--batch", "--forward"],
                           cwd=SCRATCH, stdin=handle, check=True, capture_output=True)
run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "0001-0003", cwd=SCRATCH)

path = os.path.join(SCRATCH, "ultra", "datasets.py")
text = open(path).read()

OLD = '''def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        return requests.get(url, params=params)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
'''

NEW = '''import tempfile

# Wikimedia's User-Agent policy refuses the default python-requests string.
# Every call below answers 403 without one, and the failure is silent: the
# helpers return "Failed to retrieve data" for every id, edge2id is then built
# by inverting id2relation, all those identical values collapse to a single
# entry, and the run dies 300 lines later in order_embeddings with KeyError: 0.
# Set WIKIDATA_USER_AGENT to identify your own run; the policy asks for a way to
# be contacted.
WIKIDATA_USER_AGENT = os.environ.get(
    "WIKIDATA_USER_AGENT",
    "sotaKGFMs-benchmark/1.0 (https://github.com/LckyLke/sotaKGFMs) python-requests")

# Labels are cached on disk. Two reasons, and the second is the important one.
# The API is queried once per relation instead of once per dataset build. And a
# label on Wikidata can be edited: SEMMA keys its LLM relation descriptions by
# that label, and builds half its model from the embedding of it, so an
# uncached run months later can silently produce a different relation graph
# from the same checkpoint and the same data. The cache pins what was read.
WIKIDATA_CACHE = os.environ.get(
    "SEMMA_WIKIDATA_CACHE", os.path.join(mydir, "wikidata_labels.json"))


def _wikidata_cache_read():
    try:
        with open(WIKIDATA_CACHE) as handle:
            return json.load(handle)
    except Exception:
        return {}


def _wikidata_cache_write(cache):
    try:
        os.makedirs(os.path.dirname(WIKIDATA_CACHE) or ".", exist_ok=True)
        # Unique temp name per writer. get_entities runs inside
        # fetch_in_parallel's ThreadPoolExecutor, so a fixed ".tmp" path is
        # written and renamed by several threads at once and all but one of
        # them fail with ENOENT on the rename.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(WIKIDATA_CACHE) or ".", suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(cache, handle, indent=0, sort_keys=True)
        os.replace(tmp, WIKIDATA_CACHE)
    except Exception as e:
        print(f"could not write {WIKIDATA_CACHE}: {e}")


def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        response = requests.get(
            url, params=params, headers={"User-Agent": WIKIDATA_USER_AGENT}, timeout=60)
    except Exception as e:
        raise RuntimeError(f"Wikidata request failed: {e}")
    # Raise rather than return None. A soft failure here does not stay soft: it
    # becomes a relation graph built from placeholder text, with no error.
    if response.status_code != 200:
        raise RuntimeError(
            f"Wikidata answered {response.status_code} for {params.get('ids','')[:60]}. "
            f"A 403 means the User-Agent was refused; set WIKIDATA_USER_AGENT.")
    return response
'''

assert text.count(OLD) == 1
text = text.replace(OLD, NEW)

# Cache lookup inside the two id -> label helpers. The requested list is kept
# whole: only the ids that are not cached are fetched, and the return merges
# both halves, so a partial hit cannot drop the cached ids from the answer.
for fname, kind in (("get_entities", "entity"), ("get_properties", "property")):
    marker = f"def {fname}({kind}_ids):\n    params = {{"
    assert text.count(marker) == 1, marker
    text = text.replace(marker, (
        f"def {fname}({kind}_ids):\n"
        f"    _requested = list({kind}_ids)\n"
        f"    cache = _wikidata_cache_read()\n"
        f"    missing = [i for i in _requested if i not in cache]\n"
        f"    if not missing:\n"
        f"        return {{i: cache[i] for i in _requested}}\n"
        f"    {kind}_ids = missing\n"
        f"    params = {{"))

# Fold each result into the cache before returning it.
for fname, kind, dictname in (("get_entities", "entity", "entities_dict"),
                              ("get_properties", "property", "properties_dict")):
    old_ret = (f"        return {dictname}\n"
               f"    else:\n"
               f"        return {{{kind}_id: \"Failed to retrieve data\" for {kind}_id in {kind}_ids}}")
    new_ret = (f"        cache.update({dictname})\n"
               f"        _wikidata_cache_write(cache)\n"
               f"        return {{i: cache[i] for i in _requested if i in cache}}\n"
               f"    else:\n"
               f"        raise RuntimeError(\"Wikidata returned no response for \" + str({kind}_ids[:5]))")
    assert text.count(old_ret) == 1, fname
    text = text.replace(old_ret, new_ret)

open(path, "w").write(text)

diff = run("git", "diff", "HEAD", "--", "ultra/datasets.py", cwd=SCRATCH)
assert diff.strip()
reason = (
    "make the Wikidata lookups work again, and fail loudly when they do not. "
    "ILPC2022, WKIngram and WikiTopics -- 14 of the 41 graphs -- do not use their raw "
    "relation identifiers as names. datasets.py resolves each P-code to an English label "
    "through the Wikidata API, and builds edge2id from those labels, because SEMMA's "
    "shipped relation descriptions in openrouter/descriptions/ are keyed by label too. "
    "Wikimedia now refuses the default python-requests User-Agent with 403, so every "
    "lookup returns the string 'Failed to retrieve data', inverting id2relation collapses "
    "every relation onto that one key, and the run dies far away in order_embeddings with "
    "KeyError: 0. This adds the User-Agent their policy asks for, raises instead of "
    "substituting placeholder text, and caches what it read so a later run cannot get "
    "different labels -- and therefore a different relation graph -- from the same "
    "checkpoint and the same data.")
open(os.path.join(OUT, "0004-wikidata-labels.diff"), "w").write("Reason: " + reason + "\n\n" + diff)
print("wrote patches/semma/0004-wikidata-labels.diff ({} lines)".format(len(diff.splitlines())))
