#!/usr/bin/env python3
"""Generate patches/semma/0005-relation-label-drift.diff (stacks on 0001-0004)."""
import os, re, shutil, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get("SCRATCH", "/tmp/kgfm-semma-drift")
OUT = os.path.join(ROOT, "patches", "semma")


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout


if os.path.isdir(SCRATCH):
    shutil.rmtree(SCRATCH)
shutil.copytree(os.path.join(ROOT, "repos/semma"), SCRATCH,
                ignore=shutil.ignore_patterns(".git", "__pycache__"))
for name in sorted(os.listdir(OUT)):
    if name.endswith(".diff") and not name.startswith("0005"):
        with open(os.path.join(OUT, name)) as h:
            subprocess.run(["patch", "-p1", "--batch", "--forward"],
                           cwd=SCRATCH, stdin=h, check=True, capture_output=True)
run("git", "init", "-q", cwd=SCRATCH)
run("git", "add", "-A", cwd=SCRATCH)
run("git", "-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "0001-0004", cwd=SCRATCH)

path = os.path.join(SCRATCH, "ultra", "datasets.py")
text = open(path).read()

RESOLVER = '''

# ---------------------------------------------------------------------------
# Relation-label drift
# ---------------------------------------------------------------------------
# SEMMA names a Wikidata relation by its English label, and looks that label up
# in openrouter/descriptions/<llm>/<dataset>.json, which was generated once.
# Wikidata labels are editable, and seven properties used by the 14 Wikidata
# graphs in this suite have been renamed since. The current label is then not a
# key in the description file, and the run dies in order_embeddings with a
# KeyError whose number is a relation id and says nothing about the cause.
#
# Observed drift, current label on the left, description-file key on the right:
#
#     P112   founder                            founded by
#     P355   child organization or unit         has subsidiary
#     P410   military, police or special rank   military or police rank
#     P488   chairman                           chairperson
#     P607   participated in conflict           conflict
#     P749   parent organization or unit        parent organization
#     P1029  crew members                       crew member
#
# The authors saw at least one of these: datasets.py carries the bare comment
# "parent organization/unit -> parent organization" above the ILPC2022 class.
#
# Resolution order, per property, per graph:
#
#   1. The current label is a key in the description file. Use it.
#   2. Otherwise, take the property's own Wikidata aliases that are keys, and
#      drop any alias that is the *current label of another property in the
#      same graph*. This is not a nicety. P112's aliases include "creator",
#      which is a key -- but it is P170's label, and P170 appears in the same
#      graphs. Matching on it would attach one relation's description and
#      embedding to a different relation, quietly, with no error.
#   3. Exactly one candidate survives. Use it, and print what was remapped.
#   4. Nothing survives, or several do. Consult LABEL_OVERRIDES, then raise.
#
# Rule 2 settles six of the seven. Only P112 needs the table below.
LABEL_OVERRIDES = {
    # Wikidata renamed P112 from "founded by" to "founder". Its current aliases
    # do not carry the old label, so rule 2 has nothing to find. "founded by"
    # is a key in every affected description file, and no other property in
    # those graphs holds it as a label or an alias, so the pairing is forced.
    "P112": "founded by",
}

_PROPS_CACHE = WIKIDATA_CACHE + ".properties.json"


def _props_cache_read():
    try:
        with open(_PROPS_CACHE) as handle:
            return json.load(handle)
    except Exception:
        return {}


def _props_cache_write(cache):
    try:
        os.makedirs(os.path.dirname(_PROPS_CACHE) or ".", exist_ok=True)
        # Unique temp name per writer. get_entities runs inside
        # fetch_in_parallel's ThreadPoolExecutor, so a fixed ".tmp" path is
        # written and renamed by several threads at once and all but one of
        # them fail with ENOENT on the rename.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_PROPS_CACHE) or ".", suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(cache, handle, indent=0, sort_keys=True)
        os.replace(tmp, _PROPS_CACHE)
    except Exception as e:
        print(f"could not write {_PROPS_CACHE}: {e}")


def get_property_records(property_ids):
    """{id: {"label": str, "aliases": [str]}} for each property id, cached."""
    cache = _props_cache_read()
    missing = [i for i in property_ids if i not in cache]
    for start in range(0, len(missing), 50):
        chunk = missing[start:start + 50]
        response = fetch_wikidata({
            'action': 'wbgetentities', 'ids': '|'.join(chunk),
            'format': 'json', 'languages': 'en', 'props': 'labels|aliases',
        })
        data = response.json()
        for pid in chunk:
            entity = data.get('entities', {}).get(pid, {})
            cache[pid] = {
                "label": entity.get('labels', {}).get('en', {}).get('value'),
                "aliases": [a["value"] for a in entity.get('aliases', {}).get('en', [])],
            }
    if missing:
        _props_cache_write(cache)
    return {i: cache[i] for i in property_ids if i in cache}


def _description_keys(dataset_name):
    """The relation names this dataset's description file is keyed by."""
    folder = {"gpt4o": "gpt-4o-2024-11-20",
              "qwen3-32b": "qwen3-32b",
              "deepseekv3": "deepseek-chat-v3-0324"}.get(flags.LLM)
    if folder is None:
        return set()
    path = os.path.join(mydir, "openrouter", "descriptions", folder, str(dataset_name) + ".json")
    try:
        with open(path) as handle:
            return set(json.load(handle)["cleaned_relations"])
    except Exception:
        return set()


def resolve_properties(property_ids, dataset_name):
    """Wikidata property id -> the name this dataset's descriptions are keyed by."""
    property_ids = list(property_ids)
    records = get_property_records(property_ids)
    keys = _description_keys(dataset_name)
    if not keys:
        # No description file to reconcile against; behave as before.
        return {p: (records.get(p) or {}).get("label") for p in property_ids}

    labels_in_graph = {(records.get(p) or {}).get("label") for p in property_ids}
    out, remapped, unresolved = {}, [], []
    for pid in property_ids:
        record = records.get(pid) or {}
        label = record.get("label")
        if label in keys:
            out[pid] = label
            continue
        candidates = [a for a in record.get("aliases", [])
                      if a in keys and a not in labels_in_graph]
        if len(candidates) == 1:
            out[pid] = candidates[0]
            remapped.append(f"{pid}: {label!r} -> {candidates[0]!r} (alias)")
        elif pid in LABEL_OVERRIDES and LABEL_OVERRIDES[pid] in keys:
            out[pid] = LABEL_OVERRIDES[pid]
            remapped.append(f"{pid}: {label!r} -> {LABEL_OVERRIDES[pid]!r} (override)")
        else:
            unresolved.append((pid, label, candidates))
    if unresolved:
        detail = "; ".join(f"{p} label={l!r} candidates={c}" for p, l, c in unresolved)
        raise RuntimeError(
            f"{dataset_name}: {len(unresolved)} Wikidata properties have no matching key "
            f"in the description file. Their labels changed on Wikidata since it was "
            f"generated. Add them to LABEL_OVERRIDES in ultra/datasets.py: {detail}")
    if remapped:
        print(f"{dataset_name}: reconciled {len(remapped)} renamed Wikidata label(s): "
              + "; ".join(remapped))
    return out
'''

anchor = '\ndef fetch_in_parallel(ids_list, fetch_func):'
assert text.count(anchor) == 1
text = text.replace(anchor, RESOLVER + anchor)

# Every property lookup goes through the resolver. current_dataset is assigned
# earlier in each process() that calls this, checked at all eight sites.
pattern = re.compile(r"fetch_in_parallel\((list\([^)]*\)|[A-Za-z_][\w\.\[\]\"'()]*), get_properties\)")
text, n = pattern.subn(lambda m: f"resolve_properties({m.group(1)}, current_dataset)", text)
assert n == 8, "expected 8 get_properties call sites, replaced {}".format(n)

open(path, "w").write(text)
diff = run("git", "diff", "HEAD", "--", "ultra/datasets.py", cwd=SCRATCH)
assert diff.strip()
reason = (
    "reconcile Wikidata relation labels that have been renamed since SEMMA generated its "
    "relation descriptions. SEMMA keys openrouter/descriptions/<llm>/<dataset>.json by a "
    "relation's English Wikidata label and looks the live label up at dataset-build time. "
    "Labels are editable and seven properties used by the 14 Wikidata graphs in this suite "
    "have changed, so the lookup misses and order_embeddings raises a KeyError naming a "
    "relation id and nothing else. Six are recovered from the property's own Wikidata "
    "aliases, after discarding any alias that is another property's current label in the "
    "same graph -- without that guard P112 matches 'creator', which belongs to P170, and one "
    "relation silently receives another's description. The seventh, P112 itself, is the one "
    "entry in LABEL_OVERRIDES. Anything unresolved raises and names the property, rather "
    "than being guessed at.")
open(os.path.join(OUT, "0005-relation-label-drift.diff"), "w").write("Reason: " + reason + "\n\n" + diff)
print("wrote patches/semma/0005-relation-label-drift.diff ({} lines)".format(len(diff.splitlines())))
