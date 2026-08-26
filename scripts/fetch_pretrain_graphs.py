#!/usr/bin/env python
"""Fetch ULTRA's pre-training mixture and record it. These are NOT evaluated.

``config/transductive/pretrain_3g.yaml`` states the mixture behind
``ultra_3g.pth``::

    graphs: [FB15k237, WN18RR, CoDExMedium]

They are held for provenance, not for scoring. Every number this project
reports is zero-shot, and a model cannot be zero-shot on the graphs it was
trained on. So these three stay out of ``shared/suite.py``: the suite is the
frozen definition of the 54 evaluation graphs, and nothing that is not evaluated
belongs in it. Keeping them here instead makes the separation impossible to lose
by accident -- no runner can pick them up, because no group contains them.

What they are good for is the question the suite cannot answer on its own:
which evaluation graph overlaps which training graph, and by how much. Answering
that later needs the training graphs on disk, mirrored and hashed, exactly as
they were when the checkpoint was made.

Two of the three are functions, not classes. ``FB15k237`` and ``WN18RR`` wrap
PyG's ``RelLinkPredDataset`` and ``WordNet18RR`` and take only ``root``;
``CoDExMedium`` is an ordinary ``TransductiveDataset`` subclass. ULTRA's own
``build_dataset`` resolves both shapes through ``getattr``, and so does this.

    usage: fetch_pretrain_graphs.py --ultra <patched tree> [--root <processed root>]
                                    [--raw data/raw] [--no-copy]

Writes ``data/raw/MANIFEST-ultra-pretrain.json``: for every raw file, its
sha256, its size, and the URL its dataset class declares.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)

# The mixture, spelled as pretrain_3g.yaml spells it. Not a suite group.
PRETRAIN_3G = ["FB15k237", "WN18RR", "CoDExMedium"]


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def declared_urls(obj):
    """The URLs a dataset class declares, or [] for a factory function."""
    urls = getattr(obj, "urls", None)
    if not urls:
        single = getattr(obj, "url", None)
        return [single] if single else []
    return list(urls)


def raw_files_under(root, name):
    """Every file below root that belongs to this graph, relative to root."""
    found = []
    for base, _dirs, files in os.walk(root):
        if os.sep + "processed" in base + os.sep:
            continue
        for filename in sorted(files):
            path = os.path.join(base, filename)
            found.append(path)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultra", required=True)
    parser.add_argument("--root", default=os.path.join(WORKSPACE, "data", "roots", "ultra"))
    parser.add_argument("--raw", default=os.path.join(WORKSPACE, "data", "raw"))
    parser.add_argument("--no-copy", action="store_true",
                        help="manifest only, do not copy bytes into data/raw/")
    args = parser.parse_args(argv)

    sys.path.insert(0, args.ultra)
    from ultra import datasets as ultra_datasets

    os.makedirs(args.root, exist_ok=True)
    mirror = os.path.join(args.raw, "ultra-pretrain")
    if not args.no_copy:
        os.makedirs(mirror, exist_ok=True)

    manifest = {
        "repo": "ultra",
        "mixture": "pretrain_3g",
        "checkpoint": "ckpts/ultra_3g.pth",
        "config": "config/transductive/pretrain_3g.yaml",
        "evaluated": False,
        "note": ("ULTRA's pre-training graphs. Held for provenance and for "
                 "overlap analysis against the evaluation suite. Never scored: "
                 "a model is not zero-shot on what it was trained on."),
        "processed_root": args.root,
        "graphs": {},
    }

    failures = []
    for name in PRETRAIN_3G:
        factory = getattr(ultra_datasets, name, None)
        if factory is None:
            failures.append("{}: not found in ultra.datasets".format(name))
            continue

        print("== {}".format(name), flush=True)
        before = set(raw_files_under(args.root, name))
        try:
            dataset = factory(root=args.root)
        except Exception as exc:
            failures.append("{}: {}: {}".format(name, type(exc).__name__, exc))
            print("   FAILED: {}: {}".format(type(exc).__name__, exc), flush=True)
            continue

        new_files = sorted(set(raw_files_under(args.root, name)) - before)
        entry = {
            "class": name,
            "is_factory_function": not isinstance(factory, type),
            "declared_urls": declared_urls(factory),
            "files": [],
        }
        try:
            entry["num_nodes"] = int(dataset[0].num_nodes)
            entry["num_relations"] = int(dataset[0].num_relations)
            entry["num_test_triples"] = int(dataset[2].target_edge_index.shape[1])
        except Exception:
            pass

        for path in new_files:
            record = {
                "path": os.path.relpath(path, args.root),
                "bytes": os.path.getsize(path),
                "sha256": sha256(path),
            }
            if not args.no_copy:
                destination = os.path.join(mirror, os.path.relpath(path, args.root))
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                if not os.path.exists(destination):
                    shutil.copy2(path, destination)
                    os.chmod(destination, 0o444)
            entry["files"].append(record)

        manifest["graphs"][name] = entry
        print("   {} raw files, {} nodes, {} relations".format(
            len(entry["files"]), entry.get("num_nodes", "?"),
            entry.get("num_relations", "?")), flush=True)

    os.makedirs(args.raw, exist_ok=True)
    out = os.path.join(args.raw, "MANIFEST-ultra-pretrain.json")
    with open(out, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print("\nwrote {}".format(out))
    for line in failures:
        print("  " + line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
