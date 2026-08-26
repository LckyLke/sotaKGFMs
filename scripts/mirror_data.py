#!/usr/bin/env python
"""Mirror the raw dataset downloads into data/raw/ and record where they came from.

Two jobs, kept separate on purpose:

  sources -- the download URL(s) each suite graph declares, read out of ULTRA's
             own dataset classes rather than retyped, with the version
             substituted in.  This is the record that survives a dead link: if a
             dataset later fails to download, the URL that failed is here.

  files   -- the bytes actually on disk under data/roots/<repo>/, copied into
             data/raw/ with a sha256 each, so a later container can be pointed
             at a local mirror instead of the open internet, and so two repos
             can be proven to have consumed identical input.

Nothing here processes anything: data/raw/ holds downloads, and each repo keeps
its own processed root, because PyG keys the cached relation graph in
processed/data.pt by directory and not by which pre_transform built it.

    usage: mirror_data.py --ultra <patched ultra tree> [--root data/roots/ultra]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))

import suite  # noqa: E402


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def declared_urls(datasets_module, graph):
    """The URL(s) ULTRA would fetch for this graph, version substituted."""
    cls = getattr(datasets_module, graph.dataset, None)
    if cls is None:
        return []
    version = graph.version
    raw = []
    for attr in ("urls", "url"):
        value = getattr(cls, attr, None)
        if isinstance(value, str):
            raw.append(value)
        elif isinstance(value, (list, tuple)):
            raw.extend(value)
    out = []
    for url in raw:
        if "%s" in url:
            if version is None:
                # single-version classes carry their version in a class attribute
                versions = getattr(cls, "versions", None)
                pick = None
                if isinstance(versions, (list, tuple)) and versions:
                    pick = versions[0]
                elif isinstance(versions, dict) and versions:
                    pick = sorted(versions.values())[0]
                url = url % pick if pick is not None else url
            else:
                versions = getattr(cls, "versions", None)
                resolved = versions[version] if isinstance(versions, dict) else version
                url = url % resolved
        out.append(url)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultra", required=True, help="patched ULTRA tree (for ultra.datasets)")
    parser.add_argument("--root", default=os.path.join(WORKSPACE, "data", "roots", "ultra"))
    parser.add_argument("--raw", default=os.path.join(WORKSPACE, "data", "raw"))
    parser.add_argument("--repo", default="ultra")
    parser.add_argument("--no-copy", action="store_true", help="manifest only, do not copy bytes")
    args = parser.parse_args(argv)

    sys.path.insert(0, args.ultra)
    from ultra import datasets as ultra_datasets  # noqa: E402

    manifest = {
        "repo": args.repo,
        "processed_root": os.path.relpath(args.root, WORKSPACE),
        "mirror": os.path.relpath(args.raw, WORKSPACE),
        "sources": {},
        "files": [],
    }

    for graph in suite.GRAPHS:
        manifest["sources"][graph.id] = declared_urls(ultra_datasets, graph)

    dest_base = os.path.join(args.raw, args.repo)
    for dirpath, _dirnames, filenames in os.walk(args.root):
        if os.path.basename(dirpath) != "raw":
            continue
        for name in sorted(filenames):
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, args.root)
            entry = {"path": rel, "bytes": os.path.getsize(src), "sha256": sha256(src)}
            manifest["files"].append(entry)
            if not args.no_copy:
                dst = os.path.join(dest_base, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst) or os.path.getsize(dst) != entry["bytes"]:
                    shutil.copy2(src, dst)
                os.chmod(dst, 0o444)  # data/raw is read-only by contract

    manifest["files"].sort(key=lambda e: e["path"])

    os.makedirs(args.raw, exist_ok=True)
    out = os.path.join(args.raw, "MANIFEST-{}.json".format(args.repo))
    with open(out, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("{} files, {:.1f} MiB mirrored -> {}".format(
        len(manifest["files"]),
        sum(f["bytes"] for f in manifest["files"]) / 2**20,
        out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
