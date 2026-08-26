#!/usr/bin/env python
"""Check every mirrored raw file against the byte count the server reports.

PyG's ``download_url`` streams to disk without verifying Content-Length, so a
connection cut mid-transfer leaves a short file and no error. That is not
hypothetical here: INDIGO-BM's inference graph arrived 1.7 MB short, and it only
surfaced because the truncation happened to land mid-record and raised a
ValueError. Had it landed on a line boundary the graph would simply have been
missing its tail, every metric would have been quietly wrong, and nothing would
have complained.

So the check is a byte count, not a parse. For every dataset class that declares
a list of ``urls`` (one URL per raw file, zipped together in order by its own
``download``), this HEADs each URL and compares against the file on disk.

Single-archive datasets (MTDEA's zip) are reported as unverifiable: the archive
is extracted and deleted, so there is nothing left to compare.

    usage: verify_downloads.py --ultra <patched tree> --root <processed root>
           [--fix]        re-download anything short, then re-check
"""

import argparse
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))

import suite  # noqa: E402


def remote_size(url, timeout=60):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        return int(length) if length is not None else None


def download(url, dest, timeout=300):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def resolve_url(cls, url, version):
    if "%s" not in url:
        return url
    versions = getattr(cls, "versions", None)
    if isinstance(versions, dict):
        return url % versions[version]
    return url % version


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="ultra",
                        help="python package inside the tree: ultra, motif, semma, ...")
    parser.add_argument("--ultra", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--group", action="append", default=None,
                        help="restrict to a suite group; repeatable")
    parser.add_argument("--fix", action="store_true", help="re-download short files")
    args = parser.parse_args(argv)

    sys.path.insert(0, args.ultra)
    # The package is named after the repo (ultra, motif, semma, ...). Import it
    # by name rather than hardcoding `ultra`, so one checker serves every repo.
    import importlib
    ultra_datasets = importlib.import_module(args.package + ".datasets")

    groups = args.group or list(suite.GROUPS)
    graphs = [g for g in suite.GRAPHS if g.group in groups]

    checked = short = missing = unverifiable = fixed = 0
    problems = []

    for graph in graphs:
        cls = getattr(ultra_datasets, graph.dataset, None)
        urls = getattr(cls, "urls", None) if cls is not None else None
        if not urls:
            unverifiable += 1
            continue
        try:
            instance = object.__new__(cls)
            instance.root = args.root
            instance.version = str(
                cls.versions[graph.version]
                if isinstance(getattr(cls, "versions", None), dict) and graph.version
                else (graph.version if graph.version is not None else "")
            )
            raw_dir = cls.raw_dir.fget(instance)
            names = cls.raw_file_names.fget(instance)
        except Exception as exc:  # pragma: no cover - class shapes vary
            problems.append("{}: cannot resolve raw paths ({})".format(graph.id, exc))
            unverifiable += 1
            continue

        for url, name in zip(urls, names):
            path = os.path.join(raw_dir, name)
            resolved = resolve_url(cls, url, graph.version)
            if not os.path.exists(path):
                missing += 1
                continue
            checked += 1
            local = os.path.getsize(path)
            try:
                remote = remote_size(resolved)
            except Exception as exc:
                problems.append("{} {}: HEAD failed ({})".format(graph.id, name, exc))
                continue
            if remote is None or local == remote:
                continue
            short += 1
            problems.append("{} {}: local {} bytes, server {} bytes ({:+d})  {}".format(
                graph.id, name, local, remote, local - remote, resolved))
            if args.fix:
                got = download(resolved, path)
                if got == remote:
                    fixed += 1
                    problems[-1] += "  -> REFETCHED ok"
                    processed = os.path.join(os.path.dirname(raw_dir), "processed")
                    if os.path.isdir(processed):
                        shutil.rmtree(processed)
                        problems[-1] += ", processed/ cleared"
                else:
                    problems[-1] += "  -> REFETCH STILL SHORT ({} bytes)".format(got)

    print("checked {} files: {} short, {} absent, {} datasets unverifiable (single archive)".format(
        checked, short, missing, unverifiable))
    if args.fix:
        print("re-fetched {}".format(fixed))
    for line in problems:
        print("  " + line)
    return 1 if (short - fixed) or any("HEAD failed" in p for p in problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
