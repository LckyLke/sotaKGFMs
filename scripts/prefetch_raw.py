#!/usr/bin/env python
"""Fetch every raw file a suite group needs, before any GPU time is spent.

PyG downloads a dataset's raw files the first time the dataset is constructed,
which is inside the run. That couples two unrelated failure modes: a download
problem then surfaces as a failed *graph*, after the model is loaded and the GPU
is occupied.

It is not hypothetical. ``raw.githubusercontent.com`` rate-limits a burst, and a
run of this suite is a burst: 41 graphs, most of them several files, requested
back to back. The first GPU run here died on ``FB15k237Inductive:v3`` with
``HTTP Error 429``, having already claimed the graph. The limit cleared within
minutes -- it was a burst window, not a ban -- but the graph was lost from that
pass and ``run_ultra.sh`` had to be run again to pick it up.

So the downloads are separated out and made patient:

* one request at a time, with a pause between them,
* exponential backoff on 429 and 5xx, honouring ``Retry-After`` when the server
  sends it,
* every file checked against ``Content-Length`` after it lands, for the same
  reason ``verify_downloads.py`` exists -- PyG's ``download_url`` does not check,
  and a cut connection leaves a short file and raises nothing.

Datasets that arrive as a single archive (MTDEA's zip) declare no per-file
``urls``, so there is nothing to fetch here and they are reported as deferred.
They download inside the run as before; they come from S3, not from GitHub, and
are not what the rate limit is about.

    usage: prefetch_raw.py --ultra <patched tree> --root <processed root>
           [--group ind_e] [--group ind_er]     default: every group
           [--sleep 1.0]    pause between requests, seconds
           [--retries 6]    attempts per file before giving up

Exit status is non-zero if any file could not be fetched, so a caller can stop
before starting a run that would fail partway through.
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))
sys.path.insert(0, HERE)

import suite  # noqa: E402
from verify_downloads import download, remote_size, resolve_url  # noqa: E402

RETRY_STATUS = (429, 500, 502, 503, 504)


def retry_after(error, attempt, floor=2.0, ceiling=120.0):
    """Seconds to wait: the server's Retry-After if it sent one, else backoff."""
    header = error.headers.get("Retry-After") if error.headers else None
    if header:
        try:
            return min(float(header), ceiling)
        except ValueError:
            pass
    return min(floor * (2 ** attempt), ceiling)


def fetch(url, dest, retries, pause, log):
    """Download url to dest, retrying on rate limits. Return bytes, or None."""
    for attempt in range(retries):
        try:
            got = download(url, dest)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == retries - 1:
                log("      HTTP {} {}".format(exc.code, url))
                return None
            wait = retry_after(exc, attempt)
            log("      HTTP {}, waiting {:.0f}s (attempt {}/{})".format(
                exc.code, wait, attempt + 1, retries))
            time.sleep(wait)
            continue
        except Exception as exc:  # transport-level: reset, timeout, DNS
            if attempt == retries - 1:
                log("      {}: {}".format(type(exc).__name__, exc))
                return None
            wait = min(2.0 * (2 ** attempt), 120.0)
            log("      {}, waiting {:.0f}s (attempt {}/{})".format(
                type(exc).__name__, wait, attempt + 1, retries))
            time.sleep(wait)
            continue
        time.sleep(pause)
        return got
    return None


def raw_targets(cls, graph, root):
    """(raw_dir, [(url, filename)]) for one graph, or None if single-archive."""
    urls = getattr(cls, "urls", None)
    if not urls:
        return None
    instance = object.__new__(cls)
    instance.root = root
    versions = getattr(cls, "versions", None)
    instance.version = str(
        versions[graph.version]
        if isinstance(versions, dict) and graph.version
        else (graph.version if graph.version is not None else "")
    )
    raw_dir = cls.raw_dir.fget(instance)
    names = cls.raw_file_names.fget(instance)
    pairs = [(resolve_url(cls, u, graph.version), n) for u, n in zip(urls, names)]
    return raw_dir, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="ultra",
                        help="python package inside the tree: ultra, motif, semma, ...")
    parser.add_argument("--ultra", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--group", action="append", default=None)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args(argv)

    sys.path.insert(0, args.ultra)
    # The package is named after the repo (ultra, motif, semma, ...). Import it
    # by name rather than hardcoding `ultra`, so one checker serves every repo.
    import importlib
    ultra_datasets = importlib.import_module(args.package + ".datasets")

    groups = args.group or list(suite.GROUPS)
    graphs = [g for g in suite.GRAPHS if g.group in groups]

    def log(message):
        print(message, flush=True)

    present = fetched = deferred = 0
    failures = []

    for graph in graphs:
        cls = getattr(ultra_datasets, graph.dataset, None)
        if cls is None:
            failures.append("{}: no dataset class".format(graph.id))
            continue
        try:
            targets = raw_targets(cls, graph, args.root)
        except Exception as exc:
            failures.append("{}: cannot resolve raw paths ({})".format(graph.id, exc))
            continue
        if targets is None:
            deferred += 1
            continue

        raw_dir, pairs = targets
        os.makedirs(raw_dir, exist_ok=True)
        log("== {}".format(graph.id))
        for url, name in pairs:
            path = os.path.join(raw_dir, name)
            try:
                remote = remote_size(url)
            except Exception as exc:
                failures.append("{} {}: HEAD failed ({})".format(graph.id, name, exc))
                continue
            if os.path.exists(path) and (remote is None or os.path.getsize(path) == remote):
                present += 1
                continue
            log("   fetch {}".format(name))
            got = fetch(url, path, args.retries, args.sleep, log)
            if got is None:
                failures.append("{} {}: download failed  {}".format(graph.id, name, url))
            elif remote is not None and got != remote:
                failures.append("{} {}: short after fetch, {} of {} bytes".format(
                    graph.id, name, got, remote))
            else:
                fetched += 1

    print("\n{} already present, {} fetched, {} datasets deferred (single archive)".format(
        present, fetched, deferred))
    for line in failures:
        print("  " + line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
