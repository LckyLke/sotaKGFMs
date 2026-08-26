#!/usr/bin/env python
"""Fetch the Freebase MID->name table SEMMA needs, and record where it came from.

``repos/semma/ultra/datasets.py`` opens ``fb_mid2name.tsv`` unconditionally in
the code paths for ``FB15k237Inductive``, ``FBIngram`` and ``HM`` -- 12 of the 41
graphs in groups 1 and 2. The file is not in the repository. ``setup.sh`` fetches
it from Google Drive, and that step does not work:

  1. It is a bare ``wget`` against a Drive link for an 81 MB file. Drive answers
     large files with an HTML interstitial ("too large for Google to scan for
     viruses"), so the wget writes *that page* into ``fb_mid2name.tsv``. The run
     then fails while parsing a malformed TSV rather than on a missing file,
     which is the harder failure to read.
  2. The stored object is ``mid2name.tsv.gz``, gzipped. ``setup.sh`` echoes
     "Unzipping fb_mid2name.tsv" and contains no command that unzips anything.

So this script does what setup.sh means: follow the confirmation form, verify the
bytes are gzip and not HTML, decompress, and record a sha256 next to the URL in
``data/raw/MANIFEST-semma.json``. The mirror lives under ``data/raw/semma/``
alongside every other raw download, so the tree in ``repos/semma`` stays exactly
as cloned.

    usage: fetch_semma_mid2name.py [--raw data/raw] [--force]
"""

import argparse
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)

# The id in repos/semma/setup.sh. Kept here rather than parsed out of the shell
# script, so that a change upstream shows up as a diff instead of silently
# redirecting this download somewhere else.
FILE_ID = "0B52yRXcdpG6MaHA5ZW9CZ21MbVk"
SOURCE = "https://drive.google.com/uc?export=download&id=" + FILE_ID
GZIP_MAGIC = b"\x1f\x8b"


def opener():
    jar = urllib.request.HTTPCookieProcessor()
    build = urllib.request.build_opener(jar)
    build.addheaders = [("User-Agent", "Mozilla/5.0 (kgfm dataset mirror)")]
    return build


def resolve_download(open_url):
    """Return the real download URL behind Drive's confirmation page."""
    with open_url.open(SOURCE, timeout=120) as response:
        head = response.read(200000)
        if head[:2] == GZIP_MAGIC:
            return SOURCE, head, response          # served directly
        page = head.decode("utf-8", "replace")

    action = re.search(r'action="([^"]+)"', page)
    if not action:
        raise RuntimeError("no download form on Drive's page; the link may be dead")
    url = html.unescape(action.group(1))
    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', page))
    if fields:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(fields)
    return url, None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=os.path.join(WORKSPACE, "data", "raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    mirror = os.path.join(args.raw, "semma")
    os.makedirs(mirror, exist_ok=True)
    gz_path = os.path.join(mirror, "mid2name.tsv.gz")
    tsv_path = os.path.join(mirror, "fb_mid2name.tsv")

    if os.path.exists(tsv_path) and not args.force:
        print("already present: {} ({} bytes)".format(tsv_path, os.path.getsize(tsv_path)))
        return 0

    build = opener()
    url, head, _ = resolve_download(build)
    print("resolved download url:\n   {}".format(url[:160]))

    with build.open(url, timeout=600) as response, open(gz_path, "wb") as out:
        first = head if head else response.read(len(GZIP_MAGIC) * 512)
        # Refuse HTML masquerading as the payload -- the exact failure setup.sh
        # produces, and the reason it fails later at a confusing place.
        if first[:2] != GZIP_MAGIC:
            preview = first[:200].decode("utf-8", "replace").replace("\n", " ")
            raise SystemExit(
                "refusing to save: the response is not gzip.\n"
                "  first bytes: {!r}\n"
                "  Drive returned a page, not the file. Nothing was written.".format(preview))
        out.write(first)
        shutil.copyfileobj(response, out)

    size = os.path.getsize(gz_path)
    print("downloaded {} ({:.1f} MB)".format(gz_path, size / 1e6))

    with gzip.open(gz_path, "rb") as gz, open(tsv_path, "wb") as out:
        shutil.copyfileobj(gz, out)
    print("decompressed to {} ({:.1f} MB)".format(tsv_path, os.path.getsize(tsv_path) / 1e6))

    def sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    with open(tsv_path, "r", errors="replace") as handle:
        first_line = handle.readline().rstrip("\n")
        handle.seek(0)
        rows = sum(1 for _ in handle)

    manifest = {
        "repo": "semma",
        "why": ("ultra/datasets.py opens fb_mid2name.tsv unconditionally for "
                "FB15k237Inductive, FBIngram and HM: 12 of the 41 graphs in "
                "groups 1 and 2 cannot be built without it."),
        "source": SOURCE,
        "note": ("setup.sh's wget saves Drive's virus-scan interstitial instead of "
                 "the file, and never decompresses. scripts/fetch_semma_mid2name.py "
                 "follows the confirmation form and checks for gzip magic."),
        "files": [
            {"path": "semma/mid2name.tsv.gz", "bytes": size, "sha256": sha256(gz_path)},
            {"path": "semma/fb_mid2name.tsv", "bytes": os.path.getsize(tsv_path),
             "sha256": sha256(tsv_path), "rows": rows,
             "first_row": first_line[:120]},
        ],
    }
    out_path = os.path.join(args.raw, "MANIFEST-semma.json")
    with open(out_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print("wrote {}\n{} rows, first: {}".format(out_path, rows, first_line[:80]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
