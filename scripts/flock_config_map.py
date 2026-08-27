#!/usr/bin/env python
"""Read the graph-to-config assignment out of FLOCK's own zero-shot script.

FLOCK does not evaluate the suite with one config. Walk count rises and batch
size falls as graphs grow -- n16 for most, n32, n64 and n128 for the largest --
and the transductive half goes further, to n256 and n512 with smaller ensembles.
A runner that picked one config for all 41 would not be measuring FLOCK.

The map is parsed rather than transcribed, so a change upstream shows up as a
different map instead of a silent disagreement with a copy made once by hand.

    usage: flock_config_map.py [--group ind_e|ind_er|transductive]
           flock_config_map.py --check      # every suite graph is covered
"""

import argparse
import os
import re
import sys

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE, "shared"))
import suite  # noqa: E402

SCRIPT = os.path.join(WORKSPACE, "repos", "flock", "scripts", "entity_zeroshot.sh")
LINE = re.compile(r"--config\s+(\S+).*?-d\s+(\S+)")


def parse(path=SCRIPT):
    """Return {suite id: config path relative to the repo root}."""
    out = {}
    for raw in open(path):
        line = raw.strip()
        if line.startswith("#") or "run_many.py" not in line:
            continue
        found = LINE.search(line)
        if not found:
            continue
        config, spelling = found.group(1), found.group(2)
        # FLOCK writes `-d Metafam` and `-d FBNELL`; the suite id is
        # Metafam:Metafam and FBNELL:FBNELL_v1. by_run_id normalises both.
        try:
            graph = suite.by_run_id(spelling)
        except Exception:
            continue
        if graph.id in out and out[graph.id][0] != config:
            raise ValueError("{} assigned two configs: {} and {}".format(
                graph.id, out[graph.id][0], config))
        # Keep FLOCK's own -d spelling next to the config. It is not always the
        # suite run_id: FLOCK writes `Metafam` where ULTRA writes
        # `Metafam:Metafam`, and the runner must hand each repo its own.
        out[graph.id] = (config, spelling)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    mapping = parse()
    if args.check:
        missing = [i for i in suite.ids() if i not in mapping]
        print("parsed {} assignments, {} distinct configs".format(
            len(mapping), len(set(c for c, _ in mapping.values()))))
        for group in ("ind_e", "ind_er", "transductive"):
            ids = suite.ids(group)
            have = [i for i in ids if i in mapping]
            print("  {:14} {}/{}".format(group, len(have), len(ids)))
        if missing:
            print("MISSING: " + ", ".join(missing))
            return 1
        print("every suite graph is covered")
        return 0

    ids = suite.ids(args.group) if args.group else suite.ids()
    for gid in ids:
        if gid in mapping:
            config, spelling = mapping[gid]
            print("{}\t{}\t{}".format(gid, spelling, config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
