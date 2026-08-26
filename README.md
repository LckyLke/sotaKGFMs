# kgfm — KG foundation model benchmark harness

Seven knowledge-graph foundation model repositories, each run on the stack its
authors pinned, compared through four shared artifacts and nothing else.

## Design

**One container per repo.** No shared environment. The stacks genuinely
conflict — Python 3.9 through 3.12, torch 2.1 through 2.5, CUDA 11.8 through
12.6 (see `containers/STACKS.md`) — so reconciling them would mean running at
least one model on a stack its authors never tested.

**Scoring and metrics are split.** A container produces per-query ranks and
stops. One metric implementation outside every container turns ranks into MRR
and Hits@k. No container computes a reported metric, so no container can
disagree with another about what a metric means.

Comparability rests on exactly four artifacts:

| artifact | what it fixes |
| --- | --- |
| `data/raw/` | one mirrored copy of every dataset |
| `shared/suite.py` | one frozen definition of the 54 graphs |
| `ranks/` | one rank schema every container writes into |
| `shared/metrics.py` | one metric implementation |

ULTRA is the reference. Every later repo is validated against it.

## Layout

```
repos/            seven clones, untouched, pinned by SHA (gitignored; see PINS.json)
patches/<repo>/   patch files, applied at container build — never a commit
containers/<repo> one Dockerfile per repo
shared/           suite.py, metrics.py, analyse.py
data/raw/         mirrored downloads, read-only
data/roots/<repo> one processed root per repo
ranks/<model>/<dataset>.parquet
```

`patches/` rather than commits, so `git diff` against upstream stays readable.

A **separate processed root per repo is mandatory**. PyG's loader caches the
relation graph into `processed/data.pt` keyed by directory, not by which
`pre_transform` built it. A shared root makes MOTIF silently load ULTRA's
relation graph, with no error and no warning.

## Reproducing

```bash
scripts/clone_repos.sh                 # seven clones at the pinned SHAs
scripts/prepare_ultra_workdir.sh       # upstream + patches, without touching repos/
scripts/run_ultra.sh ind_e             # zero-shot, dumps ranks/ultra/*.parquet
python shared/analyse.py --ranks ranks/ultra
```

`shared/suite.py` is executable: `python shared/suite.py ind_er` prints the
exact `-d` argument for that group, and running it with no argument lists all
54 graphs with their group, family and tail-only flag.

## Reading the results

`baseline_report.md` — criterion A (metrics.py vs ULTRA's own CSV), per-dataset
MRR and Hits@10, group means against targets, and what was and was not run.
