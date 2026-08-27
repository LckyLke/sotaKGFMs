# FLOCK container.
#
# Build context is the workspace root (kgfm/), not this directory:
#     scripts/clone_repos.sh          # repos/ is gitignored; the build needs it
#     docker build -f containers/flock/Dockerfile -t kgfm/flock:f35103d2 .
#
# The only model here that is not an ULTRA fork on the ULTRA stack. FLOCK scores
# by sampling random walks, so it carries a compiled C++ walker instead of
# rspmm, and it runs on torch 2.8 / CUDA 12.6 / Python 3.11 rather than
# 2.1 / 11.8 / 3.9. Its README names the base image below, so it is used as
# given rather than rebuilt on the stack the other four share.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git patch \
    && rm -rf /var/lib/apt/lists/*

# FLOCK's own list, from README "Setup", in its order.
RUN pip3 install --no-cache-dir \
        torch_geometric==2.4.0 easydict pybind11 pyyaml jinja2
RUN pip3 install --no-cache-dir \
        torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

# Added by this project, not by FLOCK: the rank dump writes parquet.
RUN pip3 install --no-cache-dir pyarrow

# torch-sparse is deliberately absent. FLOCK's README does not list it, and
# nothing on the entity inference path imports it -- the sparse message passing
# the other four repos need is replaced here by the random-walk sampler below.

WORKDIR /kgfm

COPY shared/ /kgfm/shared/
ENV PYTHONPATH=/kgfm/shared

COPY repos/flock/ /kgfm/repos/flock/
COPY patches/flock/ /kgfm/patches/flock/

RUN find /kgfm/repos/flock -name "__pycache__" -type d -prune -exec rm -rf {} + \
    && cd /kgfm/repos/flock \
    && for p in /kgfm/patches/flock/*.diff; do \
         echo "applying $p"; patch -p1 --batch --forward < "$p" || exit 1; \
       done

# Build the walker at image build time, not at run time. install_walker.sh does
# `pip install -e .`, which compiles a pybind11 extension; left to the first run
# it would land inside the timing of whichever graph happened to go first, and
# would recompile in every container.
RUN cd /kgfm/repos/flock && bash install_walker.sh

# Import everything the inference path needs, so a broken stack fails here
# rather than after a GPU has been reserved.
RUN python3 -c "\
import numpy, torch, pyarrow, torch_geometric, torch_scatter, graph_walker; \
print('stack ok:', 'numpy', numpy.__version__, '| torch', torch.__version__, \
      '| pyg', torch_geometric.__version__, '| pyarrow', pyarrow.__version__)"

# Per-repo extensions directory, for symmetry with the ULTRA-stack images. FLOCK
# JITs nothing today; the variable costs nothing and removes one way for a
# future addition to collide with another repo's build directory.
ENV TORCH_EXTENSIONS_DIR=/kgfm/output/.torch_extensions-flock

RUN mkdir -p /kgfm/data/roots/flock /kgfm/data/raw /kgfm/ranks/flock \
             /kgfm/output /kgfm/output/.torch_extensions-flock
VOLUME ["/kgfm/data", "/kgfm/ranks", "/kgfm/output"]

ENV FLOCK_ROOT=/kgfm/repos/flock
WORKDIR /kgfm/repos/flock

CMD ["python3", "src_entity/run_many.py", "--help"]
