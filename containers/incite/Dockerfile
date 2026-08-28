# INCITE container.
#
# Build context is the workspace root (kgfm/), not this directory:
#     scripts/clone_repos.sh          # repos/ is gitignored; the build needs it
#     docker build -f containers/incite/Dockerfile -t kgfm/incite:incite01 .
#
# PIN / TAG RULE -- INCITE wraps no upstream repo, so there is no upstream SHA
# to pin. repos/PINS.json carries an "incite" entry whose "commit" is a
# manually bumped version string; the image tag is its first 8 characters
# (scripts/docker_run.sh reads it exactly like every other model's pin).
# Bump the string ("incite01..." -> "incite02...") whenever the image
# CONTENTS change in a way a run depends on: this Dockerfile, the baked TRIX
# tree or its patches, or the Python stack. incite/ code and configs are NOT
# baked as the source of truth -- runners cd into the bind-mounted work tree,
# so the mounted copy always wins (see scripts/run_incite.sh) -- but the
# baked copy must never be allowed to drift silently either, hence the bump
# rule covers it too.
#
# One environment, the TRIX stack (same choices and reasons as
# containers/trix/Dockerfile and containers/crest/Dockerfile): Python 3.9,
# torch 2.1.0, CUDA 11.8, PyG 2.4.0. The pinned TRIX repo must run inside --
# datasets.py loads every suite graph, the layer gate compares against the
# pinned layers, and rank dumping goes through trix.rank_dump -- so the
# patched tree is baked exactly as the trix image bakes it.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-11.8/ \
    PATH=/usr/local/cuda-11.8/bin:$PATH

# get-pip.py from the version-specific path: the generic bootstrap.pypa.io copy
# now aborts on anything below 3.10. A build-tool URL, not a pin of TRIX's.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-dev python3.9-distutils \
        build-essential git patch \
    && curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && rm -rf /var/lib/apt/lists/*

# setuptools below 70: torch's cpp_extension imports pkg_resources.packaging,
# which setuptools removed in 70, and rspmm will not build without it.
RUN python3.9 -m pip install --no-cache-dir 'setuptools==69.5.1' wheel

# numpy below 2 -- torch 2.1.0, torch-scatter, torch-sparse and pyarrow 14 are
# built against the NumPy 1.x C ABI; an unpinned resolve takes 2.0.2 and breaks
# `import pyarrow`, which takes the parquet rank dump with it.
RUN python3.9 -m pip install --no-cache-dir 'numpy==1.26.4'

RUN python3.9 -m pip install --no-cache-dir \
        torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN python3.9 -m pip install --no-cache-dir \
        torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 \
        -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN python3.9 -m pip install --no-cache-dir ninja easydict pyyaml

# matplotlib: src/run_entity.py imports it at module scope, and incite's
# drivers import trix.util/tasks from the same tree, so it must be importable
# even though INCITE never plots. Same pin as the TRIX image.
RUN python3.9 -m pip install --no-cache-dir 'matplotlib==3.9.4'

# Added by this project, not by TRIX: the rank dump writes parquet, and the
# test suite runs inside this image (scripts/test_incite.sh).
RUN python3.9 -m pip install --no-cache-dir 'pyarrow==14.0.2' 'pytest==8.3.4'

# Fail the build, not the run.
RUN python3.9 -c "\
import numpy, torch, pyarrow, torch_geometric, torch_scatter, torch_sparse, matplotlib, yaml, easydict, pytest; \
assert numpy.__version__.startswith('1.'), 'numpy 2.x breaks the torch/pyarrow C ABI: ' + numpy.__version__; \
assert torch.__version__.startswith('2.1.0'), torch.__version__; \
print('stack ok:', 'numpy', numpy.__version__, '| torch', torch.__version__, '| pyarrow', pyarrow.__version__)"

WORKDIR /kgfm

# Per-repo extensions directory (see containers/trix/Dockerfile for why a
# shared one deadlocks concurrent runs on a FileBaton).
ENV TORCH_EXTENSIONS_DIR=/kgfm/output/.torch_extensions-incite

COPY shared/ /kgfm/shared/
# /kgfm on the path is what makes `python -m incite.run` resolve the package.
ENV PYTHONPATH=/kgfm:/kgfm/shared

COPY repos/trix/ /kgfm/repos/trix/
COPY patches/trix/ /kgfm/patches/trix/
RUN cd /kgfm/repos/trix \
    && for p in /kgfm/patches/trix/*.diff; do \
         echo "applying $p"; patch -p1 --batch --forward < "$p" || exit 1; \
       done

COPY incite/ /kgfm/incite/
COPY configs/ /kgfm/configs/

# Compile rspmm at build time, not at run time (crest precedent, same two
# obstacles handled the same way):
#   * no GPU is visible during `docker build`, and trix's load_extension gates
#     the CUDA sources on torch.cuda.is_available(); the call below passes the
#     CUDA sources and the exact flag set load_extension chooses at run time
#     on a CUDA machine -- byte for byte, or the JIT hash differs and the run
#     recompiles anyway.
#   * without a device torch cannot detect an architecture, so
#     TORCH_CUDA_ARCH_LIST pins 8.9 (the RTX 4070 Ti SUPER this project runs
#     on) both here and, via ENV, at run time.
ENV TORCH_CUDA_ARCH_LIST=8.9
RUN python3.9 -c "\
from torch.utils import cpp_extension; \
src = '/kgfm/repos/trix/src/trix/rspmm/source'; \
cpp_extension.load('rspmm', [src + '/rspmm.cpp', src + '/rspmm.cu'], \
    extra_cflags=['-Ofast', '-fopenmp', '-DAT_PARALLEL_OPENMP', '-DCUDA_OP'], \
    extra_cuda_cflags=['-O3']); \
print('rspmm compiled at build time')"

RUN mkdir -p /kgfm/data/roots/incite/support /kgfm/data/raw /kgfm/ranks/incite \
             /kgfm/output /kgfm/output/.torch_extensions-incite
VOLUME ["/kgfm/data", "/kgfm/ranks", "/kgfm/output"]

ENV TRIX_ROOT=/kgfm/repos/trix
WORKDIR /kgfm

CMD ["python3.9", "-m", "incite.run", "--help"]
