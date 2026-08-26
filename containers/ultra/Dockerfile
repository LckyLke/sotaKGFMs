# ULTRA reference container.
#
# Build context is the workspace root (kgfm/), not this directory:
#     scripts/clone_repos.sh          # repos/ is gitignored; the build needs it
#     docker build -f containers/ultra/Dockerfile -t kgfm/ultra:427966ad .
#
# Base is a CUDA *devel* image, not runtime: ULTRA's rspmm kernel is JIT-built by
# torch.utils.cpp_extension at first use and needs nvcc. 11.8 matches the CUDA
# version ULTRA pins in its README install block.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-11.8/ \
    PATH=/usr/local/cuda-11.8/bin:$PATH

# Python 3.9 -- ULTRA's pinned interpreter. Ubuntu 22.04 ships 3.10, so it comes
# from deadsnakes rather than from the distro.
# get-pip.py comes from the version-specific path. The generic
# bootstrap.pypa.io/get-pip.py now aborts on anything below 3.10; only the
# /pip/3.9/ copy still installs. A build-tool URL, not one of ULTRA's pins.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-dev python3.9-distutils \
        build-essential git patch \
    && curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && rm -rf /var/lib/apt/lists/*
# Deliberately not repointing /usr/bin/python3 at 3.9: the distro's own tooling
# (apt, add-apt-repository) runs on 3.10 and breaks if it moves. Everything here
# calls python3.9 explicitly.

# torch's cpp_extension in this version imports pkg_resources.packaging, which
# setuptools removed in 70. Pinning below 70 is required for rspmm to build at
# all; it is a build-tool pin, not a change to any of ULTRA's own pins.
RUN python3.9 -m pip install --no-cache-dir 'setuptools==69.5.1' wheel

# numpy below 2. torch 2.1.0, torch-scatter, torch-sparse and pyarrow 14 are all
# compiled against the NumPy 1.x C ABI. Unpinned, pip resolves numpy 2.0.2 here
# and `import pyarrow` dies with "numpy.core.multiarray failed to import", which
# takes the parquet rank dump with it. 1.26.4 is the version the CPU baseline
# ran on, recorded in environment.json. A build-tool pin, not one of ULTRA's.
RUN python3.9 -m pip install --no-cache-dir 'numpy==1.26.4'

# ULTRA's pins, exactly as its README states them. If a wheel below has been
# withdrawn, build it from source -- do not relax the version to make pip pass.
RUN python3.9 -m pip install --no-cache-dir \
        torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN python3.9 -m pip install --no-cache-dir \
        torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 \
        -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN python3.9 -m pip install --no-cache-dir ninja easydict pyyaml

# Added by this project, not by ULTRA: the rank dump writes parquet.
# An addition, never an upgrade of anything above.
RUN python3.9 -m pip install --no-cache-dir 'pyarrow==14.0.2'

# Fail the build, not the run. Every import below is on the critical path of a
# rank dump, and a NumPy major-version drift breaks them at import time only.
RUN python3.9 -c "\
import numpy, torch, pyarrow, torch_geometric, torch_scatter, torch_sparse; \
assert numpy.__version__.startswith('1.'), 'numpy 2.x breaks the torch/pyarrow C ABI: ' + numpy.__version__; \
assert torch.__version__.startswith('2.1.0'), torch.__version__; \
print('stack ok:', 'numpy', numpy.__version__, '| torch', torch.__version__, '| pyarrow', pyarrow.__version__)"

WORKDIR /kgfm

# Cache the JIT-compiled rspmm kernel on a volume so it survives container
# restarts; compiling it is a few minutes and only needs doing once.
ENV TORCH_EXTENSIONS_DIR=/kgfm/output/.torch_extensions

# shared/ is mounted read-only in normal use, but is baked in as well so that the
# image is self-contained; suite.py is imported by the rank dump patch.
COPY shared/ /kgfm/shared/
ENV PYTHONPATH=/kgfm/shared

# Upstream at the pinned SHA, then every modification as a patch file. Nothing is
# edited in place: `git diff` against upstream stays readable.
COPY repos/ultra/ /kgfm/repos/ultra/
COPY patches/ultra/ /kgfm/patches/ultra/
RUN cd /kgfm/repos/ultra \
    && for p in /kgfm/patches/ultra/*.diff; do \
         echo "applying $p"; patch -p1 --batch --forward < "$p" || exit 1; \
       done

# Separate processed root per repo is mandatory: PyG keys the cached relation
# graph in processed/data.pt by directory, not by which pre_transform built it,
# so a shared root would silently hand another model's relation graph to ULTRA.
RUN mkdir -p /kgfm/data/roots/ultra /kgfm/data/raw /kgfm/ranks/ultra \
             /kgfm/output /kgfm/output/.torch_extensions
VOLUME ["/kgfm/data", "/kgfm/ranks", "/kgfm/output"]

ENV ULTRA_ROOT=/kgfm/repos/ultra
WORKDIR /kgfm/repos/ultra

# Compile rspmm once at first run rather than racing on it across parallel
# launches -- see docs/RUNBOOK.md step 4.
CMD ["python3.9", "script/run_many.py", "--help"]
