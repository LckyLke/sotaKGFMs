# MOTIF container.
#
# Build context is the workspace root (kgfm/), not this directory:
#     scripts/clone_repos.sh          # repos/ is gitignored; the build needs it
#     docker build -f containers/motif/Dockerfile -t kgfm/motif:aab68802 .
#
# MOTIF is a direct ULTRA fork and pins the same stack, so this file tracks
# containers/ultra/Dockerfile deliberately. Where it differs, the difference is
# marked and has a reason. Base is a CUDA *devel* image for two reasons here,
# not one: MOTIF's rspmm is JIT-built by torch.utils.cpp_extension, and its
# Triton kernel also compiles at runtime and wants a matching toolchain.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-11.8/ \
    PATH=/usr/local/cuda-11.8/bin:$PATH

# Python 3.9 -- the interpreter ULTRA pins and MOTIF inherits. Ubuntu 22.04
# ships 3.10, so it comes from deadsnakes rather than from the distro.
# get-pip.py comes from the version-specific path. The generic
# bootstrap.pypa.io/get-pip.py now aborts on anything below 3.10; only the
# /pip/3.9/ copy still installs. A build-tool URL, not one of MOTIF's pins.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-dev python3.9-distutils \
        build-essential git patch \
    && curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && rm -rf /var/lib/apt/lists/*

# torch's cpp_extension in this version imports pkg_resources.packaging, which
# setuptools removed in 70. Pinning below 70 is required for rspmm to build at
# all; it is a build-tool pin, not a change to any of MOTIF's own pins.
RUN python3.9 -m pip install --no-cache-dir 'setuptools==69.5.1' wheel

# numpy below 2, for the same C-ABI reason as the ULTRA image: torch 2.1.0,
# torch-scatter, torch-sparse and pyarrow 14 are all built against NumPy 1.x,
# and an unpinned resolve installs 2.0.2, which breaks `import pyarrow` and
# takes the parquet rank dump with it.
RUN python3.9 -m pip install --no-cache-dir 'numpy==1.26.4'

# MOTIF's pins, as its README states them.
RUN python3.9 -m pip install --no-cache-dir \
        torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN python3.9 -m pip install --no-cache-dir \
        torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 \
        -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN python3.9 -m pip install --no-cache-dir ninja easydict pyyaml

# DEVIATION from MOTIF's README, stated rather than hidden.
#
# The README's last install line reads `pip triton-nightly` -- missing the word
# `install`, so it never ran for anyone, and `triton-nightly` is not a release
# on PyPI. Meanwhile torch 2.1.0 declares `triton (==2.1.0)` as a hard
# dependency and has already installed it by this point. So the Triton kernel
# runs against the version torch itself pins.
#
# Taking the nightly instead would mean pinning nothing: a nightly is a moving
# target, it would not be reproducible from this file six months from now, and
# it would silently replace a package that torch pins exactly. If MOTIF's Triton
# path ever needs a feature 2.1.0 lacks, that is a real finding and belongs in a
# patch with a reason -- not in an unpinned upgrade here.
RUN python3.9 -c "import triton; assert triton.__version__.startswith('2.1'), triton.__version__; print('triton', triton.__version__, '(from torch 2.1.0 pin, not triton-nightly)')"

# Added by this project, not by MOTIF: the rank dump writes parquet.
RUN python3.9 -m pip install --no-cache-dir 'pyarrow==14.0.2'

# Fail the build, not the run. Every import below is on the critical path of a
# rank dump, and a NumPy major-version drift breaks them at import time only.
RUN python3.9 -c "\
import numpy, torch, pyarrow, torch_geometric, torch_scatter, torch_sparse, triton; \
assert numpy.__version__.startswith('1.'), 'numpy 2.x breaks the torch/pyarrow C ABI: ' + numpy.__version__; \
assert torch.__version__.startswith('2.1.0'), torch.__version__; \
print('stack ok:', 'numpy', numpy.__version__, '| torch', torch.__version__, '| triton', triton.__version__, '| pyarrow', pyarrow.__version__)"

WORKDIR /kgfm

# Cache the JIT-compiled kernels on a volume so they survive container restarts.
# MOTIF has two compiled paths, rspmm and Triton, and both land here.
#
# Per-repo directory, not the shared one ULTRA uses. MOTIF's rspmm is ULTRA's
# kernel *modified*, and torch names the build directory after the extension --
# "rspmm" for both. Pointing them at one directory makes every alternation
# between the two images recompile, and makes two concurrent runs contend on the
# same FileBaton lock, which does not fail loudly: the loser blocks at ~0% CPU
# with no error at all.
ENV TORCH_EXTENSIONS_DIR=/kgfm/output/.torch_extensions-motif

# shared/ is mounted read-only in normal use, but is baked in as well so that the
# image is self-contained; suite.py is imported by the rank dump patch.
COPY shared/ /kgfm/shared/
ENV PYTHONPATH=/kgfm/shared

# Upstream at the pinned SHA, then every modification as a patch file.
COPY repos/motif/ /kgfm/repos/motif/
COPY patches/motif/ /kgfm/patches/motif/
RUN cd /kgfm/repos/motif \
    && for p in /kgfm/patches/motif/*.diff; do \
         echo "applying $p"; patch -p1 --batch --forward < "$p" || exit 1; \
       done

# Separate processed root per repo is mandatory, and MOTIF is the reason the
# rule exists rather than a precaution: its pre_transform builds a higher-order
# motif relation graph, while PyG keys the cached graph in processed/data.pt by
# directory alone. A root shared with ULTRA would hand MOTIF ULTRA's relation
# graph, with no error and no warning.
RUN mkdir -p /kgfm/data/roots/motif /kgfm/data/raw /kgfm/ranks/motif \
             /kgfm/output /kgfm/output/.torch_extensions-motif
VOLUME ["/kgfm/data", "/kgfm/ranks", "/kgfm/output"]

ENV MOTIF_ROOT=/kgfm/repos/motif
WORKDIR /kgfm/repos/motif

CMD ["python3.9", "script/run_many.py", "--help"]
