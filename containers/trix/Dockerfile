# TRIX container.
#
# Build context is the workspace root (kgfm/), not this directory:
#     scripts/clone_repos.sh          # repos/ is gitignored; the build needs it
#     docker build -f containers/trix/Dockerfile -t kgfm/trix:7596e14e .
#
# TRIX pins nothing. Its requirements.txt states floors only -- torch>=2.1.0,
# torch-scatter>=2.1.2, torch-geometric>=2.4.0 -- with no Python version, no
# CUDA version, and a README that is just `pip install -r requirements.txt`.
# A container has to choose. This one chooses ULTRA's exact stack (Python 3.9,
# torch 2.1.0, cu118, PyG 2.4.0), because that satisfies every floor TRIX states
# and because TRIX is an ULTRA fork whose rspmm source is byte-identical to
# ULTRA's. The choice is ours, not the authors', and is recorded here so that a
# later disagreement can be traced to this line rather than guessed at.
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

# matplotlib is a hard requirement, not an optional one: src/run_entity.py
# imports it at module scope (line 21) alongside numpy, so the script will not
# start without it. requirements.txt does not mention it.
RUN python3.9 -m pip install --no-cache-dir 'matplotlib==3.9.4'

# google-generativeai is in TRIX's requirements.txt for the llm/ directory,
# which generates relation descriptions. Nothing on the entity-prediction path
# imports it, and installing it would pull a large dependency tree and invite an
# outbound API call into a benchmark run. Left out deliberately; if a future task
# needs llm/, it belongs in its own image.

# Added by this project, not by TRIX: the rank dump writes parquet.
RUN python3.9 -m pip install --no-cache-dir 'pyarrow==14.0.2'

# Fail the build, not the run.
RUN python3.9 -c "\
import numpy, torch, pyarrow, torch_geometric, torch_scatter, torch_sparse, matplotlib; \
assert numpy.__version__.startswith('1.'), 'numpy 2.x breaks the torch/pyarrow C ABI: ' + numpy.__version__; \
assert torch.__version__.startswith('2.1.0'), torch.__version__; \
print('stack ok:', 'numpy', numpy.__version__, '| torch', torch.__version__, '| pyarrow', pyarrow.__version__)"

WORKDIR /kgfm

# Per-repo extensions directory. TRIX's rspmm.cpp is byte-identical to ULTRA's,
# but torch names the build directory after the extension -- "rspmm" for both --
# so a shared directory makes concurrent runs contend on one FileBaton lock,
# which does not fail loudly: the loser blocks at ~0% CPU with no error.
ENV TORCH_EXTENSIONS_DIR=/kgfm/output/.torch_extensions-trix

COPY shared/ /kgfm/shared/
ENV PYTHONPATH=/kgfm/shared

COPY repos/trix/ /kgfm/repos/trix/
COPY patches/trix/ /kgfm/patches/trix/
RUN cd /kgfm/repos/trix \
    && for p in /kgfm/patches/trix/*.diff; do \
         echo "applying $p"; patch -p1 --batch --forward < "$p" || exit 1; \
       done

RUN mkdir -p /kgfm/data/roots/trix /kgfm/data/raw /kgfm/ranks/trix \
             /kgfm/output /kgfm/output/.torch_extensions-trix
VOLUME ["/kgfm/data", "/kgfm/ranks", "/kgfm/output"]

# entity_prediction.pth, never relation_prediction.pth. TRIX splits the two
# tasks across separate runners and separate checkpoints; only entity prediction
# is what this suite measures and what ULTRA's numbers are.
ENV TRIX_ROOT=/kgfm/repos/trix
WORKDIR /kgfm/repos/trix

CMD ["python3.9", "src/run_entity.py", "--help"]
