# KGPFN container.
#
# Build context is the workspace root (kgfm/), not this directory:
#     scripts/clone_repos.sh          # repos/ is gitignored; the build needs it
#     docker build -f containers/kgpfn/Dockerfile -t kgfm/kgpfn:af415c33 .
#
# NOT an ULTRA fork (see containers/STACKS.md): a prior-fitted-network that
# couples an ULTRA-style structure encoder with a tabular in-context
# transformer (TabICL here). It is also the only repo in this suite with a
# fully pinned requirements.txt -- every line is `==` -- so unlike TRIX there
# is nothing to choose: the stack below is the authors', taken verbatim, and
# any deviation is called out where it happens.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-12.1/ \
    PATH=/usr/local/cuda-12.1/bin:$PATH

# Python 3.12 from deadsnakes (README: `conda create -n kgpfn python=3.12`;
# ubuntu22.04 ships 3.10). No distutils package: 3.12 removed distutils, and
# the generic get-pip bootstrap supports 3.12 directly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential git patch \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --no-cache-dir setuptools wheel

# numpy 2.4.3 is the authors' pin. First so nothing below resolves another.
RUN python3.12 -m pip install --no-cache-dir 'numpy==2.4.3'

# Core torch trio, exactly requirements.txt's +cu121 builds.
RUN python3.12 -m pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

# PyG five, exactly requirements.txt's +pt25cu121 builds.
RUN python3.12 -m pip install --no-cache-dir \
        torch-geometric==2.7.0 \
        torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
        pyg-lib==0.4.0 \
        -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# The rest of requirements.txt in one resolver pass so the pins co-resolve.
# The two unpinned lines (matplotlib, xgboost) stay unpinned as upstream wrote
# them. tabicl/tabpfn are needed as PIP packages, not only as the vendored
# model/ copies: model/tabicl/model/custom_tabicl.py imports
# `tabicl.model.interaction` from the installed package.
RUN python3.12 -m pip install --no-cache-dir \
        'transformers==5.4.0' 'sentence-transformers==5.3.0' 'huggingface-hub==1.8.0' \
        'tabicl==2.0.3' 'tabpfn==7.1.1' 'tabpfn-common-utils==0.2.19' \
        'accelerate==1.13.0' 'wandb==0.25.1' 'einops==0.8.2' \
        'scipy==1.17.1' 'pandas==3.0.2' 'scikit-learn==1.8.0' 'networkx==3.6.1' \
        matplotlib xgboost \
        'PyYAML==6.0.3' 'Jinja2==3.1.6' 'easydict==1.13' 'tqdm==4.67.3' ninja

# flash-attn is REQUIRED (TabICL's attention path) and is installed from a
# prebuilt wheel, never compiled -- the SEMMA-era wheel-hunt pattern. The
# README's example wheel is cu12torch2.7/cp312, which does not match the
# pinned torch 2.5.1; the matching coordinates are:
#   cu12       : this image's CUDA 12.1
#   torch2.5   : the pinned 2.5.1
#   cxx11abiFALSE : PyPI/download.pytorch.org torch 2.5.1 builds with the old ABI
#   cp312      : the pinned Python
# v2.8.0.post2 is the release family the README itself points at; its
# torch2.5 wheel exists and was verified downloadable before this line was
# written. A copy of the wheel and its sha256 live in
# data/raw/kgpfn/ + data/raw/MANIFEST-kgpfn.json (data/ is dockerignored, so
# the build installs from the release URL directly).
RUN python3.12 -m pip install --no-cache-dir \
    'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl'

# Added by this project, not by KGPFN: the rank dump writes parquet.
# pyarrow 21 is built against the NumPy 2.x ABI that this stack pins.
RUN python3.12 -m pip install --no-cache-dir 'pyarrow==21.0.0'

# Fail the build, not the run.
RUN python3.12 -c "\
import numpy, torch, pyarrow, torch_geometric, torch_scatter, flash_attn, \
       tabicl, tabpfn, transformers, wandb, easydict, jinja2, yaml; \
assert numpy.__version__ == '2.4.3', numpy.__version__; \
assert torch.__version__.startswith('2.5.1'), torch.__version__; \
assert torch.version.cuda == '12.1', torch.version.cuda; \
print('stack ok:', 'numpy', numpy.__version__, '| torch', torch.__version__, \
      '| flash_attn', flash_attn.__version__, '| pyarrow', pyarrow.__version__)"

WORKDIR /kgfm

# Per-repo extensions directory. KGPFN carries TWO byte-identical copies of
# ULTRA's rspmm (model/ultra/rspmm/, pfn/rspmm/), both JIT-compiled under the
# name "rspmm"; a directory shared with another repo's container would contend
# on one FileBaton lock and the loser blocks silently at ~0% CPU.
ENV TORCH_EXTENSIONS_DIR=/kgfm/output/.torch_extensions-kgpfn

COPY shared/ /kgfm/shared/
ENV PYTHONPATH=/kgfm/shared

COPY repos/kgpfn/ /kgfm/repos/kgpfn/
COPY patches/kgpfn/ /kgfm/patches/kgpfn/

# Stale __pycache__ from the authors' machines could shadow a patched module:
# the patch would land on the source while the cached bytecode still runs.
RUN find /kgfm/repos/kgpfn -name "__pycache__" -type d -prune -exec rm -rf {} + \
    && cd /kgfm/repos/kgpfn \
    && for p in /kgfm/patches/kgpfn/*.diff; do \
         echo "applying $p"; patch -p1 --batch --forward < "$p" || exit 1; \
       done

RUN mkdir -p /kgfm/data/roots/kgpfn /kgfm/data/raw /kgfm/ranks/kgpfn \
             /kgfm/output /kgfm/output/.torch_extensions-kgpfn
VOLUME ["/kgfm/data", "/kgfm/ranks", "/kgfm/output"]

# Checkpoints are NOT baked in. kgpfn_icl.pth (~304 MB, HF Eason-nuo/KGPFN)
# and the TabICL config extracted from tabicl-regressor-v2-20260212.ckpt live
# in data/raw/kgpfn/ on the bind mount, recorded with sha256 in
# data/raw/MANIFEST-kgpfn.json; scripts/run_kgpfn.sh points the eval config at
# them by absolute path, so nothing here resolves against os.getcwd().

ENV KGPFN_ROOT=/kgfm/repos/kgpfn
WORKDIR /kgfm/repos/kgpfn

CMD ["python3.12", "script/test_kgpfn.py", "--help"]
