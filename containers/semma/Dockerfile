# SEMMA container.
#
# Build context is the workspace root (kgfm/), not this directory:
#     scripts/clone_repos.sh          # repos/ is gitignored; the build needs it
#     docker build -f containers/semma/Dockerfile -t kgfm/semma:ed525ac1 .
#
# Direct ULTRA fork: the package is still called `ultra` and rspmm.cpp is
# byte-identical, so the stack tracks containers/ultra/Dockerfile. What SEMMA
# adds is a semantic half -- relation descriptions from an LLM, embedded with a
# sentence encoder and turned into a second relation graph -- so this image also
# carries transformers and sentence-transformers.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-11.8/ \
    PATH=/usr/local/cuda-11.8/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-dev python3.9-distutils \
        build-essential git patch \
    && curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --no-cache-dir 'setuptools==69.5.1' wheel

# numpy==1.24.1 is SEMMA's own pin, and the only hard pin in its
# requirements.txt. It is honoured rather than aligned with the 1.26.4 the other
# images use: it is the authors' stated version, it satisfies the NumPy 1.x C ABI
# that torch, torch-scatter and pyarrow are all built against, and nothing this
# project reports is computed by numpy inside the container -- shared/metrics.py
# runs outside every container, over the dumped ranks.
RUN python3.9 -m pip install --no-cache-dir 'numpy==1.24.1'

RUN python3.9 -m pip install --no-cache-dir \
        torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN python3.9 -m pip install --no-cache-dir \
        torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 \
        -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN python3.9 -m pip install --no-cache-dir ninja easydict pyyaml

# SEMMA's semantic half. flash-attn is in requirements.txt but commented out by
# the authors, with a note that it is awkward to install; nothing on the
# inference path imports it, so it stays out.
#
# transformers is pinned, and requirements.txt does not pin it. Current releases
# call torch.utils._pytree.register_pytree_node, which only exists from torch
# 2.2; against SEMMA's pinned torch 2.1.0 the import dies with
#   AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
# so an unpinned install cannot work with SEMMA's own torch pin at all. 4.37.2 is
# the newest release that still uses the private name. jina-embeddings-v3 is
# unaffected: its config.json records transformers_version 4.30.2, so it needs
# nothing newer than this.
RUN python3.9 -m pip install --no-cache-dir \
        'transformers==4.37.2' 'sentence-transformers==2.7.0' \
        einops nltk python-dotenv httpx aiohttp

# Added by this project, not by SEMMA: the rank dump writes parquet.
RUN python3.9 -m pip install --no-cache-dir 'pyarrow==14.0.2'

RUN python3.9 -c "\
import numpy, torch, pyarrow, torch_geometric, torch_scatter, torch_sparse, transformers, sentence_transformers; \
assert numpy.__version__.startswith('1.24'), 'SEMMA pins numpy 1.24.1: ' + numpy.__version__; \
assert torch.__version__.startswith('2.1.0'), torch.__version__; \
print('stack ok:', 'numpy', numpy.__version__, '| torch', torch.__version__, '| pyarrow', pyarrow.__version__)"

# Both encoders flags.model_embed can select, cached at build time so that a run
# needs no network at all.
#
# `sentbert` -> all-mpnet-base-v2, ordinary weights, no code execution.
RUN python3.9 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-mpnet-base-v2'); print('cached all-mpnet-base-v2')"

# `jinaai` -> jina-embeddings-v3, which SEMMA's own flags.yaml selects, and which
# requires trust_remote_code=True: transformers downloads custom_st.py from the
# model repository and executes it. Two things make that a bounded risk rather
# than an open one, and both are deliberate:
#
#   1. It is fetched HERE, during the build. A build has no bind mount, so the
#      remote code cannot see or touch the workspace at the moment it first runs.
#   2. REVISION pins the exact commit. Without it, `main` is whatever the
#      repository holds on the day of the run, and "reproducible" would mean
#      nothing for the half of SEMMA that this code builds.
#
# Licence is cc-by-nc-4.0: non-commercial, which academic benchmarking satisfies.
# Runs set HF_HUB_OFFLINE=1 so the pinned copy below is the one that executes.
ENV JINA_REVISION=ab036b023d30b4d1138c4c3bfa9f0c445ab455d6
RUN python3.9 -c "\
import os; from transformers import AutoModel, AutoTokenizer; \
rev = os.environ['JINA_REVISION']; name='jinaai/jina-embeddings-v3'; \
AutoTokenizer.from_pretrained(name, revision=rev, trust_remote_code=True); \
AutoModel.from_pretrained(name, revision=rev, trust_remote_code=True); \
print('cached', name, 'at', rev)"

# Everything the run needs is now in the image. Offline from here on, so a run
# cannot silently pick up a newer revision of the encoder or its remote code
# than the one this image was built and pinned against.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /kgfm

# Per-repo extensions directory: torch names the build directory after the
# extension, "rspmm" for every one of these repos, so a shared directory makes
# concurrent runs contend on one FileBaton lock and the loser blocks silently.
ENV TORCH_EXTENSIONS_DIR=/kgfm/output/.torch_extensions-semma

COPY shared/ /kgfm/shared/
ENV PYTHONPATH=/kgfm/shared

COPY repos/semma/ /kgfm/repos/semma/
COPY patches/semma/ /kgfm/patches/semma/

# The clone ships a stale __pycache__ built by upstream's own Python 3.9. Left
# in place it can shadow a patched module, which would apply the patch to the
# source and run the cached bytecode.
RUN find /kgfm/repos/semma -name "__pycache__" -type d -prune -exec rm -rf {} + \
    && cd /kgfm/repos/semma \
    && for p in /kgfm/patches/semma/*.diff; do \
         echo "applying $p"; patch -p1 --batch --forward < "$p" || exit 1; \
       done

RUN mkdir -p /kgfm/data/roots/semma /kgfm/data/raw /kgfm/ranks/semma \
             /kgfm/output /kgfm/output/.torch_extensions-semma
VOLUME ["/kgfm/data", "/kgfm/ranks", "/kgfm/output"]

# fb_mid2name.tsv is NOT baked in. It is 230 MB, it is not part of the
# repository, and ultra/datasets.py resolves it against os.getcwd() -- so it
# belongs next to the work tree, which scripts/prepare_semma_workdir.sh puts
# there from data/raw/semma/. See scripts/fetch_semma_mid2name.py for why
# upstream's setup.sh does not produce a usable copy.

ENV SEMMA_ROOT=/kgfm/repos/semma
WORKDIR /kgfm/repos/semma

CMD ["python3.9", "script/run_many.py", "--help"]
