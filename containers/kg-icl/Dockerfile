# KG-ICL container.
#
# Build context is the workspace root (kgfm/), not this directory:
#     scripts/clone_repos.sh          # repos/ is gitignored; the build needs it
#     docker build -f containers/kg-icl/Dockerfile -t kgfm/kg-icl:6a3166e3 .
#
# Not an ULTRA fork. KG-ICL answers a query by building a prompt graph around an
# example fact and encoding it in context, so it shares no code with the other
# repos -- including, and this matters most, its ranking. See
# patches/kg-icl/0001 for what that means and what is done about it.
#
# Its own pins: torch 2.2.0 on CUDA 11.8, Python 3.9, PyG 2.4.0.
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
        build-essential git patch unzip \
    && curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --no-cache-dir 'setuptools==69.5.1' wheel

# The README's own fallback pin. numpy 2.x breaks the C ABI that torch,
# torch-scatter and pyarrow are all built against.
RUN python3.9 -m pip install --no-cache-dir 'numpy==1.24.0'

RUN python3.9 -m pip install --no-cache-dir \
        torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
RUN python3.9 -m pip install --no-cache-dir \
        torch-scatter==2.1.2 torch-sparse==0.6.18 torch-geometric==2.4.0 \
        -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
RUN python3.9 -m pip install --no-cache-dir ninja easydict pyyaml tqdm networkx scipy

# Added by this project, not by KG-ICL: the rank dump writes parquet.
RUN python3.9 -m pip install --no-cache-dir 'pyarrow==14.0.2'

RUN python3.9 -c "\
import numpy, torch, pyarrow, torch_geometric, torch_scatter, torch_sparse, scipy, networkx; \
assert numpy.__version__.startswith('1.'), 'numpy 2.x breaks the torch/pyarrow C ABI: ' + numpy.__version__; \
assert torch.__version__.startswith('2.2.0'), torch.__version__; \
print('stack ok:', 'numpy', numpy.__version__, '| torch', torch.__version__, '| pyarrow', pyarrow.__version__)"

WORKDIR /kgfm

ENV TORCH_EXTENSIONS_DIR=/kgfm/output/.torch_extensions-kg-icl

COPY shared/ /kgfm/shared/
ENV PYTHONPATH=/kgfm/shared

COPY repos/kg-icl/ /kgfm/repos/kg-icl/
COPY patches/kg-icl/ /kgfm/patches/kg-icl/

RUN find /kgfm/repos/kg-icl -name "__pycache__" -type d -prune -exec rm -rf {} + \
    && cd /kgfm/repos/kg-icl \
    && for p in /kgfm/patches/kg-icl/*.diff; do \
         echo "applying $p"; patch -p1 --batch --forward < "$p" || exit 1; \
       done

# datasets.zip ships inside the repository and holds 55 graphs, of which 27 are
# in this suite. It is unpacked at build time so a run needs no unzip step, and
# so that scripts/verify_kgicl_data.py can compare what is in here against the
# copies the other repos download. They are NOT assumed to be the same graphs.
RUN cd /kgfm/repos/kg-icl && unzip -q -o datasets.zip -d datasets_unpacked \
    && ls datasets_unpacked | head

RUN mkdir -p /kgfm/data/roots/kg-icl /kgfm/data/raw /kgfm/ranks/kg-icl \
             /kgfm/output /kgfm/output/.torch_extensions-kg-icl
VOLUME ["/kgfm/data", "/kgfm/ranks", "/kgfm/output"]

ENV KGICL_ROOT=/kgfm/repos/kg-icl
WORKDIR /kgfm/repos/kg-icl/src

CMD ["python3.9", "evaluation.py", "--help"]
