"""The rspmm kernel path and the torch fallback compute the same sums.

The model picks TRIX's compiled rspmm kernel on CUDA (memory: custom
backward keeps O(V*d), the difference between fitting batch-32 training in
16 GiB and not) and plain index_add everywhere else. Both paths must agree,
or CPU tests would validate arithmetic the GPU never runs. Skips without a
GPU or without the compiled extension.
"""

import pytest
import torch

from incite.layers import distmult_sum, pair_sum


def _need_kernel():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        from trix.rspmm import generalized_rspmm  # noqa: F401
    except Exception:
        pytest.skip("trix rspmm extension not importable")


def test_pair_sum_kernel_matches_fallback():
    _need_kernel()
    torch.manual_seed(0)
    b, n_in, n_out, d, p = 3, 20, 15, 8, 60
    x = torch.randn(b, n_in, d)
    src = torch.randint(n_in, (p,))
    dst = torch.randint(n_out, (p,))
    cpu = pair_sum(x, src, dst, n_out)
    gpu = pair_sum(x.cuda(), src.cuda(), dst.cuda(), n_out).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-5), (cpu - gpu).abs().max()


def test_distmult_sum_kernel_matches_fallback():
    _need_kernel()
    torch.manual_seed(0)
    b, v, r, d, e = 3, 20, 6, 8, 50
    x = torch.randn(b, v, d)
    rel = torch.randn(b, r, d)
    edge_index = torch.stack([torch.randint(v, (e,)), torch.randint(v, (e,))])
    edge_type = torch.randint(r, (e,))
    boundary = torch.randn(b, v, d)
    cpu = distmult_sum(x, rel, edge_index, edge_type, boundary)
    gpu = distmult_sum(x.cuda(), rel.cuda(), edge_index.cuda(),
                       edge_type.cuda(), boundary.cuda()).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-5), (cpu - gpu).abs().max()
