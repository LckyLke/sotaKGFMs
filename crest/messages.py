"""Track A: an order-sensitive message that starts exactly at DistMult.

TRIX's convolution passes DistMult messages, ``m = x * z_r`` (see
``message_func: distmult`` in its configs and ``layers.py``). DistMult is
symmetric in a way that blinds it to relation direction; track A replaces it
with the bilinear form

    m = U (z_r * (V x))

one (U, V) pair of ``d x d`` matrices per layer, both initialised to the
identity so that at step 0 the message *is* DistMult -- exactly, not
approximately: multiplying by an exact identity matrix sums one product
``x_j * 1`` with exact zeros, which no accumulation order can change.

The keep rule (docs/CREST_PLAN.md, track A): keep it if mean relation MRR
rises by at least 0.02 over the phase 1 baseline.
"""

from __future__ import annotations

import torch
from torch import nn


class BilinearMessage(nn.Module):
    """``m = U (z * (V x))`` with U = V = I at init."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        self.u = nn.Parameter(torch.eye(dim))
        self.v = nn.Parameter(torch.eye(dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``x [..., d]`` node states, ``z [..., d]`` relation vectors."""
        return (z * (x @ self.v.t())) @ self.u.t()

    def is_identity(self) -> bool:
        eye = torch.eye(self.dim, device=self.u.device, dtype=self.u.dtype)
        return bool(torch.equal(self.u, eye) and torch.equal(self.v, eye))
