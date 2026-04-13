"""
V2X Communication Feature Quantizer.

Simulates bandwidth compression by quantizing features at V2X communication points:
  - Agent query fusion (infra track_query -> ego)
  - Lane query fusion (infra lane_query -> ego)
  - BEV scatter (infra BEV features -> ego)
"""

import torch
import torch.nn as nn

try:
    from .quant_layer import UniformAffineQuantizer
except ImportError:
    from quant_layer import UniformAffineQuantizer


class CommQuantizer(nn.Module):
    """Quantize V2X communication features to simulate bandwidth compression."""

    def __init__(self, n_bits: int = 8, symmetric: bool = True,
                 scale_method: str = 'minmax'):
        super().__init__()
        self.n_bits = n_bits
        self.enabled = True
        # Only create quantizer for valid bit-widths (2-8); >= 16 means passthrough
        if 2 <= n_bits <= 8:
            self.quantizer = UniformAffineQuantizer(
                n_bits=n_bits, symmetric=symmetric,
                channel_wise=False, scale_method=scale_method)
        else:
            self.quantizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.quantizer is None:
            return x
        # Reset inited so scale is recomputed per-tensor (communication features
        # vary across frames, so we calibrate fresh each forward pass).
        self.quantizer.inited = False
        return self.quantizer(x)

    def __repr__(self) -> str:
        return f'CommQuantizer(bits={self.n_bits}, enabled={self.enabled})'
