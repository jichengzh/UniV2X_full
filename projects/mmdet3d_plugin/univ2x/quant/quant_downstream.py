"""
UniV2X downstream head QuantBlock registration.

The downstream heads (BEVFormerTrackHead, MotionHead, OccHead, PlanningHead)
consist of stacked nn.Linear + LayerNorm layers with no MSDAPlugin-adjacent tensors.
Generic QuantModel.quant_module_refactor handles all their Linear layers correctly.

This file provides:
  register_downstream_specials()  — registers any head-specific skip names
  DOWNSTREAM_SKIP_NAMES           — set of child attr names to keep in FP16

Design note (ADR-001 extension):
  - `bev_embedding` is nn.Embedding (not Linear) → ignored by quant_module_refactor
  - `positional_encoding` → ignored (non-Linear)
  - All cls_branches / reg_branches / traj_reg_branches Linear layers → quantized
  - transformer.decoder sub-modules → handled by quant_bevformer QuantCustomMSDA
"""

from .quant_model import QuantModel

# Head child attributes that should remain FP16.
# Extend before calling register_downstream_specials() if needed.
DOWNSTREAM_SKIP_NAMES: set = set()


def register_downstream_specials():
    """
    Register downstream head quantization settings into QuantModel.
    Currently all head Linear layers are safe to quantize → no special blocks.
    """
    if DOWNSTREAM_SKIP_NAMES:
        QuantModel.register_skip_names(DOWNSTREAM_SKIP_NAMES)
