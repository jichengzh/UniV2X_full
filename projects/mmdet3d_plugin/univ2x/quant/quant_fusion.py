"""
UniV2X V2X fusion module QuantBlock registration.

The fusion MLPs (LaneQueryFusion, AgentQueryFusion and their TRT variants) consist
entirely of standard nn.Linear layers with no MSDAPlugin-adjacent tensors.
Generic QuantModel.quant_module_refactor handles all their Linear layers correctly
without any custom blocks.

This file provides:
  register_fusion_specials()  — no-op registration (placeholder for future extension)
  FUSION_SKIP_NAMES           — set of child attr names to keep in FP16 if needed
"""

from .quant_model import QuantModel

# Fusion module child attributes that should remain FP16 (extend if needed).
FUSION_SKIP_NAMES: set = set()


def register_fusion_specials():
    """
    Register V2X fusion module quantization settings into QuantModel.
    Currently all fusion Linear layers are safe to quantize → no special blocks.

    Add entries to FUSION_SKIP_NAMES before calling this function to exclude
    specific sub-module names from quantization.
    """
    if FUSION_SKIP_NAMES:
        QuantModel.register_skip_names(FUSION_SKIP_NAMES)
