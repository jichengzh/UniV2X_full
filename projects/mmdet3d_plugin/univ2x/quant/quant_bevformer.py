"""
UniV2X BEVFormer-specific QuantBlocks.

Design (ADR-001):
  - sampling_offsets and attention_weights feed directly into MSDAPlugin → keep FP16
  - value_proj and output_proj are safe to quantize

We register QuantBlock wrappers for the four MSDA module families:
  1. MSDeformableAttention3D / TRT  (inside SpatialCrossAttention)
  2. SpatialCrossAttention / TRT
  3. TemporalSelfAttention / TRT
  4. CustomMSDeformableAttention / TRT  (decoder)

Usage:
    from .quant_bevformer import register_bevformer_specials
    register_bevformer_specials()   # call once before constructing QuantModel
"""

import torch.nn as nn
from .quant_model import QuantModel, BaseQuantBlock
from .quant_layer import QuantModule


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _wrap_linears(module, quant_attrs, weight_quant_params, act_quant_params):
    """Replace named Linear attrs on module with QuantModule in-place."""
    for attr in quant_attrs:
        layer = getattr(module, attr, None)
        if layer is not None and isinstance(layer, nn.Linear):
            setattr(module, attr,
                    QuantModule(layer, weight_quant_params, act_quant_params))


def _set_quant_state_on_attrs(module, quant_attrs, weight_quant, act_quant):
    for attr in quant_attrs:
        m = getattr(module, attr, None)
        if isinstance(m, QuantModule):
            m.set_quant_state(weight_quant, act_quant)


# ---------------------------------------------------------------------------
# 1. MSDeformableAttention3D / TRT  (nested inside SpatialCrossAttention)
#    Has: sampling_offsets ✗, attention_weights ✗, value_proj ✓, output_proj N/A
# ---------------------------------------------------------------------------

class QuantMSDA3D(BaseQuantBlock):
    """Quantized wrapper for MSDeformableAttention3D (and TRT variant)."""

    _QUANT = ('value_proj',)          # output_proj is None in MSDeformableAttention3D
    _SKIP  = ('sampling_offsets', 'attention_weights')

    def __init__(self, orig_module: nn.Module,
                 weight_quant_params: dict, act_quant_params: dict):
        super().__init__()
        self.orig_module = orig_module
        _wrap_linears(orig_module, self._QUANT, weight_quant_params, act_quant_params)

    def forward(self, *args, **kwargs):
        return self.orig_module(*args, **kwargs)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        _set_quant_state_on_attrs(self.orig_module, self._QUANT, weight_quant, act_quant)


# ---------------------------------------------------------------------------
# 2. SpatialCrossAttention / TRT
#    Has: output_proj ✓; deformable_attention will be handled by QuantMSDA3D
# ---------------------------------------------------------------------------

class QuantSCA(BaseQuantBlock):
    """Quantized wrapper for SpatialCrossAttention (and TRT variant)."""

    _QUANT = ('output_proj',)

    def __init__(self, orig_module: nn.Module,
                 weight_quant_params: dict, act_quant_params: dict):
        super().__init__()
        self.orig_module = orig_module
        _wrap_linears(orig_module, self._QUANT, weight_quant_params, act_quant_params)
        # Recursively handle nested deformable_attention
        # (MSDeformableAttention3D / TRT → QuantMSDA3D)
        _refactor_nested(orig_module, weight_quant_params, act_quant_params)

    def forward(self, *args, **kwargs):
        return self.orig_module(*args, **kwargs)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        _set_quant_state_on_attrs(self.orig_module, self._QUANT, weight_quant, act_quant)
        for m in self.orig_module.modules():
            if isinstance(m, QuantMSDA3D):
                m.set_quant_state(weight_quant, act_quant)


# ---------------------------------------------------------------------------
# 3. TemporalSelfAttention / TRT
#    Has: sampling_offsets ✗, attention_weights ✗, value_proj ✓, output_proj ✓
# ---------------------------------------------------------------------------

class QuantTSA(BaseQuantBlock):
    """Quantized wrapper for TemporalSelfAttention (and TRT variant)."""

    _QUANT = ('value_proj', 'output_proj')
    _SKIP  = ('sampling_offsets', 'attention_weights')

    def __init__(self, orig_module: nn.Module,
                 weight_quant_params: dict, act_quant_params: dict):
        super().__init__()
        self.orig_module = orig_module
        _wrap_linears(orig_module, self._QUANT, weight_quant_params, act_quant_params)

    def forward(self, *args, **kwargs):
        return self.orig_module(*args, **kwargs)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        _set_quant_state_on_attrs(self.orig_module, self._QUANT, weight_quant, act_quant)


# ---------------------------------------------------------------------------
# 4. CustomMSDeformableAttention / TRT  (decoder.py)
#    Has: sampling_offsets ✗, attention_weights ✗, value_proj ✓, output_proj ✓
# ---------------------------------------------------------------------------

class QuantCustomMSDA(BaseQuantBlock):
    """Quantized wrapper for CustomMSDeformableAttention (and TRT variant)."""

    _QUANT = ('value_proj', 'output_proj')
    _SKIP  = ('sampling_offsets', 'attention_weights')

    def __init__(self, orig_module: nn.Module,
                 weight_quant_params: dict, act_quant_params: dict):
        super().__init__()
        self.orig_module = orig_module
        _wrap_linears(orig_module, self._QUANT, weight_quant_params, act_quant_params)

    def forward(self, *args, **kwargs):
        return self.orig_module(*args, **kwargs)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        _set_quant_state_on_attrs(self.orig_module, self._QUANT, weight_quant, act_quant)


# ---------------------------------------------------------------------------
# Nested refactor helper (for SCA's deformable_attention)
# ---------------------------------------------------------------------------

def _refactor_nested(module, wqp, aqp):
    """Replace known MSDA module types inside `module` with their Quant wrappers."""
    from projects.mmdet3d_plugin.univ2x.modules.spatial_cross_attention import (
        MSDeformableAttention3D, MSDeformableAttention3DTRT,
    )
    _map = {
        MSDeformableAttention3D: QuantMSDA3D,
        MSDeformableAttention3DTRT: QuantMSDA3D,
    }
    for name, child in module.named_children():
        if type(child) in _map:
            setattr(module, name, _map[type(child)](child, wqp, aqp))
        else:
            _refactor_nested(child, wqp, aqp)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_bevformer_specials():
    """
    Register BEVFormer MSDA specials into QuantModel._specials.
    Call once before constructing any QuantModel.
    """
    from projects.mmdet3d_plugin.univ2x.modules.spatial_cross_attention import (
        MSDeformableAttention3D, MSDeformableAttention3DTRT,
        SpatialCrossAttention, SpatialCrossAttentionTRT,
    )
    from projects.mmdet3d_plugin.univ2x.modules.temporal_self_attention import (
        TemporalSelfAttention, TemporalSelfAttentionTRT,
    )
    from projects.mmdet3d_plugin.univ2x.modules.decoder import (
        CustomMSDeformableAttention, CustomMSDeformableAttentionTRT,
    )

    QuantModel.register_specials({
        MSDeformableAttention3D:       QuantMSDA3D,
        MSDeformableAttention3DTRT:    QuantMSDA3D,
        SpatialCrossAttention:         QuantSCA,
        SpatialCrossAttentionTRT:      QuantSCA,
        TemporalSelfAttention:         QuantTSA,
        TemporalSelfAttentionTRT:      QuantTSA,
        CustomMSDeformableAttention:   QuantCustomMSDA,
        CustomMSDeformableAttentionTRT: QuantCustomMSDA,
    })
