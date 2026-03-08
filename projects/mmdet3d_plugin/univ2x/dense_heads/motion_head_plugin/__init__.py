try:
    from .motion_optimization import MotionNonlinearSmoother
except ImportError:
    pass  # casadi not installed; MotionNonlinearSmoother unavailable (TRT path only)
from .modules import (MotionTransformerDecoder,
                      MotionTransformerDecoderTRT, MotionTransformerDecoderTRTP)
from .motion_deformable_attn import (
    MotionTransformerAttentionLayer, MotionDeformableAttention,
    MotionTransformerAttentionLayerTRT, MotionTransformerAttentionLayerTRTP,
    MotionDeformableAttentionTRT, MotionDeformableAttentionTRTP,
)
from .motion_utils import *