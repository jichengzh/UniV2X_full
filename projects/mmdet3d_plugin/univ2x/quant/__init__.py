"""
UniV2X quantization package.

Public API:
    Primitives:   UniformAffineQuantizer, QuantModule, AdaRoundQuantizer, StraightThrough
    Model wrap:   QuantModel, BaseQuantBlock
    BN folding:   search_fold_and_remove_bn, fold_bn_into_conv
    Params:       set_weight_quantize_params, set_act_quantize_params, get_init, get_dc_fp_init
    Calib data:   save_inp_oup_data, save_dc_fp_data, GetLayerInpOut
    Reconstruct:  layer_reconstruction, LossFunction, LinearTempDecay
"""

from .quant_layer import (
    UniformAffineQuantizer,
    QuantModule,
    StraightThrough,
    round_ste,
    lp_loss,
)
from .adaptive_rounding import AdaRoundQuantizer
from .fold_bn import (
    search_fold_and_remove_bn,
    search_fold_and_reset_bn,
    fold_bn_into_conv,
)
from .quant_model import QuantModel, BaseQuantBlock
from .data_utils import (
    save_inp_oup_data,
    save_dc_fp_data,
    GetLayerInpOut,
    GetDcFpLayerInpOut,
    StopForwardException,
    DataSaverHook,
    _to_device,
)
from .quant_params import (
    get_init,
    get_dc_fp_init,
    set_weight_quantize_params,
    save_quantized_weight,
    set_act_quantize_params,
)
from .layer_recon import (
    layer_reconstruction,
    LossFunction,
    LinearTempDecay,
    find_unquantized_module,
)
from .quant_bevformer import (
    QuantMSDA3D,
    QuantSCA,
    QuantTSA,
    QuantCustomMSDA,
    register_bevformer_specials,
)
from .quant_fusion import register_fusion_specials, FUSION_SKIP_NAMES
from .quant_downstream import register_downstream_specials, DOWNSTREAM_SKIP_NAMES
from .comm_quant import CommQuantizer

__all__ = [
    # quant_layer
    'UniformAffineQuantizer', 'QuantModule', 'StraightThrough', 'round_ste', 'lp_loss',
    # adaptive_rounding
    'AdaRoundQuantizer',
    # fold_bn
    'search_fold_and_remove_bn', 'search_fold_and_reset_bn', 'fold_bn_into_conv',
    # quant_model
    'QuantModel', 'BaseQuantBlock',
    # data_utils
    'save_inp_oup_data', 'save_dc_fp_data', 'GetLayerInpOut', 'GetDcFpLayerInpOut',
    'StopForwardException', 'DataSaverHook', '_to_device',
    # quant_params
    'get_init', 'get_dc_fp_init', 'set_weight_quantize_params',
    'save_quantized_weight', 'set_act_quantize_params',
    # layer_recon
    'layer_reconstruction', 'LossFunction', 'LinearTempDecay', 'find_unquantized_module',
    # quant_bevformer
    'QuantMSDA3D', 'QuantSCA', 'QuantTSA', 'QuantCustomMSDA', 'register_bevformer_specials',
    # quant_fusion
    'register_fusion_specials', 'FUSION_SKIP_NAMES',
    # quant_downstream
    'register_downstream_specials', 'DOWNSTREAM_SKIP_NAMES',
    # comm_quant
    'CommQuantizer',
]
