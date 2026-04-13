"""
QuantModel for UniV2X.
Ported from QuantV2X/opencood/quant/quant_model.py.
Changes:
  - Removed opencood_specials / specials_unquantized_names imports (UniV2X has no model-specific
    QuantBlock specials at this base layer; they will be registered via register_specials())
  - Added register_specials() classmethod so callers can inject model-specific replacements
  - quant_module_refactor/quant_module_refactor_wo_fuse now consult self._specials dict
"""

import torch.nn as nn

try:
    from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
    from .fold_bn import search_fold_and_remove_bn
except ImportError:
    from quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
    from fold_bn import search_fold_and_remove_bn


class BaseQuantBlock(nn.Module):
    """
    Base class for special quantized blocks that need custom forward logic.
    UniV2X-specific QuantBlocks (e.g. QuantBEVFormerLayer) should subclass this.
    """

    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.trained = False
        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantModule,)):
                m.set_quant_state(weight_quant, act_quant)


class QuantModel(nn.Module):
    # Registry: maps original nn.Module subclass → QuantBlock subclass
    _specials: dict = {}
    # Set of child attribute names to skip during quantization traversal
    _skip_names: set = set()

    @classmethod
    def register_specials(cls, specials_dict: dict):
        """
        Register model-specific QuantBlock replacements.
        Call this before constructing QuantModel.

        :param specials_dict: {OriginalModuleClass: QuantBlockClass, ...}
        """
        cls._specials.update(specials_dict)

    @classmethod
    def register_skip_names(cls, names: set):
        """
        Register child attribute names that should NOT be quantized.
        """
        cls._skip_names.update(names)

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {},
                 is_fusing: bool = True):
        super().__init__()
        if is_fusing:
            search_fold_and_remove_bn(model, skip_names=self._skip_names)
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        else:
            self.model = model
            self.quant_module_refactor_wo_fuse(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {},
                              act_quant_params: dict = {}):
        """
        Recursively replace special/conv/linear modules with quantized variants and fuse BatchNorm.
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if name in self._skip_names:
                continue

            if type(child_module) in self._specials:
                setattr(module, name,
                        self._specials[type(child_module)](child_module, weight_quant_params, act_quant_params))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_module_refactor_wo_fuse(self, module: nn.Module, weight_quant_params: dict = {},
                                      act_quant_params: dict = {}):
        """
        Recursively replace modules with quantized variants but leave BatchNorm unchanged.
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if name in self._skip_names:
                continue

            if type(child_module) in self._specials:
                setattr(module, name,
                        self._specials[type(child_module)](child_module, weight_quant_params, act_quant_params))

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, nn.BatchNorm2d):
                if prev_quantmodule is not None:
                    prev_quantmodule.norm_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor_wo_fuse(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        a_list[-2].bitwidth_refactor(8)

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        if len(module_list) >= 3:
            module_list[-1].disable_act_quant = True
            module_list[-2].disable_act_quant = True
            module_list[-3].disable_act_quant = True

    def get_memory_footprint(self):
        """Calculate the total memory footprint of the model's parameters and buffers."""
        total_size = 0
        for param in self.parameters():
            total_size += param.nelement() * param.element_size()
        for buffer in self.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        total_size_MB = total_size / (1024 ** 2)
        return f"Model Memory Footprint: {total_size_MB:.2f} MB"
