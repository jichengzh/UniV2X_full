"""ONNX symbolic wrapper for MSDAPlugin (MultiScaleDeformableAttention TRT plugin).

During normal training/evaluation the real CUDA kernel is called via ext_module.
During torch.onnx.export() the symbolic() method emits an MSDAPlugin ONNX node
that the TRT plugin picks up.
"""
import torch
from torch.autograd import Function
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class _MSDAPlugin(Function):
    @staticmethod
    def symbolic(g, value, spatial_shapes, level_start_index,
                 sampling_locations, attention_weights, im2col_step):
        return g.op("MSDAPlugin", value, spatial_shapes, level_start_index,
                    sampling_locations, attention_weights)

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        ctx.fp16 = value.dtype == torch.float16
        N, _, M, D = value.shape
        num_queries = sampling_locations.shape[1]

        # During ONNX export, the symbolic() method emits the MSDAPlugin node.
        # The forward() is only executed for shape inference — return a zero tensor
        # of the correct shape using only standard ops (avoids custom CUDA ext in
        # the JIT graph, which causes _jit_pass_peephole to fail).
        if torch.onnx.is_in_onnx_export():
            return value.new_zeros(N, num_queries, M, D)

        if ctx.fp16:
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()
        output = ext_module.ms_deform_attn_forward(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights,
            im2col_step=ctx.im2col_step,
        ).view(N, -1, M, D)
        return output.half() if ctx.fp16 else output


_MSDAPlugin_gpu = _MSDAPlugin.apply


def MSDAPlugin(value, spatial_shapes, level_start_index,
               sampling_locations, attention_weights, im2col_step):
    """Call the MSDA CUDA kernel; emits MSDAPlugin ONNX node during export."""
    assert value.is_cuda
    return _MSDAPlugin_gpu(value, spatial_shapes, level_start_index,
                           sampling_locations, attention_weights, im2col_step)
