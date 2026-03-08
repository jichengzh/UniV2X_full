"""ONNX symbolic wrapper for InversePlugin (batched matrix inverse TRT plugin).

During normal forward the standard torch.linalg.inv is used.
During torch.onnx.export() the symbolic() emits an InversePlugin ONNX node.

Additionally, register_custom_op_symbolic() intercepts both torch.inverse and
torch.linalg.inv calls made anywhere in the model during ONNX export.
"""
import torch
from torch.autograd import Function


class _InversePlugin(Function):
    @staticmethod
    def symbolic(g, x):
        return g.op("InversePlugin", x)

    @staticmethod
    def forward(ctx, x):
        return torch.linalg.inv(x)


_InversePlugin_gpu = _InversePlugin.apply


def InversePlugin(x):
    """Compute matrix inverse; emits InversePlugin ONNX node during export."""
    return _InversePlugin_gpu(x)


# ---------------------------------------------------------------------------
# Intercept torch.inverse / torch.linalg.inv during ONNX export
# ---------------------------------------------------------------------------
def _custom_inverse_handler(g, *args, **kwargs):
    return g.op("InversePlugin", args[0])


def register_inverse_symbolic():
    """Register ONNX symbolic handlers for built-in inverse ops.

    Call this once before torch.onnx.export() so that any torch.inverse or
    torch.linalg.inv calls in the model graph are converted to InversePlugin nodes.
    """
    torch.onnx.register_custom_op_symbolic("::inverse", _custom_inverse_handler, 1)
    try:
        torch.onnx.register_custom_op_symbolic("aten::linalg_inv", _custom_inverse_handler, 1)
    except Exception:
        pass
    try:
        torch.onnx.register_custom_op_symbolic("linalg::inv", _custom_inverse_handler, 1)
    except Exception:
        pass
