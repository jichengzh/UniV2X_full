#!/usr/bin/env python3
"""Patch intermediate_multiclass_fusion_dataset.py to fix deepcopy of CUDA tensors.

In torch 2.x, copy.deepcopy fails on non-leaf (graph-internal) CUDA tensors.
The post_process_multiclass method deepcopies data_dict and output_dict which
contain model output CUDA tensors. Replace with _safe_dict_copy that detaches
and clones tensors instead of deepcopying.
"""
import py_compile, sys

path = "/data/jichengzhi_v2x/V2Xverse/opencood/data_utils/datasets/intermediate_multiclass_fusion_dataset.py"

with open(path) as f:
    content = f.read()

# 1. Add helper function at top of file after imports
# Find "import copy" line
if "_safe_dict_copy" in content:
    print("helper already present, skipping injection")
else:
    # Find a good insertion point: after the last import
    import_end = max(
        content.rfind("\nimport "),
        content.rfind("\nfrom ")
    )
    # Find end of that line
    line_end = content.find("\n", import_end + 1)

    helper = """

def _safe_dict_copy(obj):
    \"\"\"Recursively copy obj, replacing CUDA Tensors with detach().clone().
    Avoids copy.deepcopy which fails on non-leaf tensors in torch 2.x.
    \"\"\"
    import torch as _torch
    import copy as _copy
    if isinstance(obj, _torch.Tensor):
        return obj.detach().clone()
    elif isinstance(obj, dict):
        return {k: _safe_dict_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_dict_copy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_safe_dict_copy(v) for v in obj)
    else:
        try:
            return _copy.copy(obj)  # shallow copy for scalars/numpy arrays
        except Exception:
            return obj  # last resort: return reference
"""
    content = content[:line_end+1] + helper + content[line_end+1:]
    print(f"helper injected after position {line_end}")

# 2. Replace the two deepcopy calls in post_process_multiclass
OLD_LINES = (
    "                data_dict_single = copy.deepcopy(data_dict)\n"
    "                output_dict_single = copy.deepcopy(output_dict)"
)
NEW_LINES = (
    "                data_dict_single = _safe_dict_copy(data_dict)\n"
    "                output_dict_single = _safe_dict_copy(output_dict)"
)

if OLD_LINES not in content:
    print("ERROR: target lines not found in file")
    for ln, line in enumerate(content.split('\n'), 1):
        if 'data_dict_single' in line or 'output_dict_single' in line:
            print(f"  line {ln}: {repr(line)}")
    sys.exit(1)

content2 = content.replace(OLD_LINES, NEW_LINES)

with open(path, "w") as f:
    f.write(content2)

try:
    py_compile.compile(path, doraise=True)
    print("syntax OK")
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR:", e)
    sys.exit(1)

print("patch OK: deepcopy replaced with _safe_dict_copy in post_process_multiclass")
