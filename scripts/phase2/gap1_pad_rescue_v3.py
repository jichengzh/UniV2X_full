"""
Gap1 任务2: pad_rescue v3
把 trap25 backbone (channels [48,96,192]) stage0 输出 48->64 (÷32 对齐恢复),
stage1(96) / stage2(192) 已是 ÷32 倍数无需 pad。
产出: trap25_pad64_backbone.onnx

架构说明 (trap25 = PyramidFusion backbone 25% 宽度):
  stage0: plane=48, 2×plane=96 (48 = 3×16, NOT ÷32 -> kernel-cliff)
  stage1: plane=96              (÷32 ✓)
  stage2: plane=192             (÷32 ✓)

pad 策略 (stage0 only):
  conv1 (g=1): out 96->128; if in=48, in->64 (first block in=64 unchanged)
  conv2 (g=32, in_per_g=3): out 96->128, in_per_g 3->4
  conv3 (g=1): out 48->64; if in=96, in->128
  downsample (g=1, in=64): out 48->64
  cross-stage (layer1.0 conv1/downsample, in=48): in->64 ONLY (out unchanged)

代价: stage0 FLOPs 增约 (64/48)^2 ≈ 1.78x (诚实标出)
"""
import onnx, numpy as np, copy, os
from onnx import numpy_helper, helper, TensorProto, shape_inference

SRC = "/exdata/jichengzhi/s2_tvm/models/trap25_backbone.onnx"
DST = "/exdata/jichengzhi/s2_tvm/models/trap25_pad64_backbone.onnx"


def pad_t(arr, axis, new_size):
    old = arr.shape[axis]
    if old == new_size:
        return arr
    ps = list(arr.shape)
    ps[axis] = new_size - old
    return np.concatenate([arr, np.zeros(ps, dtype=arr.dtype)], axis=axis)


def is_stage0(node):
    return any("layer0" in o for o in node.output)


m = onnx.load(SRC)
init_map = {i.name: i for i in m.graph.initializer}
new_inits = {n: numpy_helper.to_array(i).copy() for n, i in init_map.items()}

changes = []
for node in m.graph.node:
    if node.op_type != "Conv":
        continue
    wname = node.input[1] if len(node.input) > 1 else None
    if not wname or wname not in new_inits:
        continue
    w = new_inits[wname]
    out_ch, in_ch_g = w.shape[0], w.shape[1]
    g = next((a.i for a in node.attribute if a.name == "group"), 1)
    in_ch = in_ch_g * g
    w2 = w.copy()
    note = ""
    s0 = is_stage0(node)

    if g == 32 and in_ch_g == 3 and out_ch == 96 and s0:
        # stage0 grouped conv2
        w2 = pad_t(pad_t(w2, 0, 128), 1, 4)
        note = "stage0_conv2_grouped"
    elif g == 1 and out_ch == 96 and s0:
        # stage0 conv1: pad out->128; in: 64->64 (unchanged), 48->64
        w2 = pad_t(w2, 0, 128)
        if in_ch == 48:
            w2 = pad_t(w2, 1, 64)
        note = "stage0_conv1"
    elif g == 1 and out_ch == 48 and s0:
        # stage0 conv3 or downsample: out->64; in: 96->128, 64->64
        w2 = pad_t(w2, 0, 64)
        if in_ch == 96:
            w2 = pad_t(w2, 1, 128)
        note = "stage0_conv3_or_ds"
    elif g == 1 and in_ch == 48 and not s0:
        # cross-stage: layer1.0 conv1/downsample, in 48->64 ONLY
        w2 = pad_t(w2, 1, 64)
        note = "xstage_in_only"

    if note:
        new_inits[wname] = w2
        changes.append(f"  {note}: {list(w.shape)} -> {list(w2.shape)} g={g}")
        if len(node.input) > 2 and node.input[2] in new_inits:
            b = new_inits[node.input[2]]
            if w2.shape[0] != w.shape[0]:
                new_inits[node.input[2]] = pad_t(b, 0, w2.shape[0])

for c in changes:
    print(c)

new_init_protos = [numpy_helper.from_array(arr, name=n) for n, arr in new_inits.items()]
new_nodes = [copy.deepcopy(nd) for nd in m.graph.node]

new_outputs = []
for out in m.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    if shape[1] == 48:
        shape[1] = 64   # stage0 output restored to /32
    # stage1(96) and stage2(192) unchanged
    new_outputs.append(helper.make_tensor_value_info(out.name, TensorProto.FLOAT, shape))
    print(f"output -> {shape}")

new_graph = helper.make_graph(new_nodes, "trap25_pad64", m.graph.input, new_outputs, new_init_protos)
new_model = helper.make_model(new_graph)
new_model.ir_version = m.ir_version
for op in m.opset_import:
    new_model.opset_import.append(op)
try:
    new_model = shape_inference.infer_shapes(new_model)
    print("Shape inference OK")
except Exception as e:
    print(f"Shape inference warn: {e}")

onnx.save(new_model, DST)
print(f"Saved {DST} ({os.path.getsize(DST) // 1024} KB)")
