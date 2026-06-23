#!/usr/bin/env bash
# S2.1 run: confirm CUDA on driver 535 (te vector-add) + Pyramid base.onnx import/align.
set -u
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
LOG=/exdata/jichengzhi/s2_1_run.log
exec > "$LOG" 2>&1
PREFIX=/exdata/jichengzhi/tvm310
PYBIN=$PREFIX/bin/python
export LD_LIBRARY_PATH="$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}"
echo "=== $(date) S2.1 run ==="
echo "=== CUDA confirm (driver 535 vs cu124 minor-compat) ==="
"$PYBIN" - <<'PY'
import traceback
try:
    import tvm, numpy as np
    from tvm import te
    dev=tvm.cuda(0); print("CUDA_EXIST", dev.exist)
    n=4096
    A=te.placeholder((n,),name="A"); B=te.placeholder((n,),name="B")
    C=te.compute((n,),lambda i:A[i]*2.0+B[i],name="C")
    s=te.create_schedule(C.op)
    bx,tx=s[C].split(C.op.axis[0],factor=128)
    s[C].bind(bx,te.thread_axis("blockIdx.x")); s[C].bind(tx,te.thread_axis("threadIdx.x"))
    f=tvm.build(s,[A,B,C],target="cuda")
    a=tvm.nd.array(np.random.rand(n).astype("float32"),dev)
    b=tvm.nd.array(np.random.rand(n).astype("float32"),dev)
    c=tvm.nd.array(np.zeros(n,"float32"),dev)
    f(a,b,c)
    md=float(np.max(np.abs(c.numpy()-(a.numpy()*2.0+b.numpy()))))
    print("CUDA_SMOKE", "PASS" if md<1e-5 else "FAIL", "maxdiff", md)
except Exception as e:
    traceback.print_exc(); print("CUDA_SMOKE EXCEPTION", repr(e))
PY
echo "=== S2.1 align (relay-first, relax fallback) ==="
"$PYBIN" /exdata/jichengzhi/s2_tvm/s2_1_tvm_align.py
echo "=== $(date) S2.1 run done ==="
