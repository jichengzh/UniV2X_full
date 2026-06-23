#!/usr/bin/env bash
# Finalize H800 TVM env: pin ffi 0.1.11 + pytest/onnx/onnxruntime, CUDA smoke, then S2.1 align.
set -u
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
LOG=/exdata/jichengzhi/tvm_finalize.log
exec > "$LOG" 2>&1
PREFIX=/exdata/jichengzhi/tvm310
PYBIN=$PREFIX/bin/python
export LD_LIBRARY_PATH="$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}"
echo "=== $(date) finalize ==="
"$PYBIN" -m pip install -q "apache-tvm-ffi==0.1.11" pytest onnx "onnxruntime" 2>&1 | tail -2
echo "PIP_RC=$?"
echo "=== CUDA smoke ==="
"$PYBIN" - <<'PY'
import sys, traceback
try:
    import tvm
    print("TVM_VERSION", tvm.__version__)
    dev=tvm.cuda(0); print("CUDA_EXISTS", dev.exists)
    import numpy as np
    from tvm import te
    n=1024
    A=te.placeholder((n,),name="A"); B=te.placeholder((n,),name="B")
    C=te.compute((n,),lambda i:A[i]+B[i],name="C")
    s=te.create_schedule(C.op)
    bx,tx=s[C].split(C.op.axis[0],factor=64)
    s[C].bind(bx,te.thread_axis("blockIdx.x")); s[C].bind(tx,te.thread_axis("threadIdx.x"))
    f=tvm.build(s,[A,B,C],target="cuda")
    a=tvm.nd.array(np.random.rand(n).astype("float32"),dev)
    b=tvm.nd.array(np.random.rand(n).astype("float32"),dev)
    c=tvm.nd.array(np.zeros(n,dtype="float32"),dev)
    f(a,b,c)
    md=float(np.max(np.abs(c.numpy()-a.numpy()-b.numpy())))
    print("SMOKE_RESULT","PASS" if md<1e-5 else "FAIL","maxdiff",md)
except Exception as e:
    traceback.print_exc(); print("SMOKE_RESULT EXCEPTION", repr(e))
PY
echo "=== S2.1 align (Pyramid base.onnx import + numerical) ==="
"$PYBIN" /exdata/jichengzhi/s2_tvm/s2_1_tvm_align.py
echo "=== $(date) finalize done ==="
