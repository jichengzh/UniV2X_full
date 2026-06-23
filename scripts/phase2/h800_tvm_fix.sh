#!/usr/bin/env bash
# Fix: cu124 TVM runtime needs CUDA 12.4 userspace libs (libcudart w/ cudaGraphAddDependencies_v2),
# but H800 system is CUDA 12.2. Install pip nvidia cuda-12.4 libs into the env + LD_LIBRARY_PATH.
# Driver 535(12.2) supports the underlying graph driver API → minor-version-compat should run.
set -u
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
LOG=/exdata/jichengzhi/tvm_fix.log
exec > "$LOG" 2>&1
echo "=== $(date) TVM cuda-12.4 userspace fix ==="
PREFIX=/exdata/jichengzhi/tvm310
PYBIN=$PREFIX/bin/python
echo "=== install nvidia cuda 12.4 userspace libs ==="
"$PYBIN" -m pip install \
  "nvidia-cuda-runtime-cu12==12.4.127" \
  "nvidia-cuda-nvrtc-cu12==12.4.127" \
  "nvidia-cublas-cu12==12.4.5.8" \
  "nvidia-cudnn-cu12==9.1.0.70"
echo "PIP_RC=$?"
# build LD_LIBRARY_PATH from the env's nvidia libs
NVLIBS=$("$PYBIN" - <<'PY'
import os, glob, site
base=os.path.join(site.getsitepackages()[0],"nvidia")
print(":".join(sorted(glob.glob(os.path.join(base,"*","lib")))))
PY
)
echo "NVLIBS=$NVLIBS"
echo "$NVLIBS" > /exdata/jichengzhi/tvm_nvlibs.path
export LD_LIBRARY_PATH="$NVLIBS:${LD_LIBRARY_PATH:-}"
echo "=== re-run import + CUDA smoke with patched LD_LIBRARY_PATH ==="
"$PYBIN" - <<'PY'
import sys, traceback
try:
    import tvm
    print("TVM_VERSION", tvm.__version__)
    dev = tvm.cuda(0)
    print("CUDA_EXISTS", dev.exists)
    if dev.exists:
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
        print("KERNEL_RUN_OK", md<1e-5, "maxdiff", md)
        print("SMOKE_RESULT","PASS" if md<1e-5 else "FAIL")
    else:
        print("SMOKE_RESULT FAIL_NO_CUDA")
except Exception as e:
    traceback.print_exc(); print("SMOKE_RESULT EXCEPTION", repr(e)); sys.exit(3)
PY
echo "=== $(date) FIX DONE ==="
