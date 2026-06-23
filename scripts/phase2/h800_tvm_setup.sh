#!/usr/bin/env bash
# S2.1 on H800: isolated conda env (prefix) + TVM (mlc cu124 nightly) + driver-compat smoke.
# Uses /data/user01/miniconda3/bin/conda to build env at MY prefix /exdata/jichengzhi/tvm310.
# Driver 535(CUDA12.2) vs cu124 wheel — empirical minor-version-compat test.
set -u
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
export CONDA_PKGS_DIRS=/exdata/jichengzhi/conda_pkgs
LOG=/exdata/jichengzhi/tvm_setup.log
exec > "$LOG" 2>&1
echo "=== $(date) H800 TVM setup start (cu124) ==="
PREFIX=/exdata/jichengzhi/tvm310
PYBIN=$PREFIX/bin/python
if [ -x "$PYBIN" ] && "$PYBIN" -m pip --version >/dev/null 2>&1; then
  echo "[env] $PREFIX exists with pip, reuse"
else
  echo "[env] creating venv $PREFIX (system py3.10) + get-pip"
  rm -rf "$PREFIX"
  /usr/bin/python3 -m venv --without-pip "$PREFIX"
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /exdata/jichengzhi/get-pip.py
  "$PYBIN" /exdata/jichengzhi/get-pip.py
fi
"$PYBIN" --version
"$PYBIN" -m pip --version
"$PYBIN" -m pip install --upgrade pip -q
echo "=== pip install mlc-ai-nightly-cu124 ==="
"$PYBIN" -m pip install --pre -U -f https://mlc.ai/wheels "mlc-ai-nightly-cu124"
echo "PIP_RC=$?"
echo "=== TVM import + CUDA smoke ==="
"$PYBIN" - <<'PY'
import sys
try:
    import tvm
    print("TVM_VERSION", tvm.__version__)
    dev = tvm.cuda(0)
    print("CUDA_EXISTS", dev.exists)
    if dev.exists:
        import numpy as np
        from tvm import te
        n = 1024
        A = te.placeholder((n,), name="A"); B = te.placeholder((n,), name="B")
        C = te.compute((n,), lambda i: A[i] + B[i], name="C")
        s = te.create_schedule(C.op)
        bx, tx = s[C].split(C.op.axis[0], factor=64)
        s[C].bind(bx, te.thread_axis("blockIdx.x")); s[C].bind(tx, te.thread_axis("threadIdx.x"))
        f = tvm.build(s, [A, B, C], target="cuda")
        a = tvm.nd.array(np.random.rand(n).astype("float32"), dev)
        b = tvm.nd.array(np.random.rand(n).astype("float32"), dev)
        c = tvm.nd.array(np.zeros(n, dtype="float32"), dev)
        f(a, b, c)
        md = float(np.max(np.abs(c.numpy()-a.numpy()-b.numpy())))
        print("KERNEL_RUN_OK", md < 1e-5, "maxdiff", md)
        print("SMOKE_RESULT", "PASS" if md < 1e-5 else "FAIL")
    else:
        print("SMOKE_RESULT FAIL_NO_CUDA")
except Exception as e:
    import traceback; traceback.print_exc()
    print("SMOKE_RESULT EXCEPTION", repr(e))
    sys.exit(3)
PY
echo "=== $(date) DONE ==="
