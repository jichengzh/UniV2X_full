#!/usr/bin/env bash
# Find an apache-tvm-ffi version that imports cleanly with mlc tvm 0.20.dev1070.
set -u
export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
LOG=/exdata/jichengzhi/tvm_ffisweep.log
exec > "$LOG" 2>&1
PREFIX=/exdata/jichengzhi/tvm310
PYBIN=$PREFIX/bin/python
export LD_LIBRARY_PATH="$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}"
echo "=== $(date) ffi sweep ==="
for V in 0.1.11 0.1.10 0.1.9 0.1.8.post2 0.1.7 0.1.6 0.1.5 0.1.4; do
  echo "--- trying apache-tvm-ffi==$V ---"
  "$PYBIN" -m pip install -q "apache-tvm-ffi==$V" 2>&1 | tail -1
  R=$("$PYBIN" -c "import tvm; print('IMPORT_OK', tvm.__version__)" 2>&1 | tail -3)
  echo "$R"
  if echo "$R" | grep -q IMPORT_OK; then
    echo "GOOD_FFI_VERSION $V"
    break
  fi
done
echo "=== $(date) sweep done ==="
