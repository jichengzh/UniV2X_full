from pathlib import Path


def test_backend_tail_runs_full_backend_ap_for_schedule_only() -> None:
    source = Path("scripts/stage6_backend_tail_queue_v1.sh").read_text()
    assert "stage6_export_tvm_fp32_schedule_artifact_v1.py" in source
    assert "stage2_h800_fp16_rewritten_activation_bridge.py" in source
    assert "stage3_trt_multiscale_ap_bridge_v3.py" in source
    assert "--num-samples 1789" in source


def test_tvm_export_is_directly_importable_and_tail_is_resumable() -> None:
    exporter = Path("scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py").read_text()
    tail = Path("scripts/stage6_backend_tail_queue_v1.sh").read_text()

    assert "sys.path.insert(0, str(REPO))" in exporter
    assert 'parser.add_argument("--skip-database", action="store_true")' in exporter
    assert 'TVM_PY=${TVM_PY:-/exdata/jichengzhi/tvm310/bin/python}' in tail
    assert '"$TVM_PY" scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py' in tail
    assert 'LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:' in tail
    assert 'if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/performance_result.json" ]]' in tail
    assert 'if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/ap/full_1789/full_ap_eval_report.json" ]]' in tail
    assert 'BASE_ONNX="$REPO/results/stage6_pyramid_launch_gate_20260720/artifacts/pyramid_base_multiscale.onnx"' in tail
    assert '--checkpoint-path "$CKPT_PATH"' in tail
    assert 'assert d.get("smoke_gate_passed") is True' in tail
    assert '"$TVM_PY" scripts/stage6_tvm_schedule_transfer_probe_v1.py' in tail


def test_schedule_independent_export_uses_the_same_tvm_runtime() -> None:
    runner = Path("scripts/stage6_run_schedule_independent_v1.sh").read_text()

    assert 'TVM_PY=${TVM_PY:-/exdata/jichengzhi/tvm310/bin/python}' in runner
    assert '"$TVM_PY" scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py' in runner
    assert '--checkpoint-path "$CKPT_PATH"' in runner
    assert 'LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:' in runner
    finalizer = Path("scripts/stage6_finalize_schedule_independent_v1.py").read_text()
    assert 'ap_payload.get("smoke_gate_passed") is not True' in finalizer
