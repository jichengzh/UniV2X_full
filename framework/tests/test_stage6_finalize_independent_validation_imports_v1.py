import importlib.util
from pathlib import Path


def test_stage6_finalizer_imports_reusable_result_normalizers() -> None:
    script = Path("scripts/stage6_finalize_independent_validation_v1.py")
    spec = importlib.util.spec_from_file_location("stage6_finalizer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.evaluate_consistency)
    assert callable(module.normalize_ap_report)
    assert callable(module.normalize_performance_repeat)


def test_stage6_schedule_finalizer_imports_reusable_consistency_check() -> None:
    script = Path("scripts/stage6_finalize_schedule_independent_v1.py")
    spec = importlib.util.spec_from_file_location("stage6_schedule_finalizer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.evaluate_consistency)


def test_stage6_finalizer_derives_task_identity_from_frozen_arm_scope() -> None:
    script = Path("scripts/stage6_finalize_independent_validation_v1.py")
    spec = importlib.util.spec_from_file_location("stage6_finalizer_identity", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._validation_task_id(
        {"backend": "tvm", "arm_id": "compression_only"}
    ) == "tvm:compression_only"
