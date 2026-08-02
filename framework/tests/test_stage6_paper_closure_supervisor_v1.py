from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/stage6_paper_closure_supervisor_v1.sh"


def test_closure_supervisor_keeps_backends_separate_and_freezes_thresholds() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "stage6_collect_paper_evidence_v1.py" in text
    assert "stage6_prepare_independent_validation_v1.py" in text
    assert "stage6_run_independent_validation_v1.sh" in text
    assert "stage6_run_schedule_independent_v1.sh" in text
    assert "stage6_build_paper_main_tables_v1.py" in text
    assert "stage5_pyramid_actual_v3_20260720" in text
    assert "JOINT_VALIDATION_ROOT" in text
    assert "pyramid_stage6" not in text
    assert "TRT_GPU=\"$schedule_gpu\" TVM_GPU=\"$schedule_gpu\"" in text
    assert "any(. == \"complete\")" in text
    assert '[[ -z "$schedule_pid" ]] || wait "$schedule_pid"' in text


def test_independent_ap_executor_limits_parallel_shards_to_gpu_count() -> None:
    runner = SCRIPT.with_name("stage6_run_independent_validation_v1.sh")
    text = runner.read_text(encoding="utf-8")

    assert 'if (( ${#pids[@]} == GPU_COUNT )); then' in text
    assert 'pids=()' in text
