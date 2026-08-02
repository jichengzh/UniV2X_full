from pathlib import Path


def test_fixed_controller_quarantines_performance_failures_before_ap() -> None:
    source = Path("scripts/stage5_task_round_controller_v3.sh").read_text()
    quarantine = source.index("stage6_performance_failure_quarantine_v1")
    ap_plan = source.index("scripts/stage5_ap_plan_v2.py")
    advance = source.index("scripts/stage5_advance_task_round_v2.py")
    assert quarantine < ap_plan < advance
    assert 'status == "confirmed_failure"' in source


def test_resume_supervisor_preserves_completed_backend_tail_markers() -> None:
    source = Path("scripts/stage6_resume_parallel_queues_v1.sh").read_text()
    assert 'backend_tail_queue.{done,failed}' not in source
    assert 'backend_tail_queue.failed' in source


def test_paper_closure_reuses_only_matching_passed_validation_audits() -> None:
    source = Path("scripts/stage6_paper_closure_supervisor_v1.sh").read_text()
    assert "all_tasks_passed" in source
    assert "expected_ids" in source
    assert "audit_ids" in source
    assert 'validation_audits+=("$audit")' in source


def test_paper_closure_scopes_joint_audit_to_pyramid_tasks() -> None:
    source = Path("scripts/stage6_paper_closure_supervisor_v1.sh").read_text()
    assert "stage6_scope_joint_validation_audit_v1.py" in source
    assert "S5-PYR-TVM" in source
    assert "S5-PYR-TRT" in source
