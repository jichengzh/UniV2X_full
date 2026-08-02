from pathlib import Path


def test_post_t16_watcher_prepares_graph_evidence_before_stage6() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts/fcooper_stage6_post_t16_watcher_v2.sh"
    ).read_text()

    assert "feedback_history_final_t16.json" in source
    assert "fcooper_prepare_stage6_graph_evidence_v2.py" in source
    assert "fcooper_stage6_five_arm_supervisor_v2.sh" in source
    assert source.index("fcooper_prepare_stage6_graph_evidence_v2.py") < source.index(
        "fcooper_stage6_five_arm_supervisor_v2.sh"
    )
