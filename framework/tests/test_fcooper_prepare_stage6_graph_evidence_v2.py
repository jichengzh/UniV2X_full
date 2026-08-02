import hashlib
import json

import pytest

from scripts.fcooper_prepare_stage6_graph_evidence_v2 import merge_graph_evidence


def test_stage6_graph_evidence_uses_t16_actual_graphs_and_graph_only_base() -> None:
    rows = [
        {
            "row_id": f"row-{index}",
            "task_id": "S5-FCO-TRT-V2",
            "training_source": "online_feedback",
            "graph_features": {
                "group_id": f"g-{index}",
                "model": "fcooper",
                "width": [32 + index, 64, 128, 64, 128],
                "graph_feature_provenance": "materialized_onnx_extracted_v1",
            },
        }
        for index in range(16)
    ]
    for row in rows:
        row["materialized_graph_features_sha256"] = hashlib.sha256(
            json.dumps(
                row["graph_features"],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        payload = dict(row)
        row["actual_feedback_row_sha256"] = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    base = {
        "group_id": "fcooper|original",
        "model": "fcooper",
        "width": [64, 128, 256, 128, 256],
        "graph_feature_provenance": "probe_graph_only_not_a_label",
        "parameter_elements": 10,
        "conv_flops": 20,
    }

    result = merge_graph_evidence(rows, base)

    assert len(result["graph_features"]) == 17
    assert result["probe_metric_labels_used"] == []
    assert result["t16_actual_graph_count"] == 16
    assert len(result["t16_graph_bindings"]) == 16
    assert result["t16_graph_bindings"][0]["row_id"] == "row-0"
    assert "actual_feedback_row_sha256" in result["t16_graph_bindings"][0]


def test_stage6_graph_evidence_rejects_duplicate_or_unbound_graphs() -> None:
    def row(index: int) -> dict:
        graph = {
            "group_id": f"g-{index}",
            "model": "fcooper",
            "width": [32 + index, 64, 128, 64, 128],
            "graph_feature_provenance": "materialized_onnx_extracted_v1",
        }
        source = {
            "row_id": f"row-{index}",
            "task_id": "S5-FCO-TRT-V2",
            "training_source": "online_feedback",
            "graph_features": graph,
            "materialized_graph_features_sha256": hashlib.sha256(
                json.dumps(
                    graph,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        source["actual_feedback_row_sha256"] = hashlib.sha256(
            json.dumps(
                source,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return source

    base = {
        "width": [64, 128, 256, 128, 256],
        "graph_feature_provenance": "probe_graph_only_not_a_label",
    }
    rows = [row(index) for index in range(16)]
    rows[1]["row_id"] = rows[0]["row_id"]
    with pytest.raises(ValueError, match="unique"):
        merge_graph_evidence(rows, base)

    rows = [row(index) for index in range(16)]
    rows[0]["materialized_graph_features_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="graph feature SHA"):
        merge_graph_evidence(rows, base)
