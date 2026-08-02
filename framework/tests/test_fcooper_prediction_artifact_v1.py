import json
from pathlib import Path

import numpy as np
import pytest

from scripts.fcooper_prediction_artifact_v1 import PredictionArtifactWriter


def test_prediction_writer_publishes_canonical_jsonl(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    result = {
        "pred_box_tensor": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "pred_score": np.asarray([0.9, 0.4], dtype=np.float32),
        "gt_box_tensor": None,
    }

    with PredictionArtifactWriter(output) as writer:
        writer.append(sample_index=0, result=result)
        writer.append(sample_index=1, result=result)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_index"] for row in rows] == [0, 1]
    assert rows[0]["pred_box_tensor"] == [[1.0, 2.0], [3.0, 4.0]]
    assert rows[0]["pred_score"] == pytest.approx([0.9, 0.4])
    assert rows[0]["gt_box_tensor"] is None


def test_prediction_writer_removes_partial_file_on_failure(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"

    try:
        with PredictionArtifactWriter(output) as writer:
            writer.append(
                sample_index=0,
                result={
                    "pred_box_tensor": np.asarray([[float("nan")]]),
                    "pred_score": np.asarray([1.0]),
                    "gt_box_tensor": np.asarray([[0.0]]),
                },
            )
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite prediction should fail")

    assert not output.exists()
