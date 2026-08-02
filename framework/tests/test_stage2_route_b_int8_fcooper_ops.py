from __future__ import annotations

import pytest

from scripts.stage2_route_b_int8_auto_decomp import (
    depth_to_space_source_index,
    validate_concat_shapes,
    validate_depth_to_space_spec,
)


def test_depth_to_space_dcr_index_mapping() -> None:
    assert depth_to_space_source_index(
        input_channels=512,
        output_channel=3,
        output_y=5,
        output_x=6,
        block_size=2,
    ) == (259, 2, 3)


def test_depth_to_space_contract_rejects_non_dcr_or_invalid_channels() -> None:
    assert validate_depth_to_space_spec([5, 512, 64, 64], 2, "DCR") == [
        5,
        128,
        128,
        128,
    ]
    with pytest.raises(ValueError, match="DCR"):
        validate_depth_to_space_spec([5, 512, 64, 64], 2, "CRD")
    with pytest.raises(ValueError, match="divisible"):
        validate_depth_to_space_spec([5, 510, 64, 64], 2, "DCR")


def test_concat_contract_accepts_channel_axis_and_rejects_shape_drift() -> None:
    assert validate_concat_shapes(
        [[5, 128, 256, 256], [5, 64, 256, 256], [5, 32, 256, 256]],
        axis=1,
    ) == [5, 224, 256, 256]
    with pytest.raises(ValueError, match="channel axis"):
        validate_concat_shapes([[5, 8, 16, 16], [5, 8, 16, 16]], axis=2)
    with pytest.raises(ValueError, match="non-channel"):
        validate_concat_shapes([[5, 8, 16, 16], [5, 8, 8, 16]], axis=1)
