from __future__ import annotations

import torch

from scripts.fcooper_recovery_v2 import (
    build_recovery_contract,
    project_conv2d_weight,
    project_conv_transpose2d_weight,
    select_l1_channels,
)
from scripts.fcooper_recovery_train_v2 import update_early_stopping
from scripts.fcooper_recovery_train_v2 import validate_recovery_contract


def test_l1_selection_is_deterministic_and_sorted_by_source_index() -> None:
    weight = torch.tensor(
        [
            [[[1.0]]],
            [[[4.0]]],
            [[[2.0]]],
            [[[3.0]]],
        ]
    )

    selected = select_l1_channels(weight, count=2, output_axis=0)

    assert selected.tolist() == [1, 3]


def test_conv_projection_respects_selected_input_and_output_channels() -> None:
    weight = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5, 1, 1)

    projected = project_conv2d_weight(
        weight,
        output_indices=torch.tensor([1, 3]),
        input_indices=torch.tensor([0, 2, 4]),
    )

    assert projected.shape == (2, 3, 1, 1)
    assert torch.equal(projected[:, :, 0, 0], weight[[1, 3]][:, [0, 2, 4], 0, 0])


def test_transposed_conv_projection_uses_pytorch_weight_layout() -> None:
    weight = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4, 1, 1)

    projected = project_conv_transpose2d_weight(
        weight,
        output_indices=torch.tensor([1, 3]),
        input_indices=torch.tensor([0, 2, 4]),
    )

    assert projected.shape == (3, 2, 1, 1)
    assert torch.equal(projected[:, :, 0, 0], weight[[0, 2, 4]][:, [1, 3], 0, 0])


def test_recovery_contract_freezes_training_and_forbids_prefix_only_source() -> None:
    contract = build_recovery_contract(seed=20260723)

    assert contract["initialization_policy"] == "scanner_dependency_l1_v2"
    assert contract["recovery_epochs"] == 8
    assert contract["start_epoch"] == 23
    assert contract["minimum_epochs"] == 4
    assert contract["early_stopping_patience"] == 3
    assert contract["prefix_projection_is_final_measurement_source"] is False
    assert contract["full_train_split"] is True
    assert contract["full_validation_split"] is True
    assert validate_recovery_contract(contract) == contract


def test_recovery_contract_rejects_zero_epoch_or_other_drift() -> None:
    contract = build_recovery_contract(seed=20260723)

    for field, value in (
        ("recovery_epochs", 0),
        ("minimum_epochs", 0),
        ("full_train_split", False),
        ("initialization_policy", "scanner_guided_prefix_channel_projection_v1"),
    ):
        drifted = {**contract, field: value}
        try:
            validate_recovery_contract(drifted)
        except ValueError as error:
            assert "contract drift" in str(error)
        else:
            raise AssertionError(f"{field} drift was accepted")


def test_early_stopping_state_is_updated_without_mutation() -> None:
    state = {"best_loss": 2.0, "bad_epochs": 1, "best_epoch": 23}

    improved = update_early_stopping(
        state, validation_loss=1.5, epoch=24, min_delta=1.0e-4
    )
    stale = update_early_stopping(
        improved, validation_loss=1.50001, epoch=25, min_delta=1.0e-4
    )

    assert state == {"best_loss": 2.0, "bad_epochs": 1, "best_epoch": 23}
    assert improved == {"best_loss": 1.5, "bad_epochs": 0, "best_epoch": 24}
    assert stale == {"best_loss": 1.5, "bad_epochs": 1, "best_epoch": 24}
