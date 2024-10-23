"""Unit tests for the DiceLoss module."""

import pytest
import torch

from histopipe.ml.loss import DiceLoss


@pytest.fixture
def inputs() -> torch.Tensor:
    return torch.tensor([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2]])


@pytest.fixture
def targets() -> torch.Tensor:
    return torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])


def test_identical_inputs(targets: torch.Tensor) -> None:
    dice_loss = DiceLoss()
    loss = dice_loss(targets, targets, smooth=0)
    assert torch.allclose(loss, torch.tensor([0.0]), atol=1e-5)


def test_complementary_inputs(targets: torch.Tensor) -> None:
    dice_loss = DiceLoss()
    loss = dice_loss(targets, 1.0 - targets, smooth=0)
    assert torch.allclose(loss, torch.tensor([1.0]), atol=1e-5)


def test_arbitrary_inputs(inputs: torch.Tensor, targets: torch.Tensor) -> None:
    dice_loss = DiceLoss()

    # Calculate the expected result manually
    inputs_flat = inputs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (inputs_flat * targets_flat).sum()
    dice_coefficient = (2.0 * intersection) / (inputs_flat.sum() + targets_flat.sum())
    expected_loss = 1.0 - dice_coefficient

    # Calculate the actual result using the DiceLoss module
    actual_loss = dice_loss(inputs, targets, smooth=0)

    # Check if the actual result matches the expected result
    assert torch.allclose(actual_loss, expected_loss, atol=1e-5)
