"""Tests for loss functions."""

import pytest
import torch
from stegaqr.models.losses import (
    DecodeLoss,
    PerceptualLoss,
    DecodabilityLoss,
    ConfidenceLoss,
    StegaQRLoss,
)


@pytest.fixture
def tensors():
    """Standard test tensors."""
    B, C, H, W = 4, 100, 33, 33
    return {
        "logits": torch.randn(B, C),
        "targets": torch.randint(0, 2, (B, C)).float(),
        "stego": torch.rand(B, 3, H, W),
        "cover": torch.rand(B, 3, H, W),
        "mask": torch.ones(B, 1, H, W),
        "confidence": torch.rand(B, 1).clamp(0.01, 0.99),
    }


class TestDecodeLoss:
    def test_perfect_prediction_low_loss(self):
        loss_fn = DecodeLoss()
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        logits = torch.tensor([[10.0, -10.0, 10.0]])  # very confident correct
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_wrong_prediction_high_loss(self):
        loss_fn = DecodeLoss()
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        logits = torch.tensor([[-10.0, 10.0, -10.0]])  # very confident wrong
        loss = loss_fn(logits, targets)
        assert loss.item() > 5.0

    def test_gradient_exists(self, tensors):
        loss_fn = DecodeLoss()
        tensors["logits"].requires_grad_(True)
        loss = loss_fn(tensors["logits"], tensors["targets"])
        loss.backward()
        assert tensors["logits"].grad is not None


class TestPerceptualLoss:
    def test_identical_images_zero_loss(self):
        loss_fn = PerceptualLoss()
        img = torch.rand(2, 3, 33, 33)
        loss = loss_fn(img, img)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_images_positive_loss(self, tensors):
        loss_fn = PerceptualLoss()
        loss = loss_fn(tensors["stego"], tensors["cover"])
        assert loss.item() > 0.0


class TestDecodabilityLoss:
    def test_no_violation_zero_loss(self):
        loss_fn = DecodabilityLoss(margin=0.3)
        img = torch.rand(2, 3, 33, 33)
        mask = torch.ones(2, 1, 33, 33)  # all data, no protected
        loss = loss_fn(img, img, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestStegaQRLoss:
    def test_returns_all_keys(self, tensors):
        loss_fn = StegaQRLoss(use_confidence=False)
        result = loss_fn(
            tensors["logits"], tensors["targets"],
            tensors["stego"], tensors["cover"], tensors["mask"],
        )
        assert "total" in result
        assert "decode" in result
        assert "perceptual" in result
        assert "decodability" in result
        assert "bit_accuracy" in result

    def test_with_confidence(self, tensors):
        loss_fn = StegaQRLoss(use_confidence=True)
        result = loss_fn(
            tensors["logits"], tensors["targets"],
            tensors["stego"], tensors["cover"], tensors["mask"],
            tensors["confidence"],
        )
        assert "confidence" in result

    def test_total_is_differentiable(self, tensors):
        loss_fn = StegaQRLoss()
        tensors["logits"].requires_grad_(True)
        tensors["stego"].requires_grad_(True)
        result = loss_fn(
            tensors["logits"], tensors["targets"],
            tensors["stego"], tensors["cover"], tensors["mask"],
        )
        result["total"].backward()
        assert tensors["logits"].grad is not None

    def test_bit_accuracy_range(self, tensors):
        loss_fn = StegaQRLoss()
        result = loss_fn(
            tensors["logits"], tensors["targets"],
            tensors["stego"], tensors["cover"], tensors["mask"],
        )
        assert 0.0 <= result["bit_accuracy"].item() <= 1.0
