"""Tests for evaluation metrics."""

import numpy as np
import pytest
from stegaqr.utils.metrics import (
    bit_accuracy,
    bit_error_rate,
    full_decode_rate,
    psnr,
    ssim,
    wilson_score_ci,
)


class TestBitMetrics:
    def test_perfect_accuracy(self):
        bits = np.array([0, 1, 1, 0, 1])
        assert bit_accuracy(bits, bits) == 1.0
        assert bit_error_rate(bits, bits) == 0.0

    def test_zero_accuracy(self):
        pred = np.array([0, 0, 0, 0, 0])
        gt = np.array([1, 1, 1, 1, 1])
        assert bit_accuracy(pred, gt) == 0.0

    def test_partial_accuracy(self):
        pred = np.array([0, 1, 1, 0])
        gt = np.array([0, 1, 0, 0])
        assert bit_accuracy(pred, gt) == 0.75

    def test_full_decode_rate_perfect(self):
        pred = np.array([[0, 1], [1, 0]])
        gt = np.array([[0, 1], [1, 0]])
        assert full_decode_rate(pred, gt) == 1.0

    def test_full_decode_rate_partial(self):
        pred = np.array([[0, 1], [1, 1]])
        gt = np.array([[0, 1], [1, 0]])
        assert full_decode_rate(pred, gt) == 0.5


class TestImageMetrics:
    def test_psnr_identical(self):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        assert psnr(img, img) == float("inf")

    def test_psnr_noisy(self):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        noisy = img + np.random.randn(*img.shape).astype(np.float32) * 0.01
        noisy = np.clip(noisy, 0, 1)
        p = psnr(img, noisy)
        assert 30 < p < 60  # reasonable range for small noise

    def test_ssim_identical(self):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_ssim_range(self):
        img1 = np.random.rand(32, 32).astype(np.float32)
        img2 = np.random.rand(32, 32).astype(np.float32)
        s = ssim(img1, img2)
        assert -1.0 <= s <= 1.0


class TestWilsonCI:
    def test_perfect_success(self):
        lo, hi = wilson_score_ci(100, 100)
        assert lo > 0.95
        assert hi == 1.0

    def test_zero_success(self):
        lo, hi = wilson_score_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_empty_trials(self):
        lo, hi = wilson_score_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0
