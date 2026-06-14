"""Tests for the real (non-differentiable) eval-time distortions."""

import numpy as np
import pytest

from stegaqr.realistic_distortion import (
    real_jpeg, gaussian_blur, resize_roundtrip, brightness, gaussian_noise,
    make_presets,
)
from stegaqr.utils.qr_utils import generate_cover_qr_rgb, verify_qr_decodable


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.random((64, 64, 3)).astype(np.float32)


class TestOps:
    def test_shapes_and_range(self, img):
        for fn in (lambda x: real_jpeg(x, 70), lambda x: gaussian_blur(x, 1.0),
                   lambda x: resize_roundtrip(x, 0.5), lambda x: brightness(x, 1.2),
                   lambda x: gaussian_noise(x, 0.02)):
            out = fn(img)
            assert out.shape == img.shape
            assert out.min() >= 0.0 and out.max() <= 1.0

    def test_jpeg_actually_changes_image(self, img):
        out = real_jpeg(img, 30)
        assert not np.allclose(out, img)

    def test_jpeg_quality_monotonic(self, img):
        """Lower quality should deviate more from the original."""
        d30 = np.mean((real_jpeg(img, 30) - img) ** 2)
        d90 = np.mean((real_jpeg(img, 90) - img) ** 2)
        assert d30 > d90

    def test_all_presets_run(self, img):
        for name, fn in make_presets().items():
            out = fn(img)
            assert out.shape == img.shape


class TestQRSurvivesRealJpeg:
    def test_clean_qr_decodes_after_jpeg(self):
        """A real QR (no hidden data) must still scan after moderate JPEG."""
        rgb, _ = generate_cover_qr_rgb("HELLO123", version=4, ec_level="M",
                                       module_size=8, quiet_zone=4)
        jp = real_jpeg(rgb, 70)
        assert verify_qr_decodable(jp, "HELLO123")
