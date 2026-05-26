"""Tests for QR utility functions."""

import numpy as np
import pytest
from stegaqr.utils.qr_utils import (
    generate_qr,
    generate_cover_qr_rgb,
    get_qr_structure_mask,
    verify_qr_decodable,
)


class TestQRGeneration:
    """Tests for QR code generation."""

    @pytest.mark.parametrize("version", [1, 2, 4, 6, 10])
    def test_qr_dimensions(self, version: int):
        """QR matrix dimensions must be (4v+17) x (4v+17)."""
        matrix, actual_v = generate_qr("test", version=version)
        expected_d = 4 * version + 17
        assert matrix.shape == (expected_d, expected_d)
        assert actual_v == version

    def test_qr_binary_values(self):
        """QR matrix values must be 0 or 255."""
        matrix, _ = generate_qr("hello", version=2)
        unique = np.unique(matrix)
        assert set(unique).issubset({0, 255})

    def test_rgb_output_shape(self):
        """RGB cover should have 3 channels in [0, 1]."""
        rgb, _ = generate_cover_qr_rgb("hello", version=2)
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3
        assert rgb.dtype == np.float32
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    @pytest.mark.parametrize("ec", ["L", "M", "Q", "H"])
    def test_ec_levels(self, ec: str):
        """All error correction levels should produce valid QR codes."""
        matrix, _ = generate_qr("test payload", version=4, ec_level=ec)
        assert matrix.shape[0] > 0


class TestStructureMask:
    """Tests for QR structural mask generation."""

    @pytest.mark.parametrize("version", [1, 2, 4, 7, 10])
    def test_mask_dimensions(self, version: int):
        """Mask dimensions must match QR dimensions."""
        mask = get_qr_structure_mask(version)
        expected_d = 4 * version + 17
        assert mask.shape == (expected_d, expected_d)

    def test_mask_binary(self):
        """Mask values must be 0.0 or 1.0."""
        mask = get_qr_structure_mask(4)
        unique = np.unique(mask)
        assert set(unique).issubset({0.0, 1.0})

    def test_finder_patterns_protected(self):
        """Finder patterns (corners) must be marked as protected (0)."""
        mask = get_qr_structure_mask(4)
        # Top-left finder: rows 0-6, cols 0-6
        assert mask[0:7, 0:7].sum() == 0.0
        # Top-right finder
        d = mask.shape[0]
        assert mask[0:7, d - 7 : d].sum() == 0.0
        # Bottom-left finder
        assert mask[d - 7 : d, 0:7].sum() == 0.0

    def test_data_modules_exist(self):
        """There must be data modules (mask=1) available for embedding."""
        mask = get_qr_structure_mask(4)
        assert mask.sum() > 0, "No data modules available"

    def test_protected_count_increases_with_version(self):
        """Higher versions have more total modules but similar protected ratio."""
        mask_v2 = get_qr_structure_mask(2)
        mask_v10 = get_qr_structure_mask(10)
        # Both should have meaningful data regions
        assert mask_v2.sum() > 50
        assert mask_v10.sum() > mask_v2.sum()


class TestQRDecodability:
    """Tests for QR decode verification."""

    def test_standard_qr_decodable(self):
        """A standard QR image should be decodable by pyzbar."""
        from PIL import Image

        matrix, _ = generate_qr("hello world", version=4)
        # Scale up for reliable decoding
        matrix_scaled = np.kron(matrix, np.ones((10, 10), dtype=np.uint8))
        # Add white border
        border = 40
        h, w = matrix_scaled.shape
        padded = np.full((h + 2 * border, w + 2 * border), 255, dtype=np.uint8)
        padded[border : border + h, border : border + w] = matrix_scaled

        result = verify_qr_decodable(padded, "hello world")
        assert result is True

    def test_wrong_payload_fails(self):
        """Verification should fail if expected payload doesn't match."""
        matrix, _ = generate_qr("hello world", version=4)
        matrix_scaled = np.kron(matrix, np.ones((10, 10), dtype=np.uint8))
        border = 40
        h, w = matrix_scaled.shape
        padded = np.full((h + 2 * border, w + 2 * border), 255, dtype=np.uint8)
        padded[border : border + h, border : border + w] = matrix_scaled

        result = verify_qr_decodable(padded, "wrong payload")
        assert result is False
