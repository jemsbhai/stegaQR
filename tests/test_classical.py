"""Tests for classical steganographic encoder."""

import numpy as np
import pytest
from stegaqr.models.classical import ClassicalSegregatedEncoder


class TestClassicalEncoder:
    def test_roundtrip_clean(self):
        """Classical encode->decode should recover exact bits on clean images."""
        encoder = ClassicalSegregatedEncoder(capacity_bits=99, num_lsb=1)
        cover = np.random.randint(0, 256, (33, 33, 3), dtype=np.uint8)
        payload = np.random.randint(0, 2, 99, dtype=np.uint8)

        stego = encoder.encode(cover, payload)
        recovered = encoder.decode(stego)

        np.testing.assert_array_equal(recovered, payload)

    def test_mask_respected(self):
        """Protected regions should not be modified."""
        encoder = ClassicalSegregatedEncoder(capacity_bits=99, num_lsb=1)
        cover = np.full((33, 33, 3), 128, dtype=np.uint8)
        payload = np.ones(99, dtype=np.uint8)
        mask = np.zeros((33, 33), dtype=np.float32)
        # Only allow embedding in bottom-right quadrant
        mask[20:, 20:] = 1.0

        stego = encoder.encode(cover, payload, mask)

        # Protected region should be unchanged
        np.testing.assert_array_equal(stego[:20, :20], cover[:20, :20])

    def test_stego_visual_similarity(self):
        """Stego image should differ from cover by at most 1 per pixel (LSB)."""
        encoder = ClassicalSegregatedEncoder(capacity_bits=99, num_lsb=1)
        cover = np.random.randint(0, 256, (33, 33, 3), dtype=np.uint8)
        payload = np.random.randint(0, 2, 99, dtype=np.uint8)

        stego = encoder.encode(cover, payload)
        diff = np.abs(stego.astype(int) - cover.astype(int))
        assert diff.max() <= 1
