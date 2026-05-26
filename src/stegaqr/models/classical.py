"""Classical (non-neural) steganographic encoder.

Embeds hidden bits using hand-crafted spatial/color manipulation:
  - LSB embedding in color channels
  - DCT coefficient modification
  - Color quantization residual encoding

Used as a baseline comparison against the neural encoder,
and as the "classical encode + neural decode" pipeline variant.
"""

from __future__ import annotations

import numpy as np


class ClassicalSegregatedEncoder:
    """LSB-based per-channel steganographic encoder.

    Embeds bits in the least significant bit(s) of each color channel,
    constrained to data modules only (structural modules untouched).
    """

    def __init__(self, capacity_bits: int = 100, num_lsb: int = 1):
        self.capacity_bits = capacity_bits
        self.num_lsb = num_lsb

    def encode(
        self,
        cover: np.ndarray,
        payload: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Embed payload bits into cover image.

        Parameters
        ----------
        cover : np.ndarray, shape (H, W, 3), dtype uint8
            Cover QR image.
        payload : np.ndarray, shape (capacity_bits,), dtype uint8
            Hidden bits (0 or 1).
        mask : np.ndarray, shape (H, W), dtype float32, optional
            1 = embeddable, 0 = protected.

        Returns
        -------
        np.ndarray, shape (H, W, 3), dtype uint8
            Stego image.
        """
        stego = cover.copy()
        H, W, C = stego.shape

        if mask is None:
            mask = np.ones((H, W), dtype=np.float32)

        # Collect embeddable positions
        positions = np.argwhere(mask > 0.5)
        bits_per_channel = self.capacity_bits // 3

        for ch in range(3):
            start = ch * bits_per_channel
            end = start + bits_per_channel
            ch_bits = payload[start:end]

            for i, bit in enumerate(ch_bits):
                if i >= len(positions):
                    break
                r, c = positions[i]
                val = int(stego[r, c, ch])
                # Clear LSB, set to payload bit
                val = (val & ~1) | int(bit)
                stego[r, c, ch] = np.uint8(val)

        return stego

    def decode(
        self,
        stego: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract hidden bits from stego image.

        Parameters
        ----------
        stego : np.ndarray, shape (H, W, 3), dtype uint8
        mask : np.ndarray, shape (H, W), optional

        Returns
        -------
        np.ndarray, shape (capacity_bits,), dtype uint8
        """
        H, W, C = stego.shape

        if mask is None:
            mask = np.ones((H, W), dtype=np.float32)

        positions = np.argwhere(mask > 0.5)
        bits_per_channel = self.capacity_bits // 3
        payload = np.zeros(self.capacity_bits, dtype=np.uint8)

        for ch in range(3):
            start = ch * bits_per_channel
            end = start + bits_per_channel

            for i in range(bits_per_channel):
                if i >= len(positions):
                    break
                r, c = positions[i]
                payload[start + i] = stego[r, c, ch] & 1

        return payload
