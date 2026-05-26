"""Core API for StegaQR encoding and decoding.

This module defines the public interface. The three steganographic modes
(segregated, cross-channel, hybrid) are switchable via the StegoMode enum.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image


class StegoMode(Enum):
    """Steganographic embedding mode.

    SEGREGATED:    Independent per-channel residuals. Each channel carries
                   its own hidden bitstream. Fault-isolated.
    CROSS_CHANNEL: Hidden data encoded in inter-channel relationships
                   (ratios, differences, learned latent space). Higher capacity.
    HYBRID:        QR-anchored steganography. Standard QR remains decodable;
                   hidden data lives in color perturbations within decodable
                   bounds. Finder patterns serve as calibration anchors.
    """

    SEGREGATED = "segregated"
    CROSS_CHANNEL = "cross_channel"
    HYBRID = "hybrid"


class StegaQREncoder:
    """Encodes hidden data into a QR code image.

    Supports both neural and classical embedding strategies, switchable
    via encoder_type. The neural encoder learns the embedding end-to-end;
    the classical encoder uses DCT/spatial manipulation with hand-crafted
    rules.

    Parameters
    ----------
    mode : StegoMode
        Which channel mode to use.
    capacity_bits : int
        Target hidden payload size in bits.
    encoder_type : str
        'neural' or 'classical'.
    perturbation_bound : float
        Maximum per-channel perturbation in [0, 1] scale.
    device : str
        PyTorch device string ('cuda', 'cpu').
    """

    def __init__(
        self,
        mode: StegoMode = StegoMode.HYBRID,
        capacity_bits: int = 100,
        encoder_type: str = "neural",
        perturbation_bound: float = 0.1,
        device: str = "cuda",
    ) -> None:
        self.mode = mode
        self.capacity_bits = capacity_bits
        self.encoder_type = encoder_type
        self.perturbation_bound = perturbation_bound
        self.device = device
        self._model = None  # Lazy-loaded

    def encode(
        self,
        public_payload: str,
        hidden_payload: bytes,
        *,
        qr_version: int = 4,
        ec_level: str = "M",
    ) -> Image.Image:
        """Encode public + hidden payloads into a single QR image.

        Parameters
        ----------
        public_payload : str
            The visible payload (decoded by any standard QR reader).
        hidden_payload : bytes
            The hidden bitstream to embed steganographically.
        qr_version : int
            QR version (1-40).
        ec_level : str
            Error correction level ('L', 'M', 'Q', 'H').

        Returns
        -------
        PIL.Image.Image
            The stego QR code image (RGB).
        """
        raise NotImplementedError("Encoder implementation pending.")

    def load_model(self, path: str) -> None:
        """Load pretrained encoder weights."""
        raise NotImplementedError

    @property
    def model(self):
        """Lazy-load the neural encoder model."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self._model


class StegaQRDecoder:
    """Decodes hidden data from a stego QR code image.

    Parameters
    ----------
    mode : StegoMode
        Must match the mode used for encoding.
    capacity_bits : int
        Expected hidden payload size in bits.
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        mode: StegoMode = StegoMode.HYBRID,
        capacity_bits: int = 100,
        device: str = "cuda",
    ) -> None:
        self.mode = mode
        self.capacity_bits = capacity_bits
        self.device = device
        self._model = None

    def decode(
        self, image: Image.Image
    ) -> tuple[Optional[str], Optional[bytes], dict]:
        """Decode both public and hidden payloads from a stego QR image.

        Parameters
        ----------
        image : PIL.Image.Image
            The (possibly distorted) stego QR code image.

        Returns
        -------
        tuple of (public_payload, hidden_payload, metadata)
            public_payload : str or None
                The visible QR payload (via standard QR reader).
            hidden_payload : bytes or None
                The recovered hidden bitstream.
            metadata : dict
                Decode confidence, bit error estimates, etc.
        """
        raise NotImplementedError("Decoder implementation pending.")

    def load_model(self, path: str) -> None:
        """Load pretrained decoder weights."""
        raise NotImplementedError


def encode_hidden(
    public_payload: str,
    hidden_payload: bytes,
    *,
    mode: str = "hybrid",
    capacity_bits: int = 100,
    encoder_type: str = "neural",
    qr_version: int = 4,
    ec_level: str = "M",
) -> Image.Image:
    """Convenience function for one-shot encoding.

    Parameters
    ----------
    public_payload : str
        Visible QR payload.
    hidden_payload : bytes
        Hidden steganographic payload.
    mode : str
        'segregated', 'cross_channel', or 'hybrid'.
    capacity_bits : int
        Target capacity.
    encoder_type : str
        'neural' or 'classical'.
    qr_version : int
        QR version.
    ec_level : str
        Error correction level.

    Returns
    -------
    PIL.Image.Image
        Stego QR code.
    """
    encoder = StegaQREncoder(
        mode=StegoMode(mode),
        capacity_bits=capacity_bits,
        encoder_type=encoder_type,
    )
    return encoder.encode(
        public_payload, hidden_payload, qr_version=qr_version, ec_level=ec_level
    )


def decode_hidden(
    image: Image.Image,
    *,
    mode: str = "hybrid",
    capacity_bits: int = 100,
) -> tuple[Optional[str], Optional[bytes], dict]:
    """Convenience function for one-shot decoding.

    Parameters
    ----------
    image : PIL.Image.Image
        Stego QR code image.
    mode : str
        Must match encoding mode.
    capacity_bits : int
        Expected capacity.

    Returns
    -------
    tuple of (public_payload, hidden_payload, metadata)
    """
    decoder = StegaQRDecoder(
        mode=StegoMode(mode),
        capacity_bits=capacity_bits,
    )
    return decoder.decode(image)
