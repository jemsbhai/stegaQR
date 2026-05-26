"""QR code generation and structural analysis utilities.

Provides functions to:
  - Generate standard monochrome QR codes as cover images
  - Extract QR structural masks (finder, timing, alignment, format, data)
  - Verify public payload decodability
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import qrcode
from PIL import Image


# QR error correction level mapping
EC_LEVELS = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}


def generate_qr(
    payload: str,
    version: int = 4,
    ec_level: str = "M",
    module_size: int = 1,
) -> tuple[np.ndarray, int]:
    """Generate a standard QR code as a binary numpy array.

    Parameters
    ----------
    payload : str
        Data to encode.
    version : int
        QR version (1-40).
    ec_level : str
        Error correction level: 'L', 'M', 'Q', 'H'.
    module_size : int
        Pixels per module (for output image).

    Returns
    -------
    (matrix, actual_version)
        matrix : np.ndarray, shape (d, d), dtype uint8, values {0, 255}
        actual_version : int
    """
    qr = qrcode.QRCode(
        version=version,
        error_correction=EC_LEVELS[ec_level.upper()],
        box_size=module_size,
        border=0,  # No border — we control padding ourselves
    )
    qr.add_data(payload)
    qr.make(fit=False)

    # Extract the module matrix (list of lists of bools)
    modules = qr.get_matrix()
    matrix = np.array(modules, dtype=np.uint8) * 255
    return matrix, qr.version


def generate_cover_qr_rgb(
    payload: str,
    version: int = 4,
    ec_level: str = "M",
) -> tuple[np.ndarray, int]:
    """Generate a standard QR code as an RGB numpy array.

    The QR is rendered as black modules on white background,
    as a standard phone camera would see it.

    Returns
    -------
    (image, actual_version)
        image: np.ndarray, shape (d, d, 3), dtype float32, values in [0, 1]
    """
    matrix, actual_version = generate_qr(payload, version, ec_level)
    # Convert to RGB float: 0 = black module, 1 = white module
    rgb = np.stack([matrix, matrix, matrix], axis=-1).astype(np.float32) / 255.0
    return rgb, actual_version


def get_qr_structure_mask(version: int) -> np.ndarray:
    """Generate a binary mask of QR structural vs. data modules.

    Returns a mask where:
      0 = protected structural module (finder, timing, alignment, format, version)
      1 = data/error-correction module (safe to perturb)

    Parameters
    ----------
    version : int
        QR version (1-40).

    Returns
    -------
    np.ndarray, shape (d, d), dtype float32, values {0.0, 1.0}
    """
    d = 4 * version + 17
    mask = np.ones((d, d), dtype=np.float32)

    # Finder patterns (3 corners): 7x7 each + 1-module separator
    for (r, c) in [(0, 0), (0, d - 7), (d - 7, 0)]:
        # Finder pattern itself
        mask[r : r + 7, c : c + 7] = 0.0

    # Separators around finder patterns
    # Top-left
    if d > 7:
        mask[7, 0:8] = 0.0
        mask[0:8, 7] = 0.0
    # Top-right
    if d > 7:
        mask[7, d - 8 : d] = 0.0
        mask[0:8, d - 8] = 0.0
    # Bottom-left
    if d > 7:
        mask[d - 8, 0:8] = 0.0
        mask[d - 8 : d, 7] = 0.0

    # Timing patterns (row 6 and column 6)
    mask[6, :] = 0.0
    mask[:, 6] = 0.0

    # Format information (around finder patterns)
    mask[8, 0:9] = 0.0
    mask[0:9, 8] = 0.0
    mask[8, d - 8 : d] = 0.0
    mask[d - 7 : d, 8] = 0.0

    # Dark module
    mask[d - 8, 8] = 0.0

    # Alignment patterns (version >= 2)
    if version >= 2:
        positions = _alignment_pattern_positions(version)
        for r in positions:
            for c in positions:
                # Skip if overlapping with finder patterns
                if (r < 9 and c < 9):
                    continue
                if (r < 9 and c > d - 9):
                    continue
                if (r > d - 9 and c < 9):
                    continue
                # 5x5 alignment pattern
                mask[r - 2 : r + 3, c - 2 : c + 3] = 0.0

    # Version information (version >= 7)
    if version >= 7:
        mask[0:6, d - 11 : d - 8] = 0.0
        mask[d - 11 : d - 8, 0:6] = 0.0

    return mask


def _alignment_pattern_positions(version: int) -> list[int]:
    """Return alignment pattern center positions for a given QR version.

    Based on ISO/IEC 18004:2015 Table E.1.
    """
    if version == 1:
        return []

    d = 4 * version + 17
    # Number of alignment patterns per axis
    num = version // 7 + 2

    if num == 2:
        positions = [6, d - 7]
    else:
        # Calculate evenly spaced positions
        first = 6
        last = d - 7
        step = (last - first) // (num - 1)
        # Round step to nearest even number
        if step % 2 != 0:
            step += 1
        positions = [first]
        for i in range(1, num - 1):
            positions.append(last - (num - 1 - i) * step)
        positions.append(last)

    return positions


def verify_qr_decodable(image: np.ndarray | Image.Image, expected: str) -> bool:
    """Check if a QR image can be decoded by a standard reader.

    Parameters
    ----------
    image : np.ndarray or PIL.Image.Image
        QR code image.
    expected : str
        Expected decoded payload.

    Returns
    -------
    bool
        True if pyzbar successfully decodes the expected payload.
    """
    from pyzbar.pyzbar import decode as pyzbar_decode

    if isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)

    results = pyzbar_decode(image)
    if not results:
        return False
    decoded = results[0].data.decode("utf-8", errors="replace")
    return decoded == expected
