"""QR code generation and structural analysis utilities.

Provides functions to:
  - Generate standard monochrome QR codes as cover images
  - Extract QR structural masks (finder, timing, alignment, format, data)
  - Verify public payload decodability
  - Scale QR codes to multi-pixel-per-module resolution
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
    quiet_zone: int = 0,
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
    quiet_zone : int
        Width, in modules, of the white quiet zone (border) added on all sides.
        The ISO/IEC 18004 standard mandates >=4 for reliable scanning. Default 0
        keeps the bare symbol; pass >=4 to produce a scannable cover.

    Returns
    -------
    (matrix, actual_version)
        matrix : np.ndarray, shape (D*module_size, D*module_size), dtype uint8,
                 values {0, 255}, where D = 4*version + 17 + 2*quiet_zone.
                 Dark modules are 0 (black), light modules are 255 (white).
        actual_version : int
    """
    qr = qrcode.QRCode(
        version=version,
        error_correction=EC_LEVELS[ec_level.upper()],
        box_size=module_size,
        border=0,
    )
    qr.add_data(payload)
    qr.make(fit=False)

    # qrcode.get_matrix() returns True for DARK modules. A scannable QR renders
    # dark modules black (0) and light modules white (255), so invert the boolean.
    modules = np.array(qr.get_matrix(), dtype=np.uint8)
    matrix = np.where(modules, 0, 255).astype(np.uint8)

    if quiet_zone > 0:
        matrix = np.pad(matrix, quiet_zone, mode="constant", constant_values=255)

    if module_size > 1:
        matrix = np.kron(matrix, np.ones((module_size, module_size), dtype=np.uint8))

    return matrix, qr.version


def generate_cover_qr_rgb(
    payload: str,
    version: int = 4,
    ec_level: str = "M",
    module_size: int = 1,
    quiet_zone: int = 0,
) -> tuple[np.ndarray, int]:
    """Generate a standard QR code as an RGB numpy array.

    Returns
    -------
    (image, actual_version)
        image: np.ndarray, shape (D*module_size, D*module_size, 3), dtype float32,
               values in [0, 1], where D = 4*version + 17 + 2*quiet_zone.
    """
    matrix, actual_version = generate_qr(payload, version, ec_level, module_size, quiet_zone)
    rgb = np.stack([matrix, matrix, matrix], axis=-1).astype(np.float32) / 255.0
    return rgb, actual_version


def get_qr_structure_mask(version: int, module_size: int = 1) -> np.ndarray:
    """Generate a binary mask of QR structural vs. data modules.

    Returns a mask where:
      0 = protected structural module (finder, timing, alignment, format, version)
      1 = data/error-correction module (safe to perturb)

    Parameters
    ----------
    version : int
        QR version (1-40).
    module_size : int
        Pixels per module. Mask is scaled accordingly.

    Returns
    -------
    np.ndarray, shape (d*module_size, d*module_size), dtype float32, values {0.0, 1.0}
    """
    d = 4 * version + 17
    mask = np.ones((d, d), dtype=np.float32)

    # Finder patterns (3 corners): 7x7 each + 1-module separator
    for (r, c) in [(0, 0), (0, d - 7), (d - 7, 0)]:
        mask[r : r + 7, c : c + 7] = 0.0

    # Separators around finder patterns
    if d > 7:
        mask[7, 0:8] = 0.0
        mask[0:8, 7] = 0.0
        mask[7, d - 8 : d] = 0.0
        mask[0:8, d - 8] = 0.0
        mask[d - 8, 0:8] = 0.0
        mask[d - 8 : d, 7] = 0.0

    # Timing patterns (row 6 and column 6)
    mask[6, :] = 0.0
    mask[:, 6] = 0.0

    # Format information
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
                if (r < 9 and c < 9):
                    continue
                if (r < 9 and c > d - 9):
                    continue
                if (r > d - 9 and c < 9):
                    continue
                mask[r - 2 : r + 3, c - 2 : c + 3] = 0.0

    # Version information (version >= 7)
    if version >= 7:
        mask[0:6, d - 11 : d - 8] = 0.0
        mask[d - 11 : d - 8, 0:6] = 0.0

    # Scale up if module_size > 1
    if module_size > 1:
        mask = np.kron(mask, np.ones((module_size, module_size), dtype=np.float32))

    return mask


def _alignment_pattern_positions(version: int) -> list[int]:
    """Return alignment pattern center positions for a given QR version."""
    if version == 1:
        return []

    d = 4 * version + 17
    num = version // 7 + 2

    if num == 2:
        positions = [6, d - 7]
    else:
        first = 6
        last = d - 7
        step = (last - first) // (num - 1)
        if step % 2 != 0:
            step += 1
        positions = [first]
        for i in range(1, num - 1):
            positions.append(last - (num - 1 - i) * step)
        positions.append(last)

    return positions


def verify_qr_decodable(image: np.ndarray | Image.Image, expected: str) -> bool:
    """Check if a QR image can be decoded by a standard reader."""
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
