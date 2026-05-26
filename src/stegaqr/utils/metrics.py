"""Evaluation metrics for steganographic QR codes.

Metrics cover three aspects:
  1. Hidden payload recovery (bit accuracy, full decode rate)
  2. Visual quality (PSNR, SSIM, LPIPS)
  3. Public QR decodability (standard reader success rate)
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def bit_accuracy(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Per-bit accuracy between predicted and ground truth payloads.

    Parameters
    ----------
    predicted : np.ndarray, shape (N,) or (B, N)
        Predicted bits (0 or 1).
    ground_truth : np.ndarray, shape (N,) or (B, N)
        Ground truth bits.

    Returns
    -------
    float
        Fraction of correctly predicted bits.
    """
    return float(np.mean(predicted == ground_truth))


def bit_error_rate(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Bit error rate (BER) = 1 - bit_accuracy."""
    return 1.0 - bit_accuracy(predicted, ground_truth)


def full_decode_rate(
    predicted_batch: np.ndarray, ground_truth_batch: np.ndarray
) -> float:
    """Fraction of samples where ALL bits are correct.

    Parameters
    ----------
    predicted_batch : np.ndarray, shape (B, N)
    ground_truth_batch : np.ndarray, shape (B, N)

    Returns
    -------
    float
        Fraction of samples with zero bit errors.
    """
    per_sample = np.all(predicted_batch == ground_truth_batch, axis=1)
    return float(np.mean(per_sample))


def psnr(original: np.ndarray, modified: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    original : np.ndarray
        Original image, shape (H, W, 3) or (H, W), values in [0, max_val].
    modified : np.ndarray
        Modified image, same shape.
    max_val : float
        Maximum pixel value.

    Returns
    -------
    float
        PSNR in dB. Higher is better (less distortion).
    """
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(max_val**2 / mse))


def ssim(
    original: np.ndarray,
    modified: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """Structural Similarity Index (simplified, single-scale).

    For publication, use torchmetrics.StructuralSimilarityIndexMeasure
    or skimage.metrics.structural_similarity instead.

    Parameters
    ----------
    original, modified : np.ndarray
        Images, shape (H, W) or (H, W, 3).
    max_val : float
        Maximum pixel value.

    Returns
    -------
    float
        SSIM in [0, 1]. Higher is better.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu_x = np.mean(original)
    mu_y = np.mean(modified)
    sigma_x2 = np.var(original)
    sigma_y2 = np.var(modified)
    sigma_xy = np.mean((original - mu_x) * (modified - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return float(numerator / denominator)


def qr_public_decode_rate(
    images: list[Image.Image], expected_payloads: list[str]
) -> float:
    """Check that standard QR readers can still decode the public payload.

    Uses pyzbar as the standard reader.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Stego QR code images.
    expected_payloads : list of str
        Expected public payloads.

    Returns
    -------
    float
        Fraction of images where pyzbar successfully decodes the correct payload.
    """
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
    except ImportError:
        raise ImportError("pyzbar required for QR decode rate metric")

    correct = 0
    for img, expected in zip(images, expected_payloads):
        results = pyzbar_decode(img)
        if results:
            decoded_text = results[0].data.decode("utf-8", errors="replace")
            if decoded_text == expected:
                correct += 1
    return correct / len(images) if images else 0.0


def wilson_score_ci(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Better coverage than normal approximation near 0% and 100%.

    Parameters
    ----------
    successes : int
    trials : int
    confidence : float

    Returns
    -------
    (lower, upper) bounds of the confidence interval.
    """
    from scipy.stats import norm

    if trials == 0:
        return (0.0, 0.0)

    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials
    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom

    lower = float(max(0.0, center - spread))
    upper = float(min(1.0, center + spread))
    # Clamp near-zero floating point artifacts
    if lower < 1e-10:
        lower = 0.0
    if upper > 1.0 - 1e-10:
        upper = 1.0
    return (lower, upper)
