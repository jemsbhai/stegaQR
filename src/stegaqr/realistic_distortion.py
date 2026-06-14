"""Realistic, NON-differentiable distortions for *evaluation* (not training).

The training-time DifferentiableDistortion approximates JPEG with a calibrated
noise model (it must be differentiable). For credible robustness numbers we should
evaluate against the *real* operators. This module applies true JPEG (via PIL DCT
codec), real Gaussian blur, resize round-trips, brightness/contrast and sensor
noise -- the kind of degradation a printed/displayed-then-photographed QR sees.

All functions operate on float images in [0,1], shape (H, W, 3), and return the
same. Used at eval time only (no gradients needed).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def _to_np(im: Image.Image) -> np.ndarray:
    return np.asarray(im).astype(np.float32) / 255.0


def real_jpeg(img: np.ndarray, quality: int = 70) -> np.ndarray:
    """True JPEG compression via the PIL DCT codec (round-trips through bytes)."""
    buf = io.BytesIO()
    _to_pil(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return _to_np(Image.open(buf).convert("RGB"))


def gaussian_blur(img: np.ndarray, radius: float = 1.0) -> np.ndarray:
    if radius <= 0:
        return img
    return _to_np(_to_pil(img).filter(ImageFilter.GaussianBlur(radius)))


def resize_roundtrip(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Downscale then upscale -- models capture/transmission resampling."""
    im = _to_pil(img)
    w, h = im.size
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return _to_np(small.resize((w, h), Image.BILINEAR))


def brightness(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    return np.clip(img * factor, 0, 1)


def gaussian_noise(img: np.ndarray, sigma: float = 0.02, rng=None) -> np.ndarray:
    rng = rng or np.random
    return np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 1)


# Named distortion presets: each maps a float image -> float image.
def make_presets(rng=None):
    rng = rng or np.random.default_rng(0)
    return {
        "jpeg90": lambda x: real_jpeg(x, 90),
        "jpeg70": lambda x: real_jpeg(x, 70),
        "jpeg50": lambda x: real_jpeg(x, 50),
        "jpeg30": lambda x: real_jpeg(x, 30),
        "blur1": lambda x: gaussian_blur(x, 1.0),
        "blur2": lambda x: gaussian_blur(x, 2.0),
        "resize50": lambda x: resize_roundtrip(x, 0.5),
        "bright80": lambda x: brightness(x, 0.8),
        "bright120": lambda x: brightness(x, 1.2),
        "noise": lambda x: gaussian_noise(x, 0.03, rng),
        # a realistic combined "photographed print" pipeline
        "combined": lambda x: real_jpeg(
            gaussian_noise(brightness(gaussian_blur(resize_roundtrip(x, 0.6), 0.8), 0.9), 0.015, rng),
            quality=60),
    }


def apply_preset(img: np.ndarray, name: str, rng=None) -> np.ndarray:
    return make_presets(rng)[name](img)
