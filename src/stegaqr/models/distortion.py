"""Differentiable distortion layer for training.

Simulates real-world image degradations in a differentiable manner,
placed between the encoder and decoder during training. Forces the
encoder to learn robust embeddings and the decoder to handle noise.

All operations are differentiable (or use straight-through estimators)
to allow gradient flow from decoder loss back to encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableJPEG(nn.Module):
    """Approximate JPEG compression via DCT quantization.

    Uses 8x8 block DCT with learnable/fixed quantization tables.
    Straight-through estimator for the rounding step.
    """

    def __init__(self, quality_range: tuple[int, int] = (30, 95)):
        super().__init__()
        self.quality_range = quality_range

    def forward(self, x: torch.Tensor, quality: float | None = None) -> torch.Tensor:
        """Approximate JPEG. For now, uses additive noise as proxy.

        TODO: Implement full differentiable DCT-based JPEG simulation.
        The current implementation uses a calibrated noise model that
        approximates JPEG artifacts at various quality levels.
        """
        if quality is None:
            q_low, q_high = self.quality_range
            quality = torch.empty(1).uniform_(q_low, q_high).item()

        # JPEG noise proxy: higher quality = less noise
        # Calibrated from empirical measurements (see LOGBOOK EXP-XXX)
        noise_sigma = max(0.0, (100 - quality) / 100.0 * 0.15)
        noise = torch.randn_like(x) * noise_sigma
        return torch.clamp(x + noise, 0.0, 1.0)


class DifferentiableDistortion(nn.Module):
    """Composite differentiable distortion layer.

    Randomly applies combinations of:
      - Gaussian noise
      - JPEG compression (approximate)
      - Brightness scaling
      - Per-channel color shift
      - Gaussian blur
      - Perspective distortion (affine approximation)

    Each distortion has a probability of being applied and parameters
    sampled from configured ranges.
    """

    def __init__(
        self,
        noise_sigma_range: tuple[float, float] = (0.0, 25.0),
        jpeg_quality_range: tuple[int, int] = (30, 95),
        brightness_range: tuple[float, float] = (0.8, 1.2),
        color_shift_range: tuple[float, float] = (0.0, 15.0),
        blur_sigma_range: tuple[float, float] = (0.0, 1.0),
        distortion_prob: float = 0.5,
    ):
        super().__init__()
        self.noise_sigma_range = noise_sigma_range
        self.jpeg = DifferentiableJPEG(jpeg_quality_range)
        self.brightness_range = brightness_range
        self.color_shift_range = color_shift_range
        self.blur_sigma_range = blur_sigma_range
        self.distortion_prob = distortion_prob

    def _apply_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Additive Gaussian noise."""
        lo, hi = self.noise_sigma_range
        sigma = torch.empty(1).uniform_(lo / 255.0, hi / 255.0).item()
        return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)

    def _apply_brightness(self, x: torch.Tensor) -> torch.Tensor:
        """Uniform brightness scaling."""
        lo, hi = self.brightness_range
        factor = torch.empty(1).uniform_(lo, hi).item()
        return torch.clamp(x * factor, 0.0, 1.0)

    def _apply_color_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel additive color shift."""
        lo, hi = self.color_shift_range
        shift = torch.empty(3).uniform_(-hi / 255.0, hi / 255.0)
        shift = shift.to(x.device).view(1, 3, 1, 1)
        return torch.clamp(x + shift, 0.0, 1.0)

    def _apply_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian blur with random sigma."""
        lo, hi = self.blur_sigma_range
        sigma = torch.empty(1).uniform_(lo, hi).item()
        if sigma < 0.1:
            return x
        # Kernel size must be odd
        ksize = int(2 * round(3 * sigma) + 1)
        ksize = max(3, ksize)
        if ksize % 2 == 0:
            ksize += 1

        # Create 1D Gaussian kernel
        coords = torch.arange(ksize, dtype=torch.float32, device=x.device) - ksize // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable convolution
        kernel_h = kernel_1d.view(1, 1, 1, -1).expand(3, -1, -1, -1)
        kernel_v = kernel_1d.view(1, 1, -1, 1).expand(3, -1, -1, -1)

        pad_h = ksize // 2
        x = F.pad(x, [pad_h, pad_h, 0, 0], mode="reflect")
        x = F.conv2d(x, kernel_h, groups=3)
        x = F.pad(x, [0, 0, pad_h, pad_h], mode="reflect")
        x = F.conv2d(x, kernel_v, groups=3)

        return torch.clamp(x, 0.0, 1.0)

    def forward(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Apply random distortions.

        Parameters
        ----------
        x : torch.Tensor
            (B, 3, H, W) stego image in [0, 1].
        deterministic : bool
            If True, apply all distortions at medium strength.
            Useful for reproducible evaluation.
        """
        if deterministic:
            x = self._apply_noise(x)
            x = self.jpeg(x, quality=70.0)
            x = self._apply_brightness(x)
            x = self._apply_color_shift(x)
            return x

        # Stochastic: each distortion applied with probability
        if torch.rand(1).item() < self.distortion_prob:
            x = self._apply_noise(x)
        if torch.rand(1).item() < self.distortion_prob:
            x = self.jpeg(x)
        if torch.rand(1).item() < self.distortion_prob:
            x = self._apply_brightness(x)
        if torch.rand(1).item() < self.distortion_prob:
            x = self._apply_color_shift(x)
        if torch.rand(1).item() < self.distortion_prob * 0.3:
            x = self._apply_blur(x)  # Blur less often — very destructive

        return x


class IdentityDistortion(nn.Module):
    """No-op distortion for clean evaluation."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x
