"""Steganographic decoder networks — v2.

Key changes from v1:
  1. Replaces AdaptiveAvgPool2d(1) (which destroys spatial info) with
     AdaptiveAvgPool2d to an intermediate spatial size, preserving
     WHERE perturbations are located.
  2. Multi-scale feature extraction with spatial pyramid.
  3. Cross-layer attention retained for cross-channel mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class DecoderBlock(nn.Module):
    """Residual conv block with optional attention."""

    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.attention = ChannelAttention(out_ch) if use_attention else nn.Identity()
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        return F.relu(out + identity)


class SpatialDecodeHead(nn.Module):
    """Decode head that preserves spatial information.

    Instead of global avg pool → linear, we:
      1. Pool to an intermediate spatial size (e.g., 8×8)
      2. Flatten spatial + channel dims
      3. MLP to output bits

    This preserves WHERE perturbations are while keeping parameter count manageable.
    """

    def __init__(self, in_channels: int, capacity_bits: int, pool_size: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flat_dim = in_channels * pool_size * pool_size
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, capacity_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.mlp(x)


class SegregatedDecoder(nn.Module):
    """Segregated-channel decoder — extracts hidden bits per channel independently.

    Uses SpatialDecodeHead instead of global avg pool.
    """

    def __init__(self, capacity_bits: int = 99, hidden_channels: int = 64):
        super().__init__()
        bits_per_channel = capacity_bits // 3
        self.bits_per_channel = bits_per_channel

        self.channel_decoders = nn.ModuleList()
        for _ in range(3):
            self.channel_decoders.append(
                nn.Sequential(
                    DecoderBlock(1, hidden_channels, use_attention=True),
                    DecoderBlock(hidden_channels, hidden_channels, use_attention=True),
                    DecoderBlock(hidden_channels, hidden_channels, use_attention=True),
                )
            )

        self.channel_heads = nn.ModuleList([
            SpatialDecodeHead(hidden_channels, bits_per_channel)
            for _ in range(3)
        ])

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        sub_payloads = []
        for ch_idx in range(3):
            channel = stego[:, ch_idx : ch_idx + 1, :, :]
            features = self.channel_decoders[ch_idx](channel)
            sub = self.channel_heads[ch_idx](features)
            sub_payloads.append(sub)
        return torch.cat(sub_payloads, dim=1)


class CrossChannelDecoder(nn.Module):
    """Cross-channel decoder — extracts hidden bits from all channels jointly.

    Uses SpatialDecodeHead for spatially-aware readout.
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        hidden_channels: int = 64,
    ):
        super().__init__()

        self.enc1 = DecoderBlock(3, hidden_channels, use_attention=True)
        self.enc2 = DecoderBlock(hidden_channels, hidden_channels * 2, use_attention=True)
        self.enc3 = DecoderBlock(hidden_channels * 2, hidden_channels * 2, use_attention=True)

        self.head = SpatialDecodeHead(hidden_channels * 2, capacity_bits)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        x = self.enc1(stego)
        x = self.enc2(x)
        x = self.enc3(x)
        return self.head(x)


class HybridDecoder(nn.Module):
    """Hybrid decoder — QR-anchor-aware extraction with self-calibration.

    Uses the known QR structure for color calibration, then
    spatially-aware decoding.
    """

    def __init__(self, capacity_bits: int = 100, hidden_channels: int = 64):
        super().__init__()

        # Calibration network
        self.calibrator = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 12),  # 3x3 affine color correction
        )

        self.enc1 = DecoderBlock(3, hidden_channels, use_attention=True)
        self.enc2 = DecoderBlock(hidden_channels, hidden_channels * 2, use_attention=True)
        self.enc3 = DecoderBlock(hidden_channels * 2, hidden_channels * 2, use_attention=True)

        self.payload_head = SpatialDecodeHead(hidden_channels * 2, capacity_bits)

        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, 1),
            nn.Sigmoid(),
        )

    def _apply_calibration(self, image: torch.Tensor, cal_params: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        matrix = cal_params[:, :9].view(B, 3, 3)
        bias = cal_params[:, 9:12].view(B, 3, 1, 1)
        matrix = matrix + torch.eye(3, device=matrix.device).unsqueeze(0)
        calibrated = torch.einsum("bij,bjhw->bihw", matrix, image) + bias
        return torch.clamp(calibrated, 0.0, 1.0)

    def forward(self, stego: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cal_params = self.calibrator(stego)
        calibrated = self._apply_calibration(stego, cal_params)

        x = self.enc1(calibrated)
        x = self.enc2(x)
        x = self.enc3(x)

        payload = self.payload_head(x)
        confidence = self.confidence_head(x.detach())

        return payload, confidence
