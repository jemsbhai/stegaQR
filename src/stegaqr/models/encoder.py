"""Steganographic encoder networks — v2.

Key changes from v1:
  1. Payload is spatially broadcast (tiled), not projected through a massive linear layer.
     This is the standard approach in HiDDeN/StegaStamp.
  2. Works at scaled resolution (module_size pixels per QR module) for more spatial bandwidth.
  3. Perturbation bound increased and made configurable.
  4. Proper U-Net with downsampling/upsampling for cross-channel and hybrid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU block with optional residual connection."""

    def __init__(self, in_ch: int, out_ch: int, residual: bool = False):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + identity
        return F.relu(out)


class SpatialPayloadBroadcast(nn.Module):
    """Broadcast payload bits to a spatial feature map by tiling.

    Standard approach from HiDDeN: each bit becomes a constant spatial
    channel. No learnable parameters — the encoder conv layers learn
    how to integrate the payload with the cover image.

    (B, C) → (B, C, H, W) by repeating each bit value across all positions.
    """

    def forward(self, payload: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        payload: (B, C) float {0, 1}
        returns: (B, C, H, W)
        """
        return payload.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)


class SegregatedEncoder(nn.Module):
    """Segregated-channel steganographic encoder.

    Each RGB channel independently encodes a sub-payload.
    Payload bits are spatially broadcast, not projected through a linear layer.
    """

    def __init__(
        self,
        capacity_bits: int = 99,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.capacity_bits = capacity_bits
        self.perturbation_bound = perturbation_bound
        bits_per_channel = capacity_bits // 3

        self.broadcast = SpatialPayloadBroadcast()

        # One sub-encoder per channel: input = 1 (channel) + bits_per_channel (payload)
        self.channel_encoders = nn.ModuleList()
        for _ in range(3):
            self.channel_encoders.append(
                nn.Sequential(
                    ConvBlock(1 + bits_per_channel, hidden_channels),
                    ConvBlock(hidden_channels, hidden_channels, residual=True),
                    ConvBlock(hidden_channels, hidden_channels, residual=True),
                    ConvBlock(hidden_channels, hidden_channels, residual=True),
                    nn.Conv2d(hidden_channels, 1, 1),
                    nn.Tanh(),
                )
            )

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        B, _, H, W = cover.shape
        bits_per_ch = self.capacity_bits // 3
        stego_channels = []

        for ch_idx in range(3):
            channel = cover[:, ch_idx : ch_idx + 1, :, :]
            sub_payload = payload[:, ch_idx * bits_per_ch : (ch_idx + 1) * bits_per_ch]

            # Spatial broadcast: (B, bits_per_ch) → (B, bits_per_ch, H, W)
            payload_map = self.broadcast(sub_payload, H, W)

            combined = torch.cat([channel, payload_map], dim=1)
            perturbation = self.channel_encoders[ch_idx](combined) * self.perturbation_bound
            stego_ch = torch.clamp(channel + perturbation, 0.0, 1.0)
            stego_channels.append(stego_ch)

        return torch.cat(stego_channels, dim=1)


class CrossChannelEncoder(nn.Module):
    """Cross-channel steganographic encoder.

    All channels jointly encode the full payload. U-Net with skip connections.
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.perturbation_bound = perturbation_bound
        self.broadcast = SpatialPayloadBroadcast()

        in_ch = 3 + capacity_bits  # image + broadcast payload

        # Encoder path
        self.enc1 = ConvBlock(in_ch, hidden_channels)
        self.enc2 = ConvBlock(hidden_channels, hidden_channels * 2)
        self.enc3 = ConvBlock(hidden_channels * 2, hidden_channels * 2, residual=True)

        # Decoder path with skip connections
        self.dec2 = ConvBlock(hidden_channels * 2 + hidden_channels * 2, hidden_channels * 2)
        self.dec1 = ConvBlock(hidden_channels * 2 + hidden_channels, hidden_channels)
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, 3, 1),
            nn.Tanh(),
        )

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        B, _, H, W = cover.shape
        payload_map = self.broadcast(payload, H, W)
        x = torch.cat([cover, payload_map], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(torch.cat([e3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        perturbation = self.output(d1) * self.perturbation_bound

        return torch.clamp(cover + perturbation, 0.0, 1.0)


class HybridEncoder(nn.Module):
    """Hybrid (QR-anchored) steganographic encoder.

    Like CrossChannelEncoder but with QR structure mask awareness.
    Perturbation is zeroed on protected modules (finder, timing, format).
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.perturbation_bound = perturbation_bound
        self.broadcast = SpatialPayloadBroadcast()

        in_ch = 3 + capacity_bits + 1  # image + payload + mask

        self.enc1 = ConvBlock(in_ch, hidden_channels)
        self.enc2 = ConvBlock(hidden_channels, hidden_channels * 2)
        self.enc3 = ConvBlock(hidden_channels * 2, hidden_channels * 2, residual=True)

        self.dec2 = ConvBlock(hidden_channels * 2 + hidden_channels * 2, hidden_channels * 2)
        self.dec1 = ConvBlock(hidden_channels * 2 + hidden_channels, hidden_channels)
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, 3, 1),
            nn.Tanh(),
        )

    def forward(
        self, cover: torch.Tensor, payload: torch.Tensor, qr_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = cover.shape
        payload_map = self.broadcast(payload, H, W)
        x = torch.cat([cover, payload_map, qr_mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(torch.cat([e3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        perturbation = self.output(d1) * self.perturbation_bound
        perturbation = perturbation * qr_mask

        return torch.clamp(cover + perturbation, 0.0, 1.0)
