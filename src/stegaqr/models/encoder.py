"""Steganographic encoder networks.

Each encoder takes a cover QR image (RGB, H x W x 3) and a hidden payload
(bitstring of length C) and produces a stego QR image that:
  1. Looks visually identical to the cover (perceptual loss)
  2. Still decodes correctly with standard QR readers (decodability constraint)
  3. Contains the hidden payload recoverable by the corresponding decoder
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


class PayloadExpander(nn.Module):
    """Expand a 1D bitstring payload into a spatial feature map.

    Takes (B, C) payload bits -> (B, channels, H, W) feature map
    that can be concatenated with image features.
    """

    def __init__(self, capacity_bits: int, spatial_size: int, out_channels: int = 32):
        super().__init__()
        self.spatial_size = spatial_size
        self.fc = nn.Linear(capacity_bits, out_channels * spatial_size * spatial_size)
        self.out_channels = out_channels

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        """payload: (B, capacity_bits) float tensor of {0, 1}."""
        B = payload.shape[0]
        out = self.fc(payload)
        out = out.view(B, self.out_channels, self.spatial_size, self.spatial_size)
        return out


class SegregatedEncoder(nn.Module):
    """Segregated-channel steganographic encoder.

    Encodes hidden bits independently into each RGB channel.
    The payload is split into 3 sub-payloads, one per channel.
    Each channel gets its own encoding sub-network.

    Architecture:
        Input: cover_image (B, 3, H, W) + payload (B, C)
        -> Split payload into 3 parts of C//3 bits
        -> For each channel: expand payload + encode with channel-specific net
        -> Per-channel perturbation (clamped to perturbation_bound)
        -> Output: stego_image (B, 3, H, W)
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        spatial_size: int = 33,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.1,
    ):
        super().__init__()
        self.capacity_bits = capacity_bits
        self.perturbation_bound = perturbation_bound
        bits_per_channel = capacity_bits // 3

        # One sub-encoder per channel
        self.channel_encoders = nn.ModuleList()
        for _ in range(3):
            self.channel_encoders.append(
                nn.Sequential(
                    # Input: 1 (channel) + payload_channels
                    ConvBlock(1 + 16, hidden_channels),
                    ConvBlock(hidden_channels, hidden_channels, residual=True),
                    ConvBlock(hidden_channels, hidden_channels, residual=True),
                    nn.Conv2d(hidden_channels, 1, 1),  # output: perturbation for this channel
                    nn.Tanh(),
                )
            )

        self.payload_expanders = nn.ModuleList([
            PayloadExpander(bits_per_channel, spatial_size, out_channels=16)
            for _ in range(3)
        ])

    def forward(
        self, cover: torch.Tensor, payload: torch.Tensor
    ) -> torch.Tensor:
        """
        cover: (B, 3, H, W) cover QR image, values in [0, 1]
        payload: (B, capacity_bits) hidden bits as float {0, 1}
        returns: (B, 3, H, W) stego QR image
        """
        B, _, H, W = cover.shape
        bits_per_ch = self.capacity_bits // 3
        stego_channels = []

        for ch_idx in range(3):
            # Extract channel
            channel = cover[:, ch_idx : ch_idx + 1, :, :]  # (B, 1, H, W)

            # Extract sub-payload for this channel
            start = ch_idx * bits_per_ch
            end = start + bits_per_ch
            sub_payload = payload[:, start:end]  # (B, bits_per_ch)

            # Expand payload to spatial
            payload_map = self.payload_expanders[ch_idx](sub_payload)  # (B, 16, H, W)

            # Concatenate channel + payload map
            combined = torch.cat([channel, payload_map], dim=1)  # (B, 17, H, W)

            # Encode: produce perturbation
            perturbation = self.channel_encoders[ch_idx](combined)  # (B, 1, H, W)

            # Scale and clamp perturbation
            perturbation = perturbation * self.perturbation_bound

            # Apply perturbation
            stego_ch = torch.clamp(channel + perturbation, 0.0, 1.0)
            stego_channels.append(stego_ch)

        return torch.cat(stego_channels, dim=1)  # (B, 3, H, W)


class CrossChannelEncoder(nn.Module):
    """Cross-channel steganographic encoder.

    Encodes hidden bits using inter-channel relationships. The full
    payload is embedded across all channels simultaneously, exploiting
    the 3D color space for higher capacity.

    Architecture:
        Input: cover_image (B, 3, H, W) + payload (B, C)
        -> Expand payload to spatial feature map
        -> Concatenate with cover image
        -> U-Net-like encoder-decoder producing perturbation map
        -> Output: stego_image (B, 3, H, W)
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        spatial_size: int = 33,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.1,
    ):
        super().__init__()
        self.perturbation_bound = perturbation_bound

        self.payload_expander = PayloadExpander(
            capacity_bits, spatial_size, out_channels=32
        )

        # Encoder path
        self.enc1 = ConvBlock(3 + 32, hidden_channels)
        self.enc2 = ConvBlock(hidden_channels, hidden_channels * 2)

        # Decoder path (with skip connections)
        self.dec1 = ConvBlock(hidden_channels * 2 + hidden_channels, hidden_channels)
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, 3, 1),
            nn.Tanh(),
        )

    def forward(
        self, cover: torch.Tensor, payload: torch.Tensor
    ) -> torch.Tensor:
        """
        cover: (B, 3, H, W) cover QR image
        payload: (B, capacity_bits) hidden bits
        returns: (B, 3, H, W) stego QR image
        """
        payload_map = self.payload_expander(payload)
        x = torch.cat([cover, payload_map], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        # Decoder with skip
        d1 = self.dec1(torch.cat([e2, e1], dim=1))
        perturbation = self.output(d1) * self.perturbation_bound

        return torch.clamp(cover + perturbation, 0.0, 1.0)


class HybridEncoder(nn.Module):
    """Hybrid (QR-anchored) steganographic encoder.

    The most sophisticated mode. Encodes hidden data while explicitly
    preserving QR decodability. Uses a QR structure mask to:
      1. Protect finder/timing/format modules (zero perturbation)
      2. Allow larger perturbations in data modules
      3. Use finder patterns as known reference points for calibration

    Architecture:
        Input: cover_image (B, 3, H, W) + payload (B, C) + qr_mask (B, 1, H, W)
        -> Expand payload
        -> Cross-channel encoding (like CrossChannelEncoder)
        -> Mask perturbation: zero out protected regions
        -> Output: stego_image (B, 3, H, W)
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        spatial_size: int = 33,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.1,
    ):
        super().__init__()
        self.perturbation_bound = perturbation_bound

        self.payload_expander = PayloadExpander(
            capacity_bits, spatial_size, out_channels=32
        )

        # Include QR mask as additional input channel
        self.enc1 = ConvBlock(3 + 32 + 1, hidden_channels)  # +1 for mask
        self.enc2 = ConvBlock(hidden_channels, hidden_channels * 2)
        self.enc3 = ConvBlock(hidden_channels * 2, hidden_channels * 2, residual=True)

        self.dec1 = ConvBlock(hidden_channels * 2 + hidden_channels, hidden_channels)
        self.output = nn.Sequential(
            nn.Conv2d(hidden_channels, 3, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        cover: torch.Tensor,
        payload: torch.Tensor,
        qr_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        cover: (B, 3, H, W) cover QR image
        payload: (B, capacity_bits) hidden bits
        qr_mask: (B, 1, H, W) binary mask where 1 = data module, 0 = protected
        returns: (B, 3, H, W) stego QR image
        """
        payload_map = self.payload_expander(payload)
        x = torch.cat([cover, payload_map, qr_mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d1 = self.dec1(torch.cat([e3, e1], dim=1))
        perturbation = self.output(d1) * self.perturbation_bound

        # Mask: zero perturbation on protected regions
        perturbation = perturbation * qr_mask

        return torch.clamp(cover + perturbation, 0.0, 1.0)
