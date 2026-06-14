"""Steganographic encoder networks — v3 (spatial bit-grid).

Key change from v2 (and the fix for the EXP-001 learning failure):
  v2 broadcast every payload bit as a *constant global* channel. That forces the
  network to learn L independent global patterns (one per bit), which cold-starts
  pathologically slowly and caps bit-accuracy near chance for L~100 (see
  experiments/DIAGNOSTICS.md, findings D2-D5).

  v3 lays each bit into its own cell of a Gh x Gw grid, upsampled (nearest) to
  image resolution. Because convolutions are translation-equivariant with shared
  weights, the network learns ONE "write a bit into a cell" operation that applies
  to all L cells at once. Learning 100 bits then costs the same as learning a few:
  100% held-out bit accuracy in ~250 steps, clean and under distortion.

Public API (constructor + forward signatures) is unchanged from v2, so the
training pipeline and tests are drop-in compatible.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_dims(num_bits: int) -> tuple[int, int]:
    """Near-square grid (gh, gw) with gh*gw >= num_bits."""
    gw = int(math.ceil(math.sqrt(num_bits)))
    gh = int(math.ceil(num_bits / gw))
    return gh, gw


def payload_to_gridmap(payload: torch.Tensor, gh: int, gw: int, H: int, W: int) -> torch.Tensor:
    """(B, L) bits -> (B, 1, H, W) with each bit occupying one upsampled grid cell."""
    B, L = payload.shape
    pad = gh * gw - L
    if pad:
        payload = F.pad(payload, (0, pad))
    grid = payload.view(B, 1, gh, gw)
    return F.interpolate(grid, size=(H, W), mode="nearest")


def _norm(num_channels: int) -> nn.Module:
    """GroupNorm (no running stats) — train/eval consistent, which matters for the
    tiny-perturbation steganographic regime where BatchNorm's running statistics
    diverge from per-batch statistics and break eval-mode decoding (see
    experiments/DIAGNOSTICS.md finding D6)."""
    num_groups = max(1, num_channels // 8)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU block with optional residual connection."""

    def __init__(self, in_ch: int, out_ch: int, residual: bool = False):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = _norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = _norm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + identity
        return F.relu(out)


def _encoder_body(in_ch: int, hidden: int, num_blocks: int) -> nn.Sequential:
    blocks = [ConvBlock(in_ch, hidden)]
    for _ in range(num_blocks - 1):
        blocks.append(ConvBlock(hidden, hidden, residual=True))
    return nn.Sequential(*blocks)


class SegregatedEncoder(nn.Module):
    """Segregated-channel encoder — each RGB channel independently carries a
    sub-payload via its own spatial bit-grid. Fault-isolated: a channel's bits
    depend only on that channel.
    """

    def __init__(
        self,
        capacity_bits: int = 99,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.3,
        num_blocks: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.capacity_bits = capacity_bits
        self.perturbation_bound = perturbation_bound
        self.bits_per_channel = capacity_bits // 3
        self.gh, self.gw = grid_dims(self.bits_per_channel)

        self.channel_encoders = nn.ModuleList()
        for _ in range(3):
            self.channel_encoders.append(
                nn.Sequential(
                    _encoder_body(1 + 1, hidden_channels, num_blocks),  # channel + bitmap
                    nn.Conv2d(hidden_channels, 1, 1),
                    nn.Tanh(),
                )
            )

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        B, _, H, W = cover.shape
        bpc = self.bits_per_channel
        stego_channels = []
        for ch in range(3):
            channel = cover[:, ch : ch + 1, :, :]
            sub = payload[:, ch * bpc : (ch + 1) * bpc]
            bitmap = payload_to_gridmap(sub, self.gh, self.gw, H, W)
            combined = torch.cat([channel, bitmap], dim=1)
            pert = self.channel_encoders[ch](combined) * self.perturbation_bound
            stego_channels.append(torch.clamp(channel + pert, 0.0, 1.0))
        return torch.cat(stego_channels, dim=1)


class CrossChannelEncoder(nn.Module):
    """Cross-channel encoder — the full payload is laid out on a single grid and
    embedded jointly across all three channels.
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.3,
        num_blocks: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.capacity_bits = capacity_bits
        self.perturbation_bound = perturbation_bound
        self.gh, self.gw = grid_dims(capacity_bits)
        self.body = _encoder_body(3 + 1, hidden_channels, num_blocks)
        self.head = nn.Sequential(nn.Conv2d(hidden_channels, 3, 1), nn.Tanh())

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        B, _, H, W = cover.shape
        bitmap = payload_to_gridmap(payload, self.gh, self.gw, H, W)
        x = torch.cat([cover, bitmap], dim=1)
        pert = self.head(self.body(x)) * self.perturbation_bound
        return torch.clamp(cover + pert, 0.0, 1.0)


class HybridEncoder(nn.Module):
    """Hybrid (QR-anchored) encoder — like CrossChannelEncoder but perturbation is
    zeroed on protected QR modules (finder/timing/format) via the structure mask,
    keeping the public QR standard-decodable.
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        hidden_channels: int = 64,
        perturbation_bound: float = 0.3,
        num_blocks: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.capacity_bits = capacity_bits
        self.perturbation_bound = perturbation_bound
        self.gh, self.gw = grid_dims(capacity_bits)
        self.body = _encoder_body(3 + 1 + 1, hidden_channels, num_blocks)  # cover + bitmap + mask
        self.head = nn.Sequential(nn.Conv2d(hidden_channels, 3, 1), nn.Tanh())

    def forward(
        self, cover: torch.Tensor, payload: torch.Tensor, qr_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H, W = cover.shape
        bitmap = payload_to_gridmap(payload, self.gh, self.gw, H, W)
        x = torch.cat([cover, bitmap, qr_mask], dim=1)
        pert = self.head(self.body(x)) * self.perturbation_bound
        pert = pert * qr_mask
        return torch.clamp(cover + pert, 0.0, 1.0)
