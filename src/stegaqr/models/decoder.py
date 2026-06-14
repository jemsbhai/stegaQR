"""Steganographic decoder networks — v3 (spatial bit-grid readout).

Matches the v3 encoders (see encoder.py). The decoder runs a convolutional stack,
projects to a single map, then average-pools that map down to the Gh x Gw grid so
each cell yields one bit logit. Because the read-out is shared across cells, it
generalises across all bit positions and learns in a few hundred steps.

Replaces v2's global-pool -> MLP(in*8*8 -> 256 -> L) head, which was a learning
bottleneck (see experiments/DIAGNOSTICS.md finding D4).

Public API is unchanged from v2.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_dims(num_bits: int) -> tuple[int, int]:
    gw = int(math.ceil(math.sqrt(num_bits)))
    gh = int(math.ceil(num_bits / gw))
    return gh, gw


def _norm(num_channels: int) -> nn.Module:
    """GroupNorm (no running stats) — see encoder._norm and DIAGNOSTICS finding D6."""
    num_groups = max(1, num_channels // 8)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> ReLU with optional residual."""

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


def _decoder_body(in_ch: int, hidden: int, num_blocks: int) -> nn.Sequential:
    blocks = [ConvBlock(in_ch, hidden)]
    for _ in range(num_blocks - 1):
        blocks.append(ConvBlock(hidden, hidden, residual=True))
    return nn.Sequential(*blocks)


class GridReadHead(nn.Module):
    """Conv map -> average-pool to (gh, gw) -> flatten -> first L logits."""

    def __init__(self, in_channels: int, capacity_bits: int):
        super().__init__()
        self.capacity_bits = capacity_bits
        self.gh, self.gw = grid_dims(capacity_bits)
        self.to_map = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.to_map(x)
        cells = F.adaptive_avg_pool2d(m, (self.gh, self.gw))
        return cells.view(cells.size(0), -1)[:, : self.capacity_bits]


class MaskAwareReadHead(nn.Module):
    """Read head for the mask-aware hybrid: pool to grid x grid, gather the same
    data-rich cells the encoder wrote to."""

    def __init__(self, in_channels: int, qr_version: int, capacity_bits: int):
        super().__init__()
        from stegaqr.models.encoder import select_data_cells
        self.grid = int(np.ceil(np.sqrt(2 * capacity_bits)))
        self.register_buffer("cells", select_data_cells(qr_version, self.grid, capacity_bits))
        self.to_map = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.to_map(x)
        cells = F.adaptive_avg_pool2d(m, (self.grid, self.grid))
        flat = cells.view(cells.size(0), -1)
        return flat[:, self.cells]


class BroadcastReadHead(nn.Module):
    """HiDDeN-style readout: 1x1 conv to L channels -> global average pool -> L logits."""

    def __init__(self, in_channels: int, capacity_bits: int):
        super().__init__()
        self.to_bits = nn.Conv2d(in_channels, capacity_bits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_bits(x).mean(dim=(2, 3))


class BroadcastDecoder(nn.Module):
    """Neural BASELINE decoder (HiDDeN/StegaStamp-style global readout)."""

    def __init__(self, capacity_bits: int = 100, hidden_channels: int = 64, num_blocks: int = 7):
        super().__init__()
        self.body = _decoder_body(3, hidden_channels, num_blocks)
        self.head = BroadcastReadHead(hidden_channels, capacity_bits)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(stego))


class SegregatedDecoder(nn.Module):
    """Per-channel decoder — each channel's bits are read only from that channel."""

    def __init__(self, capacity_bits: int = 99, hidden_channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.bits_per_channel = capacity_bits // 3
        self.channel_decoders = nn.ModuleList(
            [_decoder_body(1, hidden_channels, num_blocks) for _ in range(3)]
        )
        self.channel_heads = nn.ModuleList(
            [GridReadHead(hidden_channels, self.bits_per_channel) for _ in range(3)]
        )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        sub_payloads = []
        for ch in range(3):
            channel = stego[:, ch : ch + 1, :, :]
            feat = self.channel_decoders[ch](channel)
            sub_payloads.append(self.channel_heads[ch](feat))
        return torch.cat(sub_payloads, dim=1)


class CrossChannelDecoder(nn.Module):
    """Joint decoder — reads the full payload grid from all three channels."""

    def __init__(self, capacity_bits: int = 100, hidden_channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.body = _decoder_body(3, hidden_channels, num_blocks)
        self.head = GridReadHead(hidden_channels, capacity_bits)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(stego))


class HybridDecoder(nn.Module):
    """QR-anchor-aware decoder with self-calibration and a confidence head.

    A small network predicts a 3x3 affine colour correction (calibration) from the
    stego image; the corrected image is then decoded with the grid read-out. The
    confidence head predicts the probability the payload was recovered correctly.
    """

    def __init__(self, capacity_bits: int = 100, hidden_channels: int = 64, num_blocks: int = 6,
                 mask_aware: bool = False, qr_version: int = 4):
        super().__init__()
        self.calibrator = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 12),  # 3x3 matrix + 3 bias
        )
        self.body = _decoder_body(3, hidden_channels, num_blocks)
        self.payload_head = (MaskAwareReadHead(hidden_channels, qr_version, capacity_bits)
                             if mask_aware else GridReadHead(hidden_channels, capacity_bits))
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )

    def _apply_calibration(self, image: torch.Tensor, cal: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        matrix = cal[:, :9].view(B, 3, 3) + torch.eye(3, device=cal.device).unsqueeze(0)
        bias = cal[:, 9:12].view(B, 3, 1, 1)
        calibrated = torch.einsum("bij,bjhw->bihw", matrix, image) + bias
        return torch.clamp(calibrated, 0.0, 1.0)

    def forward(self, stego: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cal = self.calibrator(stego)
        calibrated = self._apply_calibration(stego, cal)
        feat = self.body(calibrated)
        payload = self.payload_head(feat)
        confidence = self.confidence_head(feat.detach())
        return payload, confidence
