"""Steganographic decoder networks.

Each decoder takes a (possibly distorted) stego QR image and recovers
the hidden payload bitstring. The decoders mirror the encoder modes:

  - SegregatedDecoder: per-channel independent extraction
  - CrossChannelDecoder: joint extraction from all channels
  - HybridDecoder: QR-anchor-aware extraction with self-calibration
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


class CrossLayerAttention(nn.Module):
    """Attention across predicted layers to resolve inter-layer correlations.

    Each layer's features attend to all other layers, learning the
    color-space correlations introduced by the encoding process.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        # Reshape to sequence: (B, H*W, C)
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_seq)

        qkv = self.qkv(x_norm).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out) + x_seq

        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


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


class SegregatedDecoder(nn.Module):
    """Segregated-channel decoder — extracts hidden bits per channel independently.

    Architecture:
        Input: stego_image (B, 3, H, W)
        -> Split into 3 channels
        -> Per-channel CNN extracts sub-payload
        -> Concatenate sub-payloads
        -> Output: (B, capacity_bits) predicted bits
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        hidden_channels: int = 64,
    ):
        super().__init__()
        bits_per_channel = capacity_bits // 3
        self.bits_per_channel = bits_per_channel

        self.channel_decoders = nn.ModuleList()
        for _ in range(3):
            self.channel_decoders.append(
                nn.Sequential(
                    DecoderBlock(1, hidden_channels, use_attention=True),
                    DecoderBlock(hidden_channels, hidden_channels, use_attention=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(hidden_channels, bits_per_channel),
                )
            )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        stego: (B, 3, H, W)
        returns: (B, capacity_bits) logits
        """
        sub_payloads = []
        for ch_idx in range(3):
            channel = stego[:, ch_idx : ch_idx + 1, :, :]
            sub = self.channel_decoders[ch_idx](channel)
            sub_payloads.append(sub)
        return torch.cat(sub_payloads, dim=1)


class CrossChannelDecoder(nn.Module):
    """Cross-channel decoder — extracts hidden bits from all channels jointly.

    Uses cross-layer attention to model inter-channel correlations.

    Architecture:
        Input: stego_image (B, 3, H, W)
        -> Multi-scale feature extraction
        -> Cross-layer attention
        -> Global pooling + FC -> payload bits
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        hidden_channels: int = 64,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        self.enc1 = DecoderBlock(3, hidden_channels, use_attention=True)
        self.enc2 = DecoderBlock(hidden_channels, hidden_channels * 2, use_attention=True)
        self.enc3 = DecoderBlock(hidden_channels * 2, hidden_channels * 2, use_attention=True)

        self.cross_attn = (
            CrossLayerAttention(hidden_channels * 2)
            if use_cross_attention
            else nn.Identity()
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, capacity_bits),
        )

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        stego: (B, 3, H, W)
        returns: (B, capacity_bits) logits
        """
        x = self.enc1(stego)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.cross_attn(x)
        return self.head(x)


class HybridDecoder(nn.Module):
    """Hybrid decoder — QR-anchor-aware extraction with self-calibration.

    Exploits the known QR structure (finder patterns, timing patterns)
    as reference points for self-calibration. The finder patterns have
    known, predictable colors (black/white), so deviation from expected
    values indicates color distortion that can be compensated.

    Architecture:
        Input: stego_image (B, 3, H, W)
        -> Detect finder pattern regions (known positions from QR structure)
        -> Estimate color calibration from finder deviations
        -> Apply learned calibration
        -> Cross-channel decode with attention
        -> Output: (B, capacity_bits) predicted bits + confidence
    """

    def __init__(
        self,
        capacity_bits: int = 100,
        hidden_channels: int = 64,
    ):
        super().__init__()

        # Calibration network: estimates color correction from image statistics
        self.calibrator = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 12),  # 3x3 matrix + 3 bias = 12 params
        )

        # Main decoder
        self.enc1 = DecoderBlock(3, hidden_channels, use_attention=True)
        self.enc2 = DecoderBlock(hidden_channels, hidden_channels * 2, use_attention=True)
        self.enc3 = DecoderBlock(hidden_channels * 2, hidden_channels * 2, use_attention=True)

        self.cross_attn = CrossLayerAttention(hidden_channels * 2)

        # Payload head
        self.payload_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, capacity_bits),
        )

        # Confidence head (auxiliary: estimates decode reliability)
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, 1),
            nn.Sigmoid(),
        )

    def _apply_calibration(
        self, image: torch.Tensor, cal_params: torch.Tensor
    ) -> torch.Tensor:
        """Apply learned 3x3 affine color calibration.

        cal_params: (B, 12) -> 3x3 matrix + 3 bias
        """
        B = image.shape[0]
        matrix = cal_params[:, :9].view(B, 3, 3)
        bias = cal_params[:, 9:12].view(B, 3, 1, 1)

        # Initialize near identity
        matrix = matrix + torch.eye(3, device=matrix.device).unsqueeze(0)

        # Apply: out[c] = sum_j(matrix[c,j] * image[j]) + bias[c]
        calibrated = torch.einsum("bij,bjhw->bihw", matrix, image) + bias
        return torch.clamp(calibrated, 0.0, 1.0)

    def forward(
        self, stego: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        stego: (B, 3, H, W)
        returns: (payload_logits, confidence)
            payload_logits: (B, capacity_bits)
            confidence: (B, 1) estimated decode reliability
        """
        # Self-calibrate
        cal_params = self.calibrator(stego)
        calibrated = self._apply_calibration(stego, cal_params)

        # Decode
        x = self.enc1(calibrated)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.cross_attn(x)

        payload = self.payload_head(x)
        confidence = self.confidence_head(x.detach())  # detach: confidence shouldn't affect features

        return payload, confidence
