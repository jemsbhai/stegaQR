"""Candidate v3 encoder/decoder variants for A/B testing against current models.

Hypothesis (from DIAGNOSTICS D1-D3): the current decoder's global-pool ->
MLP(in*8*8 -> 256 -> L) bottleneck and the lack of a clean spatial readout make
cold-start slow and cap a ceiling well below target. The proven HiDDeN/StegaStamp
readout is fully convolutional: conv stack -> 1x1 conv to L channels -> global
average pool -> L logits. No MLP bottleneck.

We test, head to head, at fixed compute, on fresh random data:
  - EncV3:  broadcast payload + cover -> conv stack (full res) -> 3ch perturbation
  - DecV3:  conv stack -> 1x1 conv to L -> global avg pool -> L logits

These are deliberately simple/standard so a positive result isolates the cause.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class EncV3(nn.Module):
    """HiDDeN-style encoder: broadcast payload, concat with cover, conv stack."""

    def __init__(self, capacity_bits=100, hidden=64, depth=6, perturbation_bound=0.5):
        super().__init__()
        self.capacity_bits = capacity_bits
        self.perturbation_bound = perturbation_bound
        layers = [conv_bn_relu(3 + capacity_bits, hidden)]
        for _ in range(depth - 1):
            layers.append(conv_bn_relu(hidden, hidden))
        self.body = nn.Sequential(*layers)
        # final: concat cover again (HiDDeN trick) then 1x1 to 3
        self.head = nn.Conv2d(hidden + 3 + capacity_bits, 3, 1)

    def forward(self, cover, payload):
        B, _, H, W = cover.shape
        pmap = payload.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        x = torch.cat([cover, pmap], dim=1)
        feat = self.body(x)
        out = self.head(torch.cat([feat, x], dim=1))
        pert = torch.tanh(out) * self.perturbation_bound
        return torch.clamp(cover + pert, 0.0, 1.0)


class DecV3(nn.Module):
    """HiDDeN-style decoder: conv stack -> 1x1 to L -> global avg pool -> L logits."""

    def __init__(self, capacity_bits=100, hidden=64, depth=7):
        super().__init__()
        layers = [conv_bn_relu(3, hidden)]
        for _ in range(depth - 1):
            layers.append(conv_bn_relu(hidden, hidden))
        self.body = nn.Sequential(*layers)
        self.to_bits = nn.Conv2d(hidden, capacity_bits, 1)

    def forward(self, stego):
        feat = self.body(stego)
        bitmap = self.to_bits(feat)            # (B, L, H, W)
        return bitmap.mean(dim=(2, 3))          # (B, L) global average -> logits


def get_v3(capacity_bits, perturbation_bound, device, hidden=64):
    enc = EncV3(capacity_bits, hidden=hidden, perturbation_bound=perturbation_bound).to(device)
    dec = DecV3(capacity_bits, hidden=hidden).to(device)
    return enc, dec, capacity_bits, False
