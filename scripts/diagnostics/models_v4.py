"""Candidate v4: spatially-structured (grid) bit encoding.

Insight from DIAGNOSTICS: global broadcast forces the network to learn L distinct
*global* patterns (one per bit). That scales terribly with L. Instead, lay each
bit out in its own spatial CELL of a Gh x Gw grid upsampled to image size. Because
convolutions are translation-equivariant with shared weights, the network learns a
SINGLE "write a bit into a cell" / "read a bit from a cell" operation that applies
to all L cells simultaneously. Learning 100 bits then costs the same as learning a
few -- locality gives strong, consistent gradients.

  - EncV4: cover(3) + upsampled bit-grid(1) -> conv stack -> 3ch perturbation
  - DecV4: stego(3) -> conv stack -> 1ch map -> adaptive_avg_pool(Gh,Gw) -> L logits
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_dims(L):
    gw = int(math.ceil(math.sqrt(L)))
    gh = int(math.ceil(L / gw))
    return gh, gw


def _norm(c):
    g = max(1, c // 8)
    while c % g != 0:
        g -= 1
    return nn.GroupNorm(g, c)


def conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        _norm(cout),  # GroupNorm: train==eval, critical in tiny-perturbation regime (D6)
        nn.ReLU(inplace=True),
    )


def payload_to_grid(payload, gh, gw, H, W):
    """(B, L) -> (B, 1, H, W) where each bit occupies one upsampled grid cell."""
    B, L = payload.shape
    pad = gh * gw - L
    if pad:
        payload = F.pad(payload, (0, pad))
    grid = payload.view(B, 1, gh, gw)
    return F.interpolate(grid, size=(H, W), mode="nearest")


class EncV4(nn.Module):
    def __init__(self, capacity_bits=100, hidden=64, depth=5, perturbation_bound=0.5):
        super().__init__()
        self.L = capacity_bits
        self.gh, self.gw = grid_dims(capacity_bits)
        self.perturbation_bound = perturbation_bound
        layers = [conv_bn_relu(3 + 1, hidden)]
        for _ in range(depth - 1):
            layers.append(conv_bn_relu(hidden, hidden))
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, 3, 1)

    def forward(self, cover, payload):
        B, _, H, W = cover.shape
        bitmap = payload_to_grid(payload, self.gh, self.gw, H, W)
        x = torch.cat([cover, bitmap], dim=1)
        pert = torch.tanh(self.head(self.body(x))) * self.perturbation_bound
        return torch.clamp(cover + pert, 0.0, 1.0)


class DecV4(nn.Module):
    def __init__(self, capacity_bits=100, hidden=64, depth=6):
        super().__init__()
        self.L = capacity_bits
        self.gh, self.gw = grid_dims(capacity_bits)
        layers = [conv_bn_relu(3, hidden)]
        for _ in range(depth - 1):
            layers.append(conv_bn_relu(hidden, hidden))
        self.body = nn.Sequential(*layers)
        self.to_map = nn.Conv2d(hidden, 1, 1)

    def forward(self, stego):
        feat = self.body(stego)
        m = self.to_map(feat)                                  # (B,1,H,W)
        cells = F.adaptive_avg_pool2d(m, (self.gh, self.gw))   # (B,1,gh,gw)
        return cells.view(cells.size(0), -1)[:, : self.L]       # (B, L) logits


def get_v4(capacity_bits, perturbation_bound, device, hidden=64):
    enc = EncV4(capacity_bits, hidden=hidden, perturbation_bound=perturbation_bound).to(device)
    dec = DecV4(capacity_bits, hidden=hidden).to(device)
    return enc, dec, capacity_bits, False
