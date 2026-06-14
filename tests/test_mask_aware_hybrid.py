"""Tests for the mask-aware hybrid (data-rich cell placement)."""

import numpy as np
import torch

from stegaqr.models.encoder import HybridEncoder, select_data_cells
from stegaqr.models.decoder import HybridDecoder
from stegaqr.utils.qr_utils import get_qr_structure_mask


def test_select_data_cells_prefers_data():
    """Selected cells should have higher mean data-fraction than the grid average."""
    version, grid, L = 4, 15, 100
    cells = select_data_cells(version, grid, L)
    assert cells.shape[0] == L
    mask = get_qr_structure_mask(version, 1)
    m = torch.from_numpy(mask)[None, None].float()
    import torch.nn.functional as F
    frac = F.adaptive_avg_pool2d(m, grid).flatten()
    assert frac[cells].mean() > frac.mean()        # picks data-rich cells
    assert frac[cells].min() >= frac.mean() - 1e-6  # all chosen cells above-ish average


def test_mask_aware_hybrid_shapes_and_grad():
    B, H, W, L = 2, 132, 132, 100
    cover = torch.rand(B, 3, H, W)
    payload = torch.randint(0, 2, (B, L)).float()
    mask = torch.ones(B, 1, H, W)
    enc = HybridEncoder(capacity_bits=L, mask_aware=True, qr_version=4)
    dec = HybridDecoder(capacity_bits=L, mask_aware=True, qr_version=4)
    stego = enc(cover, payload, mask)
    assert stego.shape == cover.shape
    logits, conf = dec(stego)
    assert logits.shape == (B, L)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, payload)
    loss.backward()
    assert any(p.grad is not None for p in enc.parameters())


def test_default_hybrid_unchanged():
    """mask_aware=False must keep the original behaviour/shape."""
    B, H, W, L = 2, 132, 132, 100
    cover = torch.rand(B, 3, H, W)
    payload = torch.randint(0, 2, (B, L)).float()
    mask = torch.ones(B, 1, H, W)
    enc = HybridEncoder(capacity_bits=L)         # default mask_aware=False
    dec = HybridDecoder(capacity_bits=L)
    logits, _ = dec(enc(cover, payload, mask))
    assert logits.shape == (B, L)
