"""Overfit diagnostic for StegaQR.

Question: can a single encoder/decoder pair drive bit accuracy to ~100% on a
TINY fixed dataset, with NO distortion and decode-only loss?

  - If YES  -> architecture/optimization is fundamentally capable; the failure
              in real training is generalization / data / loss-balance.
  - If NO   -> the architecture or optimization cannot represent/learn the
              embed->recover mapping at all. Fix that first.

This deliberately removes every confound: no distortion, no perceptual loss,
no decodability loss, fixed data, many steps.

Usage:
    python scripts/diagnostics/overfit.py --mode cross_channel
    python scripts/diagnostics/overfit.py --mode segregated --n 4 --steps 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from stegaqr.utils.seed import set_all_seeds
from stegaqr.utils.qr_utils import generate_cover_qr_rgb, get_qr_structure_mask


def build_fixed_batch(n, capacity_bits, qr_version, module_size, ec_level, device):
    """Build n fixed (cover, payload, mask) triples."""
    import random
    import string

    set_all_seeds(0)
    covers, payloads = [], []
    mask_np = get_qr_structure_mask(qr_version, module_size)
    for i in range(n):
        text = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
        rgb, _ = generate_cover_qr_rgb(text, qr_version, ec_level, module_size)
        covers.append(torch.from_numpy(rgb).permute(2, 0, 1))
        bits = np.random.randint(0, 2, size=capacity_bits).astype(np.float32)
        payloads.append(torch.from_numpy(bits))
    cover = torch.stack(covers).to(device)
    payload = torch.stack(payloads).to(device)
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1).to(device)
    return cover, payload, mask


def get_models(mode, capacity_bits, perturbation_bound, device):
    if mode == "segregated":
        from stegaqr.models.encoder import SegregatedEncoder
        from stegaqr.models.decoder import SegregatedDecoder
        capacity_bits = (capacity_bits // 3) * 3
        enc = SegregatedEncoder(capacity_bits=capacity_bits, perturbation_bound=perturbation_bound).to(device)
        dec = SegregatedDecoder(capacity_bits=capacity_bits).to(device)
        return enc, dec, capacity_bits, False
    elif mode == "cross_channel":
        from stegaqr.models.encoder import CrossChannelEncoder
        from stegaqr.models.decoder import CrossChannelDecoder
        enc = CrossChannelEncoder(capacity_bits=capacity_bits, perturbation_bound=perturbation_bound).to(device)
        dec = CrossChannelDecoder(capacity_bits=capacity_bits).to(device)
        return enc, dec, capacity_bits, False
    elif mode == "hybrid":
        from stegaqr.models.encoder import HybridEncoder
        from stegaqr.models.decoder import HybridDecoder
        enc = HybridEncoder(capacity_bits=capacity_bits, perturbation_bound=perturbation_bound).to(device)
        dec = HybridDecoder(capacity_bits=capacity_bits).to(device)
        return enc, dec, capacity_bits, True
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="cross_channel",
                   choices=["segregated", "cross_channel", "hybrid"])
    p.add_argument("--n", type=int, default=4, help="number of fixed samples")
    p.add_argument("--capacity", type=int, default=100)
    p.add_argument("--qr-version", type=int, default=4)
    p.add_argument("--module-size", type=int, default=8)
    p.add_argument("--ec-level", default="M")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--perturbation-bound", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"=== OVERFIT DIAGNOSTIC: mode={args.mode} n={args.n} "
          f"cap={args.capacity} module_size={args.module_size} "
          f"pbound={args.perturbation_bound} lr={args.lr} ===")

    cover, payload, mask = build_fixed_batch(
        args.n, args.capacity, args.qr_version, args.module_size, args.ec_level, device
    )
    enc, dec, capacity_bits, is_hybrid = get_models(
        args.mode, args.capacity, args.perturbation_bound, device
    )
    payload = payload[:, :capacity_bits]
    print(f"cover {tuple(cover.shape)} | payload {tuple(payload.shape)} | "
          f"enc params {sum(p.numel() for p in enc.parameters()):,} | "
          f"dec params {sum(p.numel() for p in dec.parameters()):,}")

    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    enc.train(); dec.train()
    for step in range(args.steps):
        if is_hybrid:
            stego = enc(cover, payload, mask)
            logits, _ = dec(stego)
        else:
            stego = enc(cover, payload)
            logits = dec(stego)

        loss = F.binary_cross_entropy_with_logits(logits, payload)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred == payload).float().mean().item()
                pert = (stego - cover).abs()
                print(f"step {step:5d} | loss {loss.item():.4f} | "
                      f"train_bit_acc {acc:.4f} | "
                      f"pert mean {pert.mean().item():.4f} max {pert.max().item():.4f}")

    # Final eval in eval() mode -- exposes BatchNorm running-stat problems
    enc.eval(); dec.eval()
    with torch.no_grad():
        if is_hybrid:
            stego = enc(cover, payload, mask)
            logits, _ = dec(stego)
        else:
            stego = enc(cover, payload)
            logits = dec(stego)
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc_eval = (pred == payload).float().mean().item()
    print(f"\nFINAL train()-mode last acc above; eval()-mode acc = {acc_eval:.4f}")
    print("VERDICT:", "CAN overfit" if acc_eval > 0.99 else
          ("train-only overfit (BN/eval issue)" if acc > 0.99 else "CANNOT overfit"))


if __name__ == "__main__":
    main()
