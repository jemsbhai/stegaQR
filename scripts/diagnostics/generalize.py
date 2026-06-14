"""Generalization diagnostic for StegaQR.

The overfit test showed the architecture CAN memorize a tiny fixed set at 100%.
The real failure is generalization. This script sweeps the difficulty knob:

  - --train-pool K : draw training batches from a FIXED pool of K distinct
                     (cover, payload) pairs. K=4 ~ overfit; K=inf (0) = fresh
                     random every step = true generalization.

We always evaluate on a HELD-OUT fresh random batch (never seen in training),
so eval bit-acc measures real generalization. Decode-only loss, no distortion.

By sweeping K we locate where generalization breaks:
    K small  -> high train & low eval  => memorization, no generalization
    K large  -> train==eval            => genuine learned code

Usage:
    python scripts/diagnostics/generalize.py --mode cross_channel --train-pool 0 --steps 4000
    python scripts/diagnostics/generalize.py --mode cross_channel --train-pool 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))  # for sibling 'overfit'

import math
import random
import string
import time
import numpy as np
import torch
import torch.nn.functional as F

from stegaqr.utils.seed import set_all_seeds
from stegaqr.utils.qr_utils import generate_cover_qr_rgb, get_qr_structure_mask
from overfit import get_models  # reuse


def make_sample(capacity_bits, qr_version, module_size, ec_level):
    text = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    rgb, _ = generate_cover_qr_rgb(text, qr_version, ec_level, module_size)
    cover = torch.from_numpy(rgb).permute(2, 0, 1)
    bits = np.random.randint(0, 2, size=capacity_bits).astype(np.float32)
    return cover, torch.from_numpy(bits)


def make_batch(b, capacity_bits, qr_version, module_size, ec_level, mask1, device, pool=None):
    if pool is not None:
        idx = [random.randrange(len(pool)) for _ in range(b)]
        covers = torch.stack([pool[i][0] for i in idx])
        payloads = torch.stack([pool[i][1] for i in idx])
    else:
        cs, ps = zip(*[make_sample(capacity_bits, qr_version, module_size, ec_level) for _ in range(b)])
        covers = torch.stack(cs)
        payloads = torch.stack(ps)
    covers = covers.to(device)
    payloads = payloads.to(device)
    mask = mask1.expand(b, -1, -1, -1).to(device)
    return covers, payloads, mask


def run_decode(enc, dec, cover, payload, mask, is_hybrid, distortion=None):
    if is_hybrid:
        stego = enc(cover, payload, mask)
    else:
        stego = enc(cover, payload)
    x = distortion(stego) if distortion is not None else stego
    if is_hybrid:
        logits, _ = dec(x)
    else:
        logits = dec(x)
    return stego, logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="cross_channel",
                   choices=["segregated", "cross_channel", "hybrid"])
    p.add_argument("--arch", default="current", choices=["current", "v3", "v4"],
                   help="current=existing; v3=HiDDeN fully-conv; v4=spatial bit-grid")
    p.add_argument("--distortion", action="store_true",
                   help="insert the differentiable distortion layer between enc and dec")
    p.add_argument("--train-pool", type=int, default=0,
                   help="size of fixed training pool; 0 = fresh random every step")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--capacity", type=int, default=100)
    p.add_argument("--qr-version", type=int, default=4)
    p.add_argument("--module-size", type=int, default=8)
    p.add_argument("--ec-level", default="M")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--perturbation-bound", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    set_all_seeds(42)
    # Diagnostics don't need bitwise determinism; enable fast cuDNN autotuning.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"=== GENERALIZE: mode={args.mode} train_pool={args.train_pool or 'inf'} "
          f"batch={args.batch} cap={args.capacity} ms={args.module_size} "
          f"pbound={args.perturbation_bound} lr={args.lr} steps={args.steps} ===")

    if args.arch == "v3":
        from models_v3 import get_v3
        enc, dec, capacity_bits, is_hybrid = get_v3(
            args.capacity, args.perturbation_bound, device)
    elif args.arch == "v4":
        from models_v4 import get_v4
        enc, dec, capacity_bits, is_hybrid = get_v4(
            args.capacity, args.perturbation_bound, device)
    else:
        enc, dec, capacity_bits, is_hybrid = get_models(
            args.mode, args.capacity, args.perturbation_bound, device)
    print(f"arch={args.arch} | enc {sum(p.numel() for p in enc.parameters()):,} | "
          f"dec {sum(p.numel() for p in dec.parameters()):,}")
    mask1 = torch.from_numpy(
        get_qr_structure_mask(args.qr_version, args.module_size)).unsqueeze(0).unsqueeze(0)

    pool = None
    if args.train_pool > 0:
        pool = [make_sample(capacity_bits, args.qr_version, args.module_size, args.ec_level)
                for _ in range(args.train_pool)]

    # Fixed held-out eval batch (fresh random, never in pool)
    eval_cover, eval_payload, eval_mask = make_batch(
        32, capacity_bits, args.qr_version, args.module_size, args.ec_level, mask1, device)

    distortion = None
    if args.distortion:
        from stegaqr.models.distortion import DifferentiableDistortion
        distortion = DifferentiableDistortion().to(device)
        print("distortion: ON (noise/jpeg/brightness/colorshift/blur)")

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)

    t0 = time.time()
    for step in range(args.steps):
        enc.train(); dec.train()
        cover, payload, mask = make_batch(
            args.batch, capacity_bits, args.qr_version, args.module_size,
            args.ec_level, mask1, device, pool=pool)
        stego, logits = run_decode(enc, dec, cover, payload, mask, is_hybrid, distortion)
        loss = F.binary_cross_entropy_with_logits(logits, payload)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 250 == 0 or step == args.steps - 1:
            enc.eval(); dec.eval()
            with torch.no_grad():
                tr_acc = ((torch.sigmoid(logits) > 0.5).float() == payload).float().mean().item()
                # eval under distortion too (robustness), if enabled
                ev_stego = enc(eval_cover, eval_payload, eval_mask) if is_hybrid else enc(eval_cover, eval_payload)
                ev_in = distortion(ev_stego) if distortion is not None else ev_stego
                ev_logits = dec(ev_in)[0] if is_hybrid else dec(ev_in)
                ev_acc = ((torch.sigmoid(ev_logits) > 0.5).float() == eval_payload).float().mean().item()
                mse = F.mse_loss(ev_stego, eval_cover).item()
                ev_psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
            sps = (step + 1) / (time.time() - t0)
            print(f"step {step:5d} | loss {loss.item():.4f} | "
                  f"train_acc {tr_acc:.4f} | EVAL_acc {ev_acc:.4f} | "
                  f"PSNR {ev_psnr:.1f}dB | {sps:.1f} it/s")


if __name__ == "__main__":
    main()
