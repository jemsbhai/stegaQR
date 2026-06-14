"""Run the full StegaQR experiment matrix (EXP-001 + studies).

Resumable: each run writes results.json; completed runs are skipped. Designed for
the 16GB GPU (module_size=4, batch 16 -- the validated config).

Matrix:
  * Main mode comparison: {segregated, cross_channel, hybrid} x {clean, distortion}
    x SEEDS, capacity 100. Tests whether end-to-end distortion training buys
    robustness (the EXP-001 hypothesis).
  * Capacity scaling: cross_channel x CAPACITIES x SEEDS_CAP, distortion.
  * Classical LSB baseline (no training): bit acc / PSNR / public decode,
    clean vs distortion -- the non-neural reference point.

Each trained run is evaluated with the full suite (clean + distorted + ECC) via
stegaqr.evaluation.evaluate_checkpoint.

Usage:
    python scripts/run_experiments.py --quick      # tiny smoke matrix
    python scripts/run_experiments.py              # full matrix
    python scripts/run_experiments.py --only main  # main | capacity | classical
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from stegaqr.training import train
from stegaqr.evaluation import evaluate_checkpoint, build_models

ROOT = Path(__file__).parent.parent
OUT = ROOT / "experiments" / "full"

# --- shared training recipe (validated for the 16GB GPU) ---
MS = 4
BATCH = 16
EPOCHS = 24
NUM_TRAIN = 1500
NUM_VAL = 256
WARMUP = 3
RAMP = 8
LAMBDA_PERC = 4.0
PBOUND = 0.3
EVAL_N = 128
ECCS = ("rep3", "rep5", "hamming74")

MODES = ["segregated", "cross_channel", "hybrid"]
SEEDS = [42, 123, 7, 99, 2024]
CAPACITIES = [25, 50, 150, 200]   # 100 comes from the main matrix
SEEDS_CAP = [42, 123, 7]


def _run_one(mode, capacity, seed, use_distortion, tag, arch="grid", mask_aware=False):
    out_dir = OUT / tag
    res_path = out_dir / "results.json"
    if res_path.exists():
        print(f"[skip] {tag} (results.json exists)")
        return json.loads(res_path.read_text())

    print(f"\n{'='*70}\n[run] {tag}\n{'='*70}", flush=True)
    t0 = time.time()
    train(
        mode=mode, capacity_bits=capacity, qr_version=4, module_size=MS,
        ec_level="M", epochs=EPOCHS, batch_size=BATCH, lr=1e-3,
        num_train=NUM_TRAIN, num_val=NUM_VAL, seed=seed, device="cuda",
        output_dir=str(out_dir), checkpoint_every=EPOCHS,
        lambda_perceptual=LAMBDA_PERC, perturbation_bound=PBOUND,
        use_distortion=use_distortion, warmup_decode_only=WARMUP,
        perc_ramp_epochs=RAMP, arch=arch, mask_aware=mask_aware,
    )
    metrics = evaluate_checkpoint(str(out_dir / "best_model.pt"), n=EVAL_N,
                                  eccs=ECCS, seed=1234 + seed)
    metrics["tag"] = tag
    metrics["train_seconds"] = round(time.time() - t0, 1)
    res_path.write_text(json.dumps(metrics, indent=2))
    print(f"[done] {tag} in {metrics['train_seconds']}s "
          f"| clean fdr {metrics['clean']['full_decode']:.3f} "
          f"psnr {metrics['clean']['psnr']:.1f} "
          f"| dist fdr {metrics['distorted']['full_decode']:.3f}", flush=True)
    return metrics


def run_main(seeds):
    for mode in MODES:
        for use_dist in (False, True):
            for seed in seeds:
                dtag = "distort" if use_dist else "clean"
                _run_one(mode, 100, seed, use_dist, f"main_{mode}_{dtag}_s{seed}")


def run_capacity(seeds):
    for cap in CAPACITIES:
        for seed in seeds:
            _run_one("cross_channel", cap, seed, True, f"cap_{cap}_s{seed}")


def run_ablations(seeds):
    """Neural baseline (global broadcast) and the mask-aware hybrid, vs the
    spatial-grid mode results in the main matrix (cross_channel / hybrid distort)."""
    for seed in seeds:
        _run_one("cross_channel", 100, seed, True, f"abl_broadcast_s{seed}", arch="broadcast")
        _run_one("hybrid", 100, seed, True, f"abl_hybrid_maskaware_s{seed}", mask_aware=True)


def run_classical():
    """Classical LSB baseline: encode -> (distort) -> decode. No training."""
    import random, string
    from PIL import Image
    from stegaqr.models.classical import ClassicalSegregatedEncoder
    from stegaqr.models.distortion import DifferentiableDistortion
    from stegaqr.utils.qr_utils import generate_cover_qr_rgb, get_qr_structure_mask
    from stegaqr.utils.metrics import bit_accuracy, full_decode_rate, psnr, ssim, qr_public_decode_rate
    from stegaqr.utils.seed import set_all_seeds

    set_all_seeds(1234)
    cap, n = 99, EVAL_N
    codec = ClassicalSegregatedEncoder(capacity_bits=cap, num_lsb=1)
    mask = get_qr_structure_mask(4, MS)
    dist = DifferentiableDistortion()

    out = {"tag": "classical_lsb", "config": {"mode": "classical_lsb", "capacity_bits": cap}}
    for setting in ("clean", "distorted"):
        preds, gts, psnrs, ssims, imgs, texts = [], [], [], [], [], []
        for _ in range(n):
            text = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
            cover, _ = generate_cover_qr_rgb(text, 4, "M", MS)           # (H,W,3) float
            cover_u8 = (cover * 255).astype(np.uint8)
            payload = np.random.randint(0, 2, cap).astype(np.uint8)
            stego_u8 = codec.encode(cover_u8, payload, mask)
            stego = stego_u8.astype(np.float32) / 255.0
            if setting == "distorted":
                t = torch.from_numpy(stego).permute(2, 0, 1).unsqueeze(0)
                t = dist(t)
                stego = t.squeeze(0).permute(1, 2, 0).numpy()
                stego_u8 = (stego * 255).astype(np.uint8)
            rec = codec.decode(stego_u8, mask)
            preds.append(rec); gts.append(payload)
            psnrs.append(psnr(cover, stego)); ssims.append(ssim(cover, stego))
            imgs.append(Image.fromarray(stego_u8)); texts.append(text)
        preds = np.stack(preds); gts = np.stack(gts)
        out[setting] = {
            "bit_acc": bit_accuracy(preds, gts),
            "full_decode": full_decode_rate(preds, gts),
            "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
            "public_decode": qr_public_decode_rate(imgs, texts),
        }
    out["ecc"] = {}
    (OUT / "classical_lsb").mkdir(parents=True, exist_ok=True)
    (OUT / "classical_lsb" / "results.json").write_text(json.dumps(out, indent=2))
    print(f"[done] classical_lsb | clean bit {out['clean']['bit_acc']:.3f} "
          f"psnr {out['clean']['psnr']:.1f} | dist bit {out['distorted']['bit_acc']:.3f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="tiny smoke matrix")
    p.add_argument("--only", choices=["main", "capacity", "classical", "ablations"], default=None)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.quick:
        global EPOCHS, NUM_TRAIN, NUM_VAL
        EPOCHS, NUM_TRAIN, NUM_VAL = 6, 400, 128
        print("QUICK smoke matrix")
        for mode in MODES:
            _run_one(mode, 100, 42, True, f"smoke_{mode}_distort_s42")
        run_classical()
        return

    if args.only in (None, "main"):
        run_main(SEEDS)
    if args.only in (None, "capacity"):
        run_capacity(SEEDS_CAP)
    if args.only in (None, "ablations"):
        run_ablations(SEEDS_CAP)
    if args.only in (None, "classical"):
        run_classical()
    print("\nALL RUNS COMPLETE")


if __name__ == "__main__":
    main()
