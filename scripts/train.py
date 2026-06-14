"""Run a StegaQR training experiment.

Usage:
    python scripts/train.py --mode hybrid --capacity 100 --epochs 50
    python scripts/train.py --mode segregated --capacity 99 --epochs 100 --seed 123
    python scripts/train.py --mode cross_channel --no-distortion --module-size 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stegaqr.training import train


def main():
    parser = argparse.ArgumentParser(description="Train StegaQR encoder-decoder")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["segregated", "cross_channel", "hybrid"])
    parser.add_argument("--capacity", type=int, default=100, help="Hidden bits")
    parser.add_argument("--qr-version", type=int, default=4, help="QR version")
    parser.add_argument("--module-size", type=int, default=8, help="Pixels per QR module")
    parser.add_argument("--ec-level", type=str, default="M")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-train", type=int, default=5000)
    parser.add_argument("--num-val", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--no-distortion", action="store_true")
    parser.add_argument("--perturbation-bound", type=float, default=0.3,
                        help="Max per-channel perturbation (generous bound). The "
                             "perceptual loss drives ADAPTIVE sub-bound perturbation; "
                             "see experiments/IMPERCEPTIBILITY.md")
    parser.add_argument("--lambda-perceptual", type=float, default=6.0,
                        help="Perceptual loss weight. Higher -> more imperceptible "
                             "(higher PSNR) at the cost of robustness margin.")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Epochs of decode-only loss before ramping perceptual in")
    parser.add_argument("--perc-ramp-epochs", type=int, default=10,
                        help="Epochs to linearly ramp perceptual/decodability 0->target "
                             "after warmup (stabilises adaptive embedding)")
    parser.add_argument("--arch", default="grid", choices=["grid", "broadcast"],
                        help="grid = spatial bit-grid (ours); broadcast = HiDDeN-style baseline")
    parser.add_argument("--mask-aware", action="store_true",
                        help="hybrid only: place bits on data-rich cells (avoid finder patterns)")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"experiments/exp_{args.mode}_{args.capacity}b_seed{args.seed}"

    result = train(
        mode=args.mode,
        capacity_bits=args.capacity,
        qr_version=args.qr_version,
        module_size=args.module_size,
        ec_level=args.ec_level,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
        use_distortion=not args.no_distortion,
        perturbation_bound=args.perturbation_bound,
        lambda_perceptual=args.lambda_perceptual,
        warmup_decode_only=args.warmup_epochs,
        perc_ramp_epochs=args.perc_ramp_epochs,
        arch=args.arch,
        mask_aware=args.mask_aware,
    )

    print(f"\nFinal best validation accuracy: {result['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
