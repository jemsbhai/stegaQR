"""Run a StegaQR training experiment.

Usage:
    python scripts/train.py --mode hybrid --capacity 100 --epochs 50
    python scripts/train.py --mode segregated --capacity 99 --epochs 100 --seed 123
    python scripts/train.py --mode cross_channel --no-distortion
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stegaqr.training import train


def main():
    parser = argparse.ArgumentParser(description="Train StegaQR encoder-decoder")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["segregated", "cross_channel", "hybrid"],
        help="Steganographic mode",
    )
    parser.add_argument("--capacity", type=int, default=100, help="Hidden bits")
    parser.add_argument("--qr-version", type=int, default=4, help="QR version")
    parser.add_argument("--ec-level", type=str, default="M", help="EC level")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-train", type=int, default=5000, help="Training samples")
    parser.add_argument("--num-val", type=int, default=750, help="Validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/exp_{mode}_{capacity}b)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint interval")
    parser.add_argument("--no-distortion", action="store_true", help="Disable distortion layer")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"experiments/exp_{args.mode}_{args.capacity}b_seed{args.seed}"

    result = train(
        mode=args.mode,
        capacity_bits=args.capacity,
        qr_version=args.qr_version,
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
    )

    print(f"\nFinal best validation accuracy: {result['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
