"""Evaluate a checkpoint against held out, non-differentiable image operators."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stegaqr.core import default_model_path
from stegaqr.evaluation import evaluate_real_distortions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=default_model_path())
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--ecc", default="rep3")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--presets", default="", help="comma-separated subset")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/real_distortion_default.json"),
    )
    args = parser.parse_args()

    presets = tuple(name for name in args.presets.split(",") if name) or None
    results = evaluate_real_distortions(
        args.checkpoint,
        n=args.n,
        presets=presets,
        ecc_name=args.ecc,
        seed=args.seed,
        device=args.device,
        chunk=args.chunk,
    )
    results["config"]["checkpoint_sha256"] = sha256(args.checkpoint)
    results["config"]["device_requested"] = args.device

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"checkpoint: {args.checkpoint}")
    print(f"samples per preset: {args.n}")
    print("preset       bit accuracy   full decode   message decode")
    for name, metrics in results["presets"].items():
        print(
            f"{name:12s} {metrics['bit_acc'] * 100:10.2f}%"
            f" {metrics['full_decode'] * 100:12.2f}%"
            f" {metrics['msg_decode'] * 100:15.2f}%"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
