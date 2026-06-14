"""Evaluate a trained StegaQR model (CLI wrapper over stegaqr.evaluation).

Reports hidden bit accuracy, full-decode rate, PSNR/SSIM, public QR decode rate,
and ECC message-decode rate -- clean and under the distortion layer.

Usage:
    python scripts/evaluate.py --checkpoint experiments/exp_xxx/best_model.pt
    python scripts/evaluate.py --checkpoint .../best_model.pt --n 200 --eccs rep3,rep5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stegaqr.evaluation import evaluate_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--eccs", default="rep3,rep5",
                   help="comma-separated: none,rep3,rep5,hamming74 (none has no message metrics)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", action="store_true", help="print full results as JSON")
    args = p.parse_args()

    eccs = tuple(e for e in args.eccs.split(",") if e and e != "none")
    res = evaluate_checkpoint(args.checkpoint, n=args.n, eccs=eccs,
                              seed=args.seed, device=args.device)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    c = res["config"]
    print(f"checkpoint: {c['checkpoint']}  (epoch {c['epoch']})")
    print(f"mode={c['mode']} cap={c['capacity_bits']} ms={c['module_size']} "
          f"pbound={c['perturbation_bound']} trained_distortion={c['use_distortion_train']}")
    for setting in ("clean", "distorted"):
        m = res[setting]
        print(f"\n--- {setting.upper()} (n={args.n}) ---")
        print(f"  bit accuracy     : {m['bit_acc']*100:.2f}%")
        print(f"  full-decode rate : {m['full_decode']*100:.2f}%  "
              f"(95% CI {m['full_decode_ci'][0]*100:.1f}-{m['full_decode_ci'][1]*100:.1f})")
        print(f"  PSNR / SSIM      : {m['psnr']:.2f} dB / {m['ssim']:.4f}")
        print(f"  public QR decode : {m['public_decode']*100:.2f}%")
    if res["ecc"]:
        print(f"\n--- ECC message decode (under distortion) ---")
        for name, e in res["ecc"].items():
            print(f"  {name:10s} ({e['net_bits']:3d} bits, rate {e['rate']:.2f}): "
                  f"message decode {e['msg_decode']*100:.2f}%  "
                  f"(95% CI {e['msg_decode_ci'][0]*100:.1f}-{e['msg_decode_ci'][1]*100:.1f})")


if __name__ == "__main__":
    main()
