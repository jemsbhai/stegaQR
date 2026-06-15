"""Export stego QR images for a physical print-scan / photo robustness study.

Generates N stego QR images from a trained model (each carrying a random hidden
message, ECC-coded) plus a manifest.json with ground truth. The user prints or
displays these, photographs them, and feeds the photos to decode_from_photo.py.

A quiet zone is added so the printed/displayed codes are reliably scannable, and
each image is upscaled for clean printing.

Usage:
    python scripts/export_for_capture.py --checkpoint experiments/full/main_cross_channel_distort_s42/best_model.pt \
        --n 12 --ecc rep3 --out capture/exp1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from stegaqr.utils.seed import set_all_seeds
from stegaqr.utils.qr_utils import generate_cover_qr_rgb, get_qr_structure_mask
from stegaqr.coding import get_ecc
from stegaqr.evaluation import build_models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--ecc", default="rep3")
    p.add_argument("--out", default="capture/export")
    p.add_argument("--upscale", type=int, default=12, help="output pixels per stego pixel")
    p.add_argument("--quiet-zone", type=int, default=4, help="quiet-zone modules")
    p.add_argument("--seed", type=int, default=20260614)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    set_all_seeds(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mode, cap = cfg["mode"], cfg["capacity_bits"]
    qrv, ms, ec = cfg["qr_version"], cfg["module_size"], cfg["ec_level"]
    enc, dec, is_hybrid = build_models(
        mode, cap, device, cfg.get("perturbation_bound", 0.1),
        arch=cfg.get("arch", "grid"), mask_aware=cfg.get("mask_aware", False),
        qr_version=qrv, use_calibration=cfg.get("use_calibration", True))
    enc.load_state_dict(ckpt["encoder_state"]); dec.load_state_dict(ckpt["decoder_state"])
    enc.eval()

    ecc = get_ecc(args.ecc)
    k = ecc.message_len(cap); clen = ecc.coded_len(k)
    qz = args.quiet_zone

    manifest = {"checkpoint": args.checkpoint, "mode": mode, "capacity_bits": cap,
                "ecc": args.ecc, "net_bits": int(k), "module_size": ms,
                "quiet_zone": qz, "upscale": args.upscale, "qr_version": qrv,
                "ec_level": ec, "items": []}

    import random, string
    for i in range(args.n):
        text = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
        # cover WITHOUT quiet zone for the model (matches training), add quiet zone after
        cover, _ = generate_cover_qr_rgb(text, qrv, ec, ms, quiet_zone=0)
        msg = np.random.randint(0, 2, k).astype(np.uint8)
        coded = np.zeros(cap, dtype=np.uint8); coded[:clen] = ecc.encode(msg)
        c = torch.from_numpy(cover).permute(2, 0, 1).unsqueeze(0).to(device)
        pl = torch.from_numpy(coded.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            if is_hybrid:
                mask = torch.from_numpy(get_qr_structure_mask(qrv, ms)).unsqueeze(0).unsqueeze(0).to(device)
                stego = enc(c, pl, mask)
            else:
                stego = enc(c, pl)
        s = stego.squeeze(0).permute(1, 2, 0).cpu().numpy()
        s = (np.clip(s, 0, 1) * 255).astype(np.uint8)
        if qz:
            s = np.pad(s, ((qz * ms, qz * ms), (qz * ms, qz * ms), (0, 0)), constant_values=255)
        im = Image.fromarray(s)
        im = im.resize((im.width * args.upscale, im.height * args.upscale), Image.NEAREST)
        fname = f"stego_{i:03d}.png"
        im.save(out / fname)
        manifest["items"].append({"file": fname, "public_text": text,
                                  "message_bits": msg.tolist()})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {args.n} stego QR images + manifest.json to {out}")
    print(f"Print or display these, photograph them, then run decode_from_photo.py "
          f"on each photo (or a folder of photos).")


if __name__ == "__main__":
    main()
