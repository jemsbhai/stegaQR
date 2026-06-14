"""Decode stego QR codes from real photographs (print-scan / photo robustness).

For each photo: locate the QR symbol (OpenCV detector), perspective-rectify it to
the model's canonical resolution, run the neural decoder + ECC to recover the hidden
message, and run pyzbar for the public payload. Results are matched to the export
manifest (by public text) and scored.

Usage:
    python scripts/decode_from_photo.py --photos capture/exp1_photos --manifest capture/exp1/manifest.json
    # quick self-test (no camera): synthesise "photos" via warp+JPEG from the exports
    python scripts/decode_from_photo.py --self-test --manifest capture/exp1/manifest.json --exports capture/exp1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import cv2
from PIL import Image

from stegaqr.coding import get_ecc
from stegaqr.evaluation import build_models


def locate_and_rectify(bgr, out_size):
    """Find a QR symbol in a BGR image; return an (out_size,out_size,3) RGB float
    rectified crop of the bare symbol, or None if not found."""
    det = cv2.QRCodeDetector()
    ok, pts = det.detect(bgr)
    if not ok or pts is None:
        return None
    pts = pts.reshape(4, 2).astype(np.float32)
    dst = np.array([[0, 0], [out_size - 1, 0],
                    [out_size - 1, out_size - 1], [0, out_size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warp = cv2.warpPerspective(bgr, M, (out_size, out_size))
    rgb = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def load_model(manifest, device, checkpoint=None):
    ckpt_path = checkpoint or manifest["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    enc, dec, is_hybrid = build_models(cfg["mode"], cfg["capacity_bits"], device,
                                       cfg.get("perturbation_bound", 0.1))
    dec.load_state_dict(ckpt["decoder_state"]); dec.eval()
    return dec, is_hybrid, cfg


def synth_photo(rgb01):
    """Synthesise a 'photo': mild perspective warp + JPEG + blur + brightness."""
    from stegaqr.realistic_distortion import real_jpeg, gaussian_blur, brightness
    h, w = rgb01.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    j = 0.04 * w
    dst = src + np.array([[j, j], [-j, j * 0.5], [-j * 0.5, -j], [j, -j * 0.5]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective((rgb01 * 255).astype(np.uint8), M, (w, h),
                               borderValue=(255, 255, 255))
    out = warp.astype(np.float32) / 255.0
    return real_jpeg(brightness(gaussian_blur(out, 0.8), 0.95), 70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--photos", help="dir of photos to decode")
    p.add_argument("--exports", help="export dir (for --self-test)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    manifest = json.loads(Path(args.manifest).read_text())
    dec, is_hybrid, cfg = load_model(manifest, device, args.checkpoint)
    ecc = get_ecc(manifest["ecc"])
    cap = manifest["capacity_bits"]; clen = ecc.coded_len(ecc.message_len(cap))
    module_count = 4 * manifest["qr_version"] + 17
    sym = module_count * manifest["module_size"]  # bare-symbol pixel size

    by_text = {it["public_text"]: it for it in manifest["items"]}

    n_loc = n_pub = n_msg = total = 0
    for it in manifest["items"]:
        total += 1
        if args.self_test:
            # build the exported (quiet-zoned, upscaled) image, then synth a photo
            arr = np.asarray(Image.open(Path(args.exports) / it["file"]).convert("RGB"))
            photo_rgb = synth_photo(arr.astype(np.float32) / 255.0)
            bgr = cv2.cvtColor((photo_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            fp = Path(args.photos) / it["file"]
            if not fp.exists():
                # allow arbitrary photo names; skip if absent
                continue
            bgr = cv2.imread(str(fp))

        rgb = locate_and_rectify(bgr, sym)
        if rgb is None:
            continue
        n_loc += 1
        # public decode (pyzbar) on the rectified symbol
        from pyzbar.pyzbar import decode as zbar
        zres = zbar(Image.fromarray((rgb * 255).astype(np.uint8)))
        if zres and zres[0].data.decode("utf-8", "replace") == it["public_text"]:
            n_pub += 1
        # hidden decode
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = (dec(t)[0] if is_hybrid else dec(t)).cpu().numpy()[0]
        dec_msg = ecc.decode(logits[:clen])
        if np.array_equal(dec_msg, np.array(it["message_bits"], dtype=np.uint8)):
            n_msg += 1

    print(f"items: {total}")
    print(f"QR located            : {n_loc}/{total}")
    print(f"public decode (pyzbar): {n_pub}/{total}")
    print(f"hidden MESSAGE decode : {n_msg}/{total}")


if __name__ == "__main__":
    main()
