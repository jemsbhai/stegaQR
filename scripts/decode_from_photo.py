"""Decode stego QR codes from real photographs (print-scan / photo robustness).

For each photo: locate the QR symbol (OpenCV detector), perspective-rectify it to
the model's canonical resolution, run the neural decoder + ECC to recover the hidden
message, and run pyzbar for the public payload. Results are matched to the export
manifest (by public text) and scored.

Usage:
    python scripts/decode_from_photo.py --photos capture/exp1_photos --manifest capture/exp1/manifest.json
    # quick self-test (no camera): synthesise "photos" via warp+JPEG from the exports
    python scripts/decode_from_photo.py --self-test --manifest capture/exp1/manifest.json --exports capture/exp1

The grid placement of the coded bits is taken from the manifest (written by
export_for_capture.py since 0.2.0); manifests without the key are read as native,
which is how every export before 0.2.0 was produced. --placement overrides.
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

from stegaqr.coding import PLACEMENTS, get_ecc, placement_permutation
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
    enc, dec, is_hybrid = build_models(
        cfg["mode"], cfg["capacity_bits"], device, cfg.get("perturbation_bound", 0.1),
        arch=cfg.get("arch", "grid"), mask_aware=cfg.get("mask_aware", False),
        qr_version=cfg["qr_version"], use_calibration=cfg.get("use_calibration", True))
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
    p.add_argument("--placement", default=None, choices=PLACEMENTS,
                   help="override the manifest placement (pre-0.2.0 manifests: native)")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    manifest = json.loads(Path(args.manifest).read_text())
    dec, is_hybrid, cfg = load_model(manifest, device, args.checkpoint)
    ecc = get_ecc(manifest["ecc"])
    cap = manifest["capacity_bits"]; clen = ecc.coded_len(ecc.message_len(cap))
    placement = args.placement or manifest.get("placement", "native")
    pos = placement_permutation(cap, placement, manifest.get("placement_seed", 0))
    print(f"placement: {placement}")
    module_count = 4 * manifest["qr_version"] + 17
    sym = module_count * manifest["module_size"]  # bare-symbol pixel size

    from pyzbar.pyzbar import decode as zbar
    from PIL import ImageOps
    by_text = {it["public_text"]: it for it in manifest["items"]}

    def hidden_ok(logits, item):
        return np.array_equal(ecc.decode(logits[pos[:clen]]),
                              np.array(item["message_bits"], dtype=np.uint8))

    def score_bgr(bgr, item):
        """Rectify + decode one image against a known manifest item; returns flags."""
        rgb = locate_and_rectify(bgr, sym)
        if rgb is None:
            return False, False, False
        zres = zbar(Image.fromarray((rgb * 255).astype(np.uint8)))
        pub_ok = bool(zres) and zres[0].data.decode("utf-8", "replace") == item["public_text"]
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = (dec(t)[0] if is_hybrid else dec(t)).cpu().numpy()[0]
        return True, pub_ok, hidden_ok(logits, item)

    n_loc = n_pub = n_msg = total = 0

    if args.self_test:
        for it in manifest["items"]:
            total += 1
            arr = np.asarray(Image.open(Path(args.exports) / it["file"]).convert("RGB"))
            photo_rgb = synth_photo(arr.astype(np.float32) / 255.0)
            bgr = cv2.cvtColor((photo_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            loc, pub, msg = score_bgr(bgr, it)
            n_loc += loc; n_pub += pub; n_msg += msg
    else:
        # Arbitrary phone-photo filenames: read each, EXIF-rotate, find its QR, match to
        # the manifest by the scanned PUBLIC text, then score the hidden message.
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        photos = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in exts)
        print(f"found {len(photos)} photos in {args.photos}")
        for fp in photos:
            total += 1
            try:
                pil = ImageOps.exif_transpose(Image.open(fp).convert("RGB"))
            except Exception:
                print(f"  {fp.name}: unreadable"); continue
            bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
            rect = locate_and_rectify(bgr, sym)
            if rect is None:
                print(f"  {fp.name}: no QR located"); continue
            n_loc += 1
            zres = zbar(Image.fromarray((rect * 255).astype(np.uint8)))
            pub_text = zres[0].data.decode("utf-8", "replace") if zres else None
            item = by_text.get(pub_text)
            if item is None:
                print(f"  {fp.name}: public={pub_text!r} (no manifest match)"); continue
            n_pub += 1
            t = torch.from_numpy(rect).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = (dec(t)[0] if is_hybrid else dec(t)).cpu().numpy()[0]
            msg_ok = hidden_ok(logits, item)
            n_msg += msg_ok
            print(f"  {fp.name}: public={pub_text!r}  hidden={'OK' if msg_ok else 'FAIL'}")

    print(f"\n=== {total} images ===")
    print(f"QR located            : {n_loc}/{total}")
    print(f"public decode (pyzbar): {n_pub}/{total}")
    print(f"hidden MESSAGE decode : {n_msg}/{total}")


if __name__ == "__main__":
    main()
