"""Evaluate a trained StegaQR model.

Loads a best_model.pt produced by scripts/train.py, generates fresh test samples,
and reports the metrics that matter for the paper:
  - hidden bit accuracy and full-decode rate (zero bit errors)
  - PSNR / SSIM of stego vs. cover (imperceptibility)
  - public QR decode rate via pyzbar (the cover payload must still scan)
  - optionally, all of the above under the distortion layer (robustness)

Usage:
    python scripts/evaluate.py --checkpoint experiments/exp_xxx/best_model.pt
    python scripts/evaluate.py --checkpoint .../best_model.pt --distortion --n 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from stegaqr.utils.seed import set_all_seeds
from stegaqr.utils.qr_utils import generate_cover_qr_rgb, get_qr_structure_mask
from stegaqr.utils.metrics import (
    bit_accuracy, full_decode_rate, psnr, ssim, qr_public_decode_rate,
    wilson_score_ci,
)


def build_models(mode, capacity_bits, device, perturbation_bound):
    # perturbation_bound is NOT a learned parameter — it scales the encoder output
    # in forward(). It MUST match the value used at training time or the loaded
    # weights produce a wrongly-scaled perturbation.
    if mode == "segregated":
        from stegaqr.models.encoder import SegregatedEncoder
        from stegaqr.models.decoder import SegregatedDecoder
        return SegregatedEncoder(capacity_bits, perturbation_bound=perturbation_bound).to(device), SegregatedDecoder(capacity_bits).to(device), False
    if mode == "cross_channel":
        from stegaqr.models.encoder import CrossChannelEncoder
        from stegaqr.models.decoder import CrossChannelDecoder
        return CrossChannelEncoder(capacity_bits, perturbation_bound=perturbation_bound).to(device), CrossChannelDecoder(capacity_bits).to(device), False
    if mode == "hybrid":
        from stegaqr.models.encoder import HybridEncoder
        from stegaqr.models.decoder import HybridDecoder
        return HybridEncoder(capacity_bits, perturbation_bound=perturbation_bound).to(device), HybridDecoder(capacity_bits).to(device), True
    raise ValueError(mode)


def make_test_batch(n, capacity_bits, qr_version, module_size, ec_level, device, payload_bits=None):
    """payload_bits: optional (n, capacity_bits) array to embed; random if None."""
    import random, string
    covers, payloads, texts = [], [], []
    mask = get_qr_structure_mask(qr_version, module_size)
    for i in range(n):
        text = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
        rgb, _ = generate_cover_qr_rgb(text, qr_version, ec_level, module_size)
        covers.append(torch.from_numpy(rgb).permute(2, 0, 1))
        if payload_bits is None:
            payloads.append(torch.from_numpy(np.random.randint(0, 2, capacity_bits).astype(np.float32)))
        else:
            payloads.append(torch.from_numpy(payload_bits[i].astype(np.float32)))
        texts.append(text)
    cover = torch.stack(covers).to(device)
    payload = torch.stack(payloads).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1).to(device)
    return cover, payload, mask_t, texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--distortion", action="store_true")
    p.add_argument("--ecc", default="none",
                   help="error-correction code: none | rep3 | rep5 | hamming74")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    set_all_seeds(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mode = cfg["mode"]; cap = cfg["capacity_bits"]
    qrv = cfg["qr_version"]; ms = cfg["module_size"]; ec = cfg["ec_level"]
    print(f"checkpoint: {args.checkpoint}")
    print(f"mode={mode} capacity={cap} qr_v={qrv} module_size={ms} ec={ec} "
          f"(trained val_acc={ckpt.get('val_acc')})")

    pbound = cfg.get("perturbation_bound", 0.1)
    enc, dec, is_hybrid = build_models(mode, cap, device, pbound)
    enc.load_state_dict(ckpt["encoder_state"]); dec.load_state_dict(ckpt["decoder_state"])
    enc.eval(); dec.eval()

    distortion = None
    if args.distortion:
        from stegaqr.models.distortion import DifferentiableDistortion
        distortion = DifferentiableDistortion().to(device)

    # --- ECC: build coded payloads from random messages ---
    from stegaqr.coding import get_ecc
    ecc = get_ecc(args.ecc)
    k = ecc.message_len(cap)            # net message bits
    coded_len = ecc.coded_len(k)        # <= cap
    messages = None
    payload_bits = None
    if ecc.name != "none":
        messages = np.random.randint(0, 2, size=(args.n, k)).astype(np.uint8)
        payload_bits = np.zeros((args.n, cap), dtype=np.uint8)
        for i in range(args.n):
            payload_bits[i, :coded_len] = ecc.encode(messages[i])

    cover, payload, mask, texts = make_test_batch(
        args.n, cap, qrv, ms, ec, device, payload_bits=payload_bits)

    with torch.no_grad():
        stego = enc(cover, payload, mask) if is_hybrid else enc(cover, payload)
        dec_in = distortion(stego) if distortion is not None else stego
        if is_hybrid:
            logits, conf = dec(dec_in)
        else:
            logits, conf = dec(dec_in), None
        pred = (torch.sigmoid(logits) > 0.5).float()

    logits_np = logits.cpu().numpy()
    pred_np = pred.cpu().numpy().astype(np.uint8)
    gt_np = payload.cpu().numpy().astype(np.uint8)
    stego_np = stego.permute(0, 2, 3, 1).cpu().numpy()
    cover_np = cover.permute(0, 2, 3, 1).cpu().numpy()

    # raw (uncoded) bit-level metrics
    ba = bit_accuracy(pred_np, gt_np)
    fdr = full_decode_rate(pred_np, gt_np)
    n_full = int(round(fdr * args.n))
    fdr_lo, fdr_hi = wilson_score_ci(n_full, args.n)
    psnrs = [psnr(cover_np[i], stego_np[i]) for i in range(args.n)]
    ssims = [ssim(cover_np[i], stego_np[i]) for i in range(args.n)]

    # public QR decodability on the (clean) stego images
    stego_imgs = [Image.fromarray((stego_np[i] * 255).astype(np.uint8)) for i in range(args.n)]
    pub = qr_public_decode_rate(stego_imgs, texts)

    print(f"\n--- {'DISTORTED' if args.distortion else 'CLEAN'} (n={args.n}) | ecc={ecc.name} ---")
    print(f"raw bit accuracy     : {ba*100:.2f}%   (uncoded, {cap} embedded bits)")
    print(f"raw full-decode rate : {fdr*100:.2f}%  (95% CI {fdr_lo*100:.1f}-{fdr_hi*100:.1f})")
    print(f"PSNR (stego vs cover): {np.mean(psnrs):.2f} dB")
    print(f"SSIM (stego vs cover): {np.mean(ssims):.4f}")
    print(f"public QR decode rate: {pub*100:.2f}%")
    if conf is not None:
        print(f"mean confidence     : {conf.mean().item():.4f}")

    if ecc.name != "none":
        # soft-decision ECC decode from logits -> message bits
        dec_msgs = np.stack([ecc.decode(logits_np[i, :coded_len]) for i in range(args.n)])
        msg_bit_acc = float(np.mean(dec_msgs == messages))
        msg_full = float(np.mean(np.all(dec_msgs == messages, axis=1)))
        mf = int(round(msg_full * args.n))
        m_lo, m_hi = wilson_score_ci(mf, args.n)
        print(f"--- after ECC ({ecc.name}, rate {ecc.rate:.2f}, {k} net message bits) ---")
        print(f"message bit accuracy : {msg_bit_acc*100:.2f}%")
        print(f"MESSAGE decode rate  : {msg_full*100:.2f}%  (95% CI {m_lo*100:.1f}-{m_hi*100:.1f})")


if __name__ == "__main__":
    main()
