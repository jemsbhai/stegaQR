"""Reusable evaluation for trained StegaQR models.

Loads a checkpoint and reports the full metric suite (chunked to stay within GPU
memory): hidden bit accuracy, full-decode rate, PSNR, SSIM, public QR decode rate,
and -- via the ECC layer -- message-level decode rate. Both clean and under the
stochastic distortion layer.

Used by scripts/evaluate.py (CLI) and scripts/run_experiments.py (matrix runner).

Placement of the ECC codeword on the grid is selectable (see
stegaqr.coding.placement_permutation). The harness default stays "native" so that
results.json files written before 0.2.0 (experiments/full) remain comparable; the
package API and CLI default to "interleaved". scripts/eval_ecc_layout.py measures
both placements on the same checkpoints.
"""

from __future__ import annotations

import random
import string

import numpy as np
import torch
from PIL import Image

from stegaqr.utils.seed import set_all_seeds
from stegaqr.utils.qr_utils import generate_cover_qr_rgb, get_qr_structure_mask
from stegaqr.utils.metrics import (
    bit_accuracy, full_decode_rate, psnr, ssim, qr_public_decode_rate,
    wilson_score_ci,
)
from stegaqr.coding import get_ecc, placement_permutation


def build_models(mode, capacity_bits, device, perturbation_bound,
                 arch="grid", mask_aware=False, qr_version=4, use_calibration=True):
    """Instantiate encoder/decoder for a mode. perturbation_bound MUST match the
    trained value (it scales the encoder output; it is not a learned weight).
    arch/mask_aware must also match the trained checkpoint."""
    if arch == "broadcast":
        from stegaqr.models.encoder import BroadcastEncoder
        from stegaqr.models.decoder import BroadcastDecoder
        return (BroadcastEncoder(capacity_bits, perturbation_bound=perturbation_bound).to(device),
                BroadcastDecoder(capacity_bits).to(device), False)
    if mode == "segregated":
        from stegaqr.models.encoder import SegregatedEncoder
        from stegaqr.models.decoder import SegregatedDecoder
        return (SegregatedEncoder(capacity_bits, perturbation_bound=perturbation_bound).to(device),
                SegregatedDecoder(capacity_bits).to(device), False)
    if mode == "cross_channel":
        from stegaqr.models.encoder import CrossChannelEncoder
        from stegaqr.models.decoder import CrossChannelDecoder
        return (CrossChannelEncoder(capacity_bits, perturbation_bound=perturbation_bound).to(device),
                CrossChannelDecoder(capacity_bits).to(device), False)
    if mode == "hybrid":
        from stegaqr.models.encoder import HybridEncoder
        from stegaqr.models.decoder import HybridDecoder
        return (HybridEncoder(capacity_bits, perturbation_bound=perturbation_bound,
                              mask_aware=mask_aware, qr_version=qr_version).to(device),
                HybridDecoder(capacity_bits, mask_aware=mask_aware, qr_version=qr_version,
                              use_calibration=use_calibration).to(device), True)
    raise ValueError(mode)


def _gen_samples(n, cap, qrv, ms, ec, quiet_zone, device, payload_bits=None):
    covers, payloads, texts = [], [], []
    mask = get_qr_structure_mask(qrv, ms)
    for i in range(n):
        text = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
        rgb, _ = generate_cover_qr_rgb(text, qrv, ec, ms, quiet_zone=quiet_zone)
        covers.append(torch.from_numpy(rgb).permute(2, 0, 1))
        if payload_bits is None:
            payloads.append(torch.from_numpy(np.random.randint(0, 2, cap).astype(np.float32)))
        else:
            payloads.append(torch.from_numpy(payload_bits[i].astype(np.float32)))
        texts.append(text)
    cover = torch.stack(covers).to(device)
    payload = torch.stack(payloads).to(device)
    # mask must match the (possibly quiet-zoned) cover size
    full_mask = get_qr_structure_mask(qrv, ms)
    if quiet_zone:
        full_mask = np.pad(full_mask, quiet_zone * ms, constant_values=0.0)
    mask_t = torch.from_numpy(full_mask).unsqueeze(0).unsqueeze(0).expand(n, -1, -1, -1).to(device)
    return cover, payload, mask_t, texts


def _forward(enc, dec, cover, payload, mask, is_hybrid, distortion, chunk=32):
    """Chunked forward; returns (logits_np, stego_np) for the whole batch."""
    logits_all, stego_all = [], []
    n = cover.shape[0]
    for s in range(0, n, chunk):
        c = cover[s:s + chunk]; p = payload[s:s + chunk]; m = mask[s:s + chunk]
        with torch.no_grad():
            stego = enc(c, p, m) if is_hybrid else enc(c, p)
            x = distortion(stego) if distortion is not None else stego
            lo = dec(x)[0] if is_hybrid else dec(x)
        logits_all.append(lo.cpu().numpy())
        stego_all.append(stego.permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(stego_all)


def evaluate_checkpoint(ckpt_path, n=128, eccs=("rep3", "rep5"), seed=1234,
                        device="cuda", chunk=32, quiet_zone=0,
                        placement="native", placement_seed=0):
    """Evaluate a checkpoint and return a nested metrics dict.

    placement: grid placement of the ECC codeword for the message-decode rows
    ('native' keeps pre-0.2.0 results comparable; 'interleaved' is the package default).
    """
    device = device if torch.cuda.is_available() else "cpu"
    set_all_seeds(seed)
    torch.backends.cudnn.benchmark = True

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mode, cap = cfg["mode"], cfg["capacity_bits"]
    qrv, ms, ec = cfg["qr_version"], cfg["module_size"], cfg["ec_level"]
    pbound = cfg.get("perturbation_bound", 0.1)

    enc, dec, is_hybrid = build_models(mode, cap, device, pbound,
                                       arch=cfg.get("arch", "grid"),
                                       mask_aware=cfg.get("mask_aware", False),
                                       qr_version=qrv,
                                       use_calibration=cfg.get("use_calibration", True))
    enc.load_state_dict(ckpt["encoder_state"]); dec.load_state_dict(ckpt["decoder_state"])
    enc.eval(); dec.eval()

    from stegaqr.models.distortion import DifferentiableDistortion
    dist = DifferentiableDistortion().to(device)

    out = {"config": {"mode": mode, "capacity_bits": cap, "module_size": ms,
                      "perturbation_bound": pbound, "seed_train": cfg.get("seed"),
                      "use_distortion_train": cfg.get("use_distortion"),
                      "lambda_perceptual": cfg.get("lambda_perceptual"),
                      "checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch")}}

    # raw payload (random cap bits)
    cover, payload, mask, texts = _gen_samples(n, cap, qrv, ms, ec, quiet_zone, device)
    gt = payload.cpu().numpy().astype(np.uint8)
    cover_np = cover.permute(0, 2, 3, 1).cpu().numpy()

    for setting, distortion in (("clean", None), ("distorted", dist)):
        logits, stego_np = _forward(enc, dec, cover, payload, mask, is_hybrid, distortion, chunk)
        pred = (logits > 0).astype(np.uint8)
        ba = bit_accuracy(pred, gt)
        fdr = full_decode_rate(pred, gt)
        lo, hi = wilson_score_ci(int(round(fdr * n)), n)
        psnrs = [psnr(cover_np[i], stego_np[i]) for i in range(n)]
        ssims = [ssim(cover_np[i], stego_np[i]) for i in range(n)]
        imgs = [Image.fromarray((stego_np[i] * 255).astype(np.uint8)) for i in range(n)]
        pub = qr_public_decode_rate(imgs, texts)
        out[setting] = {
            "bit_acc": ba, "full_decode": fdr, "full_decode_ci": [lo, hi],
            "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
            "public_decode": pub,
        }

    # ECC message-decode (under distortion -- the regime where it matters)
    pos = placement_permutation(cap, placement, placement_seed)
    out["ecc_placement"] = {"placement": placement, "seed": int(placement_seed)}
    out["ecc"] = {}
    for ecc_name in eccs:
        ecc = get_ecc(ecc_name)
        k = ecc.message_len(cap)
        clen = ecc.coded_len(k)
        if k <= 0:
            continue
        msgs = np.random.randint(0, 2, size=(n, k)).astype(np.uint8)
        pbits = np.zeros((n, cap), dtype=np.uint8)
        for i in range(n):
            pbits[i, pos[:clen]] = ecc.encode(msgs[i])
        c2, p2, m2, _ = _gen_samples(n, cap, qrv, ms, ec, quiet_zone, device, payload_bits=pbits)
        logits, _ = _forward(enc, dec, c2, p2, m2, is_hybrid, dist, chunk)
        dec_msgs = np.stack([ecc.decode(logits[i, pos[:clen]]) for i in range(n)])
        mfull = float(np.mean(np.all(dec_msgs == msgs, axis=1)))
        mlo, mhi = wilson_score_ci(int(round(mfull * n)), n)
        out["ecc"][ecc_name] = {
            "net_bits": int(k), "rate": ecc.rate,
            "msg_bit_acc": float(np.mean(dec_msgs == msgs)),
            "msg_decode": mfull, "msg_decode_ci": [mlo, mhi],
        }

    return out


def evaluate_real_distortions(ckpt_path, n=128, presets=None, ecc_name="rep3",
                              seed=1234, device="cuda", chunk=32, quiet_zone=0,
                              placement="native", placement_seed=0):
    """Evaluate robustness under REAL (non-differentiable) distortion presets.

    For each preset, applies the real operator (true JPEG, blur, resize, etc.) to
    the clean stego images, then decodes. Returns per-preset raw bit/full-decode and
    ECC message-decode. This is the credible robustness measurement (the training
    distortion is a differentiable approximation). placement as in evaluate_checkpoint.
    """
    from stegaqr.realistic_distortion import make_presets

    device = device if torch.cuda.is_available() else "cpu"
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    presets = presets or list(make_presets(rng).keys())
    preset_fns = make_presets(rng)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mode, cap = cfg["mode"], cfg["capacity_bits"]
    qrv, ms, ec = cfg["qr_version"], cfg["module_size"], cfg["ec_level"]
    enc, dec, is_hybrid = build_models(mode, cap, device, cfg.get("perturbation_bound", 0.1),
                                       arch=cfg.get("arch", "grid"),
                                       mask_aware=cfg.get("mask_aware", False), qr_version=qrv,
                                       use_calibration=cfg.get("use_calibration", True))
    enc.load_state_dict(ckpt["encoder_state"]); dec.load_state_dict(ckpt["decoder_state"])
    enc.eval(); dec.eval()

    ecc = get_ecc(ecc_name)
    k = ecc.message_len(cap); clen = ecc.coded_len(k)
    pos = placement_permutation(cap, placement, placement_seed)
    msgs = np.random.randint(0, 2, size=(n, k)).astype(np.uint8)
    pbits = np.zeros((n, cap), dtype=np.uint8)
    for i in range(n):
        pbits[i, pos[:clen]] = ecc.encode(msgs[i])
    cover, payload, mask, _ = _gen_samples(n, cap, qrv, ms, ec, quiet_zone, device, payload_bits=pbits)

    # clean stego (numpy) once
    _, stego_np = _forward(enc, dec, cover, payload, mask, is_hybrid, None, chunk)
    gt = payload.cpu().numpy().astype(np.uint8)

    results = {"config": {"mode": mode, "capacity_bits": cap, "ecc": ecc_name,
                          "net_bits": int(k), "checkpoint": str(ckpt_path),
                          "n": int(n), "seed_eval": int(seed),
                          "placement": placement, "placement_seed": int(placement_seed)},
               "presets": {}}
    for name in presets:
        fn = preset_fns[name]
        dist_np = np.stack([fn(stego_np[i]) for i in range(n)])      # (n,H,W,3)
        t = torch.from_numpy(dist_np).permute(0, 3, 1, 2).float().to(device)
        logits = []
        for s in range(0, n, chunk):
            with torch.no_grad():
                lo = dec(t[s:s + chunk])[0] if is_hybrid else dec(t[s:s + chunk])
            logits.append(lo.cpu().numpy())
        logits = np.concatenate(logits)
        pred = (logits > 0).astype(np.uint8)
        ba = float(np.mean(pred == gt))
        fdr = float(np.mean(np.all(pred == gt, axis=1)))
        dec_msgs = np.stack([ecc.decode(logits[i, pos[:clen]]) for i in range(n)])
        mfull = float(np.mean(np.all(dec_msgs == msgs, axis=1)))
        lo, hi = wilson_score_ci(int(round(mfull * n)), n)
        results["presets"][name] = {
            "bit_acc": ba, "full_decode": fdr,
            "msg_decode": mfull, "msg_decode_ci": [lo, hi],
        }
    return results
