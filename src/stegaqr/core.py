"""Core API for StegaQR encoding and decoding.

Public interface for the library/CLI. Loads a trained checkpoint and embeds /
recovers a hidden payload inside a standard-looking QR code:

    from stegaqr import encode_hidden, decode_hidden
    img = encode_hidden("https://example.com", b"secret", model="models/pretrained/stegaqr_default.pt")
    public, hidden, meta = decode_hidden(img, model="models/pretrained/stegaqr_default.pt")

A standard QR reader decodes `public` normally; only the StegaQR decoder recovers
`hidden`. The hidden payload is protected by an error-correcting code (ECC), so the
usable hidden capacity is smaller than the model's raw bit capacity (see
`StegaQREncoder.max_hidden_bytes`).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from stegaqr.coding import get_ecc, bits_to_bytes, bytes_to_bits
from stegaqr.evaluation import build_models
from stegaqr.utils.qr_utils import generate_cover_qr_rgb, get_qr_structure_mask


class StegoMode(Enum):
    """Steganographic embedding mode (must match the trained checkpoint)."""

    SEGREGATED = "segregated"
    CROSS_CHANNEL = "cross_channel"
    HYBRID = "hybrid"


def default_model_path() -> Path:
    """Return the trained model bundled with the installed package."""
    path = Path(__file__).resolve().parent / "assets" / "stegaqr_default.pt"
    if not path.is_file():
        raise FileNotFoundError(
            "the bundled StegaQR model is missing; reinstall stegaQR or pass model=PATH"
        )
    return path


def _load(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    enc, dec, is_hybrid = build_models(
        cfg["mode"], cfg["capacity_bits"], device, cfg.get("perturbation_bound", 0.1),
        arch=cfg.get("arch", "grid"), mask_aware=cfg.get("mask_aware", False),
        qr_version=cfg["qr_version"], use_calibration=cfg.get("use_calibration", True),
    )
    enc.load_state_dict(ckpt["encoder_state"])
    dec.load_state_dict(ckpt["decoder_state"])
    enc.eval(); dec.eval()
    return enc, dec, is_hybrid, cfg


class StegaQREncoder:
    """Encodes hidden data into a QR code image using a trained model + ECC.

    Parameters
    ----------
    model : str or pathlib.Path, optional
        Path to a checkpoint produced by training. Uses the bundled model by default.
    ecc : str
        Error-correcting code: 'rep3' | 'rep5' | 'hamming74' | 'none'.
    device : str
        'cuda' or 'cpu' (falls back to CPU if CUDA unavailable).
    """

    def __init__(
        self, model: str | Path | None = None, ecc: str = "rep3", device: str = "cuda"
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model) if model is not None else default_model_path()
        self.encoder, _, self.is_hybrid, self.cfg = _load(self.model_path, self.device)
        self.ecc = get_ecc(ecc)
        self.capacity = self.cfg["capacity_bits"]
        self._k = self.ecc.message_len(self.capacity)  # usable message bits

    @property
    def max_hidden_bytes(self) -> int:
        """Maximum hidden payload size in whole bytes for this model + ECC."""
        return self._k // 8

    def encode(self, public_payload: str, hidden_payload: bytes) -> Image.Image:
        """Embed public + hidden payloads into a single stego QR image (RGB).

        Returns the bare-symbol stego at the model's native resolution; a standard
        reader decodes `public_payload`, the StegaQR decoder recovers `hidden_payload`.
        """
        cfg = self.cfg
        if len(hidden_payload) > self.max_hidden_bytes:
            raise ValueError(
                f"hidden payload too long: {len(hidden_payload)} bytes > "
                f"{self.max_hidden_bytes} (model capacity {self.capacity} bits, "
                f"ECC {self.ecc.name})")

        cover, _ = generate_cover_qr_rgb(
            public_payload, cfg["qr_version"], cfg["ec_level"], cfg["module_size"])
        # message bytes -> bits -> ECC codeword -> pad to capacity
        msg_bits = bytes_to_bits(hidden_payload, n_bits=self._k)
        if len(msg_bits) < self._k:
            msg_bits = np.concatenate([msg_bits, np.zeros(self._k - len(msg_bits), np.uint8)])
        coded = np.zeros(self.capacity, dtype=np.float32)
        coded[: self.ecc.coded_len(self._k)] = self.ecc.encode(msg_bits)

        c = torch.from_numpy(cover).permute(2, 0, 1).unsqueeze(0).to(self.device)
        p = torch.from_numpy(coded).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.is_hybrid:
                mask = torch.from_numpy(
                    get_qr_structure_mask(cfg["qr_version"], cfg["module_size"])
                ).unsqueeze(0).unsqueeze(0).to(self.device)
                stego = self.encoder(c, p, mask)
            else:
                stego = self.encoder(c, p)
        arr = (np.clip(stego.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr)


class StegaQRDecoder:
    """Recovers hidden data from a stego QR image (clean / re-scanned).

    For photographs use scripts/decode_from_photo.py (perspective rectification);
    this class targets clean digital images and resizes to the model resolution.

    The bundled model is used unless ``model`` points to another checkpoint.
    """

    def __init__(
        self, model: str | Path | None = None, ecc: str = "rep3", device: str = "cuda"
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model) if model is not None else default_model_path()
        _, self.decoder, self.is_hybrid, self.cfg = _load(self.model_path, self.device)
        self.ecc = get_ecc(ecc)
        self.capacity = self.cfg["capacity_bits"]
        self._k = self.ecc.message_len(self.capacity)

    @staticmethod
    def _rectify(img: Image.Image, sym: int) -> Image.Image:
        """Locate + perspective-rectify the QR symbol to (sym, sym); resize on fail.
        Handles quiet zones, upscaling and mild perspective in saved/scanned images."""
        try:
            import cv2
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            ok, pts = cv2.QRCodeDetector().detect(bgr)
            if ok and pts is not None:
                pts = pts.reshape(4, 2).astype(np.float32)
                dst = np.array([[0, 0], [sym - 1, 0], [sym - 1, sym - 1], [0, sym - 1]],
                               dtype=np.float32)
                M = cv2.getPerspectiveTransform(pts, dst)
                warp = cv2.warpPerspective(bgr, M, (sym, sym))
                return Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
        return img.resize((sym, sym), Image.BILINEAR)

    def decode(self, image: Image.Image) -> tuple[Optional[str], Optional[bytes], dict]:
        """Return (public_payload, hidden_payload, metadata).

        public via a standard reader (pyzbar); hidden via the neural decoder + ECC.
        """
        cfg = self.cfg
        sym = (4 * cfg["qr_version"] + 17) * cfg["module_size"]
        img = image.convert("RGB")
        if img.size != (sym, sym):
            img = self._rectify(img, sym)

        # public payload via standard reader
        public = None
        try:
            from pyzbar.pyzbar import decode as zbar
            res = zbar(img)
            if res:
                public = res[0].data.decode("utf-8", "replace")
        except Exception:
            pass

        arr = np.asarray(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.decoder(t)
            logits = (out[0] if isinstance(out, tuple) else out).cpu().numpy()[0]
            conf = float(torch.sigmoid(out[1]).mean()) if isinstance(out, tuple) else None
        msg_bits = self.ecc.decode(logits[: self.ecc.coded_len(self._k)])
        hidden = bits_to_bytes(msg_bits)
        meta = {"mode": cfg["mode"], "capacity_bits": self.capacity, "ecc": self.ecc.name,
                "message_bits": int(self._k), "confidence": conf}
        return public, hidden, meta


def encode_hidden(public_payload: str, hidden_payload: bytes, *, model: str | Path | None = None,
                  ecc: str = "rep3", device: str = "cuda") -> Image.Image:
    """One-shot encode. See StegaQREncoder."""
    return StegaQREncoder(model, ecc=ecc, device=device).encode(public_payload, hidden_payload)


def decode_hidden(image: Image.Image, *, model: str | Path | None = None, ecc: str = "rep3",
                  device: str = "cuda") -> tuple[Optional[str], Optional[bytes], dict]:
    """One-shot decode. See StegaQRDecoder."""
    return StegaQRDecoder(model, ecc=ecc, device=device).decode(image)
