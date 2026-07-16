"""Tests for the public core API (encode/decode) and CLI plumbing."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from stegaqr.evaluation import build_models


def _make_checkpoint(tmp_path, mode="cross_channel", cap=99, ms=4, version=4):
    """Build an (untrained) checkpoint on disk for API smoke tests."""
    enc, dec, _ = build_models(mode, cap, "cpu", 0.3)
    ckpt = {
        "encoder_state": enc.state_dict(), "decoder_state": dec.state_dict(),
        "config": {"mode": mode, "capacity_bits": cap, "qr_version": version,
                   "ec_level": "M", "module_size": ms, "perturbation_bound": 0.3,
                   "arch": "grid", "mask_aware": False, "use_calibration": True},
        "epoch": 0,
    }
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return str(p)


class TestCoreAPISmoke:
    def test_encode_returns_rgb_image(self, tmp_path):
        from stegaqr.core import StegaQREncoder
        enc = StegaQREncoder(_make_checkpoint(tmp_path), ecc="rep3", device="cpu")
        img = enc.encode("HELLO", b"hi")
        assert img.mode == "RGB"
        sym = (4 * 4 + 17) * 4
        assert img.size == (sym, sym)

    def test_max_hidden_bytes(self, tmp_path):
        from stegaqr.core import StegaQREncoder
        enc = StegaQREncoder(_make_checkpoint(tmp_path), ecc="rep3", device="cpu")
        # cap 99 -> rep3 message_len 33 bits -> 4 bytes
        assert enc.max_hidden_bytes == 33 // 8

    def test_payload_too_long_raises(self, tmp_path):
        from stegaqr.core import StegaQREncoder
        enc = StegaQREncoder(_make_checkpoint(tmp_path), ecc="rep3", device="cpu")
        with pytest.raises(ValueError):
            enc.encode("HELLO", b"way too many bytes for this tiny capacity")

    def test_decode_returns_types(self, tmp_path):
        from stegaqr.core import StegaQREncoder, StegaQRDecoder
        ck = _make_checkpoint(tmp_path)
        img = StegaQREncoder(ck, ecc="rep3", device="cpu").encode("HELLO", b"hi")
        public, hidden, meta = StegaQRDecoder(ck, ecc="rep3", device="cpu").decode(img)
        assert public is None or isinstance(public, str)
        assert isinstance(hidden, (bytes, bytearray))
        assert meta["mode"] == "cross_channel" and meta["ecc"] == "rep3"

    def test_public_qr_scans_bundled_model(self):
        """The release model must preserve the standard QR payload."""
        from stegaqr.core import decode_hidden, encode_hidden
        img = encode_hidden("PUBLIC123", b"hi", ecc="rep3", device="cpu")
        public, _, _ = decode_hidden(img, ecc="rep3", device="cpu")
        assert public == "PUBLIC123"


@pytest.mark.slow
def test_roundtrip_pretrained():
    """Full hidden round-trip with the model bundled in the package."""
    from stegaqr.core import default_model_path, encode_hidden, decode_hidden
    assert default_model_path().is_file()
    img = encode_hidden("https://x.com", b"ID42", ecc="rep3", device="cpu")
    public, hidden, _ = decode_hidden(img, ecc="rep3", device="cpu")
    assert public == "https://x.com"
    assert hidden.rstrip(b"\x00") == b"ID42"
