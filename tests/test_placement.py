"""Tests for the 0.2.0 grid placement of the ECC codeword (interleaved vs native)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stegaqr.coding import (  # noqa: E402
    DEFAULT_PLACEMENT, PLACEMENTS, gather_coded_logits, get_ecc, place_coded_bits,
    placement_permutation,
)


def test_default_is_interleaved():
    assert DEFAULT_PLACEMENT == "interleaved"
    assert set(PLACEMENTS) == {"interleaved", "native"}


def test_native_is_identity():
    assert np.array_equal(placement_permutation(100, "native"), np.arange(100))


def test_interleaved_is_a_deterministic_permutation():
    a = placement_permutation(100, "interleaved", 0)
    b = placement_permutation(100, "interleaved", 0)
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == list(range(100))
    assert not np.array_equal(a, np.arange(100))
    assert not np.array_equal(a, placement_permutation(100, "interleaved", 1))


def test_interleaved_matches_eval_ecc_layout_construction():
    """scripts/eval_ecc_layout.py used np.random.default_rng(0).permutation(cap); the
    package must reproduce it so the paper's interleaved numbers describe the release."""
    assert np.array_equal(placement_permutation(100, "interleaved", 0),
                          np.random.default_rng(0).permutation(100))


def test_invalid_placement_raises():
    with pytest.raises(ValueError):
        placement_permutation(100, "diagonal")
    with pytest.raises(ValueError):
        placement_permutation(0, "native")


def test_place_and_gather_round_trip_every_code():
    cap = 100
    for placement in PLACEMENTS:
        pos = placement_permutation(cap, placement, 0)
        for code in ("none", "rep2", "rep3", "rep4", "rep5", "hamming74"):
            ecc = get_ecc(code)
            k = ecc.message_len(cap)
            msg = np.random.default_rng(3).integers(0, 2, size=k, dtype=np.uint8)
            coded = ecc.encode(msg)
            payload = place_coded_bits(coded, cap, pos)
            assert payload.shape == (cap,) and payload.dtype == np.float32
            assert int(payload.sum()) == int(coded.sum())          # nothing lost, nothing added
            logits = payload * 2.0 - 1.0                              # perfect channel
            back = gather_coded_logits(logits, ecc.coded_len(k), pos)
            assert np.array_equal((back > 0).astype(np.uint8), coded)
            assert np.array_equal(ecc.decode(back), msg)


def test_native_rep5_shares_a_column_and_interleaved_does_not():
    """The mechanism behind the 0.2.0 default, on the 10 x 10 grid of a 100-bit model."""
    cap, gw = 100, 10
    ecc = get_ecc("rep5")
    k, m = ecc.message_len(cap), ecc.repeat
    for placement, expect_single_column in (("native", True), ("interleaved", False)):
        pos = placement_permutation(cap, placement, 0)
        cols = [{int(pos[r * k + j] % gw) for r in range(m)} for j in range(k)]
        single = all(len(c) == 1 for c in cols)
        assert single == expect_single_column


def test_place_rejects_overflow():
    with pytest.raises(ValueError):
        place_coded_bits(np.ones(101, np.uint8), 100, placement_permutation(100, "native"))


def test_version_matches_pyproject():
    import stegaqr
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == stegaqr.__version__ == "0.2.0"


@pytest.mark.slow
class TestBundledModelPlacement:
    def test_interleaved_round_trip_is_default(self):
        from stegaqr.core import decode_hidden, encode_hidden
        img = encode_hidden("https://x.com", b"ID42", device="cpu")
        public, hidden, meta = decode_hidden(img, device="cpu")
        assert public == "https://x.com"
        assert hidden.rstrip(b"\x00") == b"ID42"
        assert meta["placement"] == "interleaved" and meta["placement_seed"] == 0

    def test_native_round_trip_reads_0_1_0_layout(self):
        from stegaqr.core import decode_hidden, encode_hidden
        img = encode_hidden("https://x.com", b"ID42", device="cpu", placement="native")
        public, hidden, meta = decode_hidden(img, device="cpu", placement="native")
        assert public == "https://x.com"
        assert hidden.rstrip(b"\x00") == b"ID42"
        assert meta["placement"] == "native"

    def test_placement_mismatch_does_not_recover_the_message(self):
        from stegaqr.core import decode_hidden, encode_hidden
        img = encode_hidden("https://x.com", b"ID42", device="cpu", placement="interleaved")
        _, hidden, _ = decode_hidden(img, device="cpu", placement="native")
        assert hidden.rstrip(b"\x00") != b"ID42"

    def test_cli_round_trip(self, tmp_path, capsys):
        from stegaqr.cli import main
        out = tmp_path / "stego.png"
        main(["encode", "--public", "PUB1", "--message", "ID42", "--out", str(out), "--device", "cpu"])
        main(["decode", "--image", str(out), "--device", "cpu"])
        text = capsys.readouterr().out
        assert "public payload : 'PUB1'" in text
        assert "hidden payload : 'ID42'" in text
        assert "placement      : interleaved" in text
        main(["info", "--device", "cpu"])
        assert "placement       : interleaved (seed 0)" in capsys.readouterr().out
