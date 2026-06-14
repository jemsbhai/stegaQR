"""Tests for error-correction coding of the hidden payload."""

import numpy as np
import pytest

from stegaqr.coding import (
    IdentityECC, RepetitionECC, HammingECC, get_ecc,
    bits_to_bytes, bytes_to_bits,
)


def _logits_from_bits(bits, conf=4.0):
    """Map 0/1 bits to logits (+conf for 1, -conf for 0)."""
    return (np.asarray(bits, dtype=np.float64) * 2 - 1) * conf


class TestRoundTrip:
    @pytest.mark.parametrize("ecc", [IdentityECC(), RepetitionECC(3), RepetitionECC(5), HammingECC()])
    def test_clean_roundtrip(self, ecc):
        rng = np.random.default_rng(0)
        k = ecc.message_len(105)  # 105 divisible by 3,5,7
        msg = rng.integers(0, 2, k).astype(np.uint8)
        coded = ecc.encode(msg)
        decoded = ecc.decode(_logits_from_bits(coded))
        assert np.array_equal(decoded[: len(msg)], msg)

    def test_coded_len_matches_encode(self):
        for ecc in [RepetitionECC(3), RepetitionECC(5), HammingECC()]:
            k = ecc.message_len(105)
            assert len(ecc.encode(np.zeros(k, np.uint8))) == ecc.coded_len(k)


class TestErrorCorrection:
    def test_repetition_soft_corrects_flips(self):
        """rep3 soft decoding corrects a minority of flipped copies."""
        ecc = RepetitionECC(3)
        msg = np.array([1, 0, 1, 1, 0], np.uint8)
        coded = ecc.encode(msg)               # 15 bits, 3 copies interleaved
        logits = _logits_from_bits(coded)
        # flip one copy of message bit 0 (positions 0, 5, 10) -> still majority correct
        logits[0] *= -1
        assert np.array_equal(ecc.decode(logits), msg)

    def test_hamming_corrects_single_error_per_block(self):
        ecc = HammingECC()
        rng = np.random.default_rng(1)
        msg = rng.integers(0, 2, 4 * 6).astype(np.uint8)
        coded = ecc.encode(msg)
        logits = _logits_from_bits(coded)
        # flip exactly one bit in each 7-bit block
        for blk in range(len(coded) // 7):
            logits[blk * 7 + (blk % 7)] *= -1
        assert np.array_equal(ecc.decode(logits), msg)

    def test_repetition_beats_identity_under_noise(self):
        """At a fixed bit-error rate, rep3 yields far fewer message-bit errors."""
        rng = np.random.default_rng(2)
        n = 99
        ber = 0.1
        id_ecc, rep = IdentityECC(), RepetitionECC(3)
        id_err = rep_err = 0
        trials = 200
        for _ in range(trials):
            # identity: n message bits
            m = rng.integers(0, 2, n).astype(np.uint8)
            lo = _logits_from_bits(m)
            flip = rng.random(n) < ber
            lo[flip] *= -1
            id_err += int((id_ecc.decode(lo) != m).sum())
            # repetition: n/3 message bits, n coded bits, same channel ber
            km = rng.integers(0, 2, rep.message_len(n)).astype(np.uint8)
            c = rep.encode(km)
            lc = _logits_from_bits(c)
            flip = rng.random(len(c)) < ber
            lc[flip] *= -1
            rep_err += int((rep.decode(lc) != km).sum())
        # normalise to per-message-bit error rate
        assert rep_err / (trials * rep.message_len(n)) < id_err / (trials * n)


class TestBytesBits:
    def test_bytes_roundtrip(self):
        data = b"StegaQR!"
        bits = bytes_to_bits(data)
        assert bits_to_bytes(bits) == data

    def test_get_ecc(self):
        assert get_ecc("none").name == "none"
        assert get_ecc("rep5").repeat == 5
        assert get_ecc("hamming74").name == "hamming74"
