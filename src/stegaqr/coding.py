"""Error-correction coding (ECC) for the hidden payload.

The neural decoder recovers individual bits with high but imperfect accuracy
(~98-99% under distortion at high PSNR). A full message needs ALL bits correct, so
without coding the message-decode rate collapses: at 1.7% bit-error rate a 100-bit
message decodes only ~0.983**100 ~ 18% of the time. ECC decouples bit-accuracy
from message success: it encodes k message bits into n > k coded bits, the model
embeds the n coded bits, and on decode the residual bit errors are corrected.

This lets the encoder run at high PSNR (small, imperceptible perturbation -> some
bit flips) while still recovering the full message -- the standard approach in
StegaStamp and real watermarking systems. The cost is rate: net message length
k < embedded capacity n.

The neural decoder emits per-bit LOGITS (sign = bit, magnitude = confidence), so
we use SOFT-decision decoding -- combining log-likelihoods -- which is substantially
stronger than hard-decision (majority) decoding. Convention: logit > 0 => bit 1.
"""

from __future__ import annotations

import numpy as np


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a 0/1 bit array (MSB-first) into bytes; length padded to a multiple of 8."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    pad = (-len(b)) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(b).tobytes()


def bytes_to_bits(data: bytes, n_bits: int | None = None) -> np.ndarray:
    """Unpack bytes into an MSB-first 0/1 bit array, optionally truncated to n_bits."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits[:n_bits] if n_bits is not None else bits


class IdentityECC:
    """No coding (rate 1). Baseline for comparison."""

    name = "none"
    rate = 1.0

    def coded_len(self, k: int) -> int:
        return k

    def message_len(self, n: int) -> int:
        return n

    def encode(self, msg_bits: np.ndarray) -> np.ndarray:
        return np.asarray(msg_bits, dtype=np.uint8).ravel()

    def decode(self, logits: np.ndarray) -> np.ndarray:
        """Hard threshold of logits (soft info unused without redundancy)."""
        return (np.asarray(logits, dtype=np.float64).ravel() > 0).astype(np.uint8)


class RepetitionECC:
    """Rate-1/m repetition code with soft-decision (log-likelihood) combining.

    Each message bit is embedded m times. Copies are interleaved (placed k
    positions apart) so a spatial burst in the embedding grid corrupts different
    message bits rather than all copies of one bit. Soft decoding sums the m
    per-copy logits -- the optimal combiner for repetition over independent
    channels -- before thresholding.

    Parameters
    ----------
    repeat : int
        Number of copies per message bit (>= 1). Larger = more robust, lower rate.
    """

    def __init__(self, repeat: int = 3):
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.repeat = repeat
        self.name = f"rep{repeat}"
        self.rate = 1.0 / repeat

    def coded_len(self, k: int) -> int:
        return k * self.repeat

    def message_len(self, n: int) -> int:
        return n // self.repeat

    def encode(self, msg_bits: np.ndarray) -> np.ndarray:
        msg = np.asarray(msg_bits, dtype=np.uint8).ravel()
        return np.tile(msg, self.repeat)  # [msg, msg, ...]; copy r of bit j at index r*k + j

    def decode(self, logits: np.ndarray) -> np.ndarray:
        """Soft-decision decode: sum the m logits per message bit, then threshold."""
        L = np.asarray(logits, dtype=np.float64).ravel()
        k = L.size // self.repeat
        L = L[: k * self.repeat].reshape(self.repeat, k)
        return (L.sum(axis=0) > 0).astype(np.uint8)

    def decode_hard(self, bits: np.ndarray) -> np.ndarray:
        """Hard-decision (majority vote) decode, when only bits are available."""
        b = np.asarray(bits, dtype=np.uint8).ravel()
        k = b.size // self.repeat
        b = b[: k * self.repeat].reshape(self.repeat, k)
        return (b.sum(axis=0) * 2 > self.repeat).astype(np.uint8)


# Hamming(7,4): single-error-correcting, rate 4/7 ~ 0.571. More bit-efficient than
# repetition when the per-bit error rate is low; corrects 1 error per 7-bit block.
# Column j (1-indexed) of H is the 3-bit binary representation of j (weights 4,2,1),
# so the syndrome of a single-bit error directly gives that bit's position.
_H = np.array([
    [0, 0, 0, 1, 1, 1, 1],   # weight 4
    [0, 1, 1, 0, 0, 1, 1],   # weight 2
    [1, 0, 1, 0, 1, 0, 1],   # weight 1
], dtype=np.uint8)
_PARITY_POS = [0, 1, 3]          # codeword positions 1,2,4 (powers of two)
_DATA_POS = [2, 4, 5, 6]         # codeword positions 3,5,6,7


def _gf2_inv(mat: np.ndarray) -> np.ndarray:
    """Inverse of a square binary matrix over GF(2) via Gauss-Jordan."""
    n = mat.shape[0]
    a = np.concatenate([mat % 2, np.eye(n, dtype=np.uint8)], axis=1)
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r, col])
        a[[col, piv]] = a[[piv, col]]
        for r in range(n):
            if r != col and a[r, col]:
                a[r] ^= a[col]
    return a[:, n:]


_HP_INV = _gf2_inv(_H[:, _PARITY_POS])  # so that H @ codeword == 0


class HammingECC:
    """Hamming(7,4) single-error-correcting code (rate 4/7).

    Parity bits are derived from H (H @ codeword == 0), guaranteeing the encoder and
    the syndrome decoder are consistent. Hard-decision: corrects exactly one bit
    error per 7-bit block. Best when the per-bit error rate is low; for higher error
    rates prefer RepetitionECC with soft combining. Message length is padded to a
    multiple of 4.
    """

    name = "hamming74"
    rate = 4.0 / 7.0

    def coded_len(self, k: int) -> int:
        return ((k + 3) // 4) * 7

    def message_len(self, n: int) -> int:
        return (n // 7) * 4

    def encode(self, msg_bits: np.ndarray) -> np.ndarray:
        m = np.asarray(msg_bits, dtype=np.uint8).ravel()
        pad = (-len(m)) % 4
        if pad:
            m = np.concatenate([m, np.zeros(pad, dtype=np.uint8)])
        blocks = m.reshape(-1, 4)                       # (nblk, 4)
        s = (blocks @ _H[:, _DATA_POS].T) % 2           # syndrome contribution of data
        parity = (s @ _HP_INV.T) % 2                    # parity bits zeroing the syndrome
        code = np.zeros((blocks.shape[0], 7), dtype=np.uint8)
        code[:, _DATA_POS] = blocks
        code[:, _PARITY_POS] = parity
        return code.ravel().astype(np.uint8)

    def decode(self, logits: np.ndarray) -> np.ndarray:
        bits = (np.asarray(logits, dtype=np.float64).ravel() > 0).astype(np.uint8)
        nblk = bits.size // 7
        bits = bits[: nblk * 7].reshape(nblk, 7).copy()
        synd = (bits @ _H.T) % 2                        # (nblk, 3)
        pos = (synd * np.array([4, 2, 1])).sum(axis=1)  # 1..7, 0 = no error
        for i, p in enumerate(pos):
            if p > 0:
                bits[i, p - 1] ^= 1
        return bits[:, _DATA_POS].ravel().astype(np.uint8)


def get_ecc(name: str):
    """Factory: 'none' | 'rep3' | 'rep5' | 'repN' | 'hamming74'."""
    name = name.lower()
    if name in ("none", "identity"):
        return IdentityECC()
    if name in ("hamming", "hamming74"):
        return HammingECC()
    if name.startswith("rep"):
        return RepetitionECC(int(name[3:]))
    raise ValueError(f"unknown ECC: {name}")
