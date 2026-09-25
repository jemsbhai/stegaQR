"""Error-correction coding (ECC) and grid placement for the hidden payload.

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

Placement (0.2.0). Coded bit c must also be assigned to a cell of the spatial
bit-grid. The 0.1.0 release placed coded bit c on cell c ("native"). Because the
repetition encoder tiles its copies (copy r of message bit j is coded bit r*k + j)
and the grid fills row by row, native placement puts all copies of a bit in ONE grid
column for k = 20 on the 10 x 10 grid, and the copies then fail together under
distortion (paper, Section V-E; scripts/eval_ecc_layout.py). The "interleaved"
placement sends coded bit c to cell perm[c] for a fixed pseudo-random permutation
shared by encoder and decoder, which spreads the copies over the grid and removed
every repetition-code message failure in that study. Interleaved is the default from
0.2.0; pass placement="native" to read codes produced by 0.1.0.
"""

from __future__ import annotations

import numpy as np

PLACEMENTS = ("interleaved", "native")
DEFAULT_PLACEMENT = "interleaved"
DEFAULT_PLACEMENT_SEED = 0


def placement_permutation(capacity: int, placement: str = DEFAULT_PLACEMENT,
                          seed: int = DEFAULT_PLACEMENT_SEED) -> np.ndarray:
    """Return pos with pos[c] = grid cell holding coded bit c (length `capacity`).

    'native' is the identity (0.1.0 behaviour). 'interleaved' is a fixed
    pseudo-random permutation of the cells drawn from numpy's default_rng(seed),
    so encoder and decoder agree as long as they share capacity, placement, and seed.
    The same construction is used by scripts/eval_ecc_layout.py.
    """
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    if placement == "native":
        return np.arange(capacity)
    if placement == "interleaved":
        return np.random.default_rng(int(seed)).permutation(capacity)
    raise ValueError(f"unknown placement {placement!r}; expected one of {PLACEMENTS}")


def place_coded_bits(coded: np.ndarray, capacity: int, pos: np.ndarray,
                     dtype=np.float32) -> np.ndarray:
    """Scatter a coded bit vector into a capacity-length payload at positions pos[:len(coded)].
    Cells that carry no coded bit are zero."""
    coded = np.asarray(coded).ravel()
    if coded.size > capacity:
        raise ValueError(f"{coded.size} coded bits exceed capacity {capacity}")
    payload = np.zeros(capacity, dtype=dtype)
    payload[pos[: coded.size]] = coded
    return payload


def gather_coded_logits(logits: np.ndarray, coded_len: int, pos: np.ndarray) -> np.ndarray:
    """Inverse of place_coded_bits: read the coded_len logits back in coded order."""
    return np.asarray(logits).ravel()[pos[:coded_len]]


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

    Each message bit is embedded m times. Copies are tiled (copy r of bit j at coded
    index r*k + j). Where the copies land on the spatial grid is decided separately by
    the placement (see placement_permutation); with native placement and k a multiple
    of the grid width, all copies of a bit share one grid column. Soft decoding sums
    the m per-copy logits -- the optimal combiner for repetition over independent
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
