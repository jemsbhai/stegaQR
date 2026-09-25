"""Tests for scripts/eval_ecc_layout.py (camera-ready R3-W4 experiment)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location("eval_ecc_layout", ROOT / "scripts" / "eval_ecc_layout.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from stegaqr.coding import get_ecc


def _square_grid_coords(cap: int) -> np.ndarray:
    gw = int(np.ceil(np.sqrt(cap)))
    p = np.arange(cap)
    return np.stack([p // gw, p % gw], axis=1)


def test_native_rep5_copies_share_one_column_on_10x10_grid():
    coords = _square_grid_coords(100)
    ecc = get_ecc("rep5")
    k, m = ecc.message_len(100), ecc.repeat
    geo = mod._copy_geometry(coords, np.arange(ecc.coded_len(k)), k, m)
    assert geo["mean_distinct_cols_per_bit"] == 1.0
    assert geo["mean_distinct_rows_per_bit"] == 5.0


def test_native_rep3_copies_are_dispersed_on_10x10_grid():
    coords = _square_grid_coords(100)
    ecc = get_ecc("rep3")
    k, m = ecc.message_len(100), ecc.repeat
    geo = mod._copy_geometry(coords, np.arange(ecc.coded_len(k)), k, m)
    assert geo["mean_distinct_cols_per_bit"] == 3.0
    assert geo["mean_distinct_rows_per_bit"] == 3.0


def test_interleaver_disperses_rep5_copies():
    coords = _square_grid_coords(100)
    ecc = get_ecc("rep5")
    k, m = ecc.message_len(100), ecc.repeat
    perm = np.random.default_rng(0).permutation(100)
    geo = mod._copy_geometry(coords, perm[: ecc.coded_len(k)], k, m)
    assert geo["mean_distinct_cols_per_bit"] > 3.0


def test_placement_round_trip_recovers_coded_order():
    """Placing coded bits at perm positions and reading logits back at the same
    positions must return the coded bits in coded order for every code."""
    cap = 100
    perm = np.random.default_rng(0).permutation(cap)
    for code in ("none", "rep2", "rep3", "rep4", "rep5", "hamming74"):
        ecc = get_ecc(code)
        k = ecc.message_len(cap)
        clen = ecc.coded_len(k)
        msgs = np.random.default_rng(1).integers(0, 2, size=(4, k), dtype=np.uint8)
        coded = np.stack([ecc.encode(x) for x in msgs]).astype(np.uint8)
        for pos in (np.arange(clen), perm[:clen]):
            pbits = np.zeros((4, cap), dtype=np.uint8)
            pbits[:, pos] = coded
            logits = pbits.astype(np.float64) * 2 - 1          # perfect channel
            back = logits[:, pos]
            assert np.array_equal((back > 0).astype(np.uint8), coded)
            dec = np.stack([ecc.decode(back[i]) for i in range(4)])
            assert np.array_equal(dec, msgs)


def test_independent_majority_reference():
    assert mod._independent_majority_error(0.0, 5) == 0.0
    assert mod._independent_majority_error(1.0, 5) == pytest.approx(1.0)
    # p = 0.01, m = 5: needs >= 3 wrong copies; leading term C(5,3) p^3 (1-p)^2
    assert mod._independent_majority_error(0.01, 5) == pytest.approx(9.8e-6, rel=0.05)


def test_ci95_matches_aggregate_results_convention():
    mu, h = mod._ci95([1.0, 0.9, 0.8])
    assert mu == pytest.approx(0.9)
    assert h == pytest.approx(1.96 * 0.1 / np.sqrt(3))
