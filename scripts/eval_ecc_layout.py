"""Post hoc test of error-correction copy PLACEMENT on the spatial bit-grid.

Camera-ready experiment for ICTAI 2026 Reviewer 3, weakness W4: Repetition-5 decoded
fewer messages (98.1 percent) than Repetition-3 (99.5) and than the raw codeword (99.2)
in Table IV, although soft-decision combining of independent copies should never hurt.

Hypothesis (from reading coding.py and encoder.py):
  RepetitionECC.encode tiles the copies, so copy r of message bit j sits at coded index
  r*k + j, and payload_to_gridmap fills the g x g grid row-major. For the 100-bit
  cross-channel model g = 10, so rep5 (k = 20) places all five copies of a bit in ONE
  grid column, two rows apart, and makes the whole coded map vertically periodic with
  period two rows. rep3 (k = 33) scatters its copies across rows and columns. The model
  is trained on i.i.d. bits, so the periodic map is off the training distribution and
  the copies' errors are spatially correlated, which is exactly the regime in which soft
  combining gains nothing. Evidence already in experiments/full: the seed-42 cross-channel
  checkpoint has zero coded-bit errors under random payloads but loses 3 of 128 messages
  under rep5 payloads, same weights.

Test: the SAME checkpoints, SAME covers, SAME messages and SAME distortion draws, under
two placements of the coded bits on the grid, with no retraining:
  native       coded bit c at grid position c   (exactly what Table IV measured)
  interleaved  coded bit c at grid position perm[c], perm a fixed pseudo-random permutation
If the placement hypothesis holds: interleaved rep5 >= interleaved rep3 >= raw, and the
native copy-geometry numbers (distinct columns per bit) explain the native ordering.

For repetition codes the script also reports the per-copy hard-decision error rate next
to the post-combining message-bit error rate. Independent copies with per-copy error p
give a combined error far below p; correlated copies give a combined error close to p.

n defaults to 1024 messages per checkpoint (Table IV used 128, whose Wilson intervals
overlap). Aggregation across seeds uses mean +/- 1.96 SE, matching aggregate_results.py.

Usage (PowerShell, from the repo root):
  python scripts/eval_ecc_layout.py --ckpt-root E:\\data\\code\\claudecode\\mQRstego\\experiments\\full
Outputs: experiments/ecc_layout/<tag>.json per checkpoint and experiments/ecc_layout/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from stegaqr.coding import RepetitionECC, get_ecc
from stegaqr.evaluation import _forward, _gen_samples, build_models
from stegaqr.models.distortion import DifferentiableDistortion
from stegaqr.utils.metrics import wilson_score_ci
from stegaqr.utils.seed import set_all_seeds

ROOT = Path(__file__).parent.parent
OUT = ROOT / "experiments" / "ecc_layout"

GROUPS = {
    "cross_channel": "main_cross_channel_distort_s{seed}",
    "hybrid": "main_hybrid_distort_s{seed}",
}
SEEDS = [42, 123, 7, 99, 2024]
CODES = ["none", "rep2", "rep3", "rep4", "rep5", "hamming74"]


# ----------------------------------------------------------------------------- helpers
def _load(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    enc, dec, is_hybrid = build_models(
        cfg["mode"], cfg["capacity_bits"], device, cfg.get("perturbation_bound", 0.1),
        arch=cfg.get("arch", "grid"), mask_aware=cfg.get("mask_aware", False),
        qr_version=cfg["qr_version"], use_calibration=cfg.get("use_calibration", True),
    )
    enc.load_state_dict(ckpt["encoder_state"])
    dec.load_state_dict(ckpt["decoder_state"])
    enc.eval()
    dec.eval()
    return enc, dec, is_hybrid, cfg


def _seed_torch(s: int) -> None:
    """Identical distortion-parameter and noise draws for every code and layout."""
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _grid_coords(enc, cap: int) -> np.ndarray:
    """(cap, 2) array of (row, col) grid coordinates of payload position p."""
    if hasattr(enc, "cells"):                      # mask-aware hybrid: data-rich cells
        idx = enc.cells.detach().cpu().numpy()
        g = int(enc.grid)
        return np.stack([idx // g, idx % g], axis=1)
    gw = int(enc.gw)
    p = np.arange(cap)
    return np.stack([p // gw, p % gw], axis=1)


def _copy_geometry(coords: np.ndarray, pos: np.ndarray, k: int, m: int) -> dict:
    """How the m copies of each message bit are spread over the grid.

    Copy r of message bit j is coded bit r*k + j, placed at grid position pos[r*k + j].
    """
    cols, rows, dists = [], [], []
    for j in range(k):
        pts = coords[pos[[r * k + j for r in range(m)]]]
        cols.append(len(set(pts[:, 1].tolist())))
        rows.append(len(set(pts[:, 0].tolist())))
        pair = [max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))
                for a, b in combinations(pts.tolist(), 2)]
        dists.append(float(np.mean(pair)))
    return {
        "mean_distinct_cols_per_bit": float(np.mean(cols)),
        "mean_distinct_rows_per_bit": float(np.mean(rows)),
        "mean_pairwise_chebyshev_cells": float(np.mean(dists)),
    }


def _independent_majority_error(p: float, m: int) -> float:
    """Message-bit error a hard majority vote would give if the m copies were
    independent with per-copy error p (a reference point, not a claim about soft
    decoding, which is at least as good)."""
    need = m // 2 + 1
    return float(sum(math.comb(m, i) * p ** i * (1 - p) ** (m - i) for i in range(need, m + 1)))


# ----------------------------------------------------------------------------- one checkpoint
def evaluate_layouts(ckpt_path: Path, n: int, seed: int, device: str, chunk: int,
                     interleave_seed: int) -> dict:
    device = device if torch.cuda.is_available() else "cpu"
    set_all_seeds(seed)
    enc, dec, is_hybrid, cfg = _load(ckpt_path, device)
    cap, qrv, ms, ec = cfg["capacity_bits"], cfg["qr_version"], cfg["module_size"], cfg["ec_level"]
    dist = DifferentiableDistortion().to(device)

    # One fixed set of covers, shared by every code and layout (paired comparison).
    cover, _, mask, _ = _gen_samples(n, cap, qrv, ms, ec, 0, device)
    coords = _grid_coords(enc, cap)
    perm = np.random.default_rng(interleave_seed).permutation(cap)

    out = {
        "config": {
            "mode": cfg["mode"], "capacity_bits": cap, "module_size": ms,
            "seed_train": cfg.get("seed"), "checkpoint": str(ckpt_path),
            "n": int(n), "seed_eval": int(seed), "interleave_seed": int(interleave_seed),
            "perm": perm.tolist(), "epoch": cfg.get("epoch"),
        },
        "codes": {},
    }

    for ci, code in enumerate(CODES):
        ecc = get_ecc(code)
        k = ecc.message_len(cap)
        clen = ecc.coded_len(k)
        if k <= 0:
            continue
        # Same messages for both layouts; independent of the global RNG.
        msgs = np.random.default_rng(seed * 1000 + ci).integers(0, 2, size=(n, k), dtype=np.uint8)
        coded = np.stack([ecc.encode(m) for m in msgs]).astype(np.uint8)          # (n, clen)

        layouts = {"native": np.arange(clen)}
        if code != "none":
            layouts["interleaved"] = perm[:clen]

        res = {"net_bits": int(k), "rate": float(ecc.rate), "coded_bits": int(clen)}
        for lname, pos in layouts.items():
            pbits = np.zeros((n, cap), dtype=np.uint8)
            pbits[:, pos] = coded                                                # unused cells stay 0
            payload = torch.from_numpy(pbits.astype(np.float32)).to(device)
            _seed_torch(seed + 7)
            logits, _ = _forward(enc, dec, cover, payload, mask, is_hybrid, dist, chunk)
            lg = logits[:, pos]                                                  # coded order
            pred = (lg > 0).astype(np.uint8)
            dec_msgs = np.stack([ecc.decode(lg[i]) for i in range(n)])
            ok = np.all(dec_msgs == msgs, axis=1)
            lo, hi = wilson_score_ci(int(ok.sum()), n)
            r = {
                "raw_bit_acc": float(np.mean(pred == coded)),
                "raw_full_decode": float(np.mean(np.all(pred == coded, axis=1))),
                "msg_bit_acc": float(np.mean(dec_msgs == msgs)),
                "msg_decode": float(ok.mean()),
                "msg_decode_ci": [lo, hi],
                "failed_messages": int(n - ok.sum()),
            }
            if isinstance(ecc, RepetitionECC):
                m = ecc.repeat
                copies = pred.reshape(n, m, k)                                   # copy r of bit j
                copy_wrong = copies != msgs[:, None, :]
                bit_wrong = dec_msgs != msgs
                p_copy = float(copy_wrong.mean())
                r["copy_error_rate"] = p_copy
                r["msg_bit_error_rate"] = float(bit_wrong.mean())
                r["independent_majority_error_reference"] = _independent_majority_error(p_copy, m)
                wrong_per_bit = copy_wrong.sum(axis=1)                           # (n, k)
                r["mean_wrong_copies_among_failed_bits"] = (
                    float(wrong_per_bit[bit_wrong].mean()) if bit_wrong.any() else 0.0)
                r["geometry"] = _copy_geometry(coords, pos, k, m)
            res[lname] = r
        out["codes"][code] = res
    return out


# ----------------------------------------------------------------------------- aggregation
def _ci95(vals):
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return mu, 0.0
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))
    return mu, 1.96 * sd / math.sqrt(n)


def _fmt(vals, scale=100.0):
    mu, h = _ci95(vals)
    return f"{mu * scale:.1f} +/- {h * scale:.1f}"


def write_results(results: dict[str, list[dict]], n: int) -> None:
    lines = ["# ECC copy placement on the spatial bit-grid (camera-ready, R3-W4)", "",
             f"n = {n} messages per checkpoint; mean +/- 95% CI (1.96 SE) across seeds; "
             "same covers, messages and distortion draws for native and interleaved.", ""]
    for group, runs in results.items():
        if not runs:
            continue
        lines += [f"## {group} ({len(runs)} seeds)", "",
                  "| code | net bits | rate | native msg-decode | interleaved msg-decode | "
                  "native copy-err | native msg-bit-err | interleaved copy-err | interleaved msg-bit-err | "
                  "native distinct cols/bit | interleaved distinct cols/bit |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for code in CODES:
            rs = [r["codes"][code] for r in runs if code in r["codes"]]
            if not rs:
                continue
            nb, rate = rs[0]["net_bits"], rs[0]["rate"]
            nat = _fmt([r["native"]["msg_decode"] for r in rs])
            if "interleaved" in rs[0]:
                inter = _fmt([r["interleaved"]["msg_decode"] for r in rs])
            else:
                inter = "same"
            if "copy_error_rate" in rs[0]["native"]:
                ce_n = _fmt([r["native"]["copy_error_rate"] for r in rs])
                be_n = _fmt([r["native"]["msg_bit_error_rate"] for r in rs])
                ce_i = _fmt([r["interleaved"]["copy_error_rate"] for r in rs])
                be_i = _fmt([r["interleaved"]["msg_bit_error_rate"] for r in rs])
                gc_n = f"{rs[0]['native']['geometry']['mean_distinct_cols_per_bit']:.2f}"
                gc_i = f"{rs[0]['interleaved']['geometry']['mean_distinct_cols_per_bit']:.2f}"
            else:
                ce_n = be_n = ce_i = be_i = gc_n = gc_i = "-"
            lines.append(f"| {code} | {nb} | {rate:.2f} | {nat} | {inter} | {ce_n} | {be_n} | "
                         f"{ce_i} | {be_i} | {gc_n} | {gc_i} |")
        lines.append("")
        lines.append("Per-seed failed messages (native / interleaved):")
        lines.append("")
        for r in runs:
            parts = []
            for code in CODES:
                if code not in r["codes"]:
                    continue
                c = r["codes"][code]
                if "interleaved" in c:
                    parts.append(f"{code} {c['native']['failed_messages']}/{c['interleaved']['failed_messages']}")
                else:
                    parts.append(f"{code} {c['native']['failed_messages']}")
            lines.append(f"- seed {r['config']['seed_train']}: " + ", ".join(parts))
        lines.append("")
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("wrote", OUT / "RESULTS.md")


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-root", default=str(ROOT / "experiments" / "full"),
                   help="directory holding main_<mode>_distort_s<seed>/best_model.pt")
    p.add_argument("--groups", default="cross_channel,hybrid")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234, help="evaluation seed (covers, messages, distortion)")
    p.add_argument("--interleave-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--force", action="store_true", help="recompute even if the per-checkpoint JSON exists")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    root = Path(args.ckpt_root)
    seeds = [int(s) for s in args.seeds.split(",")]
    results: dict[str, list[dict]] = {}
    for group in args.groups.split(","):
        results[group] = []
        for seed in seeds:
            tag = GROUPS[group].format(seed=seed)
            ckpt = root / tag / "best_model.pt"
            if not ckpt.exists():
                print(f"[missing] {ckpt}")
                continue
            js = OUT / f"{tag}.json"
            if js.exists() and not args.force:
                print(f"[cached] {tag}")
                results[group].append(json.loads(js.read_text(encoding="utf-8")))
                continue
            t0 = time.time()
            r = evaluate_layouts(ckpt, args.n, args.seed, args.device, args.chunk, args.interleave_seed)
            r["tag"] = tag
            r["eval_seconds"] = round(time.time() - t0, 1)
            js.write_text(json.dumps(r, indent=2), encoding="utf-8")
            results[group].append(r)
            c = r["codes"]
            print(f"[done] {tag} in {r['eval_seconds']}s | raw {c['none']['native']['msg_decode']:.4f} | "
                  f"rep3 {c['rep3']['native']['msg_decode']:.4f}/{c['rep3']['interleaved']['msg_decode']:.4f} | "
                  f"rep5 {c['rep5']['native']['msg_decode']:.4f}/{c['rep5']['interleaved']['msg_decode']:.4f} "
                  "(native/interleaved)", flush=True)
    write_results(results, args.n)


if __name__ == "__main__":
    main()
