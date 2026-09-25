"""Module-size sweep at fixed payload (camera-ready experiment for ICTAI 2026 R3-W1).

Reviewer 3 asked whether the spatial bit-grid wins over global broadcast because each
bit has a LOCATION (spatial correspondence) or because each bit occupies a BLOCK of
pixels that the decoder average-pools (block redundancy), and proposed fixing L and
sweeping the module size m, which changes pixels-per-bit without changing capacity.

This script trains the cross-channel model at L = 100 with the exact recipe of
scripts/run_experiments.py (same epochs, batch, samples, warmup, ramp, loss weights,
perturbation bound, evaluation suite), varying only module_size. Cover side is 33 m
pixels and the grid is 10 x 10, so the cell (pixels per bit, one side) is 3.3 m:
    m = 1 -> 3.3 px    m = 2 -> 6.6 px    m = 3 -> 9.9 px    m = 4 -> 13.2 px (paper)
Runs are ordered smallest-informative-first and by seed, are resumable (a run with a
results.json is skipped), and stop launching new runs once --budget-minutes is spent,
so a hard time limit yields a complete subset rather than a half-finished run.

Caveats to state in the paper: the distortion layer works in pixels, so its blur and
noise are relatively harsher on small covers; and a standard reader needs modules of
a few pixels, so the public-decode column at m = 1 measures the reader, not StegaQR.

Usage (PowerShell, repo root):
  python scripts/run_module_sweep.py --budget-minutes 25
  python scripts/run_module_sweep.py --summary     # tables only, no training
Results: experiments/full/ms_<m>_s<seed>/results.json and experiments/full/MODULE_SWEEP.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_experiments as RE          # noqa: E402  (recipe constants, by construction identical)
from stegaqr.evaluation import evaluate_checkpoint   # noqa: E402
from stegaqr.training import train                   # noqa: E402

OUT = RE.OUT
QR_MODULES = 33                      # version 4
GRID = 10                            # ceil(sqrt(100))
ORDER = [(2, 42), (3, 42), (1, 42), (2, 123), (3, 123), (1, 123), (2, 7), (3, 7), (1, 7)]


def run_one(ms: int, seed: int) -> dict:
    tag = f"ms_{ms}_s{seed}"
    out_dir = OUT / tag
    res_path = out_dir / "results.json"
    if res_path.exists():
        print(f"[skip] {tag} (results.json exists)")
        return json.loads(res_path.read_text(encoding="utf-8"))
    print(f"\n{'=' * 70}\n[run] {tag}  (cover {QR_MODULES * ms} px, cell {QR_MODULES * ms / GRID:.1f} px)\n{'=' * 70}",
          flush=True)
    t0 = time.time()
    train(
        mode="cross_channel", capacity_bits=100, qr_version=4, module_size=ms,
        ec_level="M", epochs=RE.EPOCHS, batch_size=RE.BATCH, lr=1e-3,
        num_train=RE.NUM_TRAIN, num_val=RE.NUM_VAL, seed=seed, device="cuda",
        output_dir=str(out_dir), checkpoint_every=RE.EPOCHS,
        lambda_perceptual=RE.LAMBDA_PERC, perturbation_bound=RE.PBOUND,
        use_distortion=True, warmup_decode_only=RE.WARMUP,
        perc_ramp_epochs=RE.RAMP, arch="grid",
    )
    metrics = evaluate_checkpoint(str(out_dir / "best_model.pt"), n=RE.EVAL_N,
                                  eccs=RE.ECCS, seed=1234 + seed)
    metrics["tag"] = tag
    metrics["module_size"] = ms
    metrics["train_seconds"] = round(time.time() - t0, 1)
    res_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done] {tag} in {metrics['train_seconds']}s "
          f"| dist bit-acc {metrics['distorted']['bit_acc']:.4f} "
          f"fdr {metrics['distorted']['full_decode']:.3f} "
          f"psnr {metrics['distorted']['psnr']:.1f} "
          f"public {metrics['distorted']['public_decode']:.2f}", flush=True)
    return metrics


def _ci95(vals):
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return mu, 0.0
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))
    return mu, 1.96 * sd / math.sqrt(n)


def _fmt(vals, scale=100.0):
    vals = [min(v, 99.0) if math.isinf(v) else v for v in vals]
    mu, h = _ci95(vals)
    return f"{mu * scale:.1f} +/- {h * scale:.1f}" if scale != 1 else f"{mu:.1f} +/- {h:.1f}"


def summary() -> None:
    rows = {}
    for rp in sorted(OUT.glob("ms_*_s*/results.json")):
        r = json.loads(rp.read_text(encoding="utf-8"))
        rows.setdefault(int(r["module_size"]), []).append(r)
    # the paper's m = 4 runs (same recipe) for reference
    for rp in sorted(OUT.glob("main_cross_channel_distort_s*/results.json")):
        rows.setdefault(4, []).append(json.loads(rp.read_text(encoding="utf-8")))
    lines = ["# Module-size sweep at L = 100, cross-channel, distortion-trained (R3-W1)", "",
             "Same recipe as the main matrix; only module_size varies. Cell side = 3.3 m px. "
             "Mean +/- 95% CI (1.96 SE) across seeds; m = 4 rows are the paper's main-matrix runs.", "",
             "| m | cover px | cell px | seeds | clean bit-acc | clean FDR | dist bit-acc | dist FDR | "
             "PSNR (dB) | public decode | rep3 msg-decode | train s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ms in sorted(rows):
        rs = rows[ms]
        rep3 = [r["ecc"]["rep3"]["msg_decode"] for r in rs if "rep3" in r.get("ecc", {})]
        lines.append(
            f"| {ms} | {QR_MODULES * ms} | {QR_MODULES * ms / GRID:.1f} | {len(rs)} | "
            f"{_fmt([r['clean']['bit_acc'] for r in rs])} | {_fmt([r['clean']['full_decode'] for r in rs])} | "
            f"{_fmt([r['distorted']['bit_acc'] for r in rs])} | {_fmt([r['distorted']['full_decode'] for r in rs])} | "
            f"{_fmt([r['distorted']['psnr'] for r in rs], 1)} | {_fmt([r['distorted']['public_decode'] for r in rs])} | "
            f"{_fmt(rep3) if rep3 else '-'} | "
            f"{_fmt([r.get('train_seconds', float('nan')) for r in rs], 1)} |")
    lines.append("")
    lines.append("Per-run: " + ", ".join(
        f"{r['tag']} bit {r['distorted']['bit_acc']:.4f} fdr {r['distorted']['full_decode']:.3f}"
        for ms in sorted(rows) for r in rows[ms] if r["tag"].startswith("ms_")))
    (OUT / "MODULE_SWEEP.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("wrote", OUT / "MODULE_SWEEP.md")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget-minutes", type=float, default=25.0,
                   help="stop launching new runs once this much wall time has elapsed")
    p.add_argument("--only-m", default=None, help="comma-separated module sizes, e.g. 2,3")
    p.add_argument("--only-seeds", default=None, help="comma-separated seeds, e.g. 42,123")
    p.add_argument("--summary", action="store_true", help="write the table from existing results and exit")
    args = p.parse_args()

    if args.summary:
        summary()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    order = list(ORDER)
    if args.only_m:
        keep = {int(x) for x in args.only_m.split(",")}
        order = [o for o in order if o[0] in keep]
    if args.only_seeds:
        keep = {int(x) for x in args.only_seeds.split(",")}
        order = [o for o in order if o[1] in keep]

    t0 = time.time()
    for ms, seed in order:
        elapsed = (time.time() - t0) / 60.0
        if elapsed > args.budget_minutes and not (OUT / f"ms_{ms}_s{seed}" / "results.json").exists():
            print(f"[budget] {elapsed:.1f} min elapsed, not starting ms_{ms}_s{seed}")
            continue
        run_one(ms, seed)
    summary()


if __name__ == "__main__":
    main()
