"""Aggregate per-run results.json into result tables (mean +/- 95% CI across seeds).

Reads experiments/full/*/results.json (written by run_experiments.py) and emits:
  * experiments/full/RESULTS.md      -- human-readable tables for the 4 studies
  * experiments/full/results.csv     -- flat per-run table for plotting/analysis

Usage: python scripts/aggregate_results.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
FULL = ROOT / "experiments" / "full"


def _ci95(vals):
    """Mean and half-width of a 95% CI (t-ish via 1.96; small-n falls back to range).
    Infinite PSNR (collapsed-to-zero perturbation) is clipped to 99 dB so the table
    stays readable and the collapse is still visible as an extreme value."""
    vals = [min(v, 99.0) if math.isinf(v) else v for v in vals]
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def _fmt(vals, scale=100.0, unit=""):
    m, h = _ci95(vals)
    return f"{m*scale:.1f}{unit} +/- {h*scale:.1f}" if scale != 1 else f"{m:.2f} +/- {h:.2f}"


def load_runs():
    runs = []
    for rp in sorted(FULL.glob("*/results.json")):
        runs.append(json.loads(rp.read_text()))
    return runs


def group(runs, prefix):
    """Group runs whose tag starts with prefix by the tag minus the _s<seed> suffix."""
    groups = {}
    for r in runs:
        tag = r.get("tag", "")
        if not tag.startswith(prefix):
            continue
        key = re.sub(r"_s\d+$", "", tag)
        groups.setdefault(key, []).append(r)
    return groups


def main():
    runs = load_runs()
    if not runs:
        print("No results found in", FULL)
        return
    lines = ["# StegaQR Experimental Results", "",
             "Aggregated from per-run `results.json`. Values are mean +/- 95% CI "
             "across seeds. PSNR/SSIM are stego-vs-cover (imperceptibility).", ""]

    # ---- Study 1: main mode comparison (clean vs distortion training) ----
    lines += ["## 1. Mode comparison (capacity 100)", "",
              "| mode | train | seeds | clean bit-acc | clean FDR | dist bit-acc | "
              "dist FDR | PSNR (dB) | SSIM | public-decode |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    g = group(runs, "main_")
    for key in sorted(g):
        rs = g[key]
        mode = key.replace("main_", "").rsplit("_", 1)[0]
        train_kind = key.rsplit("_", 1)[1]
        def col(path_a, path_b, scale=100.0):
            vals = [r[path_a][path_b] for r in rs]
            return _fmt(vals, scale)
        lines.append(
            f"| {mode} | {train_kind} | {len(rs)} | "
            f"{col('clean','bit_acc')} | {col('clean','full_decode')} | "
            f"{col('distorted','bit_acc')} | {col('distorted','full_decode')} | "
            f"{_fmt([r['clean']['psnr'] for r in rs],1)} | "
            f"{_fmt([r['clean']['ssim'] for r in rs],1)} | "
            f"{col('distorted','public_decode')} |")
    lines.append("")

    # ---- Study 2: ECC operating points (under distortion) ----
    lines += ["## 2. ECC message-decode under distortion (cross_channel, cap 100)", "",
              "| code | net bits | rate | message-decode |",
              "|---|---|---|---|"]
    cc = [r for r in runs if r.get("tag", "").startswith("main_cross_channel_distort")]
    if cc:
        ecc_names = list(cc[0].get("ecc", {}).keys())
        # include raw (no ECC) full-decode as the 100-bit reference
        lines.append(f"| none (raw) | 100 | 1.00 | {_fmt([r['distorted']['full_decode'] for r in cc])} |")
        for name in ecc_names:
            nb = cc[0]["ecc"][name]["net_bits"]; rate = cc[0]["ecc"][name]["rate"]
            vals = [r["ecc"][name]["msg_decode"] for r in cc if name in r["ecc"]]
            lines.append(f"| {name} | {nb} | {rate:.2f} | {_fmt(vals)} |")
    lines.append("")

    # ---- Study 3: capacity scaling ----
    lines += ["## 3. Capacity scaling (cross_channel, distortion)", "",
              "| capacity | seeds | dist bit-acc | dist FDR | PSNR (dB) | SSIM |",
              "|---|---|---|---|---|---|"]
    gc = group(runs, "cap_")
    # include cap 100 from the main matrix
    main_cc = group(runs, "main_cross_channel_distort")
    rows = []
    for key, rs in gc.items():
        cap = int(key.replace("cap_", ""))
        rows.append((cap, rs))
    for key, rs in main_cc.items():
        rows.append((100, rs))
    for cap, rs in sorted(rows):
        lines.append(
            f"| {cap} | {len(rs)} | "
            f"{_fmt([r['distorted']['bit_acc'] for r in rs])} | "
            f"{_fmt([r['distorted']['full_decode'] for r in rs])} | "
            f"{_fmt([r['distorted']['psnr'] for r in rs],1)} | "
            f"{_fmt([r['distorted']['ssim'] for r in rs],1)} |")
    lines.append("")

    # ---- Study 5: architecture ablation (spatial grid vs global broadcast) ----
    arch = [r for r in runs if r.get("tag", "").startswith("arch_")]
    if arch:
        lines += ["## 5. Architecture ablation: spatial grid vs broadcast (decode-only)", "",
                  "| arch | training | clean bit-acc | clean FDR | dist bit-acc | dist FDR | PSNR (dB) |",
                  "|---|---|---|---|---|---|---|"]
        for r in sorted(arch, key=lambda r: r["tag"]):
            t = r["tag"].replace("arch_", "").replace("_s42", "")
            a, tr = t.rsplit("_", 1)
            lines.append(
                f"| {a} | {tr} | {r['clean']['bit_acc']*100:.1f}% | {r['clean']['full_decode']*100:.1f}% | "
                f"{r['distorted']['bit_acc']*100:.1f}% | {r['distorted']['full_decode']*100:.1f}% | "
                f"{min(r['clean']['psnr'],99):.1f} |")
        lines.append("")

    # ---- Study 4: classical baseline ----
    cl = [r for r in runs if r.get("tag") == "classical_lsb"]
    if cl:
        c = cl[0]
        lines += ["## 4. Classical LSB baseline (no training)", "",
                  "| setting | bit-acc | full-decode | PSNR (dB) | SSIM | public-decode |",
                  "|---|---|---|---|---|---|"]
        for s in ("clean", "distorted"):
            m = c[s]
            lines.append(f"| {s} | {m['bit_acc']*100:.1f}% | {m['full_decode']*100:.1f}% | "
                         f"{m['psnr']:.1f} | {m['ssim']:.3f} | {m['public_decode']*100:.1f}% |")
        lines.append("")

    (FULL / "RESULTS.md").write_text("\n".join(lines))
    print("wrote", FULL / "RESULTS.md")

    # ---- flat CSV ----
    with open(FULL / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "mode", "capacity", "setting", "bit_acc", "full_decode",
                    "psnr", "ssim", "public_decode"])
        for r in runs:
            cfg = r.get("config", {})
            for s in ("clean", "distorted"):
                if s in r:
                    m = r[s]
                    w.writerow([r.get("tag"), cfg.get("mode"), cfg.get("capacity_bits"), s,
                                m["bit_acc"], m["full_decode"], m["psnr"], m["ssim"],
                                m["public_decode"]])
    print("wrote", FULL / "results.csv")


if __name__ == "__main__":
    main()
