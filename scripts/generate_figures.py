"""Generate publication figures from aggregated experiment results.

Reads experiments/full/*/results.json and writes PNG/PDF figures to figures/:
  fig_mode_comparison   -- distorted bit-acc & full-decode per mode (clean vs dist training)
  fig_imperceptibility  -- PSNR vs robust full-decode frontier (operating points)
  fig_ecc               -- message-decode rate vs ECC / net bits
  fig_capacity          -- bit-acc & PSNR vs capacity

Usage: python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
FULL = ROOT / "experiments" / "full"
FIG = ROOT / "figures"


def load():
    return [json.loads(p.read_text()) for p in sorted(FULL.glob("*/results.json"))]


def by_group(runs, prefix):
    g = defaultdict(list)
    for r in runs:
        t = r.get("tag", "")
        if t.startswith(prefix):
            g[re.sub(r"_s\d+$", "", t)].append(r)
    return g


def mean_ci(vals):
    n = len(vals); m = float(np.mean(vals))
    h = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    return m, h


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def fig_mode_comparison(runs):
    g = by_group(runs, "main_")
    modes = ["segregated", "cross_channel", "hybrid"]
    kinds = ["clean", "distort"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(modes)); w = 0.35
    for i, kind in enumerate(kinds):
        means, errs = [], []
        for mode in modes:
            key = f"main_{mode}_{kind}"
            rs = g.get(key, [])
            vals = [r["distorted"]["full_decode"] * 100 for r in rs] or [0]
            m, h = mean_ci(vals); means.append(m); errs.append(h)
        ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=4,
               label=f"{kind}-trained")
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_ylabel("distorted full-decode rate (%)")
    ax.set_title("Robustness by mode and training regime (cap 100)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_mode_comparison")


def fig_imperceptibility(runs):
    """PSNR vs distorted full-decode across all trained runs (the frontier)."""
    pts = [(r["clean"]["psnr"], r["distorted"]["full_decode"] * 100,
            r.get("config", {}).get("mode", "?"))
           for r in runs if "clean" in r and r.get("config", {}).get("mode") != "classical_lsb"]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, mk in [("segregated", "o"), ("cross_channel", "s"), ("hybrid", "^")]:
        xs = [p[0] for p in pts if p[2] == mode]
        ys = [p[1] for p in pts if p[2] == mode]
        if xs:
            ax.scatter(xs, ys, marker=mk, label=mode, alpha=0.7)
    ax.set_xlabel("PSNR (dB)  --  imperceptibility")
    ax.set_ylabel("distorted full-decode rate (%)")
    ax.set_title("Imperceptibility vs robustness frontier")
    ax.legend(); ax.grid(alpha=0.3)
    save(fig, "fig_imperceptibility")


def fig_ecc(runs):
    cc = [r for r in runs if r.get("tag", "").startswith("main_cross_channel_distort")]
    if not cc:
        return
    names = ["none"] + list(cc[0].get("ecc", {}).keys())
    labels, means, errs, bits = [], [], [], []
    for name in names:
        if name == "none":
            vals = [r["distorted"]["full_decode"] * 100 for r in cc]
            nb = cc[0]["config"]["capacity_bits"]
        else:
            vals = [r["ecc"][name]["msg_decode"] * 100 for r in cc if name in r["ecc"]]
            nb = cc[0]["ecc"][name]["net_bits"]
        if not vals:
            continue
        m, h = mean_ci(vals); labels.append(name); means.append(m); errs.append(h); bits.append(nb)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=4, color="tab:green", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n({b} bits)" for l, b in zip(labels, bits)])
    ax.set_ylabel("message-decode rate (%)")
    ax.set_title("ECC under distortion (cross_channel, cap 100)")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_ecc")


def fig_capacity(runs):
    rows = defaultdict(list)
    for r in runs:
        t = r.get("tag", "")
        if t.startswith("cap_"):
            rows[int(re.sub(r"_s\d+$", "", t).replace("cap_", ""))].append(r)
        elif t.startswith("main_cross_channel_distort"):
            rows[100].append(r)
    if not rows:
        return
    caps = sorted(rows)
    ba = [mean_ci([r["distorted"]["bit_acc"] * 100 for r in rows[c]]) for c in caps]
    ps = [mean_ci([r["distorted"]["psnr"] for r in rows[c]]) for c in caps]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.errorbar(caps, [m for m, _ in ba], yerr=[h for _, h in ba], marker="o",
                 color="tab:blue", label="distorted bit-acc")
    ax1.set_xlabel("capacity (embedded bits)")
    ax1.set_ylabel("distorted bit accuracy (%)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.errorbar(caps, [m for m, _ in ps], yerr=[h for _, h in ps], marker="s",
                 color="tab:red", label="PSNR")
    ax2.set_ylabel("PSNR (dB)", color="tab:red")
    ax1.set_title("Capacity scaling (cross_channel, distortion)")
    ax1.grid(alpha=0.3)
    save(fig, "fig_capacity")


def main():
    runs = load()
    if not runs:
        print("No results in", FULL)
        return
    fig_mode_comparison(runs)
    fig_imperceptibility(runs)
    fig_ecc(runs)
    fig_capacity(runs)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
