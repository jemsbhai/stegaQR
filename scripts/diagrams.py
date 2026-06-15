"""Schematic diagrams for the StegaQR paper (matplotlib, reproducible).

Usage:
    python scripts/diagrams.py system      # overall system pipeline
    python scripts/diagrams.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).parent.parent / "figures"

# palette
C_EMBED = "#dbe9f6"; E_EMBED = "#3b6ea5"     # embed path (blue)
C_STEGO = "#ffe9b3"; E_STEGO = "#d99b00"     # stego (gold, highlighted)
C_CHAN = "#f3d9d6"; E_CHAN = "#b5453b"       # channel (red)
C_EXTR = "#d8efd8"; E_EXTR = "#3a8f4a"       # extract path (green)
C_OUT = "#eceff1"; E_OUT = "#5b6b73"         # outputs (slate)


def box(ax, cx, cy, w, h, text, fc, ec, fs=11, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.6,rounding_size=2.5",
        linewidth=1.6, facecolor=fc, edgecolor=ec, linestyle=ls))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#15202b")


def arrow(ax, p, q, color="#3a4750", lw=1.8, style="-|>"):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=2, shrinkB=2))


def make_system():
    fig, ax = plt.subplots(figsize=(16.5, 6.2))
    ax.set_xlim(0, 180); ax.set_ylim(6, 66); ax.axis("off")

    # --- embed lane ---
    box(ax, 12, 50, 20, 10, "Public payload\n“https://…”", C_EMBED, E_EMBED)
    box(ax, 12, 28, 20, 10, "Hidden payload\n(bytes)", C_EMBED, E_EMBED)
    box(ax, 38, 50, 20, 10, "QR encoder\n(ISO/IEC 18004)", C_EMBED, E_EMBED)
    box(ax, 38, 28, 20, 10, "ECC encode\n(rep / Hamming)", C_EMBED, E_EMBED)
    box(ax, 63, 50, 18, 10, "Cover QR\n(RGB)", C_EMBED, E_EMBED)
    box(ax, 63, 28, 18, 10, "Spatial\nbit-grid", C_EMBED, E_EMBED)
    box(ax, 88, 39, 20, 14, "Neural encoder\n(GroupNorm\nconv U-net)", C_EMBED, E_EMBED, bold=True)
    box(ax, 113, 39, 18, 12, "Stego QR\n(looks normal)", C_STEGO, E_STEGO, bold=True)

    for a, b in [((22, 50), (28, 50)), ((48, 50), (54, 50)), ((72, 50), (78, 47)),
                 ((22, 28), (28, 28)), ((48, 28), (54, 28)), ((72, 28), (78, 31)),
                 ((98, 39), (104, 39))]:
        arrow(ax, a, b, E_EMBED)

    # --- channel + extract lane ---
    box(ax, 113, 16, 24, 9, "Channel\nJPEG · print-scan · photo", C_CHAN, E_CHAN)
    arrow(ax, (113, 33), (113, 20.5), E_CHAN)            # stego -> channel
    box(ax, 140, 16, 17, 9, "Localize +\nrectify (OpenCV)", C_EXTR, E_EXTR)
    arrow(ax, (125, 16), (131.5, 16), E_EXTR)

    box(ax, 140, 52, 17, 9, "Standard QR\nreader (pyzbar)", C_OUT, E_OUT)
    box(ax, 140, 34, 17, 9, "Neural decoder\n+ ECC decode", C_EXTR, E_EXTR, bold=True)

    # public is always scannable: stego -> standard reader (diagonal up-right)
    arrow(ax, (120, 43), (131.5, 51), E_OUT)
    # rectified symbol -> neural decoder
    arrow(ax, (140, 20.5), (140, 29.5), E_EXTR)

    box(ax, 165, 52, 13, 9, "Public\npayload ✓", C_OUT, E_OUT)
    box(ax, 165, 34, 13, 9, "Hidden\npayload ✓", C_EXTR, E_EXTR, bold=True)
    arrow(ax, (148.5, 52), (158.5, 52), E_OUT)
    arrow(ax, (148.5, 34), (158.5, 34), E_EXTR)

    # lane labels
    ax.text(12, 62, "EMBED", fontsize=13, fontweight="bold", color=E_EMBED)
    ax.text(150, 62, "EXTRACT", fontsize=13, fontweight="bold", color=E_EXTR)
    ax.text(90, 8.5, "A standard reader recovers the public payload; only the StegaQR "
            "decoder recovers the hidden payload.", ha="center", fontsize=10,
            style="italic", color="#5b6b73")

    _save(fig, "fig_system")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def make_architecture():
    fig, ax = plt.subplots(figsize=(16, 7.0))
    ax.set_xlim(0, 172); ax.set_ylim(2, 70); ax.axis("off")

    # ----- Encoder lane (top) -----
    ax.text(2, 66, "Encoder", fontsize=14, fontweight="bold", color=E_EMBED)
    box(ax, 13, 60, 19, 7, "Cover RGB\n3×H×W", C_EMBED, E_EMBED, fs=10)
    box(ax, 13, 51, 19, 7, "Bit-grid\n1×H×W", C_EMBED, E_EMBED, fs=10)
    box(ax, 13, 42, 19, 7, "QR mask 1×H×W\n(hybrid only)", "#eef3f8", E_EMBED, fs=9, ls="--")
    box(ax, 35, 51, 11, 8, "concat", C_EMBED, E_EMBED, fs=10)
    box(ax, 57, 51, 22, 11, "Conv × 5\n3×3 · GroupNorm · ReLU\n64 ch", C_EMBED, E_EMBED, fs=10, bold=True)
    box(ax, 85, 51, 18, 10, "1×1 conv → 3\ntanh × ε", C_EMBED, E_EMBED, fs=10)
    box(ax, 108, 51, 13, 9, "× mask\n(hybrid)", "#eef3f8", E_EMBED, fs=9, ls="--")
    box(ax, 128, 51, 14, 9, "+ cover\nclamp[0,1]", C_EMBED, E_EMBED, fs=10)
    box(ax, 150, 51, 16, 10, "Stego QR\n3×H×W", C_STEGO, E_STEGO, fs=11, bold=True)
    for a, b in [((22.5, 60), (29.5, 53)), ((22.5, 51), (29.5, 51)),
                 ((22.5, 42), (29.5, 49)), ((40.5, 51), (46, 51)),
                 ((68, 51), (76, 51)), ((94, 51), (101.5, 51)),
                 ((114.5, 51), (121, 51)), ((135, 51), (142, 51))]:
        arrow(ax, a, b, E_EMBED, lw=1.6)
    ax.text(108, 43.5, "structure mask\nzeros protected modules", ha="center", fontsize=7.5,
            color="#5b6b73")

    # ----- Decoder lane (bottom) -----
    ax.text(2, 30, "Decoder", fontsize=14, fontweight="bold", color=E_EXTR)
    box(ax, 13, 22, 19, 9, "Stego QR\n(channel output)", C_STEGO, E_STEGO, fs=9.5)
    box(ax, 39, 22, 20, 11, "Conv × 6\nGroupNorm", C_EXTR, E_EXTR, fs=10, bold=True)
    box(ax, 66, 22, 18, 10, "1×1 conv → 1\nactivation map", C_EXTR, E_EXTR, fs=10)
    box(ax, 90, 22, 17, 10, "avg-pool\n→ g×g grid", C_EXTR, E_EXTR, fs=10)
    box(ax, 113, 22, 15, 9, "read cells\n→ L logits", C_EXTR, E_EXTR, fs=10)
    box(ax, 134, 22, 14, 9, "threshold\n+ ECC", C_EXTR, E_EXTR, fs=10)
    box(ax, 156, 22, 15, 9, "Hidden\nbits ✓", C_EXTR, E_EXTR, fs=10.5, bold=True)
    for a, b in [((22.5, 22), (29, 22)), ((49, 22), (57, 22)), ((75, 22), (81.5, 22)),
                 ((98.5, 22), (104.5, 22)), ((120.5, 22), (127, 22)), ((141, 22), (148.5, 22))]:
        arrow(ax, a, b, E_EXTR, lw=1.6)
    # channel link encoder->decoder
    arrow(ax, (150, 46), (13, 27), "#b5453b", lw=1.4, style="-|>")
    ax.text(80, 37.5, "channel  (JPEG · print-scan · photo)", ha="center", fontsize=9,
            color=E_CHAN, style="italic")

    # ----- notes -----
    ax.text(86, 11.5, "Spatial bit-grid: each of the L payload bits occupies one cell of a "
            "⌈√L⌉×⌈√L⌉ grid upsampled to H×W; the decoder average-pools back to the grid "
            "and reads each cell.", ha="center", fontsize=9, color="#33414b")
    ax.text(86, 6.5, "Modes — segregated: 3 independent per-channel sub-networks · "
            "cross-channel: joint RGB (shown) · hybrid: joint + structural mask + bits on data cells.",
            ha="center", fontsize=9, color="#33414b")
    _save(fig, "fig_architecture")


def make_bitgrid():
    import numpy as np
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    g = 5
    bits = rng.integers(0, 2, (g, g))
    up = np.kron(bits, np.ones((24, 24)))                      # write: upsampled grid
    act = np.clip(up * 0.7 + 0.15 + rng.normal(0, 0.13, up.shape), 0, 1)
    act = gaussian_filter(act, 2.0)                             # decoder activation (noisy/blurred)
    pooled = act.reshape(g, 24, g, 24).mean((1, 3))            # avg-pool back to g×g
    rec = (pooled > 0.5).astype(int)

    fig = plt.figure(figsize=(15.5, 7.4))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off"); bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    def panel(rect, img, title, cmap, binary=True):
        ax = fig.add_axes(rect)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        n = img.shape[0]
        if binary:
            ax.set_xticks(np.arange(-.5, n, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n, 1), minor=True)
            ax.grid(which="minor", color="#888", lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10.5)
        return ax

    # row labels
    bg.text(0.015, 0.80, "ENCODE\n(write)", fontsize=12, fontweight="bold", color=E_EMBED, va="center")
    bg.text(0.015, 0.27, "DECODE\n(read)", fontsize=12, fontweight="bold", color=E_EXTR, va="center")

    # top (write): grid -> upsample -> stego
    panel([0.10, 0.58, 0.18, 0.30], bits, f"payload → {g}×{g} grid\n(each cell = 1 bit)", "gray_r")
    panel([0.37, 0.58, 0.18, 0.30], up, "upsample (nearest)\n→ bit-grid map  1×H×W", "gray_r", binary=False)
    # stego (real image) in the middle-right
    axs = fig.add_axes([0.66, 0.30, 0.26, 0.40])
    stego_path = FIG.parent / "capture" / "screen1" / "stego_000.png"
    if stego_path.exists():
        axs.imshow(plt.imread(str(stego_path)))
    axs.set_xticks([]); axs.set_yticks([])
    axs.set_title("Stego QR\n(encoder embeds the map into the cover)", fontsize=10.5)

    # bottom (read): activation -> pool -> recovered grid
    panel([0.37, 0.10, 0.18, 0.30], act, "decoder activation map\n1×H×W", "magma", binary=False)
    panel([0.10, 0.10, 0.18, 0.30], rec, f"avg-pool → {g}×{g}\n→ read cells → bits", "gray_r")

    # arrows (figure fraction)
    A = lambda p, q, c: bg.annotate("", xy=q, xytext=p,
                                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2,
                                                    shrinkA=3, shrinkB=3))
    A((0.285, 0.73), (0.365, 0.73), E_EMBED)        # grid -> upsample
    A((0.555, 0.73), (0.655, 0.62), E_EMBED)        # upsample -> stego
    A((0.79, 0.30), (0.55, 0.25), E_CHAN)           # stego -> activation (channel + decoder)
    A((0.365, 0.25), (0.285, 0.25), E_EXTR)         # activation -> pooled grid
    bg.text(0.70, 0.21, "channel + decoder conv", fontsize=8.5, color=E_CHAN, style="italic")

    match = int((bits == rec).sum())
    bg.text(0.5, 0.025,
            "A single weight-shared convolution writes/reads any cell, so L=100 bits are learned in "
            "~250 steps (100% recovery). Global broadcast (each bit constant over all H×W) must "
            f"disentangle L global patterns and fails at L=100 (~chance).   [demo: {match}/{g*g} cells recovered]",
            ha="center", fontsize=9.2, color="#33414b")
    _save(fig, "fig_bitgrid")


def make_examples():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import numpy as np
    from PIL import Image
    from stegaqr.core import StegaQREncoder
    from stegaqr.utils.qr_utils import generate_cover_qr_rgb
    from stegaqr.utils.metrics import psnr, ssim
    from pyzbar.pyzbar import decode as zbar

    public, hidden = "ICTAI2026", b"Hi"
    specs = [
        ("Imperceptible\n(clean-trained)", "experiments/full/main_cross_channel_clean_s42/best_model.pt"),
        ("Robust\n(distortion-trained)", "experiments/full/main_cross_channel_distort_s42/best_model.pt"),
        ("Hybrid robust\n(structural mask)", "models/pretrained/stegaqr_default.pt"),
    ]
    root = Path(__file__).parent.parent
    cols = []
    for label, rel in specs:
        enc = StegaQREncoder(str(root / rel), ecc="rep3", device="cpu")
        cfg = enc.cfg
        cover, _ = generate_cover_qr_rgb(public, cfg["qr_version"], cfg["ec_level"], cfg["module_size"])
        stego = np.asarray(enc.encode(public, hidden)).astype(np.float32) / 255.0
        resid = np.abs(stego - cover)
        ok = bool(zbar(Image.fromarray((stego * 255).astype(np.uint8))))
        cols.append(dict(label=label, cover=cover, stego=stego, resid=resid,
                         psnr=psnr(cover, stego), ssim=ssim(cover, stego),
                         pub=ok, maxd=resid.max()))

    n = len(specs)
    fig, axes = plt.subplots(2, n + 1, figsize=(3.3 * (n + 1), 7.0))
    # column 0: cover (top), legend text (bottom)
    axes[0, 0].imshow(cols[0]["cover"]); axes[0, 0].set_title("Cover QR\n(no hidden data)", fontsize=11)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "top: stego QR\n(public still scans)\n\nbottom: |stego − cover|\n"
                    "normalized to show\nwhere bits are embedded",
                    ha="center", va="center", fontsize=9.5, color="#33414b")
    for j, c in enumerate(cols, start=1):
        axes[0, j].imshow(c["stego"])
        pub = "public ✓" if c["pub"] else "public ✗"
        axes[0, j].set_title(f"{c['label']}\nPSNR {c['psnr']:.1f} dB · SSIM {c['ssim']:.3f} · {pub}",
                             fontsize=10)
        r = c["resid"] / (c["resid"].max() + 1e-9)
        axes[1, j].imshow(r)
        axes[1, j].set_title(f"residual ×{1/(c['maxd']+1e-9):.0f}  (max Δ {c['maxd']:.2f})", fontsize=9.5)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Cover vs. stego across operating points — imperceptibility ↔ robustness trade-off",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig_examples")


def make_capture():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import json
    import numpy as np
    from PIL import Image, ImageOps
    from stegaqr.core import StegaQRDecoder
    from stegaqr.coding import bits_to_bytes

    root = Path(__file__).parent.parent
    cap = root / "capture" / "screen1"
    manifest = json.loads((cap / "manifest.json").read_text())
    by_text = {it["public_text"]: it for it in manifest["items"]}
    photos = sorted((cap / "stegaqrtest").glob("*.jpg"))

    dec = StegaQRDecoder(str(root / "models/pretrained/stegaqr_default.pt"), ecc="rep3", device="cpu")
    sym = (4 * dec.cfg["qr_version"] + 17) * dec.cfg["module_size"]

    # representative photo: decode it
    rep = photos[0]
    pil = ImageOps.exif_transpose(Image.open(rep).convert("RGB"))
    rect = dec._rectify(pil, sym)
    public, hidden, meta = dec.decode(pil)
    item = by_text.get(public)
    gt = bits_to_bytes(np.array(item["message_bits"], np.uint8)) if item else b""
    match = hidden.rstrip(b"\x00") == gt.rstrip(b"\x00")

    fig, axd = plt.subplot_mosaic("ABC\nDEE", figsize=(13, 7.6))
    for key, ph in zip("ABC", photos[:3]):
        im = ImageOps.exif_transpose(Image.open(ph).convert("RGB"))
        axd[key].imshow(im); axd[key].set_xticks([]); axd[key].set_yticks([])
        axd[key].set_title(ph.name, fontsize=8)
    axd["A"].set_ylabel("raw phone photos", fontsize=11)
    fig.text(0.5, 0.93, "EXP-002 — stego QR displayed on a monitor, photographed with a phone",
             ha="center", fontsize=13, fontweight="bold")

    axd["D"].imshow(rect); axd["D"].set_xticks([]); axd["D"].set_yticks([])
    axd["D"].set_title("OpenCV localize + perspective-rectify\n→ decoder input", fontsize=10)

    axd["E"].axis("off")
    lines = [
        ("Representative decode", "header"),
        (f"public payload (pyzbar):   {public!r}", "ok"),
        (f"hidden message (neural+ECC): {hidden.rstrip(chr(0).encode()).hex()}  "
         f"{'✓ matches ground truth' if match else '✗'}", "ok" if match else "bad"),
        (f"decoder confidence:        {meta.get('confidence', 0):.3f}", "plain"),
        ("", "plain"),
        ("Aggregate over all 10 photos", "header"),
        ("QR located:            10 / 10", "ok"),
        ("public decode (pyzbar): 10 / 10", "ok"),
        ("hidden MESSAGE decode:  10 / 10", "ok"),
        ("", "plain"),
        ("100% recovery through a real display→camera channel", "note"),
        ("(perspective · glare · moiré · JPEG · screen colour),", "note"),
        ("from a model trained only on the simulated distortion layer.", "note"),
    ]
    colors = {"header": "#15202b", "ok": E_EXTR, "bad": E_CHAN, "plain": "#33414b", "note": "#5b6b73"}
    y = 0.96
    for txt, kind in lines:
        if not txt:
            y -= 0.045; continue
        axd["E"].text(0.02, y, txt, fontsize=11 if kind == "header" else 10.2,
                      fontweight="bold" if kind == "header" else "normal",
                      style="italic" if kind == "note" else "normal",
                      family="monospace" if kind in ("ok", "bad", "plain") else "sans-serif",
                      color=colors[kind], transform=axd["E"].transAxes, va="top")
        y -= 0.072 if kind == "header" else 0.066
    _save(fig, "fig_capture")


DIAGRAMS = {"system": make_system, "architecture": make_architecture,
            "bitgrid": make_bitgrid, "examples": make_examples, "capture": make_capture}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = DIAGRAMS.keys() if which == "all" else [which]
    for n in names:
        DIAGRAMS[n]()


if __name__ == "__main__":
    main()
