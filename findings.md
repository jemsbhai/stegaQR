# Findings — StegaQR

## Curated Summary

*Publication-ready prose. Every claim traces to a logged run (experiments/full/,
LOGBOOK EXP-001) or diagnostic (experiments/DIAGNOSTICS.md).*

### F1 — Spatial bit-layout is what makes neural QR steganography learnable
Broadcasting each payload bit as a constant spatial channel (the HiDDeN/StegaStamp
convention) fails for QR covers: at 100 bits it never escapes chance accuracy. Laying
each bit in its own cell of an upsampled grid — so a single weight-shared convolution
learns to read every cell — reaches 100% held-out bit accuracy in a few hundred steps.
This is the core architectural finding (DIAGNOSTICS D2–D5).

### F2 — End-to-end distortion training is necessary and sufficient for robustness
Clean-trained codecs are near-perfectly imperceptible (~62 dB PSNR) but collapse under
distortion (10–14% full-message recovery). Training through a differentiable distortion
layer raises distorted full-message recovery to **99%+** (cross_channel 99.2±1.5%,
segregated 99.1±0.8%, n=5 seeds), trading imperceptibility down to ~18 dB. This directly
remedies the clean-trained-decoder fragility observed in prior work.

### F3 — The public QR is never broken
Across all modes and training regimes, a standard reader (pyzbar) decodes the public
payload of the stego image 100% of the time — the hidden channel is transparent to
ordinary use.

### F4 — Imperceptibility, robustness and capacity form a tunable frontier
Under clean conditions, 100-bit recovery is essentially free at any perturbation budget
(PSNR 15–44 dB). Robustness is the binding constraint: robust full-100-bit recovery
without coding sits near ~20 dB. Error-correction coding decouples bit-accuracy from
message success — Hamming(7,4) yields 100% message recovery (56 net bits) and adaptive
embedding reaches 36.7 dB / SSIM 0.9996 with a robust 33-bit message. Capacity scales to
at least 200 embedded bits at ~99% distorted bit accuracy.

### F5 — Neural >> classical under distortion
A classical LSB baseline is perfect when clean (100%, 78 dB) but is destroyed by
distortion (53.8% bit accuracy, 3.9% full-decode) — the learned codec's robustness is
the difference.

### Open item — hybrid mode
The QR-anchored hybrid mode (hard finder/timing mask + self-calibration) is the
intended deployment mode but is currently the weakest: the mask prevents finder-pattern
cells from carrying bits (raw accuracy capped ~94%) and the adaptive perceptual schedule
destabilises it (perturbation collapse on several seeds). A mask-aware bit layout (bits
placed only on data-rich cells) and a gentler perceptual schedule are under evaluation.

---

## Raw Findings Log

*Chronological, append-only. Each entry is a dated observation tied to a specific experiment.*

- **2026-06-14** — v2 architecture failed (EXP-001): ~54% bit accuracy at 100 bits.
  Root-caused to global broadcast + BatchNorm + inverted covers (DIAGNOSTICS.md).
- **2026-06-14** — Spatial bit-grid + GroupNorm fix: 100% held-out bit accuracy, clean
  and under distortion.
- **2026-06-14** — Imperceptibility frontier mapped; ECC (repetition soft-decode,
  Hamming) added; 36.7 dB / SSIM 0.9996 with 100% robust 33-bit message.
- **2026-06-15** — Full EXP-001 matrix (3 modes × clean/distortion × 5 seeds) +
  capacity sweep + classical baseline completed; see LOGBOOK addendum and RESULTS.md.

---
