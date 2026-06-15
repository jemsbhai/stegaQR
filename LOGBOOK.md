# Experimental Logbook — StegaQR

**Project:** StegaQR — Neural Steganographic Data Embedding in QR Codes  
**Target:** IEEE ICTAI 2026, Best Paper Award  
**Researcher:** Muntaser Syed  
**Started:** 2026-05-26  

---

*Entries below are append-only. Corrections are added as dated addenda.*

---

## EXP-001: Baseline Training — All Three Modes (Clean + Distortion)

**Date:** 2026-05-26
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis

All three encoder-decoder architectures (segregated, cross-channel, hybrid) can learn to embed and recover hidden bits from QR code images with >95% bit accuracy at ~100-bit capacity under clean conditions. Adding the differentiable distortion layer during training will reduce clean accuracy slightly but dramatically improve robustness to degraded inputs — addressing the weakness identified in our ICMLA multispecqr paper where the ML decoder trained on clean data failed on brightness shifts because it learned absolute rather than relative color mappings.

### Independent Variables

- **Mode:** segregated (99 bits), cross_channel (100 bits), hybrid (100 bits)
- **Distortion:** none (clean) vs. full distortion layer (noise, JPEG, brightness, color shift, blur)
- Total: 6 configurations (3 modes × 2 distortion settings)

### Dependent Variables / Metrics

- Bit accuracy: fraction of correctly recovered hidden bits (per-sample, then averaged)
- Full decode rate: fraction of samples with zero bit errors
- Training loss convergence (decode, perceptual, decodability components)
- PSNR of stego vs. cover images (visual quality)
- Training time per epoch (seconds)
- Total parameter count per model

### Control Conditions

- QR version: 4 (33×33 modules)
- EC level: M
- Capacity: 99 bits (segregated, divisible by 3) / 100 bits (cross-channel, hybrid)
- Epochs: 50
- Batch size: 32
- Learning rate: 1e-3, cosine annealing
- Optimizer: Adam, weight_decay=1e-4
- Gradient clipping: max_norm=1.0
- Training samples: 5000/epoch, Validation: 750
- Seed: 42

### Protocol

**Phase A — Clean (no distortion):**
```powershell
python scripts/train.py --mode segregated --capacity 99 --epochs 50 --no-distortion --output-dir experiments/exp_001a_seg_clean --seed 42
python scripts/train.py --mode cross_channel --capacity 100 --epochs 50 --no-distortion --output-dir experiments/exp_001a_cross_clean --seed 42
python scripts/train.py --mode hybrid --capacity 100 --epochs 50 --no-distortion --output-dir experiments/exp_001a_hybrid_clean --seed 42
```

**Phase B — With distortion:**
```powershell
python scripts/train.py --mode segregated --capacity 99 --epochs 50 --output-dir experiments/exp_001b_seg_distort --seed 42
python scripts/train.py --mode cross_channel --capacity 100 --epochs 50 --output-dir experiments/exp_001b_cross_distort --seed 42
python scripts/train.py --mode hybrid --capacity 100 --epochs 50 --output-dir experiments/exp_001b_hybrid_distort --seed 42
```

### Environment

- **Hardware:** NVIDIA RTX 4090 (24GB VRAM), 64GB RAM
- **Software:** Windows 11, Python 3.12, PyTorch 2.x, CUDA 12.x
- **Git commit:** (to be recorded at run time)
- **Config file:** configs/base.yaml
- **Seeds:** master=42, python=42, numpy=42, torch=42, torch_cuda=42

### Design Note

In our ICMLA multispecqr paper, the ML decoder (trained on clean data only) failed on brightness scaling because it learned absolute color mappings. Here, the differentiable distortion layer sits between encoder and decoder during training, forcing the encoder to learn perturbations that are robust to photometric/geometric distortions. Phase A vs Phase B directly tests whether this end-to-end distortion training addresses that weakness.

### Results

See addendum 2026-06-15 below (the original v2 architecture failed; the experiment
was re-run after a root-cause fix). Aggregated tables: experiments/full/RESULTS.md.

### Observations

See addendum 2026-06-15.

### Interpretation

See addendum 2026-06-15.

### Artifacts

- Per-run metrics: experiments/full/*/results.json (best_model.pt gitignored)
- Aggregated: experiments/full/RESULTS.md, experiments/full/results.csv
- Figures: figures/
- Config/seeds: experiments/full/*/config.json, seed.json

---

## EXP-001 — Addendum (2026-06-15): root-cause fix and re-run

**Researcher:** Muntaser Syed + automated diagnosis (Claude)
**Status:** completed

### Correction to the original plan
The first EXP-001 runs (v2 architecture) **failed** — bit accuracy plateaued at
~54% (chance) instead of the hypothesised >95%. Rigorous diagnosis
(experiments/DIAGNOSTICS.md) found **three independent defects**, all fixed:

1. **Global payload broadcast** → replaced with a **spatial bit-grid** (each bit in
   its own grid cell; weight-shared conv reads all cells). v2 broadcast forced the
   net to learn L independent global patterns and could not generalise (~chance at
   L=100). Fix: 100% held-out bit accuracy in ~250 steps.
2. **BatchNorm train/eval mismatch** → **GroupNorm**. With a tiny perturbation,
   BN running stats diverged from batch stats; train 99.8% / val 50%.
3. **Colour-inverted covers** (`get_matrix()` True=dark mapped to white) → every
   synthetic cover was unscannable, voiding the public-decode premise. Fixed mapping
   + optional quiet zone. Regression-tested.

Hardware note: actual GPU is an RTX 4090 **Laptop (16GB)**, not the 24GB desktop
assumed; experiments use module_size=4 (132x132), batch 16.

### Recipe (fixed pipeline)
Spatial-grid encoder/decoder (GroupNorm), capacity 100, QR v4 / EC M, module_size 4,
batch 16, 24 epochs, Adam lr 1e-3 cosine. Adaptive embedding: decode-only warmup
(3 ep) then linear ramp of the perceptual loss (8 ep), perturbation_bound 0.3,
lambda_perceptual 4. Best model = max(robust full-decode + robust bit-acc + small
PSNR tie-break) under stochastic distortion. 5 seeds {42,123,7,99,2024}.

### Results — mode comparison (cap 100, mean +/- 95% CI, n=5 seeds)

| mode | training | clean FDR | distorted FDR | PSNR (dB) | SSIM | public-decode |
|------|----------|-----------|---------------|-----------|------|---------------|
| cross_channel | clean      | 99.5 | 10.0 +/-19.6 | 61.7 | 1.00 | 100 |
| cross_channel | distortion | 99.8 | **99.2 +/-1.5** | 19.0 | 0.97 | 100 |
| segregated    | clean      | 100.0 | 13.9 +/-11.3 | 63.5 | 1.00 | 100 |
| segregated    | distortion | 99.8 | **99.1 +/-0.8** | 17.6 | 0.96 | 100 |
| hybrid        | clean      | 99.8 | 13.9 +/-11.3 | 58.8 | 1.00 | 100 |
| hybrid        | distortion | 100.0 | **100.0 +/-0.0** | 22.3 | 0.99 | 100 |

(FDR = full-decode rate, all 100 bits correct. Values without CI are ~0 variance.
Hybrid uses the stabilized config: mask-aware grid + no calibration branch, see
addendum 2 below / DIAGNOSTICS D9.)

### Studies
- **ECC** (distortion cross_channel, under distortion): Hamming(7,4) **100%** message
  decode (56 net bits); rep3 99.5% (33 bits); rep5 98.1% (20 bits); raw 100-bit 99.2%.
- **Capacity** (cross_channel, distortion): 25-200 bits hold ~99-100% distorted
  bit-acc; FDR 97.7% @25, 99.2% @100, more variable @150 (one weak seed).
- **Classical LSB baseline**: clean 100% / distorted **53.8% bit, 3.9% FDR** — LSB is
  destroyed by distortion.
- **Architecture ablation** (decode-only, 100 bits, seed 42): the spatial grid reaches
  **100%** bit accuracy (clean & distorted); the HiDDeN-style global-broadcast baseline
  stays at **~52% (chance)** — the grid layout is the enabling contribution.
- **Mask-aware hybrid / broadcast under the adaptive recipe (lambda_perc=4)**: both
  collapse (perturbation -> 0) — the weaker decoders cannot hold signal against the
  perceptual loss. The grid cross/segregated modes do not. Hybrid stabilisation
  (gentler perceptual schedule) remains open.

### Interpretation
- **The central hypothesis holds.** End-to-end distortion training converts a
  fragile clean-trained codec (10-14% distorted message recovery) into a robust one
  (**99%+**), at the cost of imperceptibility (~62 dB clean-trained -> ~18 dB). This
  is exactly the ICMLA-multispecqr weakness (clean-trained decoder failing on
  photometric shifts) that the differentiable distortion layer was meant to fix.
- The public QR remains **100% standard-decodable** in every setting — the
  steganography does not break the cover.
- **cross_channel and segregated are equivalent** (~99% robust FDR); segregated adds
  per-channel fault isolation at no measured cost.
- **Hybrid is now stable and matches the other modes** (addendum 2 / DIAGNOSTICS D9):
  with the mask-aware grid + no calibration branch, distortion-trained hybrid reaches
  **100% +/- 0.0 distorted full-decode** (n=5) at 22.3 dB / SSIM 0.99 — slightly higher
  PSNR than cross_channel/segregated thanks to the structural mask. The QR-anchored
  mode thus gives guaranteed structural preservation AND full robust decoding.
- ECC removes the residual full-message brittleness: 100% message recovery with
  Hamming(7,4) at the distortion-trained operating point.

---

## EXP-001 — Addendum 2 (2026-06-15): hybrid stabilization

The first matrix exposed hybrid as unstable (collapse to ~0 perturbation; raw
accuracy capped ~94%). Diagnosed (DIAGNOSTICS D9) to two causes: (i) the uniform
grid placed bits on finder-pattern cells the mask cannot perturb; (ii) the
calibration network's random init applied a random affine colour transform before
decoding, giving a stochastic chance-saddle cold-start that the perceptual ramp then
collapsed. Fix: **mask-aware grid + no calibration branch** (the conv decoder handles
photometric distortion implicitly, as cross_channel does). The hybrid rows in the
table above were re-run (5 seeds) with this stabilized config: distortion-trained
hybrid now reaches **100% +/- 0.0 distorted full-decode** at 22.3 dB / SSIM 0.99,
matching cross_channel and segregated, with the public QR 100% decodable. ECC rep3
and Hamming(7,4) give 100% message decode.

---

## EXP-002: Physical screen-to-phone robustness (real capture)

**Date:** 2026-06-15
**Researcher:** Muntaser Syed
**Type:** physical / empirical
**Status:** completed

### Hypothesis
A StegaQR stego image trained only against the *simulated* differentiable distortion
layer will still yield correct hidden-message recovery when displayed on a screen and
photographed by a phone — i.e. the learned robustness transfers to a real
display->camera channel (the gap that limited prior work).

### Protocol
- Model: `models/pretrained/stegaqr_default.pt` (hybrid, distortion-trained, mask-aware,
  no calibration), ECC = rep3 (4 hidden bytes). 10 stego QR codes with distinct public
  payloads + random hidden messages exported via `scripts/export_for_capture.py`
  (upscale 12, quiet zone 4); ground truth in `capture/screen1/manifest.json`.
- Capture: each code displayed on a monitor and photographed with a Google Pixel
  (`capture/screen1/stegaqrtest/PXL_*.jpg`), free-hand, mixed angle/distance.
- Decode: `scripts/decode_from_photo.py` — OpenCV detect + perspective-rectify ->
  pyzbar (public) and neural decoder + ECC (hidden); each photo matched to its code by
  the scanned public text.

### Results (n=10)
| metric | result |
|--------|--------|
| QR located            | 10/10 |
| public decode (pyzbar)| 10/10 |
| hidden MESSAGE decode | **10/10** |

Per-photo log: `capture/screen1/decode_log.txt`; summary: `capture/screen1/capture_results.json`.

### Interpretation
**The simulated-distortion training transfers to reality.** 100% hidden-message
recovery through a genuine display->camera pipeline (perspective, glare, moiré, JPEG,
screen colour response) — and the public QR remained 100% standard-scannable. This
directly addresses the real-world weakness of the prior ICMLA multispecqr decoder.
Next: a degradation sweep (steeper angles / low light / distance) to locate the
failure boundary, and the same at higher capacity.

---
