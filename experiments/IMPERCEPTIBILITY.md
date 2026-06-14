# Capacity / Imperceptibility / Robustness Trade-off — StegaQR

**Date:** 2026-06-14
**Setup:** cross_channel (v3 spatial-grid arch), capacity=100 bits, QR v4,
module_size=4 (132x132), fresh random covers+payloads, decode-only BCE, lr=2e-3,
~1500-2500 steps. EVAL = held-out fresh batch. PSNR = stego vs cover.

The `perturbation_bound` hard-caps the per-pixel perturbation; it is the primary
imperceptibility knob. The default was 0.3 — ~30x larger than needed.

## Clean frontier (no distortion)

| perturbation_bound | bit accuracy | PSNR (dB) |
|--------------------|--------------|-----------|
| 0.30               | 100.0%       | 15.3      |
| 0.15               | 100.0%       | 21.1      |
| 0.08               | 100.0%       | 26.6      |
| 0.04               | 100.0%       | 33.6      |
| 0.02               | 100.0%       | 38.7      |
| 0.01               | 100.0%       | 44.2      |

**Under clean conditions, 100-bit recovery is essentially free at any bound** —
accuracy is 100% all the way down to 0.01 (44 dB, visually imperceptible).
Capacity is nowhere near saturated; the binding constraint is robustness, not
clean decodability.

## Robust frontier (with distortion layer)

(running — noise/jpeg/brightness/colorshift/blur; small bounds expected to fail
as the perturbation drops below the noise floor ~0.1)

| perturbation_bound | bit accuracy (distorted) | PSNR (dB) | note |
|--------------------|--------------------------|-----------|------|
| 0.30               | 100.0%                   | 14.7      |      |
| 0.15               | 100.0%                   | 20.3      |      |
| 0.10               | 100.0%                   | 23.4      | knee — robust + best PSNR |
| 0.06               | 99.6%                    | 27.6      | graceful degradation |
| 0.03               | ~55% (unstable)          | 33.5      | perturbation < noise floor (~0.1); training collapses |

The distortion layer's noise reaches ~0.1 (sigma up to 25/255). When the
perturbation bound falls below that floor the hidden signal is drowned, accuracy
collapses, and training destabilises (acc fell from 1.0 -> 0.55 within the run at
bound 0.03).

## End-to-end validation at the recommended operating point

cross_channel, capacity=100, ms=4, **perturbation_bound=0.1**, trained WITH the
distortion layer AND the full perceptual + decodability loss (16 epochs, quick),
evaluated with `scripts/evaluate.py` (n=48). Covers use the D7 inversion fix.

| metric                   | clean   | distorted |
|--------------------------|---------|-----------|
| hidden bit accuracy      | 100.0%  | 100.0%    |
| full-decode rate         | 100.0%  | 100.0%    |
| public QR decode rate    | 100.0%  | 100.0%    |
| PSNR (stego vs cover)    | 23.15 dB| 23.15 dB  |
| SSIM (stego vs cover)    | 0.989   | 0.989     |

All three goals met simultaneously: the hidden payload is perfectly recovered
(clean and distorted), AND the public QR still scans (pyzbar) 100% of the time, at
SSIM 0.989. (PSNR is stego-vs-cover so it is identical under the two eval modes.)

Note: PSNR here (~23 dB) matches the hard-bound frontier rather than exceeding it
— the encoder uses the full bound because the perceptual weight is not strong
enough to pull perturbation below the cap. Strengthening the perceptual weight /
adding LPIPS to get *adaptive* sub-bound perturbation is the next imperceptibility
lever.

> Gotcha fixed: `scripts/evaluate.py` must rebuild the encoder with the SAME
> `perturbation_bound` as training (it scales the forward pass; it is not a learned
> weight). Reading it from the checkpoint config now.

## Adaptive perturbation (pushing PSNR) and the message-robustness wall

Goal: beat the hard-bound ceiling (23 dB) by letting the encoder use *adaptive*
sub-bound perturbation — a generous bound (0.3) + a strong perceptual loss that is
ramped in after a decode-only warmup. Findings:

- **GroupNorm is required.** With BatchNorm, strong perceptual pressure made the
  hidden signal tiny and eval-mode decoding oscillated/collapsed (the D6 effect).
  GroupNorm makes training stable; eval accuracy holds.
- **Warmup + linear ramp of the perceptual weight** prevents the decode/perceptual
  tug-of-war from collapsing accuracy. Implemented in `training.py`
  (`warmup_decode_only`, `perc_ramp_epochs`).
- **Bit-level PSNR reaches 36-39 dB** at ~98-99% *bit* accuracy under distortion.
- **But robust full-MESSAGE decode (all 100 bits) is a hard wall ~20 dB.** At
  higher PSNR the per-pixel perturbation drops below the distortion noise floor for
  a few cells, flipping ~1-2% of bits; since full-decode needs every bit,
  0.98^100 ~ 13-19% message success. Measured: a 39 dB model gives 98.3% distorted
  bit acc but only ~19% distorted full-decode.

### Checkpoint-selection bug (found & fixed)
Selecting `best_model` on `val_acc` alone kept the decode-only **warmup** model
(acc saturates at 1.0 before perceptual shrinks perturbation) -> a 14 dB
checkpoint despite the run reaching 36 dB. Then selecting on clean PSNR kept a
beautiful-but-fragile model (clean-perfect, ~19% distorted full-decode). Fix:
**validate under deterministic distortion and select to MAXIMISE PSNR SUBJECT TO
robust full-decode >= target** (`ROBUST_FDR_TARGET=0.98`). New robust val metrics
(`val_bit_acc_robust`, `val_fdr_robust`, `val_psnr`) are logged every epoch.

### Deployable operating point (cross_channel, cap=100, ms=4, with distortion)
Robust-gated selection now yields, on held-out data:

| metric                | clean | distorted |
|-----------------------|-------|-----------|
| hidden bit accuracy   | 100%  | 100%      |
| full-decode rate      | 100%  | 100%      |
| public QR decode rate | 100%  | 100%      |
| PSNR / SSIM           | 20.4 dB / 0.980 | -- |

Up from the original 14.7 dB, now with robustness-aware selection.

### To break the 20 dB message-robustness wall -> ERROR-CORRECTION CODING
Add ECC (repetition / Hamming / BCH / Reed-Solomon) to the payload so ~98-99%
bit accuracy -> ~100% message recovery. This decouples bit-accuracy from message
success and is how StegaStamp and real watermarking systems operate at 30+ dB.
Cost: net payload < raw capacity (the rate). This is the recommended next step to
get 30+ dB WITH robust full-message decode.

## Takeaways (folded into the paper)
- **Clean decodability is free; robustness is the binding constraint.** The gap
  between the two curves (e.g. 44 dB clean vs 23 dB robust at fixed 100% accuracy)
  is the paper's central trade-off figure.
- **Recommended operating point: perturbation_bound = 0.10** — 100% robust bit
  accuracy at 23.4 dB, vs the old default 0.3 at 14.7 dB (a ~9 dB imperceptibility
  gain for zero robustness cost). Default updated in configs/base.yaml,
  scripts/train.py, training.py.
- Open refinements: (a) generous bound + stronger perceptual loss for *adaptive*
  perturbation (concentrate where it hurts least) may beat the hard-bound frontier;
  (b) higher module_size adds spatial bandwidth and should raise PSNR at fixed
  robustness; (c) per-mode (segregated/hybrid) and per-capacity curves.
