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

(To be filled after experiment completes)

### Observations

(To be filled after experiment completes)

### Interpretation

(To be filled after experiment completes)

### Artifacts

- Checkpoints: experiments/exp_001{a,b}_{seg,cross,hybrid}_{clean,distort}/checkpoints/
- Logs: experiments/exp_001{a,b}_{seg,cross,hybrid}_{clean,distort}/logs/
- Config: experiments/exp_001{a,b}_{seg,cross,hybrid}_{clean,distort}/config.json
- Seeds: experiments/exp_001{a,b}_{seg,cross,hybrid}_{clean,distort}/seed.json

---
