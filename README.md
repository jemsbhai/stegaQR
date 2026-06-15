# StegaQR: Neural Steganographic Data Embedding in QR Codes with Switchable Channel Modes

**Target venue:** IEEE ICTAI 2026 (38th IEEE International Conference on Tools with Artificial Intelligence)  
**Submission deadline:** June 30, 2026  
**Author:** Muntaser Syed  

## Overview

StegaQR embeds hidden data payloads inside standard-looking QR code images using learned neural encoder-decoder networks. A standard QR reader decodes the public payload normally; only the StegaQR decoder can extract the hidden steganographic message from subtle color perturbations.

Three steganographic modes are supported, switchable at encode time:

| Mode | Hidden data location | Failure model | Use case |
|------|---------------------|---------------|----------|
| **Segregated-channel** | Independent per-channel residuals | Per-channel independent | Max fault isolation |
| **Cross-channel** | Inter-channel relationships (ratios, latent space) | Correlated | Max capacity |
| **Hybrid (QR-anchored)** | Color perturbations within standard-decodable bounds | Correlated, self-calibrating | Real-world deployment |

## Key Contributions

1. **QR codes as cover objects** — unlike prior work (StegaStamp, PPRSteg) that hides QR codes inside natural images, we hide data inside QR codes themselves, exploiting their rigid spatial structure as free alignment anchors.
2. **Switchable channel modes** — segregated-channel vs. cross-channel vs. hybrid, with configurable capacity-imperceptibility tradeoff.
3. **Dual neural encoder-decoder** — both a fully neural codec and a classical-embed + neural-decode pipeline, with comparative evaluation.
4. **Configurable capacity** — from watermark-level (tens of bits) to message-level (hundreds of bits), controlled via a capacity parameter.

## Setup

```powershell
# Clone and install
cd E:\data\code\claudecode\mQRstego
pip install -e ".[dev,ml]"
```

## Quick start (CLI)

A trained checkpoint ships at `models/pretrained/stegaqr_default.pt` (hybrid mode,
distortion-trained). The hidden payload is protected by error-correcting code, so
capacity is small (a few bytes) — run `info` to see the exact limit.

```powershell
stegaqr info   --model models/pretrained/stegaqr_default.pt
stegaqr encode --model models/pretrained/stegaqr_default.pt `
               --public "https://example.com/p" --message "ID42" --out stego.png
stegaqr decode --model models/pretrained/stegaqr_default.pt --image stego.png
```

`encode` writes a normal-looking QR: any reader scans `--public`; only StegaQR
recovers `--message`. (Use `--device cpu` if you have no GPU.)

## Library

```python
from stegaqr import encode_hidden, decode_hidden
img = encode_hidden("https://example.com", b"ID42",
                    model="models/pretrained/stegaqr_default.pt", ecc="rep3")
public, hidden, meta = decode_hidden(img, model="models/pretrained/stegaqr_default.pt")
```

## Reproducing results

Every figure and number traces to a logged run in `LOGBOOK.md` / `experiments/full/`.

```powershell
# Train one model (see scripts/train.py --help for modes, ECC-free vs adaptive, etc.)
python scripts/train.py --mode hybrid --mask-aware --no-calibration --output-dir experiments/run1

# Evaluate a checkpoint (clean + distortion + ECC message-decode)
python scripts/evaluate.py --checkpoint experiments/run1/best_model.pt

# Full experiment matrix (modes x clean/distortion x seeds, capacity sweep, baselines)
python scripts/run_experiments.py            # resumable; --quick for a smoke test

# Aggregate -> experiments/full/RESULTS.md, then figures -> figures/
python scripts/aggregate_results.py
python scripts/generate_figures.py
```

Physical print-scan robustness study:

```powershell
python scripts/export_for_capture.py --checkpoint <ckpt> --out capture/exp1   # print/display + photograph
python scripts/decode_from_photo.py  --photos capture/exp1_photos --manifest capture/exp1/manifest.json
```

## Project Structure

```
mQRstego/
├── configs/              # Experiment configurations (YAML)
├── data/                 # Datasets (raw + processed)
├── src/stegaqr/          # Source code
│   ├── models/           # Encoder, decoder, discriminator architectures
│   ├── data/             # Data loading and augmentation
│   ├── utils/            # Metrics, seeding, checkpointing
│   └── analysis/         # Statistical analysis scripts
├── scripts/              # Entry-point scripts
├── experiments/          # Experiment outputs (gitignored large files)
├── figures/              # Publication-ready figures
├── notebooks/            # Exploratory analysis
├── tests/                # Test suite
├── paper/                # LaTeX source
└── models/pretrained/    # Trained model weights
```

## License

MIT
