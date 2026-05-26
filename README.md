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

## Reproducing Results

Every figure and number in the paper traces to a logged experiment in `LOGBOOK.md`.

```powershell
# Run all experiments
python scripts/run_experiment.py --config configs/base.yaml

# Generate paper figures
python scripts/generate_figures.py
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
