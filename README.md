# StegaQR

StegaQR embeds a small private payload inside a QR code while preserving the public
payload for ordinary QR readers. A trained neural decoder recovers the private data.

The package includes a trained hybrid model, repetition and Hamming error correction,
a command line interface, and a Python API. The bundled model carries 100 coded bits.
With the default repetition code, it carries up to 4 hidden bytes.

## Install

```powershell
pip install stegaQR
```

StegaQR supports Python 3.10 through 3.12. PyTorch is installed as a required
dependency. Public QR decoding uses `pyzbar`. Some Linux and macOS environments also
need the native ZBar library supplied by the operating system.

## Command line

Inspect the bundled model:

```powershell
stegaqr info
```

Create a QR code with a public URL and a private four byte identifier:

```powershell
stegaqr encode --public "https://example.com/p" --message "ID42" --out stego.png
```

Recover both payloads:

```powershell
stegaqr decode --image stego.png
```

The generated image includes a quiet zone and nearest neighbor upscaling for reliable
scanning. Use `--model PATH` to select a different trained checkpoint.

## Python API

```python
from stegaqr import decode_hidden, encode_hidden

image = encode_hidden("https://example.com", b"ID42", device="cpu")
public, hidden, metadata = decode_hidden(image, device="cpu")

assert public == "https://example.com"
assert hidden.rstrip(b"\x00") == b"ID42"
```

CUDA is used when requested and available. Otherwise the package falls back to CPU.

## Embedding modes

The research implementation supports three trained architectures:

| Mode | Design | Purpose |
|---|---|---|
| Segregated | One encoder and decoder per color channel | Channel fault isolation |
| Cross-channel | Joint RGB encoder and decoder | Flexible signal placement |
| Hybrid | Joint RGB model with a QR structure mask | Protects finder and control modules |

The bundled model uses the hybrid architecture with a mask-aware spatial bit grid and
no color calibration branch.

## Reported results

The saved experiment matrix contains five seed comparisons for the three modes,
capacity studies from 25 to 200 coded bits, error correction comparisons, an
architecture ablation, and a classical baseline. The physical pilot recovered the
public and private payloads from all 10 screen to phone photographs after QR detection
and perspective rectification.

The robust operating point has a visible color tint. It should be described as robust
data embedding, not invisible steganography. The high PSNR operating point is visually
subtle but was not validated for physical capture.

See [`experiments/full/RESULTS.md`](experiments/full/RESULTS.md),
[`LOGBOOK.md`](LOGBOOK.md), and [`paper/main.pdf`](paper/main.pdf) for the recorded
evidence and limitations.

## Development

```powershell
git clone https://github.com/jemsbhai/stegaQR.git
cd stegaQR
pip install -e ".[dev,analysis]"
pytest
```

Train and evaluate a checkpoint:

```powershell
python scripts/train.py --mode hybrid --mask-aware --no-calibration `
  --output-dir experiments/run1
python scripts/evaluate.py --checkpoint experiments/run1/best_model.pt
python scripts/evaluate_real.py --checkpoint experiments/run1/best_model.pt
```

The complete experiment matrix is resumable:

```powershell
python scripts/run_experiments.py
python scripts/aggregate_results.py
python scripts/generate_figures.py
```

## License

MIT
