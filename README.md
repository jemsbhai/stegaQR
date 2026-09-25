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

Since 0.2.0 the coded bits are spread over the bit grid by a fixed permutation
(`--placement interleaved`, the default). Images produced with 0.1.0 placed coded bit
`c` on grid cell `c`; decode them with `--placement native`. Encoder and decoder must
use the same placement.

## Python API

```python
from stegaqr import decode_hidden, encode_hidden

image = encode_hidden("https://example.com", b"ID42", device="cpu")
public, hidden, metadata = decode_hidden(image, device="cpu")

assert public == "https://example.com"
assert hidden.rstrip(b"\x00") == b"ID42"
```

CUDA is used when requested and available. Otherwise the package falls back to CPU.

Both functions accept `placement="interleaved"` (default) or `placement="native"`
(0.1.0 layout), and `placement_seed` for the interleaved permutation.

## Placement of the error-correction copies

The repetition code tiles its copies, so copy `r` of message bit `j` is coded bit
`r*k + j`. With the 0.1.0 native placement on the bundled model's grid, all copies of
a message bit under Repetition-5 shared one grid column and failed together under
distortion. `scripts/eval_ecc_layout.py` measures both placements on the same
checkpoints, covers, messages, and distortion draws; on five cross-channel models and
1,024 messages each, interleaving removed every repetition-code message failure. See
[`experiments/ecc_layout/RESULTS.md`](experiments/ecc_layout/RESULTS.md).

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
