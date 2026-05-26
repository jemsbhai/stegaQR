# Data Provenance

## Synthetic Training Data

All training data is generated synthetically by:
1. Generating random QR codes (random alphanumeric payloads, various versions/EC levels)
2. Encoding hidden payloads using the steganographic encoder
3. Applying configurable distortions

No external datasets are required for the core experiments.

## Cover Images

Cover images are standard monochrome QR codes generated using the `qrcode` Python library,
which implements ISO/IEC 18004:2015.

## Data Splits

- Training: 70% of synthetic samples (generated on-the-fly)
- Validation: 15%
- Test: 15% (fixed seed for reproducibility)

## Checksums

Data checksums stored in `checksums.sha256` when applicable.
