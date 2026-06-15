# StegaQR Experimental Results

Aggregated from per-run `results.json`. Values are mean +/- 95% CI across seeds. PSNR/SSIM are stego-vs-cover (imperceptibility).

## 1. Mode comparison (capacity 100)

| mode | train | seeds | clean bit-acc | clean FDR | dist bit-acc | dist FDR | PSNR (dB) | SSIM | public-decode |
|---|---|---|---|---|---|---|---|---|---|
| cross_channel | clean | 5 | 100.0 +/- 0.0 | 99.5 +/- 0.9 | 63.8 +/- 10.6 | 10.0 +/- 19.6 | 61.72 +/- 7.74 | 1.00 +/- 0.00 | 100.0 +/- 0.0 |
| cross_channel | distort | 5 | 100.0 +/- 0.0 | 99.8 +/- 0.3 | 100.0 +/- 0.0 | 99.2 +/- 1.5 | 19.01 +/- 1.33 | 0.97 +/- 0.01 | 100.0 +/- 0.0 |
| hybrid | clean | 5 | 67.9 +/- 21.4 | 0.0 +/- 0.0 | 53.5 +/- 5.5 | 0.0 +/- 0.0 | 86.89 +/- 14.54 | 1.00 +/- 0.00 | 100.0 +/- 0.0 |
| hybrid | distort | 5 | 76.4 +/- 21.1 | 0.0 +/- 0.0 | 75.9 +/- 20.7 | 0.0 +/- 0.0 | 59.01 +/- 32.02 | 1.00 +/- 0.00 | 100.0 +/- 0.0 |
| segregated | clean | 5 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 65.5 +/- 5.7 | 13.9 +/- 11.3 | 63.46 +/- 3.89 | 1.00 +/- 0.00 | 100.0 +/- 0.0 |
| segregated | distort | 5 | 100.0 +/- 0.0 | 99.8 +/- 0.3 | 100.0 +/- 0.0 | 99.1 +/- 0.8 | 17.62 +/- 1.72 | 0.96 +/- 0.02 | 100.0 +/- 0.0 |

## 2. ECC message-decode under distortion (cross_channel, cap 100)

| code | net bits | rate | message-decode |
|---|---|---|---|
| none (raw) | 100 | 1.00 | 99.2 +/- 1.5 |
| rep3 | 33 | 0.33 | 99.5 +/- 0.6 |
| rep5 | 20 | 0.20 | 98.1 +/- 1.1 |
| hamming74 | 56 | 0.57 | 100.0 +/- 0.0 |

## 3. Capacity scaling (cross_channel, distortion)

| capacity | seeds | dist bit-acc | dist FDR | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| 25 | 3 | 99.9 +/- 0.1 | 97.7 +/- 2.3 | 23.70 +/- 11.32 | 0.97 +/- 0.05 |
| 50 | 3 | 100.0 +/- 0.0 | 99.7 +/- 0.5 | 20.32 +/- 1.97 | 0.98 +/- 0.01 |
| 100 | 5 | 100.0 +/- 0.0 | 99.2 +/- 1.5 | 19.01 +/- 1.33 | 0.97 +/- 0.01 |
| 150 | 3 | 99.3 +/- 1.4 | 86.5 +/- 24.2 | 23.90 +/- 10.39 | 0.98 +/- 0.02 |
| 200 | 3 | 100.0 +/- 0.0 | 95.3 +/- 7.0 | 19.00 +/- 0.84 | 0.97 +/- 0.01 |

## 5. Architecture ablation: spatial grid vs broadcast (decode-only)

| arch | training | clean bit-acc | clean FDR | dist bit-acc | dist FDR | PSNR (dB) |
|---|---|---|---|---|---|---|
| broadcast | clean | 52.5% | 0.0% | 50.9% | 0.0% | 18.6 |
| broadcast | distort | 51.4% | 0.0% | 51.6% | 0.0% | 21.1 |
| grid | clean | 100.0% | 100.0% | 100.0% | 100.0% | 17.1 |
| grid | distort | 100.0% | 100.0% | 100.0% | 100.0% | 17.0 |

## 4. Classical LSB baseline (no training)

| setting | bit-acc | full-decode | PSNR (dB) | SSIM | public-decode |
|---|---|---|---|---|---|
| clean | 100.0% | 100.0% | 78.3 | 1.000 | 100.0% |
| distorted | 53.8% | 3.9% | 29.6 | 0.985 | 100.0% |
