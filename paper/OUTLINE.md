# StegaQR — Paper Outline & Framing (decided 2026-06-15)

**Title:** StegaQR: Robust Neural Data Embedding in QR Codes with Switchable Channel Modes
**Venue:** IEEE ICTAI 2026

## Framing decision: dual-regime (imperceptibility ↔ robustness frontier)
We present a single learned system that spans an **operating frontier**, and we use
precise, honest terminology per regime (the literature does not call *visible*
machine-decodable QR perturbations "steganography"; see related_work.md):

| Regime | PSNR | Human-perceptible? | Honest term | Use case |
|--------|------|--------------------|-------------|----------|
| Covert | ~40–54 dB | no (near-invisible) | **steganographic / covert embedding** | digital channels |
| Robust | ~18–22 dB | yes (color tint) | **robust data embedding / machine-covert color watermark** | physical capture (print/screen→camera) |

Rule for the writing: say **"steganographic/covert"** only for the high-PSNR regime;
say **"robust data embedding"** (or "machine-covert color watermark") for the visible
robust regime. The frontier itself (and being able to *choose* the point) is a
contribution, not a caveat.

## Contributions (claims the paper makes)
1. **First neural data-in-QR system** that hides an arbitrary bitstring inside a
   still-standard-decodable QR via learned color perturbation (vs classical 2LQR/HiQ/
   Halftone; vs neural methods that hide *in natural images*).
2. **Spatial bit-grid payload layout** — shown to be the enabling design (global
   broadcast à la HiDDeN/StegaStamp fails at L=100; grid → 100% in ~250 steps).
3. **Switchable channel modes** (segregated / cross-channel / hybrid QR-anchored) with
   a measured comparison; hybrid preserves QR structure exactly via a module mask.
4. **Imperceptibility↔robustness↔capacity frontier**, plus ECC coupling that converts
   ~98–99% bit accuracy into ~100% message recovery.
5. **Real-world validation**: 100% hidden-message recovery from screen→phone-camera
   capture, from a model trained only on a simulated distortion layer (fixes the
   real-photometric failure of our prior ICMLA multispecqr decoder).

## Section plan (with assets)
1. **Introduction** — motivation, the data-in-QR setting, contributions list. (fig_system)
2. **Related Work** — from related_work.md (4 pillars; contrast StegaStamp & 2LQR).
3. **Method**
   - 3.1 Cover/QR + payload pipeline (fig_system)
   - 3.2 Spatial bit-grid layout (fig_bitgrid) — the core idea + why broadcast fails
   - 3.3 Encoder/decoder architecture, GroupNorm (fig_architecture)
   - 3.4 Channel modes: segregated / cross-channel / hybrid mask (fig_examples residuals)
   - 3.5 Differentiable distortion layer + adaptive perceptual training (warmup→ramp)
   - 3.6 ECC coupling; best-model selection (robustness-aware)
4. **Experiments**
   - Setup: synthetic data, QR v4/EC-M, ms=4, 16GB GPU, 5 seeds, metrics + Wilson CIs.
   - 4.1 Mode comparison, clean vs distortion-trained (Table 1 / fig_mode_comparison) — EXP-001
   - 4.2 Architecture ablation: grid vs broadcast (RESULTS §5) — 100% vs ~52%
   - 4.3 Imperceptibility–robustness frontier (fig_imperceptibility)
   - 4.4 Capacity scaling (fig_capacity) + ECC operating points (fig_ecc)
   - 4.5 Classical LSB baseline (3.9% distorted)
   - 4.6 **Physical screen→phone capture, 10/10** (fig_capture) — EXP-002
5. **Discussion / Limitations** — small net capacity (4–12 B at 100-bit models);
   single QR version/EC/resolution; one capture session/device; no steganalysis;
   the dual-regime trade-off stated honestly.
6. **Conclusion** + reproducibility statement (code, seeds, tests, logbook, model).

## Numbers to cite (from experiments/full/RESULTS.md, LOGBOOK)
- Distortion-trained distorted full-decode: cross 99.2±1.5, seg 99.1±0.8, hybrid 100.0±0.0 %; public 100%.
- Clean-trained distorted full-decode ~10–14% (the fragility motivating distortion training).
- ECC: Hamming(7,4) → 100% message; rep3 → 99.5%.
- Capacity 25–200 bits ~99–100% distorted bit-acc.
- Grid vs broadcast: 100% vs ~52% at 100 bits (decode-only).
- Classical LSB: clean 100% / distorted 3.9% full-decode.
- Physical: 10/10 located, 10/10 public, 10/10 hidden (EXP-002).
- Imperceptible regime example: 54.9 dB / SSIM 1.000, public ✓ (fig_examples).

## Pre-submission TODOs (from related_work.md + audit)
- 2LQR / HiQ exact PSNR & capacity for the comparison table.
- multispecqr (ICMLA) citation.
- Verify ★ refs; one targeted "deep-learning QR steganography" search to confirm novelty.
- Optional strengtheners: degradation-sweep capture, higher-capacity model, version/EC sweep.
