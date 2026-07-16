# Related Work & Framing - StegaQR

*Source: adversarially-verified deep-research pass (2026-06-15), 21 primary sources,
25/25 claims confirmed (one 2‑1, rest 3‑0). Drafting material for the paper's Related
Work + the framing decision. Year/venue nuances flagged inline; verify the 2 starred
refs before camera-ready.*

## Taxonomy (four pillars)

### 1. Neural encoder-decoder data hiding (the architectural lineage)
- **Baluja, "Hiding Images in Plain Sight: Deep Steganography," NeurIPS 2017** - image-in-image, trained encoder/reveal pair; natural-image cover; robustness: none (not for lossy files).
- **Zhu, Kaplan, Johnson, Fei-Fei, "HiDDeN: Hiding Data with Deep Networks," ECCV 2018** - canonical bitstring-in-image; **global message broadcast** (message replicated spatially, decoded by global avg-pool); robustness via a **differentiable noise layer** (dropout/crop/blur/JPEG-approx). No physical channel.
- **Tancik, Mildenhall, Ng, "StegaStamp: Invisible Hyperlinks in Physical Photographs," CVPR 2020** - imperceptible bitstring in a natural photo that survives **real print + photograph**; differentiable print/recapture augmentations; **BCH 56 data + 40 parity** ECC. Closest neural prior on the physical-channel + ECC axes - but hides *in a natural image*, global broadcast.
- **Bui et al., "RoSteALS: Robust Steganography using Autoencoder Latent Space," CVPRW 2023** - single content-independent latent offset added across the whole image; natural-image cover; robust to simulated ImageNet-C only (reportedly fails on printed images).
- **Zhang et al., "RivaGAN: Robust Invisible Video Watermarking with Attention," arXiv 2019** - attention-based video watermarking (commonly cited; verify bib).

**All four hide data IN natural images via global broadcast - the inverse of our "hide data IN a QR" setting, and the exact baseline our spatial bit-grid contrasts against.**

### 2. Data hiding *in* QR / 2D barcodes (our cover medium - but classical)
- **Tkachenko et al., "Two-Level QR Code for Private Message Sharing and Document Authentication," IEEE TIFS 2016** - public level = standard QR (any reader decodes it) + private level via **textured-pattern substitution** on black modules; classical (pattern correlation + Reed-Solomon). **Closest data-in-QR prior; public stays standard-decodable; private level is machine-covert (visibly textured), not human-invisible.**
- **Yang et al., "Robust and Fast Decoding of High-Capacity Color QR Codes (HiQ)," IEEE TIP 2017/2018 (arXiv:1704.06447)** - classical color multiplexing; needs a specialized ML decoder (LSVM-CMI/QDA-CMI); the color code is **not** standard-monochrome-readable.
- **Chu, Chang, Lee, Mitra, "Halftone QR Codes," ACM TOG / SIGGRAPH Asia 2013** - per-module 3×3 submodule split; embeds a **visible** image while staying machine-readable; classical, overt.
- **"Low-Cost Anti-Copying 2D Barcode (LCAC)," IEEE TMM 2020 (arXiv:2001.06203)** and **Xie et al., "Detection of Information Hiding at Anti-Copying 2D Barcodes," (arXiv:2003.09316)** - hidden info in 2D barcodes; note that ML schemes for printed barcodes "lack robustness because a printed 2D barcode is very much environmentally dependent" - directly motivates our distortion-augmented training (and explains the prior multispecqr/ICMLA real-photometric failure).

**This whole pillar is classical (substitution / color-multiplexing / per-module), not learned per-pixel perturbation.**

### 3. Physical channel - print-scan & screen-shooting robustness
- **Fang et al., "Screen-Shooting Resilient Watermarking," IEEE TIFS 2019** - formalizes the display to camera channel; three distortion classes: **lens deformation, light-source deformation, moiré**; I-SIFT localization + repeated small embedding.
- **Fang et al., "PIMoG: Screen-shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking," ACM MM 2022** - key insight: simulate only the **dominant** distortions (perspective, illumination, moiré) in a differentiable noise layer to end-to-end trainable, **>97%** screen-shooting extraction. The efficient distortion-layer design we follow.

### 4. Steganography + error-correction coding
- StegaStamp's BCH(56+40) is the canonical demonstration; we use repetition / Hamming(7,4) to lift ~98-99% bit accuracy to ~100% message recovery.

## Comparison table (fill from the source papers before camera-ready)

| Work | Cover | Neural? | Capacity | Imperceptibility | Robustness tested | Public QR still decodable? |
|------|-------|---------|----------|------------------|-------------------|----------------------------|
| Baluja '17 | natural image | yes | full RGB image | high | none | n/a |
| HiDDeN '18 | natural image | yes | 52 b (0.203 bpp) | visually indist. (no PSNR headline) | sim (blur/crop/JPEG/Gaussian) | n/a |
| StegaStamp '20 | natural photo | yes | 56 b (post-BCH) | near-invisible | **print + photo** | n/a |
| RoSteALS '23 | natural image | yes | 100 b (released ckpt) | high | sim only | n/a |
| 2LQR '16 | **QR** | no | up to 20,000 b (v40, 8-ary RS) | visible (textured) | print-scan | **yes** |
| HiQ '17/'18 | color QR | no (ML dec.) | up to 8,900 B (3 color layers) | overt color | mobile capture | **no** |
| Halftone QR '13 | QR | no | (visible image) | overt | - | yes |
| **StegaQR (ours)** | **QR** | **yes** | 100 b raw / 33-56 b net | 15-44 dB (tunable) | **sim + real screen to phone** | **yes (100%)** |

## The gap StegaQR fills
No prior work combines: **(a)** neural encoder-decoder hiding of an arbitrary bitstring,
**(b)** inside a still **standard-decodable** QR, **(c)** via a **spatial bit-grid** layout
(vs global broadcast), **(d)** with **switchable channel modes**, **(e)** **ECC-coupled**,
**(f)** validated on **real screen to camera** capture. Closest contrasts: **StegaStamp**
(neural/physical/ECC, but natural-image cover + global broadcast) and **2LQR** (data-in-QR,
public stays decodable, but classical and untested on the screen-capture channel).

## TERMINOLOGY / FRAMING GUIDANCE (the decision driver)
The literature does **not** call visible-but-machine-decodable QR perturbations
"steganography." 2LQR frames its visibly-textured private level as a *rich/two-level QR*
(data embedding for authentication); HiQ/Halftone are overt *visual/high-capacity QR*;
StegaStamp reserves *"steganography"/"invisible"* for genuinely human-imperceptible embedding.

**Implication for StegaQR:** our **robust** operating point is *visibly* color-tinted, so
calling it "steganography" (which implies human-imperceptibility) is **not defensible** there.
Honest terms for that regime: **robust data embedding** / **machine-covert color watermark**.
"Steganography / covert" is honest only at the **high-PSNR (~40-54 dB)** near-invisible
operating point.

## Research audit status
- Targeted neural data-in-QR search completed 2026-07-16. No learned encoder and decoder using a standard-decodable QR as the cover surfaced. The search did identify two important classical neighbors now cited in the paper: Lin and Chen 2016 and Koptyra and Ogiela 2024.
- ~~Extract capacity numbers from 2LQR and HiQ to complete the table.~~ DONE (2026-06-16,
 web-verified): 2LQR up to 20,000 private bits (v40, 8-ary RS); HiQ 2900/7700/8900 B over
 three color layers (arXiv:1704.06447). Neither reports a PSNR headline (both overt). These
 are now in main.tex Table~\ref{tab:related}.
- MultiSpecQR remains omitted from the anonymous draft while it is under review. Add it to an accepted version when the venue permits identification.
- Reference metadata, citation coverage, and the final bibliography build were verified.
