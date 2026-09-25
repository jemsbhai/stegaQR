# Changelog

## 0.2.0 - 2026-09-25

- Added grid placement of the error-correction codeword: `placement="interleaved"`
  spreads the copies of a repetition code over the spatial bit-grid through a fixed
  pseudo-random cell permutation shared by encoder and decoder. This is now the
  default for the Python API and the command line interface.
- Motivation: with the 0.1.0 placement (coded bit c on cell c), all copies of a
  message bit under Repetition-5 shared one grid column and failed together under
  distortion; on five cross-channel models and 1,024 messages each, interleaving
  removed every repetition-code message failure (see `experiments/ecc_layout/`
  and `scripts/eval_ecc_layout.py`).
- Compatibility: images produced with 0.1.0 decode with `placement="native"`
  (`--placement native` on the command line). The bundled model is unchanged.
- `evaluate_checkpoint` and `evaluate_real_distortions` accept `placement`; their
  default stays `native` so earlier `results.json` files remain comparable.
- `export_for_capture.py` records the placement in its manifest;
  `decode_from_photo.py` reads it (manifests without the key are native).
- Added `scripts/eval_ecc_layout.py` and `scripts/run_module_sweep.py`
  (camera-ready experiments for the ICTAI 2026 paper).

## 0.1.0 - 2026-07-16

- Bundled the trained hybrid model for installed use.
- Added default model discovery for the Python API and command line interface.
- Added held out evaluation with true JPEG, blur, resize, brightness, noise, and a combined pipeline.
- Added reproducible package builds, installation tests, CI, and release automation.
- Added the revised eight page research paper and its supporting results.
