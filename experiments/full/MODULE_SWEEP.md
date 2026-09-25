# Module-size sweep at L = 100, cross-channel, distortion-trained (R3-W1)

Same recipe as the main matrix; only module_size varies. Cell side = 3.3 m px. Mean +/- 95% CI (1.96 SE) across seeds; m = 4 rows are the paper's main-matrix runs.

| m | cover px | cell px | seeds | clean bit-acc | clean FDR | dist bit-acc | dist FDR | PSNR (dB) | public decode | rep3 msg-decode | train s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 33 | 3.3 | 2 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 99.2 +/- 0.0 | 18.5 +/- 1.1 | 0.0 +/- 0.0 | 98.8 +/- 2.3 | 220.8 +/- 23.8 |
| 2 | 66 | 6.6 | 2 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 98.4 +/- 3.1 | 22.9 +/- 8.4 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 232.2 +/- 26.8 |
| 3 | 99 | 9.9 | 2 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 18.2 +/- 0.5 | 100.0 +/- 0.0 | 99.6 +/- 0.8 | 305.4 +/- 3.3 |
| 4 | 132 | 13.2 | 5 | 100.0 +/- 0.0 | 99.8 +/- 0.3 | 100.0 +/- 0.0 | 99.2 +/- 1.5 | 19.0 +/- 1.3 | 100.0 +/- 0.0 | 99.5 +/- 0.6 | 426.8 +/- 0.2 |

Per-run: ms_1_s123 bit 0.9999 fdr 0.992, ms_1_s42 bit 0.9999 fdr 0.992, ms_2_s123 bit 0.9997 fdr 0.969, ms_2_s42 bit 1.0000 fdr 1.000, ms_3_s123 bit 1.0000 fdr 1.000, ms_3_s42 bit 1.0000 fdr 1.000