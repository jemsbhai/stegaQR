# ECC copy placement on the spatial bit-grid (camera-ready, R3-W4)

n = 1024 messages per checkpoint; mean +/- 95% CI (1.96 SE) across seeds; same covers, messages and distortion draws for native and interleaved.

## cross_channel (5 seeds)

| code | net bits | rate | native msg-decode | interleaved msg-decode | native copy-err | native msg-bit-err | interleaved copy-err | interleaved msg-bit-err | native distinct cols/bit | interleaved distinct cols/bit |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 100 | 1.00 | 99.4 +/- 0.6 | same | - | - | - | - | - | - |
| rep2 | 50 | 0.50 | 99.4 +/- 0.6 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 1.00 | 1.90 |
| rep3 | 33 | 0.33 | 99.6 +/- 0.4 | 100.0 +/- 0.1 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.1 +/- 0.1 | 0.0 +/- 0.0 | 3.00 | 2.58 |
| rep4 | 25 | 0.25 | 99.2 +/- 0.7 | 100.0 +/- 0.0 | 0.1 +/- 0.1 | 0.0 +/- 0.0 | 0.1 +/- 0.1 | 0.0 +/- 0.0 | 2.00 | 3.40 |
| rep5 | 20 | 0.20 | 98.0 +/- 1.2 | 100.0 +/- 0.0 | 0.2 +/- 0.1 | 0.1 +/- 0.1 | 0.1 +/- 0.1 | 0.0 +/- 0.0 | 1.00 | 4.35 |
| hamming74 | 56 | 0.57 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | - | - | - | - | - | - |

Per-seed failed messages (native / interleaved):

- seed 42: none 4, rep2 2/0, rep3 0/0, rep4 3/0, rep5 12/0, hamming74 0/0
- seed 123: none 2, rep2 0/0, rep3 0/0, rep4 2/0, rep5 8/0, hamming74 0/0
- seed 7: none 0, rep2 2/0, rep3 0/0, rep4 3/0, rep5 16/0, hamming74 0/0
- seed 99: none 18, rep2 15/0, rep3 8/2, rep4 19/0, rep5 24/0, hamming74 1/1
- seed 2024: none 5, rep2 11/0, rep3 10/0, rep4 13/0, rep5 44/0, hamming74 0/0

## hybrid (5 seeds)

| code | net bits | rate | native msg-decode | interleaved msg-decode | native copy-err | native msg-bit-err | interleaved copy-err | interleaved msg-bit-err | native distinct cols/bit | interleaved distinct cols/bit |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 100 | 1.00 | 99.9 +/- 0.1 | same | - | - | - | - | - | - |
| rep2 | 50 | 0.50 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 1.90 | 1.94 |
| rep3 | 33 | 0.33 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 2.06 | 2.73 |
| rep4 | 25 | 0.25 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 3.64 | 3.68 |
| rep5 | 20 | 0.20 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 3.50 | 4.55 |
| hamming74 | 56 | 0.57 | 100.0 +/- 0.0 | 100.0 +/- 0.0 | - | - | - | - | - | - |

Per-seed failed messages (native / interleaved):

- seed 42: none 0, rep2 0/0, rep3 0/0, rep4 0/0, rep5 0/0, hamming74 0/0
- seed 123: none 0, rep2 0/0, rep3 0/0, rep4 0/0, rep5 0/0, hamming74 0/0
- seed 7: none 0, rep2 0/0, rep3 0/0, rep4 0/0, rep5 0/0, hamming74 0/0
- seed 99: none 3, rep2 0/0, rep3 0/0, rep4 0/0, rep5 0/0, hamming74 0/0
- seed 2024: none 0, rep2 0/0, rep3 0/0, rep4 0/0, rep5 0/0, hamming74 0/0
