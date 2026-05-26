"""Neural encoder-decoder architectures for StegaQR.

Three model pairs, one per steganographic mode:
  - SegregatedEncoder / SegregatedDecoder
  - CrossChannelEncoder / CrossChannelDecoder
  - HybridEncoder / HybridDecoder

Plus a shared differentiable distortion layer for training.
"""
