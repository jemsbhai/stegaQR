"""StegaQR: Neural steganographic data embedding in QR codes."""

__version__ = "0.1.0"

from stegaqr.core import (
    StegaQREncoder,
    StegaQRDecoder,
    encode_hidden,
    decode_hidden,
    StegoMode,
)

__all__ = [
    "StegaQREncoder",
    "StegaQRDecoder",
    "encode_hidden",
    "decode_hidden",
    "StegoMode",
    "__version__",
]
