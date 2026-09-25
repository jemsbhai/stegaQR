"""StegaQR command-line interface.

    stegaqr info   --model M
    stegaqr encode --model M --public "https://x.com" --message "secret" --out stego.png
    stegaqr decode --model M --image stego.png

`encode` writes a stego QR PNG (a standard reader scans `--public`; only StegaQR
recovers `--message`). `decode` recovers both. A quiet zone + upscaling are added to
the saved PNG for real-world scannability; `decode` rectifies them automatically.

`--placement interleaved` (default since 0.2.0) spreads the error-correction copies
over the bit-grid; use `--placement native` to read images produced by 0.1.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from stegaqr.coding import DEFAULT_PLACEMENT, DEFAULT_PLACEMENT_SEED, PLACEMENTS


DEFAULT_MODEL = str(Path(__file__).resolve().parent / "assets" / "stegaqr_default.pt")


def _add_common(p):
    p.add_argument("--model", default=DEFAULT_MODEL, help="path to a trained checkpoint")
    p.add_argument("--ecc", default="rep3", help="rep3 | rep5 | hamming74 | none")
    p.add_argument("--placement", default=DEFAULT_PLACEMENT, choices=PLACEMENTS,
                   help="grid placement of coded bits; 'native' reads 0.1.0 images")
    p.add_argument("--placement-seed", type=int, default=DEFAULT_PLACEMENT_SEED,
                   help="seed of the interleaved permutation")
    p.add_argument("--device", default="cuda")


def _codec_kwargs(args):
    return dict(ecc=args.ecc, device=args.device, placement=args.placement,
                placement_seed=args.placement_seed)


def cmd_info(args):
    from stegaqr.core import StegaQREncoder
    enc = StegaQREncoder(args.model, **_codec_kwargs(args))
    cfg = enc.cfg
    print(f"model           : {args.model}")
    print(f"mode            : {cfg['mode']}")
    print(f"raw capacity    : {cfg['capacity_bits']} bits")
    print(f"QR version/EC   : {cfg['qr_version']} / {cfg['ec_level']}  (module_size {cfg['module_size']})")
    print(f"ECC             : {enc.ecc.name} (rate {enc.ecc.rate:.2f})")
    print(f"placement       : {enc.placement} (seed {enc.placement_seed})")
    print(f"max hidden bytes: {enc.max_hidden_bytes}")


def cmd_encode(args):
    from stegaqr.core import StegaQREncoder
    enc = StegaQREncoder(args.model, **_codec_kwargs(args))
    msg = args.message.encode("utf-8")
    img = enc.encode(args.public, msg)                    # bare symbol, model resolution
    if args.upscale > 1:
        img = img.resize((img.width * args.upscale, img.height * args.upscale), Image.NEAREST)
    if args.quiet_zone > 0:
        b = args.quiet_zone * args.upscale * enc.cfg["module_size"]
        arr = np.pad(np.asarray(img), ((b, b), (b, b), (0, 0)), constant_values=255)
        img = Image.fromarray(arr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out}  ({img.width}x{img.height})  "
          f"public='{args.public}'  hidden={len(msg)}B/{enc.max_hidden_bytes}B max  "
          f"placement={enc.placement}")


def cmd_decode(args):
    from stegaqr.core import StegaQRDecoder
    dec = StegaQRDecoder(args.model, **_codec_kwargs(args))
    public, hidden, meta = dec.decode(Image.open(args.image))
    print(f"public payload : {public!r}")
    # show hidden as text if printable, else hex; trim trailing NULs from padding
    h = hidden.rstrip(b"\x00")
    try:
        text = h.decode("utf-8")
        printable = text.isprintable()
    except UnicodeDecodeError:
        printable = False
    print(f"hidden payload : {text!r}" if printable else f"hidden payload : {h.hex()} (hex)")
    print(f"placement      : {meta['placement']}")
    if meta.get("confidence") is not None:
        print(f"confidence     : {meta['confidence']:.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="stegaqr", description="Neural steganography in QR codes")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="show model capacity"); _add_common(pi)
    pe = sub.add_parser("encode", help="embed a hidden message into a QR")
    _add_common(pe)
    pe.add_argument("--public", required=True, help="visible QR payload")
    pe.add_argument("--message", required=True, help="hidden message (utf-8)")
    pe.add_argument("--out", default="stego.png")
    pe.add_argument("--upscale", type=int, default=10, help="output pixels per stego pixel")
    pe.add_argument("--quiet-zone", type=int, default=4, help="quiet-zone modules for scannability")
    pd = sub.add_parser("decode", help="recover the hidden message from a stego QR")
    _add_common(pd)
    pd.add_argument("--image", required=True)

    args = p.parse_args(argv)
    if not Path(args.model).exists():
        sys.exit(f"model not found: {args.model}\n"
                 "Reinstall stegaQR to restore the bundled model, or pass --model PATH.")
    {"info": cmd_info, "encode": cmd_encode, "decode": cmd_decode}[args.cmd](args)


if __name__ == "__main__":
    main()
