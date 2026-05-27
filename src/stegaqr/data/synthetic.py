"""Synthetic dataset for StegaQR training.

Generates (cover_image, hidden_payload, qr_mask) triples on the fly.
No external data needed — all training data is procedurally generated
from random QR payloads and random hidden bitstrings.

v2: Supports module_size > 1 for scaled resolution (more pixels per module
    gives the encoder more spatial bandwidth for embedding).
"""

from __future__ import annotations

import random
import string
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from stegaqr.utils.qr_utils import (
    generate_cover_qr_rgb,
    get_qr_structure_mask,
)


def _random_payload(min_len: int = 5, max_len: int = 30) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def _random_hidden_bits(capacity: int) -> np.ndarray:
    return np.random.randint(0, 2, size=capacity, dtype=np.uint8)


class SyntheticStegaQRDataset(Dataset):
    """On-the-fly synthetic dataset for StegaQR training.

    Each sample consists of:
      - cover: (3, H, W) float32 tensor in [0, 1] — standard QR code as RGB
      - payload: (capacity_bits,) float32 tensor of {0.0, 1.0} — hidden bits
      - mask: (1, H, W) float32 tensor — QR structure mask (1=data, 0=protected)
      - public_text: str — the public QR payload (for decodability checks)

    Where H = W = (4*version + 17) * module_size

    Parameters
    ----------
    num_samples : int
        Number of samples per epoch.
    capacity_bits : int
        Hidden payload size in bits.
    qr_version : int
        QR version (determines module count: 4v+17).
    module_size : int
        Pixels per QR module. Larger = more spatial bandwidth for encoding.
        1 = original resolution, 8 = 264x264 for version 4.
    ec_level : str
        Error correction level.
    payload_len_range : tuple[int, int]
        Range of public payload string lengths.
    """

    def __init__(
        self,
        num_samples: int = 5000,
        capacity_bits: int = 100,
        qr_version: int = 4,
        module_size: int = 8,
        ec_level: str = "M",
        payload_len_range: tuple[int, int] = (5, 30),
    ):
        self.num_samples = num_samples
        self.capacity_bits = capacity_bits
        self.qr_version = qr_version
        self.module_size = module_size
        self.ec_level = ec_level
        self.payload_len_range = payload_len_range

        # Pre-compute QR structure mask (same for all samples at same version)
        self._mask = get_qr_structure_mask(qr_version, module_size)
        self._module_count = 4 * qr_version + 17
        self._spatial_size = self._module_count * module_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        public_text = _random_payload(*self.payload_len_range)

        try:
            cover_rgb, _ = generate_cover_qr_rgb(
                public_text,
                version=self.qr_version,
                ec_level=self.ec_level,
                module_size=self.module_size,
            )
        except Exception:
            public_text = _random_payload(3, 10)
            cover_rgb, _ = generate_cover_qr_rgb(
                public_text,
                version=self.qr_version,
                ec_level=self.ec_level,
                module_size=self.module_size,
            )

        hidden_bits = _random_hidden_bits(self.capacity_bits)

        cover_tensor = torch.from_numpy(cover_rgb).permute(2, 0, 1)
        payload_tensor = torch.from_numpy(hidden_bits).float()
        mask_tensor = torch.from_numpy(self._mask).unsqueeze(0)

        return {
            "cover": cover_tensor,
            "payload": payload_tensor,
            "mask": mask_tensor,
            "public_text": public_text,
        }

    @property
    def spatial_size(self) -> int:
        return self._spatial_size

    @property
    def module_count(self) -> int:
        return self._module_count


def create_dataloaders(
    num_train: int = 5000,
    num_val: int = 750,
    capacity_bits: int = 100,
    qr_version: int = 4,
    module_size: int = 8,
    ec_level: str = "M",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    """Create train and validation DataLoaders."""
    train_ds = SyntheticStegaQRDataset(
        num_samples=num_train,
        capacity_bits=capacity_bits,
        qr_version=qr_version,
        module_size=module_size,
        ec_level=ec_level,
    )
    val_ds = SyntheticStegaQRDataset(
        num_samples=num_val,
        capacity_bits=capacity_bits,
        qr_version=qr_version,
        module_size=module_size,
        ec_level=ec_level,
    )

    def collate_fn(batch):
        return {
            "cover": torch.stack([b["cover"] for b in batch]),
            "payload": torch.stack([b["payload"] for b in batch]),
            "mask": torch.stack([b["mask"] for b in batch]),
            "public_text": [b["public_text"] for b in batch],
        }

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=False,
    )

    return train_loader, val_loader
