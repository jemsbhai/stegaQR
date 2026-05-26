"""Deterministic seed management for reproducibility.

Per lab-runner protocol: explicit seed for Python random, NumPy, PyTorch
CPU and CUDA. Seeds are saved to JSON for every experiment run.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_all_seeds(master_seed: int = 42) -> dict[str, int]:
    """Set all random seeds deterministically and return the seed dict.

    Parameters
    ----------
    master_seed : int
        Master seed from which all framework seeds are derived.

    Returns
    -------
    dict with all seed values set.
    """
    seeds = {
        "master_seed": master_seed,
        "python_random": master_seed,
        "numpy": master_seed,
        "torch": master_seed,
        "torch_cuda": master_seed,
    }

    random.seed(seeds["python_random"])
    np.random.seed(seeds["numpy"])

    try:
        import torch

        torch.manual_seed(seeds["torch"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds["torch_cuda"])
            # Best-effort determinism (some ops remain non-deterministic)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seeds["notes"] = (
            "torch.backends.cudnn.deterministic=True; "
            "some conv backward ops may remain non-deterministic on CUDA"
        )
    except ImportError:
        seeds["notes"] = "PyTorch not installed; torch seeds not set"

    return seeds


def save_seeds(seeds: dict[str, Any], path: str | Path) -> None:
    """Save seed configuration to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(seeds, f, indent=2)


def load_seeds(path: str | Path) -> dict[str, Any]:
    """Load and apply seeds from JSON."""
    with open(path) as f:
        seeds = json.load(f)
    set_all_seeds(seeds["master_seed"])
    return seeds
