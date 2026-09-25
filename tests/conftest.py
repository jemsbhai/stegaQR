"""Shared test fixtures.

The repository source tree is put first on sys.path so the suite tests the code in
this checkout even when an older stegaqr release is installed in site-packages. Test
modules are collected alphabetically, so without this a module collected earlier would
import the installed copy and later modules (test_placement.py) would fail to import
symbols that only exist in the checkout.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def seed():
    """Standard seed for reproducible tests."""
    from stegaqr.utils.seed import set_all_seeds
    return set_all_seeds(42)
