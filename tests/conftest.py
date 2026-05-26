"""Shared test fixtures."""

import pytest


@pytest.fixture(scope="session")
def seed():
    """Standard seed for reproducible tests."""
    from stegaqr.utils.seed import set_all_seeds
    return set_all_seeds(42)
