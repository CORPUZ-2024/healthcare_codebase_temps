"""Shared fixtures. Adds the template root to sys.path so `import claims_foundation` works."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claims_foundation import data  # noqa: E402


@pytest.fixture(scope="session")
def universe():
    """Small seeded universe (fast: ~1 s)."""
    return data.generate_universe(n_members=200, seed=11, months=12)
