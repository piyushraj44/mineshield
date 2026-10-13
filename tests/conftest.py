"""Shared pytest setup: make the project root importable and provide the default config."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mineshield.config import load_config  # noqa: E402


@pytest.fixture
def cfg() -> dict:
    """A fresh copy of config/default.json."""
    return load_config()
