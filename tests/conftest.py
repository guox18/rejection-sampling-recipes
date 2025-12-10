"""
Pytest configuration and fixtures.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def model_outputs(fixtures_dir):
    """Load model outputs fixture data."""
    with open(fixtures_dir / "model_outputs.json") as f:
        return json.load(f)


@pytest.fixture
def mcq_responses(model_outputs):
    """Get MCQ response fixtures."""
    return model_outputs["mcq_responses"]
