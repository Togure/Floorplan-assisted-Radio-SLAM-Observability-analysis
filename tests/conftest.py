"""Shared pytest fixtures for the test suite."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.map_manager import MapManager, Wall


@pytest.fixture
def default_T() -> float:
    """Standard sampling period in seconds."""
    return 1.0


@pytest.fixture
def clock_params() -> dict:
    """Typical TCXO-grade clock noise coefficients."""
    return {"h0": 1e-19, "h_2": 1e-20}


@pytest.fixture
def rectangular_room() -> MapManager:
    """A 20 m × 15 m rectangular floorplan with four walls."""
    return MapManager.rectangular_room(width=20.0, height=15.0)


@pytest.fixture
def anchor_inside_room() -> np.ndarray:
    """Anchor position strictly inside the rectangular room."""
    return np.array([5.0, 7.0])
