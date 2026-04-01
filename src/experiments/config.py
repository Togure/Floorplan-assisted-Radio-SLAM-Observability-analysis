"""Experiment configuration: four prior cases for Monte Carlo comparison.

Case 1: No prior information.
Case 2: Known initial receiver position.
Case 3: Known initial clock states (time).
Case 4: Known initial receiver position AND clock states.

Prior conditions follow Table I of IPIN 2026 paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for a single Monte Carlo experiment run.

    Attributes:
        case_id: Integer 1-4 identifying the prior case.
        n_runs: Number of Monte Carlo runs.
        n_steps: Number of time steps per run.
        T: Sampling period in seconds.
        N: Number of reflected paths (virtual anchors).
        room_width: Floorplan width in metres.
        room_height: Floorplan height in metres.
        q_x: Receiver acceleration PSD in x (m²/s³).
        q_y: Receiver acceleration PSD in y (m²/s³).
        h0_r: Receiver clock h0 coefficient.
        h_2_r: Receiver clock h_{-2} coefficient.
        h0_a: Anchor clock h0 coefficient.
        h_2_a: Anchor clock h_{-2} coefficient.
        sigma_range: Range measurement noise std (metres).
        know_position: Whether initial receiver position is known.
        know_time: Whether initial clock states are known.
    """

    case_id: int
    n_runs: int = 50
    n_steps: int = 100
    T: float = 1.0
    N: int = 4   # one VA per wall (rectangular room has 4 walls)
    room_width: float = 40.0
    room_height: float = 15.0
    q_x: float = 1          # m²/s³ — velocity random-walk (inflated for
    q_y: float = 1        #          Lissajous/random-walk model mismatch)
    q_va: float = 0.01         # m²/step — small VA regularisation noise (mapless)
    h0_r: float = 1e-21       # TCXO-grade receiver oscillator
    h_2_r: float = 1e-23
    h0_a: float = 1e-23       # OCXO-grade anchor oscillator (more stable)
    h_2_a: float = 1e-25
    sigma_range: float = 1  # m — range measurement noise std-dev
    know_position: bool = False
    know_time: bool = False


CASE_1 = ExperimentConfig(case_id=1, know_position=False, know_time=False)
CASE_2 = ExperimentConfig(case_id=2, know_position=True,  know_time=False)
CASE_3 = ExperimentConfig(case_id=3, know_position=False, know_time=True)
CASE_4 = ExperimentConfig(case_id=4, know_position=True,  know_time=True)

ALL_CASES: dict[int, ExperimentConfig] = {1: CASE_1, 2: CASE_2, 3: CASE_3, 4: CASE_4}
