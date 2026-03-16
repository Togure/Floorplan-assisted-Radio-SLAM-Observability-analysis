"""Velocity random-walk and two-state clock dynamics models.

State vector layout (positions in metres, clock states in metres of range-equivalent):

  The clock bias b = c·δt and drift bd = c·δṫ are stored in metres so that
  the H-matrix clock columns are ±1 (same scale as the position unit-vector
  columns).  This eliminates the 3×10⁸ scale mismatch that would otherwise
  make the EKF numerically degenerate.

  Original seconds-domain layout (used in the paper equations):


  Mapless model  (10 + 2N dims):
    x = [xr(6), xa(4), r_va1(2), ..., r_vaN(2)]^T
    where xr = [r_r(2), rdot_r(2), delta_t_r, delta_tdot_r]^T
          xa = [r_a(2), delta_t_a, delta_tdot_a]^T

  Map-assisted model (10 dims):
    xM = [xr(6), xa(4)]^T
    (virtual anchor positions are algebraic functions of xa via MapManager)

Reference: Lyu & Zhang, "Observability Evaluation in Map-Assisted Radio SLAM",
           IPIN 2026, Equations (1)–(5).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Clock sub-matrices
# ---------------------------------------------------------------------------

def build_F_clk(T: float) -> np.ndarray:
    """Build the 2×2 clock state transition matrix.

    F_clk = [[1, T],
             [0, 1]]

    Args:
        T: Sampling period in seconds.

    Returns:
        2×2 clock transition matrix.
    """
    return np.array([[1.0, T], [0.0, 1.0]])


def build_Q_clk(T: float, h0: float, h_2: float, c_light: float = 1.0) -> np.ndarray:
    """Build the 2×2 clock process noise covariance matrix.

    Power spectral densities (from oscillator power-law model):
        S_wδt  ≈ h0 / 2
        S_wδ̇t ≈ 2π²h_{-2}

    When c_light = 1.0  (default): Q is in seconds²  (raw SI units).
    When c_light = c    (3×10⁸):  Q is in metres²    (range-equivalent units).
    Scaling by c_light² converts the time-domain covariance to the chosen unit.

    Args:
        T:       Sampling period in seconds.
        h0:      White frequency noise coefficient h_0.
        h_2:     Random-walk frequency noise coefficient h_{-2}.
        c_light: Speed of light in m/s (default 1.0 → seconds domain).

    Returns:
        2×2 symmetric positive-definite clock noise covariance.
    """
    s_wdt  = (h0 / 2.0)               * c_light**2
    s_wddt = 2.0 * (np.pi**2) * h_2   * c_light**2
    return np.array([
        [s_wdt * T + s_wddt * T**3 / 3.0, s_wddt * T**2 / 2.0],
        [s_wddt * T**2 / 2.0,             s_wddt * T],
    ])


# ---------------------------------------------------------------------------
# Position-velocity sub-matrix
# ---------------------------------------------------------------------------

def build_Q_pv(T: float, q_x: float, q_y: float) -> np.ndarray:
    """Build the 4×4 position-velocity process noise covariance.

    Derived from a velocity random-walk model driven by zero-mean white
    acceleration noise with PSDs q_x, q_y in x and y respectively.

    State order within this block: [x, y, x_dot, y_dot].

    Q_pv = [[q_x T³/3,  0,        q_x T²/2,  0       ],
            [0,         q_y T³/3, 0,         q_y T²/2],
            [q_x T²/2,  0,        q_x T,     0       ],
            [0,         q_y T²/2, 0,         q_y T   ]]

    Args:
        T: Sampling period in seconds.
        q_x: Acceleration PSD in x (m²/s³).
        q_y: Acceleration PSD in y (m²/s³).

    Returns:
        4×4 symmetric positive-semi-definite process noise covariance.
    """
    return np.array([
        [q_x * T**3 / 3.0, 0.0,              q_x * T**2 / 2.0, 0.0             ],
        [0.0,              q_y * T**3 / 3.0, 0.0,              q_y * T**2 / 2.0],
        [q_x * T**2 / 2.0, 0.0,              q_x * T,          0.0             ],
        [0.0,              q_y * T**2 / 2.0, 0.0,              q_y * T         ],
    ])


# ---------------------------------------------------------------------------
# Receiver block (6×6)
# ---------------------------------------------------------------------------

def build_F_r(T: float) -> np.ndarray:
    """Build the 6×6 receiver state transition matrix Fr.

    Receiver state: xr = [r_r(2), rdot_r(2), delta_t_r, delta_tdot_r].

    Fr = [[I2,  TI2, 02 ],
          [02,  I2,  02 ],
          [02,  02,  Fclk]]

    Args:
        T: Sampling period in seconds.

    Returns:
        6×6 receiver transition matrix.
    """
    F = np.zeros((6, 6))
    F[0:2, 0:2] = np.eye(2)        # r_r ← r_r
    F[0:2, 2:4] = T * np.eye(2)   # r_r ← rdot_r (kinematic integration)
    F[2:4, 2:4] = np.eye(2)        # rdot_r ← rdot_r (random-walk velocity)
    F[4:6, 4:6] = build_F_clk(T)  # clock ← clock
    return F


def build_Q_r(
    T: float,
    q_x: float,
    q_y: float,
    h0: float,
    h_2: float,
    c_light: float = 1.0,
) -> np.ndarray:
    """Build the 6×6 receiver process noise covariance Qr.

    Qr = block_diag(Q_pv, Q_clk_r).

    Args:
        T:       Sampling period in seconds.
        q_x:     Acceleration PSD in x (m²/s³).
        q_y:     Acceleration PSD in y (m²/s³).
        h0:      Receiver clock h0 coefficient.
        h_2:     Receiver clock h_{-2} coefficient.
        c_light: Speed of light — passed to build_Q_clk for unit conversion.

    Returns:
        6×6 receiver process noise covariance.
    """
    Q = np.zeros((6, 6))
    Q[0:4, 0:4] = build_Q_pv(T, q_x, q_y)
    Q[4:6, 4:6] = build_Q_clk(T, h0, h_2, c_light)
    return Q


# ---------------------------------------------------------------------------
# Anchor block (4×4)
# ---------------------------------------------------------------------------

def build_F_a(T: float) -> np.ndarray:
    """Build the 4×4 anchor state transition matrix Fa.

    Anchor state: xa = [r_a(2), delta_t_a, delta_tdot_a].
    Anchor position is stationary; only the clock state evolves.

    Fa = diag[I2, F_clk]

    Args:
        T: Sampling period in seconds.

    Returns:
        4×4 anchor transition matrix.
    """
    F = np.zeros((4, 4))
    F[0:2, 0:2] = np.eye(2)        # r_a stationary
    F[2:4, 2:4] = build_F_clk(T)  # clock
    return F


def build_Q_a(T: float, h0: float, h_2: float, c_light: float = 1.0) -> np.ndarray:
    """Build the 4×4 anchor process noise covariance Qa.

    Anchor position has zero noise; only the clock is driven.

    Qa = block_diag(02×2, Q_clk_a).

    Args:
        T:       Sampling period in seconds.
        h0:      Anchor clock h0 coefficient.
        h_2:     Anchor clock h_{-2} coefficient.
        c_light: Speed of light — passed to build_Q_clk for unit conversion.

    Returns:
        4×4 anchor process noise covariance.
    """
    Q = np.zeros((4, 4))
    Q[2:4, 2:4] = build_Q_clk(T, h0, h_2, c_light)
    return Q


# ---------------------------------------------------------------------------
# Full system matrices — Mapless model (10 + 2N)
# ---------------------------------------------------------------------------

def build_F_mapless(T: float, N: int) -> np.ndarray:
    """Build the (10 + 2N) × (10 + 2N) transition matrix for the mapless model.

    Full state: x = [xr(6), xa(4), r_va1(2), ..., r_vaN(2)]^T.

    F = block_diag(Fr, Fa, I_{2N})

    Virtual anchors are modelled as stationary (identity block).

    Args:
        T: Sampling period in seconds.
        N: Number of virtual anchors (reflecting surfaces).

    Returns:
        (10 + 2N) × (10 + 2N) state transition matrix.
    """
    dim = 10 + 2 * N
    F = np.zeros((dim, dim))
    F[0:6,  0:6]  = build_F_r(T)
    F[6:10, 6:10] = build_F_a(T)
    F[10:dim, 10:dim] = np.eye(2 * N)   # VAs are stationary landmarks
    return F


def build_Q_mapless(
    T: float,
    N: int,
    q_x: float,
    q_y: float,
    h0_r: float,
    h_2_r: float,
    h0_a: float,
    h_2_a: float,
    c_light: float = 1.0,
) -> np.ndarray:
    """Build the (10 + 2N) × (10 + 2N) process noise covariance for mapless model.

    Q = block_diag(Qr, Qa, 0_{2N×2N})

    Args:
        T: Sampling period in seconds.
        N: Number of virtual anchors.
        q_x: Receiver acceleration PSD in x (m²/s³).
        q_y: Receiver acceleration PSD in y (m²/s³).
        h0_r: Receiver clock h0 coefficient.
        h_2_r: Receiver clock h_{-2} coefficient.
        h0_a: Anchor clock h0 coefficient.
        h_2_a: Anchor clock h_{-2} coefficient.

    Returns:
        (10 + 2N) × (10 + 2N) process noise covariance matrix.
    """
    dim = 10 + 2 * N
    Q = np.zeros((dim, dim))
    Q[0:6,  0:6]  = build_Q_r(T, q_x, q_y, h0_r, h_2_r, c_light)
    Q[6:10, 6:10] = build_Q_a(T, h0_a, h_2_a, c_light)
    # Virtual anchors have zero process noise (Q_va = 0_{2N})
    return Q


# ---------------------------------------------------------------------------
# Full system matrices — Map-assisted model (10)
# ---------------------------------------------------------------------------

def build_F_map_aided(T: float) -> np.ndarray:
    """Build the 10×10 state transition matrix for the map-assisted model.

    Reduced state: xM = [xr(6), xa(4)]^T.

    FM = block_diag(Fr, Fa)

    Args:
        T: Sampling period in seconds.

    Returns:
        10×10 map-assisted state transition matrix.
    """
    F = np.zeros((10, 10))
    F[0:6,  0:6]  = build_F_r(T)
    F[6:10, 6:10] = build_F_a(T)
    return F


def build_Q_map_aided(
    T: float,
    q_x: float,
    q_y: float,
    h0_r: float,
    h_2_r: float,
    h0_a: float,
    h_2_a: float,
    c_light: float = 1.0,
) -> np.ndarray:
    """Build the 10×10 process noise covariance for the map-assisted model.

    QM = block_diag(Qr, Qa)

    Args:
        T: Sampling period in seconds.
        q_x: Receiver acceleration PSD in x (m²/s³).
        q_y: Receiver acceleration PSD in y (m²/s³).
        h0_r: Receiver clock h0 coefficient.
        h_2_r: Receiver clock h_{-2} coefficient.
        h0_a: Anchor clock h0 coefficient.
        h_2_a: Anchor clock h_{-2} coefficient.

    Returns:
        10×10 map-assisted process noise covariance matrix.
    """
    Q = np.zeros((10, 10))
    Q[0:6,  0:6]  = build_Q_r(T, q_x, q_y, h0_r, h_2_r, c_light)
    Q[6:10, 6:10] = build_Q_a(T, h0_a, h_2_a, c_light)
    return Q
