"""Jacobian (H) computation for Mapless and Map-assisted observation models.

Observation vector z(tk) ∈ R^{N+1}:
  z_0       : direct-path range + clock bias
  z_1..z_N  : reflected-path ranges + clock bias (N virtual anchors)

Range-plus-clock measurement for path i:
  z_i = ||r_r - p_i||_2 + c_light * (delta_t_r - delta_t_a)

where p_0 = r_a (direct) and p_i = r_vai or f_i(r_a) (reflected).

State vector layouts (see dynamics.py):
  Mapless     x  = [r_r(2), rdot_r(2), dt_r(2), r_a(2), dt_a(2),
                    r_va1(2), ..., r_vaN(2)]     shape: (10+2N,)
  Map-assisted xM = [r_r(2), rdot_r(2), dt_r(2), r_a(2), dt_a(2)]
                                                  shape: (10,)

Reference: Lyu & Zhang, IPIN 2026, Equations (7)–(10).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------

def unit_vector(r_from: np.ndarray, r_to: np.ndarray) -> np.ndarray:
    """Compute the unit direction vector pointing from r_from toward r_to.

    ê = (r_to - r_from) / ||r_to - r_from||_2

    Args:
        r_from: Source position, shape (2,).
        r_to:   Target position, shape (2,).

    Returns:
        Unit vector, shape (2,).

    Raises:
        ValueError: If the two points are numerically coincident.
    """
    delta = np.asarray(r_to, dtype=float) - np.asarray(r_from, dtype=float)
    dist = np.linalg.norm(delta)
    if dist < 1e-12:
        raise ValueError(
            f"Coincident points (dist={dist:.2e}): unit vector is undefined."
        )
    return delta / dist


def range_and_clock_jacobian_row(
    r_r: np.ndarray,
    p: np.ndarray,
    c_light: float,
) -> tuple[np.ndarray, float, float]:
    """Partial derivatives of a single range-plus-clock measurement.

    z_i = ||r_r - p||_2 + c_light * delta_t_r - c_light * delta_t_a

    Returns:
        e_hat: ê(p) = (r_r - p)/||r_r - p|| , shape (2,)
                      ∂z_i/∂r_r  =  e_hat^T
        c_r:  = +c_light   (∂z_i/∂delta_t_r)
        c_a:  = -c_light   (∂z_i/∂delta_t_a)
    """
    e_hat = unit_vector(p, r_r)   # direction FROM p TO r_r
    return e_hat, c_light, -c_light


# ---------------------------------------------------------------------------
# Mapless Jacobian — H ∈ R^{(N+1) × (10+2N)}
# ---------------------------------------------------------------------------

def build_H_mapless(
    r_r: np.ndarray,
    r_a: np.ndarray,
    r_vas: list[np.ndarray],
    c_light: float = 3e8,
) -> np.ndarray:
    """Build the full Jacobian H for the mapless model (Eq. 8).

    Column layout of H  [shape (N+1) × (10+2N)]:
      cols  0: 1  — r_r    (2)   receiver position
      cols  2: 3  — rdot_r (2)   receiver velocity   [always zero]
      cols  4: 5  — dt_r   (2)   receiver clock [delta_t_r, delta_tdot_r]
      cols  6: 7  — r_a    (2)   anchor position
      cols  8: 9  — dt_a   (2)   anchor clock   [delta_t_a, delta_tdot_a]
      cols 10:11  — r_va1  (2)   virtual anchor 1
        ...
      cols 10+2(N-1):10+2N-1 — r_vaN (2)

    Row layout:
      row 0  : direct path  (p = r_a)
      row i  : i-th reflected path  (p = r_vai),  i = 1..N

    Hr sub-block  (N+1)×6:
      [ê^T(r_a),    0_{1×2}, +c, 0]    row 0
      [ê^T(r_vai),  0_{1×2}, +c, 0]    row i

    Ha sub-block  (N+1)×4:
      [-ê^T(r_a),   -c, 0]    row 0
      [0_{1×2},     -c, 0]    row i  (VA position not a function of r_a)

    Hva sub-block  (N+1)×2N:
      [0, ..., 0]              row 0  (direct path has no VA contribution)
      [0,...,-ê^T(r_vai),...,0] row i  (only the i-th VA column is non-zero)

    Args:
        r_r:     Receiver position, shape (2,).
        r_a:     Anchor position, shape (2,).
        r_vas:   List of N virtual anchor positions, each shape (2,).
        c_light: Speed of light in m/s.

    Returns:
        H matrix, shape (N+1, 10+2N).
    """
    r_r = np.asarray(r_r, dtype=float)
    r_a = np.asarray(r_a, dtype=float)
    N = len(r_vas)
    dim = 10 + 2 * N
    H = np.zeros((N + 1, dim))

    # ---- Row 0: direct path (p = r_a) ----
    e0, c_r, c_a = range_and_clock_jacobian_row(r_r, r_a, c_light)
    H[0, 0:2] = e0          # ∂z_0/∂r_r
    # cols 2:4 (rdot_r) stay zero
    H[0, 4]   = c_r          # ∂z_0/∂delta_t_r
    # col 5 (delta_tdot_r) stays zero
    H[0, 6:8] = -e0          # ∂z_0/∂r_a  = -ê^T(r_a)
    H[0, 8]   = c_a          # ∂z_0/∂delta_t_a
    # col 9 (delta_tdot_a) stays zero
    # Hva row 0: all zeros (direct path has no VA dependence)

    # ---- Rows 1..N: reflected paths (p = r_vai) ----
    for i, r_vai in enumerate(r_vas):
        r_vai = np.asarray(r_vai, dtype=float)
        ei, c_r, c_a = range_and_clock_jacobian_row(r_r, r_vai, c_light)
        row = i + 1
        H[row, 0:2] = ei          # ∂z_i/∂r_r
        # cols 2:4 (rdot_r) stay zero
        H[row, 4]   = c_r          # ∂z_i/∂delta_t_r
        # col 5 stays zero
        # Ha: anchor position columns → zero for reflected paths in mapless
        H[row, 8]   = c_a          # ∂z_i/∂delta_t_a
        # col 9 stays zero
        # Hva: only the i-th VA block is non-zero
        va_col = 10 + 2 * i
        H[row, va_col:va_col + 2] = -ei   # ∂z_i/∂r_vai = -ê^T(r_vai)

    return H


# ---------------------------------------------------------------------------
# Map-assisted Jacobian — HM ∈ R^{(N+1) × 10}
# ---------------------------------------------------------------------------

def build_H_map_aided(
    r_r: np.ndarray,
    r_a: np.ndarray,
    r_vas: list[np.ndarray],
    M_list: list[np.ndarray],
    c_light: float = 3e8,
) -> np.ndarray:
    """Build the full Jacobian HM for the map-assisted model (Eq. 10).

    Column layout of HM  [shape (N+1) × 10]:
      cols 0:1  — r_r    (2)
      cols 2:3  — rdot_r (2)   [always zero]
      cols 4:5  — dt_r   (2)   [delta_t_r, delta_tdot_r]
      cols 6:7  — r_a    (2)
      cols 8:9  — dt_a   (2)   [delta_t_a, delta_tdot_a]

    Row layout:
      row 0  : direct path
      row i  : i-th reflected path, i = 1..N

    HM,r sub-block  (N+1)×6:  identical to the mapless Hr.

    HM,a sub-block  (N+1)×4:
      row 0: [-ê^T(r_a),         -c, 0]   (same as mapless)
      row i: [-ê^T(f_i(r_a))@M_i, -c, 0]  (chain rule: VA = f_i(r_a))

    The Householder term -ê^T(f_i(r_a))@M_i couples every reflected path
    back to the anchor position, making each reflected multipath an
    independent constraint on r_a (the observability gain of map assistance).

    Args:
        r_r:     Receiver position, shape (2,).
        r_a:     Anchor position, shape (2,).
        r_vas:   List of N virtual anchor positions f_i(r_a), each shape (2,).
        M_list:  List of N Householder matrices M_i = ∂f_i/∂r_a, each (2,2).
        c_light: Speed of light in m/s.

    Returns:
        HM matrix, shape (N+1, 10).
    """
    r_r = np.asarray(r_r, dtype=float)
    r_a = np.asarray(r_a, dtype=float)
    N = len(r_vas)
    if len(M_list) != N:
        raise ValueError(
            f"r_vas has {N} entries but M_list has {len(M_list)} entries."
        )
    HM = np.zeros((N + 1, 10))

    # ---- Row 0: direct path (p = r_a) ----
    e0, c_r, c_a = range_and_clock_jacobian_row(r_r, r_a, c_light)
    HM[0, 0:2] = e0          # ∂z_0/∂r_r
    HM[0, 4]   = c_r          # ∂z_0/∂delta_t_r
    HM[0, 6:8] = -e0          # ∂z_0/∂r_a  = -ê^T(r_a)
    HM[0, 8]   = c_a          # ∂z_0/∂delta_t_a

    # ---- Rows 1..N: reflected paths ----
    for i, (r_vai, Mi) in enumerate(zip(r_vas, M_list)):
        r_vai = np.asarray(r_vai, dtype=float)
        Mi    = np.asarray(Mi,    dtype=float)
        ei, c_r, c_a = range_and_clock_jacobian_row(r_r, r_vai, c_light)
        row = i + 1
        HM[row, 0:2] = ei                     # ∂z_i/∂r_r
        HM[row, 4]   = c_r                     # ∂z_i/∂delta_t_r
        # Chain rule: ∂z_i/∂r_a = -ê^T(f_i(r_a)) @ M_i  (1×2 row vector)
        HM[row, 6:8] = -(ei @ Mi)              # ∂z_i/∂r_a
        HM[row, 8]   = c_a                     # ∂z_i/∂delta_t_a

    return HM


# ---------------------------------------------------------------------------
# Nonlinear measurement function (for EKF predict/update)
# ---------------------------------------------------------------------------

def measure_mapless(
    r_r: np.ndarray,
    r_a: np.ndarray,
    r_vas: list[np.ndarray],
    delta_t_r: float,
    delta_t_a: float,
    c_light: float = 3e8,
) -> np.ndarray:
    """Compute the nonlinear measurement vector h(x) for the mapless model.

    z_0 = ||r_r - r_a||_2 + c*(delta_t_r - delta_t_a)
    z_i = ||r_r - r_vai||_2 + c*(delta_t_r - delta_t_a)  for i=1..N

    Args:
        r_r:       Receiver position, shape (2,).
        r_a:       Anchor position, shape (2,).
        r_vas:     List of N virtual anchor positions, each shape (2,).
        delta_t_r: Receiver clock bias in seconds.
        delta_t_a: Anchor clock bias in seconds.
        c_light:   Speed of light in m/s.

    Returns:
        Measurement vector z, shape (N+1,).
    """
    clock_term = c_light * (delta_t_r - delta_t_a)
    sources = [r_a] + list(r_vas)
    return np.array([
        np.linalg.norm(r_r - p) + clock_term for p in sources
    ])


def measure_map_aided(
    r_r: np.ndarray,
    r_a: np.ndarray,
    r_vas: list[np.ndarray],
    delta_t_r: float,
    delta_t_a: float,
    c_light: float = 3e8,
) -> np.ndarray:
    """Compute the nonlinear measurement vector hM(xM) for the map-assisted model.

    Functionally identical to measure_mapless; the difference is that
    r_vas here are computed from r_a via MapManager, not from independent states.

    Args:
        r_r:       Receiver position, shape (2,).
        r_a:       Anchor position, shape (2,).
        r_vas:     List of N virtual anchor positions f_i(r_a), each shape (2,).
        delta_t_r: Receiver clock bias in seconds.
        delta_t_a: Anchor clock bias in seconds.
        c_light:   Speed of light in m/s.

    Returns:
        Measurement vector zM, shape (N+1,).
    """
    return measure_mapless(r_r, r_a, r_vas, delta_t_r, delta_t_a, c_light)
