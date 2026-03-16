"""Map-constrained EKF for the Map-Assisted Radio SLAM model.

State vector (10 dimensions — fixed):
  xM = [r_r(2), rdot_r(2), delta_t_r, delta_tdot_r,
         r_a(2),  delta_t_a, delta_tdot_a]

Virtual anchor positions are NOT states.  They are algebraic functions of
the estimated anchor position r_a, computed on-the-fly via MapManager:
  r_vai = f_i(r_a) = M_i @ r_a + 2*c_i*n_i

The Householder matrices M_i are injected into the anchor sub-block of the
Jacobian HM,a through the chain rule (Eq. 10 of the paper).

Reference: Lyu & Zhang, IPIN 2026, Section III-A2.
"""
from __future__ import annotations

import numpy as np

from src.core.dynamics import build_F_map_aided, build_Q_map_aided
from src.core.map_manager import MapManager
from src.core.measurement import build_H_map_aided, measure_map_aided
from src.estimators.ekf_base import EKFBase


class MapAidedEKF(EKFBase):
    """EKF for the map-assisted model with a fixed 10-dimensional state.

    Args:
        x0:         Initial state estimate, shape (10,).
        P0:         Initial covariance, shape (10, 10).
        map_manager: MapManager carrying the floorplan geometry.
        dyn_params: Dict with keys q_x, q_y, h0_r, h_2_r, h0_a, h_2_a.
        c_light:    Speed of light in m/s (default 3×10⁸).
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        map_manager: MapManager,
        dyn_params: dict,
        c_light: float = 3e8,
    ) -> None:
        super().__init__(x0, P0)
        if self.state_dim != 10:
            raise ValueError(
                f"Map-aided state must be 10-dimensional, got {self.state_dim}."
            )
        self._mm   = map_manager
        self._dyn  = dyn_params
        self._c    = c_light

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def step(self, z: np.ndarray, T: float, R: np.ndarray) -> None:
        """Execute one predict-update cycle.

        Virtual anchor positions are recomputed from r_a at each step, so
        the floorplan constraint is applied implicitly at every epoch.

        Args:
            z: Measurement vector (N+1 ranges), shape (N+1,).
            T: Sampling period in seconds.
            R: Measurement noise covariance, shape (N+1, N+1).
        """
        # ---- Predict ----
        # Q built in metres² (c_light=self._c scales clock noise blocks by c²)
        F = build_F_map_aided(T)
        Q = build_Q_map_aided(
            T,
            self._dyn["q_x"], self._dyn["q_y"],
            self._dyn["h0_r"], self._dyn["h_2_r"],
            self._dyn["h0_a"], self._dyn["h_2_a"],
            c_light=self._c,
        )
        self.predict(F, Q)

        # ---- Derive VA positions from current anchor estimate ----
        # Clock states (indices 4,5,8,9) are in metres (b = c·δt)
        r_r    = self.x[0:2]
        r_a    = self.x[6:8]
        N      = len(z) - 1
        r_vas  = self._mm.virtual_anchor_positions(r_a)[:N]
        M_list = self._mm.householder_matrices()[:N]
        b_r    = self.x[4]   # [m]
        b_a    = self.x[8]

        # ---- Compute predicted measurement and Jacobian ----
        if len(z) != N + 1:
            raise ValueError(
                f"Measurement z has length {len(z)}, expected N+1={N+1}."
            )
        # c_light=1.0: clock columns of H are ±1, consistent with metres state
        z_hat = measure_map_aided(r_r, r_a, r_vas, b_r, b_a, c_light=1.0)
        H     = build_H_map_aided(r_r, r_a, r_vas, M_list, c_light=1.0)

        # ---- Update ----
        self.update(z - z_hat, H, R)


# ------------------------------------------------------------------
# Initialisation helpers
# ------------------------------------------------------------------

def build_initial_state_map_aided(
    x_true_0: np.ndarray,
    sigma_pos: float,
    sigma_vel: float,
    sigma_b:   float,
    sigma_bd:  float,
    know_position: bool = False,
    know_time: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the initial 10-dim state estimate and covariance for map-aided EKF.

    Clock states are in metres (range-equivalent), so sigma_b / sigma_bd are
    in metres / (m/s).

    Args:
        x_true_0:      True initial 10-dim state.
        sigma_pos:     Position init std-dev [m].
        sigma_vel:     Velocity init std-dev [m/s].
        sigma_b:       Clock bias init std-dev [m].
        sigma_bd:      Clock drift init std-dev [m/s].
        know_position: If True, use 0.1 m std-dev for receiver position.
        know_time:     If True, use 0.1 m std-dev for clock bias.
        rng:           Random number generator.

    Returns:
        (x0_ekf, P0_ekf): Initial state and covariance.
    """
    if rng is None:
        rng = np.random.default_rng()

    x0 = x_true_0[:10].copy()

    sig_pos_r = 0.1  if know_position else sigma_pos
    sig_b_r   = 0.1  if know_time    else sigma_b
    sig_bd_r  = 0.01 if know_time    else sigma_bd

    x0[0:2] += rng.normal(0, sig_pos_r, 2)
    x0[2:4] += rng.normal(0, sigma_vel, 2)
    x0[4]   += rng.normal(0, sig_b_r)
    x0[5]   += rng.normal(0, sig_bd_r)
    x0[6:8] += rng.normal(0, sigma_pos, 2)
    x0[8]   += rng.normal(0, sig_b_r)
    x0[9]   += rng.normal(0, sig_bd_r)

    diag_vals = np.array([
        sig_pos_r**2, sig_pos_r**2,
        sigma_vel**2, sigma_vel**2,
        sig_b_r**2,   sig_bd_r**2,
        sigma_pos**2, sigma_pos**2,
        sig_b_r**2,   sig_bd_r**2,
    ])
    return x0, np.diag(diag_vals)
