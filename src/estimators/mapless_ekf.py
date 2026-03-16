"""State-augmented EKF for the Mapless Radio SLAM model.

State vector (10 + 2N dimensions):
  x = [r_r(2), rdot_r(2), delta_t_r, delta_tdot_r,
        r_a(2),  delta_t_a, delta_tdot_a,
        r_va1(2), ..., r_vaN(2)]

Virtual anchors are treated as independent static landmarks.
Their positions are estimated jointly with the receiver and anchor.

Warm-start strategy for VA initialisation:
  When no prior map is available, VA positions cannot be inferred from a
  single measurement.  A trilateration warm-start is used: each VA is
  initialised at the receiver's estimated position displaced along a
  random bearing by the measured reflected-path range (z_i minus the
  estimated clock offset).  The initial covariance reflects the large
  positional uncertainty.

Reference: Lyu & Zhang, IPIN 2026, Section III-A1.
"""
from __future__ import annotations

import numpy as np

from src.core.dynamics import build_F_mapless, build_Q_mapless
from src.core.measurement import build_H_mapless, measure_mapless
from src.estimators.ekf_base import EKFBase


class MaplessEKF(EKFBase):
    """EKF for the mapless model with augmented virtual-anchor state.

    Args:
        x0:        Initial state estimate, shape (10 + 2N,).
        P0:        Initial covariance, shape (10+2N, 10+2N).
        N:         Number of virtual anchors (reflecting walls).
        dyn_params: Dict with keys q_x, q_y, h0_r, h_2_r, h0_a, h_2_a.
        c_light:   Speed of light in m/s (default 3×10⁸).
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        N: int,
        dyn_params: dict,
        c_light: float = 3e8,
    ) -> None:
        super().__init__(x0, P0)
        if self.state_dim != 10 + 2 * N:
            raise ValueError(
                f"x0 must have length 10+2N={10+2*N}, got {self.state_dim}."
            )
        self._N = N
        self._dyn = dyn_params
        self._c = c_light

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def step(self, z: np.ndarray, T: float, R: np.ndarray) -> None:
        """Execute one predict-update cycle.

        Args:
            z: Measurement vector (N+1 ranges), shape (N+1,).
            T: Sampling period in seconds.
            R: Measurement noise covariance, shape (N+1, N+1).
        """
        N = self._N

        # ---- Predict ----
        # Q is built in metres² (c_light=self._c scales clock blocks by c²)
        F = build_F_mapless(T, N)
        Q = build_Q_mapless(
            T, N,
            self._dyn["q_x"], self._dyn["q_y"],
            self._dyn["h0_r"], self._dyn["h_2_r"],
            self._dyn["h0_a"], self._dyn["h_2_a"],
            c_light=self._c,
        )
        # Small VA process noise prevents P_va from collapsing prematurely,
        # which would cause NEES to explode when actual VA errors are non-zero.
        q_va = self._dyn.get("q_va", 0.0)
        if q_va > 0.0:
            Q[10: 10 + 2 * N, 10: 10 + 2 * N] += q_va * np.eye(2 * N)
        self.predict(F, Q)

        # ---- Extract state components ----
        # Clock states (indices 4,5,8,9) are stored in metres (b = c·δt)
        r_r   = self.x[0:2]
        r_a   = self.x[6:8]
        r_vas = [self.x[10 + 2 * i: 12 + 2 * i] for i in range(N)]
        b_r   = self.x[4]   # range-equivalent clock bias [m]
        b_a   = self.x[8]

        # ---- Compute predicted measurement and Jacobian ----
        # c_light=1.0: clock columns of H become ±1 (consistent with metres state)
        z_hat = measure_mapless(r_r, r_a, r_vas, b_r, b_a, c_light=1.0)
        H     = build_H_mapless(r_r, r_a, r_vas, c_light=1.0)

        # ---- Update ----
        self.update(z - z_hat, H, R)


# ------------------------------------------------------------------
# Initialisation helpers
# ------------------------------------------------------------------

def build_initial_state_mapless(
    x_true_0: np.ndarray,
    N: int,
    sigma_pos: float,
    sigma_vel: float,
    sigma_b:   float,
    sigma_bd:  float,
    sigma_va:  float,
    know_position: bool = False,
    know_time: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the initial state estimate and covariance for the mapless EKF.

    Clock states (indices 4, 5, 8, 9) are in metres (range-equivalent),
    so sigma_b / sigma_bd are also in metres / (m/s).

    Args:
        x_true_0:      True initial state, shape (10 + 2N,).
        N:             Number of virtual anchors.
        sigma_pos:     Position init std-dev [m].
        sigma_vel:     Velocity init std-dev [m/s].
        sigma_b:       Clock bias init std-dev [m range-equivalent].
        sigma_bd:      Clock drift init std-dev [m/s range-equivalent].
        sigma_va:      VA position warm-start std-dev [m].
        know_position: If True, use 0.1 m std-dev for receiver position.
        know_time:     If True, use 0.1 m std-dev for clock bias.
        rng:           Random number generator.

    Returns:
        (x0_ekf, P0_ekf): Initial state and covariance.
    """
    if rng is None:
        rng = np.random.default_rng()

    x0 = x_true_0.copy()

    sig_pos_r = 0.1  if know_position else sigma_pos
    sig_b_r   = 0.1  if know_time    else sigma_b
    sig_bd_r  = 0.01 if know_time    else sigma_bd

    x0[0:2] += rng.normal(0, sig_pos_r, 2)   # receiver position
    x0[2:4] += rng.normal(0, sigma_vel, 2)   # receiver velocity
    x0[4]   += rng.normal(0, sig_b_r)        # b_r  [m]
    x0[5]   += rng.normal(0, sig_bd_r)       # bd_r [m/s]
    x0[6:8] += rng.normal(0, sigma_pos, 2)   # anchor position
    x0[8]   += rng.normal(0, sig_b_r)        # b_a  [m]
    x0[9]   += rng.normal(0, sig_bd_r)       # bd_a [m/s]
    for i in range(N):
        x0[10 + 2*i: 12 + 2*i] += rng.normal(0, sigma_va, 2)

    diag_vals = np.array(
        [sig_pos_r**2, sig_pos_r**2,   # r_r
         sigma_vel**2, sigma_vel**2,   # rdot_r
         sig_b_r**2,   sig_bd_r**2,   # b_r, bd_r
         sigma_pos**2, sigma_pos**2,  # r_a
         sig_b_r**2,   sig_bd_r**2,   # b_a, bd_a
         ] + [sigma_va**2, sigma_va**2] * N
    )
    return x0, np.diag(diag_vals)
