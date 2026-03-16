"""Base Extended Kalman Filter with numerically robust covariance updates.

Design principles:
  • Predict step  : x_pred = F @ x;  P_pred = F @ P @ F^T + Q
  • Update  step  : Joseph form  P = (I-KH) P (I-KH)^T + K R K^T
                    (numerically superior to the simple P = (I-KH)P form
                     because it remains symmetric and PSD even with float errors)
  • S inversion   : Cholesky preferred; falls back to pseudo-inverse (SVD) when
                    S is near-singular (occurs in unobservable directions)
  • Symmetry guard: P is symmetrised after every predict/update to prevent
                    numerical drift from corrupting the Cholesky decomposition

Subclasses provide:
  • build_step_matrices() — model-specific F, Q, R, z_hat, H at each epoch
  Alternatively, call predict() and update() directly with externally built matrices.
"""
from __future__ import annotations

import numpy as np


class EKFBase:
    """Abstract base EKF.  Subclasses override step() or call predict()/update().

    Attributes:
        x: Current state estimate, shape (n,).
        P: Current state covariance (symmetric PSD), shape (n, n).
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.x: np.ndarray = np.asarray(x0, dtype=float).copy()
        self.P: np.ndarray = np.asarray(P0, dtype=float).copy()
        _check_square_symmetric(self.P, "P0")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector."""
        return len(self.x)

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        """EKF time-propagation step.

        x_pred = F @ x
        P_pred = F @ P @ F^T + Q

        Args:
            F: State transition matrix, shape (n, n).
            Q: Process noise covariance (PSD), shape (n, n).
        """
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = _symmetrise(self.P)

    def update(
        self,
        innovation: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """EKF measurement-update step (Joseph form).

        Args:
            innovation: z - z_hat (pre-computed by subclass), shape (m,).
            H:          Observation Jacobian at predicted state, shape (m, n).
            R:          Measurement noise covariance (PD), shape (m, m).
        """
        P, n = self.P, self.state_dim
        S = H @ P @ H.T + R                     # innovation covariance (m×m)
        S = _symmetrise(S)
        S_inv = _robust_invert(S)               # Cholesky → SVD fallback
        K = P @ H.T @ S_inv                     # Kalman gain (n×m)

        self.x = self.x + K @ innovation

        I_KH = np.eye(n) - K @ H               # (n×n) factor for Joseph form
        self.P = I_KH @ P @ I_KH.T + K @ R @ K.T
        self.P = _symmetrise(self.P)


# ------------------------------------------------------------------
# Module-level numerical helpers (not exported to subclasses as methods
# to keep the class interface clean)
# ------------------------------------------------------------------

def _symmetrise(A: np.ndarray) -> np.ndarray:
    """Return (A + A^T) / 2 to correct floating-point asymmetry."""
    return 0.5 * (A + A.T)


def _robust_invert(S: np.ndarray) -> np.ndarray:
    """Invert a symmetric matrix with Cholesky, falling back to pseudo-inverse.

    Cholesky is O(m³/3) and numerically excellent for PD matrices.
    The SVD pseudo-inverse is used when Cholesky fails (near-singular S),
    which happens in directions orthogonal to the observable subspace.

    Args:
        S: Symmetric matrix, shape (m, m).

    Returns:
        S^{-1} or S^+ (pseudo-inverse), shape (m, m).
    """
    try:
        L = np.linalg.cholesky(S)
        # Solve L L^T X = I  →  X = (L^T)^{-1} L^{-1}
        eye = np.eye(len(S))
        return np.linalg.solve(L.T, np.linalg.solve(L, eye))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(S)


def _check_square_symmetric(A: np.ndarray, name: str) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {A.shape}.")
    if not np.allclose(A, A.T, atol=1e-9):
        raise ValueError(f"{name} must be symmetric.")
