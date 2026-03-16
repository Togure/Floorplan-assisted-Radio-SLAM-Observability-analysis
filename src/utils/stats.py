"""NEES consistency check and Chi-squared acceptance bounds.

NEES (Normalized Estimation Error Squared) for a single epoch:

  epsilon_k = (x_true - x_hat)^T  P^{+}  (x_true - x_hat)

where P^{+} is the Moore-Penrose pseudo-inverse.  Using the pseudo-inverse
is essential because unobservable directions have infinite variance in P,
yielding near-zero singular values.  The pseudo-inverse naturally projects
the error onto the observable subspace, avoiding numerical explosion.

Consistency test (Kassas & Humphreys 2014):
  For n_runs independent MC runs, the time-averaged NEES

      ε̄_k = (1/n_runs) Σ_{j=1}^{n_runs} epsilon_k^{(j)}

  follows a scaled Chi-squared distribution:

      n_runs * ε̄_k ~ χ²(d * n_runs)

  The 99% acceptance region is:

      r1 = chi2.ppf(α/2,   d*n_runs) / n_runs
      r2 = chi2.ppf(1-α/2, d*n_runs) / n_runs    (α = 0.01)

Reference:
  Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications
  to Tracking and Navigation," Wiley, 2001. Chapter 5.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def compute_nees(
    error: np.ndarray,
    P: np.ndarray,
    rcond: float = 1e-10,
) -> float:
    """Compute NEES for a single time step.

    Uses the pseudo-inverse of P to handle near-zero eigenvalues that arise
    in unobservable directions.  Directions with singular values below
    rcond * max_singular_value are treated as unobservable (excluded).

    Args:
        error: State estimation error (x_true - x_hat), shape (n,).
        P:     State covariance matrix, shape (n, n).
        rcond: Relative threshold for pseudo-inverse (default 1e-10).

    Returns:
        NEES scalar epsilon_k ≥ 0.
    """
    error = np.asarray(error, dtype=float)
    P     = np.asarray(P, dtype=float)
    if error.ndim != 1:
        raise ValueError(f"error must be 1-D, got shape {error.shape}.")
    if P.shape != (len(error), len(error)):
        raise ValueError(
            f"P shape {P.shape} inconsistent with error length {len(error)}."
        )
    P_inv = np.linalg.pinv(P, rcond=rcond)
    return float(error @ P_inv @ error)


def chi2_bounds(
    state_dim: int,
    n_runs: int,
    alpha: float = 0.01,
) -> tuple[float, float]:
    """Compute Chi-squared acceptance bounds for the average NEES.

    The degree-of-freedom (dof) for the aggregate test is d * n_runs,
    and the bounds are scaled back by n_runs to give per-run bounds on ε̄.

    Args:
        state_dim: State vector dimension d.
        n_runs:    Number of Monte Carlo runs.
        alpha:     Significance level (default 0.01 → 99% bounds).

    Returns:
        (r1, r2): Lower and upper average NEES acceptance bounds.
    """
    if state_dim <= 0:
        raise ValueError(f"state_dim must be positive, got {state_dim}.")
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}.")
    dof = state_dim * n_runs
    r1 = float(chi2.ppf(alpha / 2.0,       dof) / n_runs)
    r2 = float(chi2.ppf(1.0 - alpha / 2.0, dof) / n_runs)
    return r1, r2


def average_nees(nees_array: np.ndarray) -> np.ndarray:
    """Compute the Monte Carlo average NEES across runs at every time step.

    Args:
        nees_array: Per-run NEES values, shape (n_steps, n_runs).

    Returns:
        Mean NEES over runs at each time step, shape (n_steps,).
    """
    return np.mean(nees_array, axis=1)


def rmse(errors: np.ndarray) -> np.ndarray:
    """Compute RMS error across Monte Carlo runs at each time step.

    Args:
        errors: Error values, shape (n_steps, n_runs) or (n_steps, n_runs, n_states).

    Returns:
        RMSE, shape (n_steps,) or (n_steps, n_states).
    """
    return np.sqrt(np.mean(errors**2, axis=1))
