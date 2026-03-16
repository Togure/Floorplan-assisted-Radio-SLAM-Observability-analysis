"""l-step observability matrix builder and SVD rank/estimability analysis.

The discrete-time l-step observability matrix is:

  O(t0, tl) = [ H(t0)         ]     ← shape m × n
              [ H(t1) @ F     ]
              [ H(t2) @ F²    ]
              [      ...       ]
              [ H(tl) @ F^l   ]   total shape: (l+1)*m × n

where m = N+1 (measurements per epoch), n = state dimension,
F is the (constant) state transition matrix.

Theoretical rank bounds (IPIN 2026, Table I, N ≥ 3):
  Mapless model      (n = 2N+10): max rank = 2N+5   (l ≥ 3)
  Map-assisted model (n = 10):    max rank = 8       (l ≥ 2)

SVD-based metrics follow Kassas & Humphreys 2014:
  - Numerical rank via singular value threshold
  - Condition number (estimability proxy): σ_max / σ_min_nonzero
  - Null space dimension: n - rank

Reference:
  Z. M. Kassas & T. E. Humphreys, IEEE Trans. ITS, 2014.
  Lyu & Zhang, IPIN 2026, Section IV.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Observability matrix builder
# ---------------------------------------------------------------------------

def build_observability_matrix(
    H_list: list[np.ndarray],
    F: np.ndarray,
) -> np.ndarray:
    """Stack the l-step observability matrix O(t0, tl).

    O = vstack([ H(tk) @ F^k  for k = 0, 1, ..., l ])

    F^0 = I_n is used for k=0, so the first block is just H(t0).
    The matrix power F^k is computed by repeated matrix multiplication
    to avoid numerical drift from scipy.linalg.matrix_power for large k.

    Args:
        H_list: List of l+1 Jacobian matrices [H(t0), H(t1), ..., H(tl)],
                each shape (m, n). All must share the same (m, n).
        F:      State transition matrix, shape (n, n).

    Returns:
        O: Observability matrix, shape ((l+1)*m, n).

    Raises:
        ValueError: On shape inconsistency between H entries or F.
    """
    if not H_list:
        raise ValueError("H_list must contain at least one Jacobian matrix.")

    m, n = H_list[0].shape
    if F.shape != (n, n):
        raise ValueError(
            f"F must be ({n},{n}) to match H columns, got {F.shape}."
        )
    for k, Hk in enumerate(H_list):
        if Hk.shape != (m, n):
            raise ValueError(
                f"H_list[{k}] has shape {Hk.shape}, expected ({m},{n})."
            )

    rows = []
    Fk = np.eye(n)          # F^0 = I
    for Hk in H_list:
        rows.append(Hk @ Fk)
        Fk = Fk @ F         # F^{k+1}

    return np.vstack(rows)


# ---------------------------------------------------------------------------
# SVD rank analysis
# ---------------------------------------------------------------------------

def svd_rank_analysis(
    O: np.ndarray,
    tol: float | None = None,
) -> dict:
    """Compute SVD-based rank and estimability metrics for an observability matrix.

    Singular values below `tol` are treated as numerically zero.
    Default tolerance follows NumPy convention:
        tol = max(O.shape) * eps * sigma_max

    Returns a dict with:
      'rank'             : int   — numerical rank of O
      'singular_values'  : array — all singular values, descending
      'condition_number' : float — σ_max / σ_min_nonzero
                                   (inf if rank == 0)
      'null_dim'         : int   — n - rank  (dimension of unobservable subspace)
      'null_vectors'     : array — right singular vectors for σ < tol, shape (n, null_dim)

    Args:
        O:   Observability matrix, shape (rows, n).
        tol: Singular value threshold. None → auto (machine-epsilon based).

    Returns:
        Analysis dict as described above.
    """
    _, sigmas, Vt = np.linalg.svd(O, full_matrices=True)
    n = O.shape[1]

    if tol is None:
        eps = np.finfo(float).eps
        tol = max(O.shape) * eps * (sigmas[0] if sigmas.size > 0 else 1.0)

    rank = int(np.sum(sigmas > tol))
    nonzero_sigmas = sigmas[sigmas > tol]
    cond = float(nonzero_sigmas[0] / nonzero_sigmas[-1]) if rank > 0 else np.inf

    # Null space: right singular vectors corresponding to near-zero singular values
    # Vt rows are right singular vectors; last (n - rank) rows form null space basis
    null_vectors = Vt[rank:].T   # shape (n, null_dim)

    return {
        "rank": rank,
        "singular_values": sigmas,
        "condition_number": cond,
        "null_dim": n - rank,
        "null_vectors": null_vectors,
    }


# ---------------------------------------------------------------------------
# Convenience: analyse across increasing l steps
# ---------------------------------------------------------------------------

def rank_vs_steps(
    H_list: list[np.ndarray],
    F: np.ndarray,
    tol: float | None = None,
) -> list[dict]:
    """Compute SVD rank analysis for each prefix O(t0, tk), k = 0..l.

    Useful for verifying the minimum number of epochs needed to reach
    the theoretical maximum rank (l=3 for mapless, l=2 for map-aided).

    Args:
        H_list: List of l+1 Jacobians [H(t0), ..., H(tl)].
        F:      State transition matrix, shape (n, n).
        tol:    Singular value threshold. None → auto per epoch.

    Returns:
        List of l+1 analysis dicts, one per prefix length.
    """
    results = []
    for end in range(1, len(H_list) + 1):
        O = build_observability_matrix(H_list[:end], F)
        results.append(svd_rank_analysis(O, tol=tol))
    return results


# ---------------------------------------------------------------------------
# Theoretical rank bounds (paper Table I)
# ---------------------------------------------------------------------------

def theoretical_max_rank(model: str, N: int) -> int:
    """Return the theoretical maximum observable rank from Table I.

    Args:
        model: 'mapless' or 'map_aided'.
        N:     Number of virtual anchors (reflecting walls). Must be >= 3.

    Returns:
        Maximum achievable rank of O(t0, tl) for l large enough.

    Raises:
        ValueError: For unknown model name or N < 3.
    """
    if N < 3:
        raise ValueError(
            f"Table I results require N >= 3 multipaths; got N={N}."
        )
    if model == "mapless":
        return 2 * N + 5
    if model == "map_aided":
        return 8
    raise ValueError(f"Unknown model '{model}'. Use 'mapless' or 'map_aided'.")


def minimum_epochs(model: str) -> int:
    """Return the minimum number of epochs l to reach theoretical max rank.

    Args:
        model: 'mapless' or 'map_aided'.

    Returns:
        Minimum l (number of state transitions before max rank is achieved).
    """
    if model == "mapless":
        return 3
    if model == "map_aided":
        return 2
    raise ValueError(f"Unknown model '{model}'. Use 'mapless' or 'map_aided'.")
