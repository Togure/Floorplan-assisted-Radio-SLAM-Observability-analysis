"""Wall geometry and Householder reflection transformations for map-assisted SLAM.

Each wall is defined by:
  - n_i : unit inward-facing normal vector ∈ R²
  - c_i : scalar offset such that {r : n_i^T r = c_i} is the wall hyperplane

Mirror formula (Householder reflection):
  f_i(r_a) = r_a − 2·(n_i^T r_a − c_i)·n_i
           = M_i @ r_a + 2·c_i·n_i

Householder matrix:
  M_i = I_2 − 2·n_i·n_i^T

Jacobian of the mirroring function (used in the map-assisted Jacobian):
  ∂f_i(r_a)/∂r_a = M_i        (constant — independent of r_a)

Reference: Lyu & Zhang, IPIN 2026, Equations (9)–(10) and surrounding text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Wall dataclass
# ---------------------------------------------------------------------------

@dataclass
class Wall:
    """A single planar wall in 2D, defined by its unit inward normal and offset.

    The wall occupies the hyperplane {r ∈ R² : n^T r = c}.

    Attributes:
        normal: Unit inward-facing normal vector, shape (2,).
        offset: Scalar c in the wall equation n^T r = c.
    """

    normal: np.ndarray
    offset: float

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=float)
        if self.normal.shape != (2,):
            raise ValueError(
                f"Wall normal must be 2-D, got shape {self.normal.shape}."
            )
        mag = np.linalg.norm(self.normal)
        if not np.isclose(mag, 1.0, atol=1e-9):
            raise ValueError(
                f"Wall normal must be a unit vector; got ||n|| = {mag:.9f}. "
                "Normalise before constructing Wall."
            )


# ---------------------------------------------------------------------------
# Core reflection primitives
# ---------------------------------------------------------------------------

def householder(wall: Wall) -> np.ndarray:
    """Compute the 2×2 Householder reflection matrix for a single wall.

    M_i = I_2 − 2·n_i·n_i^T

    Key properties (n_i is a unit vector):
      • Symmetric:    M_i^T = M_i
      • Orthogonal:   M_i @ M_i^T = I_2
      • Involutory:   M_i² = I_2
      • det(M_i) = −1   (orientation-reversing reflection)
      • M_i·n_i = −n_i  (normal component flipped)
      • M_i·t_i = t_i   (tangential component preserved, t_i ⊥ n_i)

    Args:
        wall: Wall instance carrying the unit normal n_i.

    Returns:
        2×2 Householder matrix M_i.
    """
    n = wall.normal
    return np.eye(2) - 2.0 * np.outer(n, n)


def mirror_point(r_a: np.ndarray, wall: Wall) -> np.ndarray:
    """Reflect a point across a wall using the Householder transformation.

    Derivation:
      σ = n_i^T r_a − c_i        (signed distance from r_a to wall)
      f_i(r_a) = r_a − 2·σ·n_i
               = M_i @ r_a + 2·c_i·n_i

    Verification (right wall at x=W, n=[-1,0], c=-W, anchor at (ax, ay)):
      σ = −ax − (−W) = W − ax
      f = (ax, ay) − 2(W−ax)(−1, 0) = (2W−ax, ay) ✓

    Args:
        r_a: Point to reflect (typically the physical anchor position), shape (2,).
        wall: Wall to reflect across.

    Returns:
        Reflected point f_i(r_a), shape (2,).
    """
    r_a = np.asarray(r_a, dtype=float)
    signed_dist = wall.normal @ r_a - wall.offset
    return r_a - 2.0 * signed_dist * wall.normal


# ---------------------------------------------------------------------------
# MapManager
# ---------------------------------------------------------------------------

class MapManager:
    """Manages floorplan geometry and virtual anchor position generation.

    For each reflecting wall i:
      • Stores the Wall (normal n_i, offset c_i).
      • Provides the Householder matrix M_i = ∂f_i/∂r_a (used in HM,a Jacobian).
      • Computes virtual anchor positions r_vai = f_i(r_a) on demand.

    Attributes:
        walls: Ordered list of Wall objects describing the floorplan.
    """

    def __init__(self, walls: List[Wall]) -> None:
        if not walls:
            raise ValueError("MapManager requires at least one wall.")
        self.walls: List[Wall] = walls

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def rectangular_room(cls, width: float, height: float) -> "MapManager":
        """Create a MapManager for an axis-aligned rectangular room.

        Four walls are defined with inward-facing unit normals:
          Wall 0 — left   (x = 0):     n = [ 1,  0], c =  0
          Wall 1 — right  (x = W):     n = [-1,  0], c = -W
          Wall 2 — bottom (y = 0):     n = [ 0,  1], c =  0
          Wall 3 — top    (y = H):     n = [ 0, -1], c = -H

        Mirroring verification for anchor at (ax, ay):
          Left   → (−ax, ay)      Right → (2W−ax, ay)
          Bottom → (ax, −ay)      Top   → (ax, 2H−ay)

        Args:
            width:  Room width  W in metres (x-direction).
            height: Room height H in metres (y-direction).

        Returns:
            MapManager with four walls ordered [left, right, bottom, top].
        """
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Room dimensions must be positive; got width={width}, height={height}."
            )
        walls = [
            Wall(normal=np.array([ 1.0,  0.0]), offset=0.0),      # left:   x = 0
            Wall(normal=np.array([-1.0,  0.0]), offset=-width),    # right:  x = W
            Wall(normal=np.array([ 0.0,  1.0]), offset=0.0),       # bottom: y = 0
            Wall(normal=np.array([ 0.0, -1.0]), offset=-height),   # top:    y = H
        ]
        return cls(walls)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_walls(self) -> int:
        """Number of reflecting walls N (equals number of virtual anchors)."""
        return len(self.walls)

    # ------------------------------------------------------------------
    # Core outputs consumed by the EKF modules
    # ------------------------------------------------------------------

    def householder_matrices(self) -> List[np.ndarray]:
        """Return the Householder matrix M_i for every wall.

        These are the Jacobians ∂f_i/∂r_a injected into HM,a.

        Returns:
            List of N arrays, each 2×2.
        """
        return [householder(w) for w in self.walls]

    def virtual_anchor_positions(self, r_a: np.ndarray) -> List[np.ndarray]:
        """Compute all virtual anchor positions for a given physical anchor.

        r_vai = f_i(r_a) = M_i @ r_a + 2·c_i·n_i   for i = 1..N

        Args:
            r_a: Physical anchor position, shape (2,).

        Returns:
            List of N virtual anchor positions, each shape (2,).
        """
        r_a = np.asarray(r_a, dtype=float)
        if r_a.shape != (2,):
            raise ValueError(f"Anchor position must be shape (2,), got {r_a.shape}.")
        return [mirror_point(r_a, w) for w in self.walls]

    def reflection_jacobian(self, wall_index: int) -> np.ndarray:
        """Return the Jacobian ∂f_i/∂r_a = M_i for the i-th wall.

        Convenience wrapper for single-wall access inside the EKF update loop.

        Args:
            wall_index: Zero-based index into self.walls.

        Returns:
            2×2 Householder matrix M_i.

        Raises:
            IndexError: If wall_index is out of range.
        """
        if not (0 <= wall_index < self.num_walls):
            raise IndexError(
                f"wall_index {wall_index} out of range for {self.num_walls} walls."
            )
        return householder(self.walls[wall_index])
