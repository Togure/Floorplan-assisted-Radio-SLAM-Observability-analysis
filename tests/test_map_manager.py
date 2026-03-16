"""Unit tests for src/core/map_manager.py.

Validates:
  • Wall constructor rejects non-unit normals.
  • Householder matrix properties: symmetry, orthogonality, determinant = -1.
  • Householder action on normal/tangent vectors.
  • mirror_point formula against closed-form results.
  • MapManager.rectangular_room wall normals and offsets.
  • virtual_anchor_positions for known anchor positions.
  • reflection_jacobian equals householder output.
  • num_walls matches wall count.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.map_manager import MapManager, Wall, householder, mirror_point


class TestWall:
    """Tests for the Wall dataclass validation."""

    def test_valid_wall_construction(self):
        w = Wall(normal=np.array([1.0, 0.0]), offset=0.0)
        assert w.offset == 0.0

    def test_non_unit_normal_raises(self):
        with pytest.raises(ValueError, match="unit vector"):
            Wall(normal=np.array([2.0, 0.0]), offset=0.0)

    def test_wrong_dimension_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            Wall(normal=np.array([1.0, 0.0, 0.0]), offset=0.0)

    def test_normal_stored_as_float(self):
        w = Wall(normal=np.array([0, 1]), offset=5)
        assert w.normal.dtype == float


class TestHouseholder:
    """Tests for the householder() primitive."""

    @pytest.fixture(params=[
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]) / np.sqrt(2),
        np.array([3.0, 4.0]) / 5.0,
    ])
    def wall(self, request) -> Wall:
        return Wall(normal=request.param, offset=0.0)

    def test_shape(self, wall):
        M = householder(wall)
        assert M.shape == (2, 2)

    def test_symmetry(self, wall):
        M = householder(wall)
        np.testing.assert_array_almost_equal(M, M.T)

    def test_orthogonality(self, wall):
        """M @ M^T must equal I_2 (Householder is orthogonal)."""
        M = householder(wall)
        np.testing.assert_array_almost_equal(M @ M.T, np.eye(2))

    def test_involutory(self, wall):
        """M² must equal I_2."""
        M = householder(wall)
        np.testing.assert_array_almost_equal(M @ M, np.eye(2))

    def test_determinant_minus_one(self, wall):
        """Reflection reverses orientation: det(M) = -1."""
        M = householder(wall)
        assert np.linalg.det(M) == pytest.approx(-1.0)

    def test_normal_component_flipped(self, wall):
        """M·n = -n  (normal direction is reflected)."""
        M = householder(wall)
        np.testing.assert_array_almost_equal(M @ wall.normal, -wall.normal)

    def test_tangent_component_preserved(self):
        """M·t = t  (tangential direction is unchanged)."""
        wall = Wall(normal=np.array([1.0, 0.0]), offset=0.0)
        tangent = np.array([0.0, 1.0])
        M = householder(wall)
        np.testing.assert_array_almost_equal(M @ tangent, tangent)

    def test_explicit_x_wall(self):
        """Left wall (n=[1,0]): M should negate x, preserve y."""
        wall = Wall(normal=np.array([1.0, 0.0]), offset=0.0)
        M = householder(wall)
        expected = np.array([[-1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(M, expected)

    def test_explicit_y_wall(self):
        """Bottom wall (n=[0,1]): M should preserve x, negate y."""
        wall = Wall(normal=np.array([0.0, 1.0]), offset=0.0)
        M = householder(wall)
        expected = np.array([[1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_array_almost_equal(M, expected)


class TestMirrorPoint:
    """Tests for mirror_point() against closed-form results."""

    def test_left_wall_at_origin(self):
        """Reflecting (5, 3) across x=0 (n=[1,0], c=0) → (-5, 3)."""
        wall = Wall(normal=np.array([1.0, 0.0]), offset=0.0)
        result = mirror_point(np.array([5.0, 3.0]), wall)
        np.testing.assert_array_almost_equal(result, np.array([-5.0, 3.0]))

    def test_right_wall(self):
        """Reflecting (5, 3) across x=20 (n=[-1,0], c=-20) → (35, 3)."""
        wall = Wall(normal=np.array([-1.0, 0.0]), offset=-20.0)
        result = mirror_point(np.array([5.0, 3.0]), wall)
        np.testing.assert_array_almost_equal(result, np.array([35.0, 3.0]))

    def test_bottom_wall_at_origin(self):
        """Reflecting (5, 7) across y=0 (n=[0,1], c=0) → (5, -7)."""
        wall = Wall(normal=np.array([0.0, 1.0]), offset=0.0)
        result = mirror_point(np.array([5.0, 7.0]), wall)
        np.testing.assert_array_almost_equal(result, np.array([5.0, -7.0]))

    def test_top_wall(self):
        """Reflecting (5, 7) across y=15 (n=[0,-1], c=-15) → (5, 23)."""
        wall = Wall(normal=np.array([0.0, -1.0]), offset=-15.0)
        result = mirror_point(np.array([5.0, 7.0]), wall)
        np.testing.assert_array_almost_equal(result, np.array([5.0, 23.0]))

    def test_point_on_wall_maps_to_itself(self):
        """A point ON the wall reflects to itself (σ=0)."""
        wall = Wall(normal=np.array([1.0, 0.0]), offset=0.0)
        point_on_wall = np.array([0.0, 7.0])
        result = mirror_point(point_on_wall, wall)
        np.testing.assert_array_almost_equal(result, point_on_wall)

    def test_double_reflection_is_identity(self):
        """Reflecting twice across the same wall returns the original point."""
        wall = Wall(normal=np.array([1.0, 1.0]) / np.sqrt(2), offset=2.0)
        r_a = np.array([3.0, 1.0])
        twice = mirror_point(mirror_point(r_a, wall), wall)
        np.testing.assert_array_almost_equal(twice, r_a)

    def test_jacobian_matches_householder(self):
        """∂f_i/∂r_a = M_i must equal householder(wall) (verified numerically)."""
        wall = Wall(normal=np.array([3.0, 4.0]) / 5.0, offset=3.0)
        r_a = np.array([2.0, 5.0])
        eps = 1e-6
        M_numerical = np.zeros((2, 2))
        for j in range(2):
            dr = np.zeros(2)
            dr[j] = eps
            M_numerical[:, j] = (mirror_point(r_a + dr, wall) - mirror_point(r_a - dr, wall)) / (2 * eps)
        M_analytical = householder(wall)
        np.testing.assert_array_almost_equal(M_numerical, M_analytical, decimal=5)


class TestMapManager:
    """Tests for the MapManager class."""

    def test_rectangular_room_num_walls(self, rectangular_room):
        assert rectangular_room.num_walls == 4

    def test_rectangular_room_normals(self, rectangular_room):
        expected_normals = [
            np.array([ 1.0,  0.0]),  # left
            np.array([-1.0,  0.0]),  # right
            np.array([ 0.0,  1.0]),  # bottom
            np.array([ 0.0, -1.0]),  # top
        ]
        for wall, exp_n in zip(rectangular_room.walls, expected_normals):
            np.testing.assert_array_almost_equal(wall.normal, exp_n)

    def test_rectangular_room_offsets(self, rectangular_room):
        """Offsets for a 20m × 15m room."""
        expected_offsets = [0.0, -20.0, 0.0, -15.0]
        for wall, exp_c in zip(rectangular_room.walls, expected_offsets):
            assert wall.offset == pytest.approx(exp_c)

    def test_invalid_room_dimensions(self):
        with pytest.raises(ValueError):
            MapManager.rectangular_room(width=0.0, height=10.0)

    def test_empty_walls_raises(self):
        with pytest.raises(ValueError):
            MapManager(walls=[])

    def test_virtual_anchor_positions_count(self, rectangular_room, anchor_inside_room):
        vas = rectangular_room.virtual_anchor_positions(anchor_inside_room)
        assert len(vas) == 4

    def test_virtual_anchor_left_wall(self, anchor_inside_room):
        """Anchor at (5, 7) mirrored across x=0 → (-5, 7)."""
        mm = MapManager.rectangular_room(width=20.0, height=15.0)
        vas = mm.virtual_anchor_positions(anchor_inside_room)
        np.testing.assert_array_almost_equal(vas[0], np.array([-5.0, 7.0]))

    def test_virtual_anchor_right_wall(self, anchor_inside_room):
        """Anchor at (5, 7) mirrored across x=20 → (35, 7)."""
        mm = MapManager.rectangular_room(width=20.0, height=15.0)
        vas = mm.virtual_anchor_positions(anchor_inside_room)
        np.testing.assert_array_almost_equal(vas[1], np.array([35.0, 7.0]))

    def test_virtual_anchor_bottom_wall(self, anchor_inside_room):
        """Anchor at (5, 7) mirrored across y=0 → (5, -7)."""
        mm = MapManager.rectangular_room(width=20.0, height=15.0)
        vas = mm.virtual_anchor_positions(anchor_inside_room)
        np.testing.assert_array_almost_equal(vas[2], np.array([5.0, -7.0]))

    def test_virtual_anchor_top_wall(self, anchor_inside_room):
        """Anchor at (5, 7) mirrored across y=15 → (5, 23)."""
        mm = MapManager.rectangular_room(width=20.0, height=15.0)
        vas = mm.virtual_anchor_positions(anchor_inside_room)
        np.testing.assert_array_almost_equal(vas[3], np.array([5.0, 23.0]))

    def test_householder_matrices_count(self, rectangular_room):
        Ms = rectangular_room.householder_matrices()
        assert len(Ms) == 4

    def test_householder_matrices_orthogonal(self, rectangular_room):
        for M in rectangular_room.householder_matrices():
            np.testing.assert_array_almost_equal(M @ M.T, np.eye(2))

    def test_reflection_jacobian_matches_householder(self, rectangular_room):
        for i in range(rectangular_room.num_walls):
            M_via_method = rectangular_room.reflection_jacobian(i)
            M_direct = householder(rectangular_room.walls[i])
            np.testing.assert_array_almost_equal(M_via_method, M_direct)

    def test_reflection_jacobian_out_of_range(self, rectangular_room):
        with pytest.raises(IndexError):
            rectangular_room.reflection_jacobian(99)

    def test_virtual_anchor_wrong_shape(self, rectangular_room):
        with pytest.raises(ValueError):
            rectangular_room.virtual_anchor_positions(np.array([1.0, 2.0, 3.0]))
