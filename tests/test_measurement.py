"""Unit tests for src/core/measurement.py.

Validates:
  • unit_vector direction, normalisation, and coincident-point guard.
  • build_H_mapless:
      - output shape (N+1) × (10+2N)
      - zero velocity columns
      - clock bias columns = ±c_light
      - anchor position row 0 = -ê^T(r_a), rows 1..N = 0
      - VA blocks: each row i has -ê^T(r_vai) in the correct 2-column slot
      - numerical Jacobian agreement (finite-difference verification)
  • build_H_map_aided:
      - output shape (N+1) × 10
      - anchor position rows via chain rule: -ê^T(f_i(r_a)) @ M_i
      - identical Hr sub-block to mapless
      - numerical Jacobian agreement
  • measure_mapless / measure_map_aided:
      - output length N+1
      - clock-bias scaling by c_light
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.map_manager import MapManager
from src.core.measurement import (
    build_H_map_aided,
    build_H_mapless,
    measure_map_aided,
    measure_mapless,
    unit_vector,
)

# ---------------------------------------------------------------------------
# Shared geometry fixture
# ---------------------------------------------------------------------------

C_LIGHT = 3e8   # speed of light used throughout

@pytest.fixture
def geometry():
    """Canonical test geometry: receiver, anchor, 3 virtual anchors."""
    r_r = np.array([3.0, 4.0])
    r_a = np.array([10.0, 8.0])
    mm  = MapManager.rectangular_room(width=20.0, height=15.0)
    r_vas = mm.virtual_anchor_positions(r_a)         # 4 VAs
    M_list = mm.householder_matrices()               # 4 Householder matrices
    return r_r, r_a, r_vas, M_list


# ---------------------------------------------------------------------------
# unit_vector
# ---------------------------------------------------------------------------

class TestUnitVector:
    def test_known_direction(self):
        u = unit_vector(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_almost_equal(u, np.array([0.6, 0.8]))

    def test_unit_norm(self):
        for _ in range(20):
            a = np.random.randn(2)
            b = np.random.randn(2) + 5.0
            assert np.linalg.norm(unit_vector(a, b)) == pytest.approx(1.0)

    def test_opposite_directions(self):
        u_fwd = unit_vector(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        u_bwd = unit_vector(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(u_fwd, -u_bwd)

    def test_coincident_raises(self):
        with pytest.raises(ValueError, match="Coincident"):
            unit_vector(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# build_H_mapless
# ---------------------------------------------------------------------------

class TestBuildHMapless:

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_shape(self, N):
        r_r  = np.array([3.0, 4.0])
        r_a  = np.array([10.0, 8.0])
        # Use offsets far from r_r to avoid coincident-point errors
        r_vas = [np.array([20.0 + float(i), 20.0 + float(i)]) for i in range(N)]
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        assert H.shape == (N + 1, 10 + 2 * N)

    def test_velocity_columns_zero(self, geometry):
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        # cols 2:4 are rdot_r — must be zero for all rows
        np.testing.assert_array_equal(H[:, 2:4], 0.0)

    def test_clock_drift_columns_zero(self, geometry):
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        # col 5 = delta_tdot_r, col 9 = delta_tdot_a → always zero
        np.testing.assert_array_equal(H[:, 5], 0.0)
        np.testing.assert_array_equal(H[:, 9], 0.0)

    def test_clock_bias_receiver_column(self, geometry):
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        # col 4 = +c for all rows
        np.testing.assert_array_almost_equal(H[:, 4], C_LIGHT)

    def test_clock_bias_anchor_column(self, geometry):
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        # col 8 = -c for all rows
        np.testing.assert_array_almost_equal(H[:, 8], -C_LIGHT)

    def test_receiver_position_row0(self, geometry):
        """Row 0 receiver position block = ê^T(r_a)."""
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        e_expected = (r_r - r_a) / np.linalg.norm(r_r - r_a)
        np.testing.assert_array_almost_equal(H[0, 0:2], e_expected)

    def test_receiver_position_reflected_rows(self, geometry):
        """Row i receiver position block = ê^T(r_vai)."""
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        for i, r_vai in enumerate(r_vas):
            e_expected = (r_r - r_vai) / np.linalg.norm(r_r - r_vai)
            np.testing.assert_array_almost_equal(H[i + 1, 0:2], e_expected)

    def test_anchor_position_row0(self, geometry):
        """Row 0 anchor position block = -ê^T(r_a)."""
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        e = (r_r - r_a) / np.linalg.norm(r_r - r_a)
        np.testing.assert_array_almost_equal(H[0, 6:8], -e)

    def test_anchor_position_reflected_rows_zero(self, geometry):
        """Rows 1..N anchor position block = 0 (VA not function of r_a in mapless)."""
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        np.testing.assert_array_equal(H[1:, 6:8], 0.0)

    def test_va_block_diagonal_structure(self, geometry):
        """Each reflected row has -ê^T(r_vai) only in the i-th VA column slot."""
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        N = len(r_vas)
        for i, r_vai in enumerate(r_vas):
            e = (r_r - r_vai) / np.linalg.norm(r_r - r_vai)
            va_col = 10 + 2 * i
            # Correct slot
            np.testing.assert_array_almost_equal(H[i + 1, va_col:va_col + 2], -e)
            # All other VA slots must be zero
            for j in range(N):
                if j != i:
                    other_col = 10 + 2 * j
                    np.testing.assert_array_equal(H[i + 1, other_col:other_col + 2], 0.0)

    def test_va_block_row0_all_zero(self, geometry):
        """Direct-path row has no VA column contributions."""
        r_r, r_a, r_vas, _ = geometry
        H = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        np.testing.assert_array_equal(H[0, 10:], 0.0)

    def test_numerical_jacobian_mapless(self, geometry):
        """Finite-difference check of full H against the analytic formula."""
        r_r, r_a, r_vas, _ = geometry
        N = len(r_vas)
        dim = 10 + 2 * N
        eps = 1e-5

        # Assemble nominal state
        state = np.zeros(dim)
        state[0:2]  = r_r
        state[6:8]  = r_a
        for i, rv in enumerate(r_vas):
            state[10 + 2*i: 12 + 2*i] = rv

        def h(s):
            rr_  = s[0:2]
            ra_  = s[6:8]
            rvs  = [s[10 + 2*j: 12 + 2*j] for j in range(N)]
            dt_r = s[4]    # delta_t_r lives at col 4
            dt_a = s[8]    # delta_t_a lives at col 8
            return measure_mapless(rr_, ra_, rvs, dt_r, dt_a, C_LIGHT)

        J_num = np.zeros((N + 1, dim))
        for col in range(dim):
            dp = np.zeros(dim); dp[col] = eps
            J_num[:, col] = (h(state + dp) - h(state - dp)) / (2 * eps)

        H_analytic = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        np.testing.assert_array_almost_equal(J_num, H_analytic, decimal=5)


# ---------------------------------------------------------------------------
# build_H_map_aided
# ---------------------------------------------------------------------------

class TestBuildHMapAided:

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_shape(self, N):
        mm = MapManager.rectangular_room(20.0, 15.0)
        r_r = np.array([3.0, 4.0])
        r_a = np.array([10.0, 8.0])
        r_vas  = mm.virtual_anchor_positions(r_a)[:N]
        M_list = mm.householder_matrices()[:N]
        HM = build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)
        assert HM.shape == (N + 1, 10)

    def test_velocity_columns_zero(self, geometry):
        r_r, r_a, r_vas, M_list = geometry
        HM = build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)
        np.testing.assert_array_equal(HM[:, 2:4], 0.0)

    def test_clock_columns(self, geometry):
        r_r, r_a, r_vas, M_list = geometry
        HM = build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)
        np.testing.assert_array_almost_equal(HM[:, 4],  C_LIGHT)
        np.testing.assert_array_almost_equal(HM[:, 8], -C_LIGHT)
        np.testing.assert_array_equal(HM[:, 5], 0.0)
        np.testing.assert_array_equal(HM[:, 9], 0.0)

    def test_Hr_identical_to_mapless(self, geometry):
        """HM,r sub-block must equal Hr from the mapless model exactly."""
        r_r, r_a, r_vas, M_list = geometry
        HM = build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)
        H_ml = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        np.testing.assert_array_almost_equal(HM[:, 0:6], H_ml[:, 0:6])

    def test_anchor_row0_same_as_mapless(self, geometry):
        """Direct-path anchor block row 0 must be -ê^T(r_a) in both models."""
        r_r, r_a, r_vas, M_list = geometry
        HM = build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)
        H_ml = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        np.testing.assert_array_almost_equal(HM[0, 6:8], H_ml[0, 6:8])

    def test_anchor_reflected_rows_nonzero(self, geometry):
        """Reflected-path anchor rows must be non-zero (chain-rule fill-in)."""
        r_r, r_a, r_vas, M_list = geometry
        HM = build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)
        for row in range(1, HM.shape[0]):
            assert not np.allclose(HM[row, 6:8], 0.0), (
                f"Row {row} anchor block is zero — chain-rule fill-in missing."
            )

    def test_anchor_chain_rule_formula(self, geometry):
        """∂z_i/∂r_a = -ê^T(f_i(r_a)) @ M_i  (Eq. 10, chain rule)."""
        r_r, r_a, r_vas, M_list = geometry
        HM = build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)
        for i, (r_vai, Mi) in enumerate(zip(r_vas, M_list)):
            e_i = (r_r - r_vai) / np.linalg.norm(r_r - r_vai)
            expected = -(e_i @ Mi)      # 1×2 row vector
            np.testing.assert_array_almost_equal(
                HM[i + 1, 6:8], expected,
                err_msg=f"Chain-rule mismatch for reflected path {i + 1}."
            )

    def test_mapless_vs_mapAided_anchor_difference(self, geometry):
        """Mapless anchor rows 1..N are zero; map-aided are non-zero."""
        r_r, r_a, r_vas, M_list = geometry
        H_ml = build_H_mapless(r_r, r_a, r_vas, C_LIGHT)
        HM   = build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)
        # Mapless: reflected anchor position sub-block must be zero
        np.testing.assert_array_equal(H_ml[1:, 6:8], 0.0)
        # Map-aided: same rows must be non-zero
        for row in range(1, HM.shape[0]):
            assert not np.allclose(HM[row, 6:8], 0.0)

    def test_mismatched_M_list_raises(self):
        r_r = np.array([3.0, 4.0])
        r_a = np.array([10.0, 8.0])
        r_vas  = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        M_list = [np.eye(2)]   # length 1 ≠ 2
        with pytest.raises(ValueError, match="M_list"):
            build_H_map_aided(r_r, r_a, r_vas, M_list, C_LIGHT)

    def test_numerical_jacobian_map_aided(self, geometry):
        """Finite-difference check of HM against analytic formula."""
        r_r, r_a, r_vas, M_list = geometry
        mm = MapManager.rectangular_room(20.0, 15.0)
        eps = 1e-5

        # Nominal 10-dim state
        state = np.zeros(10)
        state[0:2] = r_r
        state[6:8] = r_a

        def hM(s):
            rr_  = s[0:2]
            ra_  = s[6:8]
            rvs  = mm.virtual_anchor_positions(ra_)
            dt_r = s[4]    # delta_t_r lives at col 4
            dt_a = s[8]    # delta_t_a lives at col 8
            return measure_map_aided(rr_, ra_, rvs, dt_r, dt_a, C_LIGHT)

        J_num = np.zeros((len(r_vas) + 1, 10))
        for col in range(10):
            dp = np.zeros(10); dp[col] = eps
            J_num[:, col] = (hM(state + dp) - hM(state - dp)) / (2 * eps)

        r_vas_nominal = mm.virtual_anchor_positions(r_a)
        HM_analytic = build_H_map_aided(r_r, r_a, r_vas_nominal, M_list, C_LIGHT)
        np.testing.assert_array_almost_equal(J_num, HM_analytic, decimal=4)


# ---------------------------------------------------------------------------
# measure_mapless / measure_map_aided
# ---------------------------------------------------------------------------

class TestMeasureFunctions:

    def test_length(self, geometry):
        r_r, r_a, r_vas, _ = geometry
        z = measure_mapless(r_r, r_a, r_vas, 0.0, 0.0, C_LIGHT)
        assert z.shape == (len(r_vas) + 1,)

    def test_direct_path_range(self, geometry):
        r_r, r_a, r_vas, _ = geometry
        z = measure_mapless(r_r, r_a, r_vas, 0.0, 0.0, C_LIGHT)
        expected = np.linalg.norm(r_r - r_a)
        assert z[0] == pytest.approx(expected)

    def test_clock_term_scales(self, geometry):
        r_r, r_a, r_vas, _ = geometry
        dt = 1e-9   # 1 nanosecond
        z0 = measure_mapless(r_r, r_a, r_vas, 0.0,  0.0,  C_LIGHT)
        z1 = measure_mapless(r_r, r_a, r_vas, dt,   0.0,  C_LIGHT)
        diff = z1 - z0
        np.testing.assert_array_almost_equal(diff, C_LIGHT * dt)

    def test_relative_clock_bias(self, geometry):
        """Only the relative clock bias Δt = δt_r - δt_a matters."""
        r_r, r_a, r_vas, _ = geometry
        z_a = measure_mapless(r_r, r_a, r_vas,  1e-9, 0.0,   C_LIGHT)
        z_b = measure_mapless(r_r, r_a, r_vas,  2e-9, 1e-9,  C_LIGHT)
        np.testing.assert_array_almost_equal(z_a, z_b)

    def test_map_aided_equals_mapless_numerically(self, geometry):
        """For the same VA positions the two functions return identical values."""
        r_r, r_a, r_vas, _ = geometry
        z_ml = measure_mapless(r_r, r_a, r_vas, 1e-9, 5e-10, C_LIGHT)
        z_ma = measure_map_aided(r_r, r_a, r_vas, 1e-9, 5e-10, C_LIGHT)
        np.testing.assert_array_almost_equal(z_ml, z_ma)
