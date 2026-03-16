"""Unit tests for src/core/dynamics.py.

Validates:
  • F_clk and Q_clk matrix entries against closed-form expressions.
  • Q_pv symmetry and positive-semi-definiteness.
  • Fr, Fa shapes and structural entries.
  • F_mapless / F_map_aided block-diagonal assembly.
  • Q_mapless / Q_map_aided symmetry.
  • State dimension contracts: mapless = 10+2N, map-aided = 10.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.dynamics import (
    build_F_a,
    build_F_clk,
    build_F_map_aided,
    build_F_mapless,
    build_F_r,
    build_Q_a,
    build_Q_clk,
    build_Q_map_aided,
    build_Q_mapless,
    build_Q_pv,
    build_Q_r,
)


class TestClockModel:
    """Tests for F_clk and Q_clk."""

    def test_F_clk_values(self, default_T):
        F = build_F_clk(default_T)
        expected = np.array([[1.0, default_T], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(F, expected)

    def test_F_clk_shape(self, default_T):
        assert build_F_clk(default_T).shape == (2, 2)

    def test_Q_clk_shape(self, default_T, clock_params):
        Q = build_Q_clk(default_T, **clock_params)
        assert Q.shape == (2, 2)

    def test_Q_clk_symmetry(self, default_T, clock_params):
        Q = build_Q_clk(default_T, **clock_params)
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_Q_clk_positive_definite(self, default_T, clock_params):
        Q = build_Q_clk(default_T, **clock_params)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues > 0), f"Q_clk is not PD: eigenvalues = {eigenvalues}"

    def test_Q_clk_entry_formula(self, clock_params):
        """Verify diagonal entries against the closed-form formula."""
        T = 2.0
        h0, h_2 = clock_params["h0"], clock_params["h_2"]
        s_wdt = h0 / 2.0
        s_wddt = 2.0 * np.pi**2 * h_2
        Q = build_Q_clk(T, h0, h_2)
        assert Q[0, 0] == pytest.approx(s_wdt * T + s_wddt * T**3 / 3.0)
        assert Q[1, 1] == pytest.approx(s_wddt * T)
        assert Q[0, 1] == pytest.approx(s_wddt * T**2 / 2.0)


class TestQpv:
    """Tests for the position-velocity process noise covariance."""

    def test_shape(self, default_T):
        assert build_Q_pv(default_T, 0.1, 0.1).shape == (4, 4)

    def test_symmetry(self, default_T):
        Q = build_Q_pv(default_T, 0.1, 0.2)
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_positive_semi_definite(self, default_T):
        Q = build_Q_pv(default_T, 0.1, 0.2)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-12), f"Q_pv has negative eigenvalue: {eigenvalues}"

    def test_entry_formula(self):
        """Check specific entries against analytical formula."""
        T, q_x, q_y = 1.0, 0.5, 0.3
        Q = build_Q_pv(T, q_x, q_y)
        assert Q[0, 0] == pytest.approx(q_x * T**3 / 3.0)
        assert Q[1, 1] == pytest.approx(q_y * T**3 / 3.0)
        assert Q[2, 2] == pytest.approx(q_x * T)
        assert Q[3, 3] == pytest.approx(q_y * T)
        assert Q[0, 2] == pytest.approx(q_x * T**2 / 2.0)
        assert Q[1, 3] == pytest.approx(q_y * T**2 / 2.0)
        # Cross-axis terms must be zero (independent x/y noise)
        assert Q[0, 1] == pytest.approx(0.0)
        assert Q[0, 3] == pytest.approx(0.0)


class TestReceiverBlock:
    """Tests for Fr (6×6 receiver transition matrix)."""

    def test_shape(self, default_T):
        assert build_F_r(default_T).shape == (6, 6)

    def test_kinematic_integration(self, default_T):
        """Position integrates velocity over period T."""
        F = build_F_r(default_T)
        # r_r ← rdot_r block must be T * I2
        np.testing.assert_array_almost_equal(F[0:2, 2:4], default_T * np.eye(2))

    def test_position_identity(self, default_T):
        F = build_F_r(default_T)
        np.testing.assert_array_almost_equal(F[0:2, 0:2], np.eye(2))

    def test_velocity_identity(self, default_T):
        F = build_F_r(default_T)
        np.testing.assert_array_almost_equal(F[2:4, 2:4], np.eye(2))

    def test_clock_block(self, default_T):
        F = build_F_r(default_T)
        expected_clk = build_F_clk(default_T)
        np.testing.assert_array_almost_equal(F[4:6, 4:6], expected_clk)

    def test_off_diagonal_zero(self, default_T):
        """Blocks coupling position/velocity to clock must be zero."""
        F = build_F_r(default_T)
        np.testing.assert_array_almost_equal(F[0:4, 4:6], np.zeros((4, 2)))
        np.testing.assert_array_almost_equal(F[4:6, 0:4], np.zeros((2, 4)))


class TestAnchorBlock:
    """Tests for Fa (4×4 anchor transition matrix)."""

    def test_shape(self, default_T):
        assert build_F_a(default_T).shape == (4, 4)

    def test_position_stationary(self, default_T):
        F = build_F_a(default_T)
        np.testing.assert_array_almost_equal(F[0:2, 0:2], np.eye(2))

    def test_clock_block(self, default_T):
        F = build_F_a(default_T)
        np.testing.assert_array_almost_equal(F[2:4, 2:4], build_F_clk(default_T))

    def test_cross_terms_zero(self, default_T):
        F = build_F_a(default_T)
        np.testing.assert_array_almost_equal(F[0:2, 2:4], np.zeros((2, 2)))
        np.testing.assert_array_almost_equal(F[2:4, 0:2], np.zeros((2, 2)))


class TestMaplessSystem:
    """Tests for F_mapless and Q_mapless (10 + 2N dimensions)."""

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_F_shape(self, default_T, N):
        F = build_F_mapless(default_T, N)
        assert F.shape == (10 + 2 * N, 10 + 2 * N)

    @pytest.mark.parametrize("N", [1, 3])
    def test_VA_block_is_identity(self, default_T, N):
        F = build_F_mapless(default_T, N)
        np.testing.assert_array_almost_equal(
            F[10:10 + 2 * N, 10:10 + 2 * N], np.eye(2 * N)
        )

    @pytest.mark.parametrize("N", [1, 3])
    def test_F_receiver_block(self, default_T, N):
        F = build_F_mapless(default_T, N)
        np.testing.assert_array_almost_equal(F[0:6, 0:6], build_F_r(default_T))

    @pytest.mark.parametrize("N", [1, 3])
    def test_F_anchor_block(self, default_T, N):
        F = build_F_mapless(default_T, N)
        np.testing.assert_array_almost_equal(F[6:10, 6:10], build_F_a(default_T))

    @pytest.mark.parametrize("N", [1, 3])
    def test_Q_shape(self, default_T, clock_params, N):
        Q = build_Q_mapless(
            default_T, N,
            q_x=0.1, q_y=0.1,
            h0_r=clock_params["h0"], h_2_r=clock_params["h_2"],
            h0_a=clock_params["h0"], h_2_a=clock_params["h_2"],
        )
        assert Q.shape == (10 + 2 * N, 10 + 2 * N)

    @pytest.mark.parametrize("N", [1, 3])
    def test_Q_symmetry(self, default_T, clock_params, N):
        Q = build_Q_mapless(
            default_T, N,
            q_x=0.1, q_y=0.1,
            h0_r=clock_params["h0"], h_2_r=clock_params["h_2"],
            h0_a=clock_params["h0"], h_2_a=clock_params["h_2"],
        )
        np.testing.assert_array_almost_equal(Q, Q.T)

    @pytest.mark.parametrize("N", [1, 3])
    def test_Q_VA_block_is_zero(self, default_T, clock_params, N):
        """Virtual anchors must have zero process noise."""
        Q = build_Q_mapless(
            default_T, N,
            q_x=0.1, q_y=0.1,
            h0_r=clock_params["h0"], h_2_r=clock_params["h_2"],
            h0_a=clock_params["h0"], h_2_a=clock_params["h_2"],
        )
        np.testing.assert_array_almost_equal(Q[10:, 10:], np.zeros((2 * N, 2 * N)))


class TestMapAidedSystem:
    """Tests for F_map_aided and Q_map_aided (10 dimensions)."""

    def test_F_shape(self, default_T):
        assert build_F_map_aided(default_T).shape == (10, 10)

    def test_F_receiver_block(self, default_T):
        F = build_F_map_aided(default_T)
        np.testing.assert_array_almost_equal(F[0:6, 0:6], build_F_r(default_T))

    def test_F_anchor_block(self, default_T):
        F = build_F_map_aided(default_T)
        np.testing.assert_array_almost_equal(F[6:10, 6:10], build_F_a(default_T))

    def test_Q_shape(self, default_T, clock_params):
        Q = build_Q_map_aided(
            default_T,
            q_x=0.1, q_y=0.1,
            h0_r=clock_params["h0"], h_2_r=clock_params["h_2"],
            h0_a=clock_params["h0"], h_2_a=clock_params["h_2"],
        )
        assert Q.shape == (10, 10)

    def test_Q_symmetry(self, default_T, clock_params):
        Q = build_Q_map_aided(
            default_T,
            q_x=0.1, q_y=0.1,
            h0_r=clock_params["h0"], h_2_r=clock_params["h_2"],
            h0_a=clock_params["h0"], h_2_a=clock_params["h_2"],
        )
        np.testing.assert_array_almost_equal(Q, Q.T)

    def test_state_dimension_contract(self, default_T):
        """Map-aided state dimension must be exactly 10."""
        F = build_F_map_aided(default_T)
        assert F.shape[0] == 10
        assert F.shape[1] == 10
