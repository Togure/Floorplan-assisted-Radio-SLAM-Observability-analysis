"""Unit tests for observability matrix builder and SVD rank analysis.

Validates:
  • build_observability_matrix: shape, first block = H(t0), stacking order.
  • svd_rank_analysis: rank of known matrices, condition number, null_dim.
  • rank_vs_steps: monotone-non-decreasing rank sequence.
  • theoretical_max_rank / minimum_epochs: Table I boundary values.
  • Structural rank bounds: mapless max = 2N+5, map-aided max = 8.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.dynamics import build_F_map_aided, build_F_mapless
from src.core.map_manager import MapManager
from src.core.measurement import build_H_map_aided, build_H_mapless
from src.core.observability import (
    build_observability_matrix,
    minimum_epochs,
    rank_vs_steps,
    svd_rank_analysis,
    theoretical_max_rank,
)


# ---------------------------------------------------------------------------
# build_observability_matrix
# ---------------------------------------------------------------------------

class TestBuildObservabilityMatrix:

    def test_shape_single_H(self):
        H = np.random.randn(3, 6)
        F = np.eye(6)
        O = build_observability_matrix([H], F)
        assert O.shape == (3, 6)

    def test_shape_multiple_H(self):
        m, n, l = 4, 8, 5
        H_list = [np.random.randn(m, n) for _ in range(l + 1)]
        F = np.eye(n)
        O = build_observability_matrix(H_list, F)
        assert O.shape == ((l + 1) * m, n)

    def test_first_block_equals_H0(self):
        """O[:m, :] must equal H(t0) exactly (F^0 = I)."""
        m, n = 3, 5
        H0 = np.random.randn(m, n)
        H1 = np.random.randn(m, n)
        F  = np.random.randn(n, n)
        O = build_observability_matrix([H0, H1], F)
        np.testing.assert_array_almost_equal(O[:m, :], H0)

    def test_second_block_equals_H1_F(self):
        """O[m:2m, :] must equal H(t1) @ F."""
        m, n = 3, 5
        H0 = np.random.randn(m, n)
        H1 = np.random.randn(m, n)
        F  = np.random.randn(n, n)
        O = build_observability_matrix([H0, H1], F)
        np.testing.assert_array_almost_equal(O[m:2*m, :], H1 @ F)

    def test_third_block_equals_H2_F2(self):
        m, n = 2, 4
        H_list = [np.random.randn(m, n) for _ in range(3)]
        F = np.random.randn(n, n)
        O = build_observability_matrix(H_list, F)
        np.testing.assert_array_almost_equal(O[2*m:3*m, :], H_list[2] @ F @ F)

    def test_empty_H_list_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            build_observability_matrix([], np.eye(4))

    def test_shape_mismatch_F_raises(self):
        H = np.random.randn(3, 5)
        F = np.eye(6)   # wrong size
        with pytest.raises(ValueError):
            build_observability_matrix([H], F)

    def test_inconsistent_H_shape_raises(self):
        H0 = np.random.randn(3, 5)
        H1 = np.random.randn(4, 5)   # wrong row count
        F  = np.eye(5)
        with pytest.raises(ValueError):
            build_observability_matrix([H0, H1], F)

    def test_identity_F_stacks_H_unchanged(self):
        """With F=I every block H_k @ F^k = H_k."""
        m, n = 2, 4
        H_list = [np.random.randn(m, n) for _ in range(4)]
        F = np.eye(n)
        O = build_observability_matrix(H_list, F)
        for k, Hk in enumerate(H_list):
            np.testing.assert_array_almost_equal(O[k*m:(k+1)*m, :], Hk)


# ---------------------------------------------------------------------------
# svd_rank_analysis
# ---------------------------------------------------------------------------

class TestSvdRankAnalysis:

    def test_full_rank_matrix(self):
        O = np.eye(5)
        res = svd_rank_analysis(O)
        assert res["rank"] == 5
        assert res["null_dim"] == 0

    def test_rank_deficient_matrix(self):
        O = np.zeros((6, 4))
        O[0, 0] = 1.0
        O[1, 1] = 1.0   # rank = 2
        res = svd_rank_analysis(O)
        assert res["rank"] == 2
        assert res["null_dim"] == 2

    def test_all_zero_matrix(self):
        O = np.zeros((5, 5))
        res = svd_rank_analysis(O)
        assert res["rank"] == 0
        assert res["null_dim"] == 5
        assert res["condition_number"] == np.inf

    def test_condition_number_positive(self):
        O = np.diag([10.0, 5.0, 1.0])
        res = svd_rank_analysis(O)
        assert res["condition_number"] == pytest.approx(10.0)

    def test_singular_values_descending(self):
        O = np.random.randn(8, 5)
        res = svd_rank_analysis(O)
        sv = res["singular_values"]
        assert np.all(sv[:-1] >= sv[1:])

    def test_null_vectors_shape(self):
        O = np.zeros((5, 4))
        O[0, 0] = 1.0      # rank = 1, null_dim = 3
        res = svd_rank_analysis(O)
        assert res["null_vectors"].shape == (4, 3)

    def test_null_vectors_in_null_space(self):
        """O @ null_vectors ≈ 0."""
        O = np.random.randn(8, 5)
        O[:, 3] = O[:, 0]   # introduce one linear dependence
        res = svd_rank_analysis(O)
        residual = O @ res["null_vectors"]
        np.testing.assert_array_almost_equal(residual, 0.0, decimal=8)

    def test_custom_tol(self):
        sv = np.array([100.0, 10.0, 1e-12, 1e-15])
        U = np.eye(4)
        O = U @ np.diag(sv) @ U.T
        res = svd_rank_analysis(O, tol=1e-10)
        assert res["rank"] == 2


# ---------------------------------------------------------------------------
# rank_vs_steps
# ---------------------------------------------------------------------------

class TestRankVsSteps:

    def test_monotone_non_decreasing(self):
        """Rank must be non-decreasing as more epochs are added."""
        n = 6
        F = np.random.randn(n, n)
        H_list = [np.random.randn(3, n) for _ in range(5)]
        results = rank_vs_steps(H_list, F)
        ranks = [r["rank"] for r in results]
        for a, b in zip(ranks, ranks[1:]):
            assert b >= a, f"Rank decreased: {ranks}"

    def test_length_equals_H_list(self):
        F = np.eye(4)
        H_list = [np.random.randn(2, 4) for _ in range(6)]
        results = rank_vs_steps(H_list, F)
        assert len(results) == 6


# ---------------------------------------------------------------------------
# Theoretical bounds — Table I
# ---------------------------------------------------------------------------

class TestTheoreticalBounds:

    @pytest.mark.parametrize("N,expected", [(3, 11), (4, 13), (5, 15)])
    def test_mapless_rank(self, N, expected):
        assert theoretical_max_rank("mapless", N) == expected

    def test_map_aided_rank(self):
        for N in range(3, 8):
            assert theoretical_max_rank("map_aided", N) == 8

    def test_minimum_epochs_mapless(self):
        assert minimum_epochs("mapless") == 3

    def test_minimum_epochs_map_aided(self):
        assert minimum_epochs("map_aided") == 2

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            theoretical_max_rank("wrong", 3)

    def test_N_less_than_3_raises(self):
        with pytest.raises(ValueError, match="N >= 3"):
            theoretical_max_rank("mapless", 2)


# ---------------------------------------------------------------------------
# Integration: structural rank on realistic geometry
# ---------------------------------------------------------------------------

class TestStructuralRank:
    """End-to-end rank check using actual F and H from the two models."""

    @pytest.fixture
    def scenario(self):
        T   = 1.0
        N   = 4                            # 4 reflecting walls
        mm  = MapManager.rectangular_room(20.0, 15.0)
        r_r = np.array([3.0,  4.0])
        r_a = np.array([10.0, 8.0])
        r_vas  = mm.virtual_anchor_positions(r_a)
        M_list = mm.householder_matrices()
        return T, N, mm, r_r, r_a, r_vas, M_list

    def test_map_aided_rank_reaches_8_at_l2(self, scenario):
        """Map-aided model must reach rank 8 by l=2 (N≥3, Table I)."""
        T, N, mm, r_r, r_a, r_vas, M_list = scenario
        F  = build_F_map_aided(T)
        # Simulate 3 epochs of Jacobians (l=0,1,2)
        H_list = [
            build_H_map_aided(r_r, r_a, r_vas, M_list) for _ in range(3)
        ]
        results = rank_vs_steps(H_list, F)
        rank_l2 = results[2]["rank"]     # index 2 = after 3 blocks (l=2)
        assert rank_l2 == 8, (
            f"Map-aided rank at l=2 should be 8, got {rank_l2}. "
            "Check that H chain-rule fill-in is correct."
        )

    def test_map_aided_rank_at_l1_less_than_8(self, scenario):
        """Map-aided rank at l=1 (single snapshot, 1 block) must be < 8.

        At a single epoch the velocity and clock-drift columns of H are
        structurally zero, limiting rank to at most N+1 (= 5 for N=4).
        Only when F is applied (l≥2) do those columns become non-zero.
        """
        T, N, mm, r_r, r_a, r_vas, M_list = scenario
        F  = build_F_map_aided(T)
        # Single block: O = [H(t0)]  →  at most (N+1) = 5 rows, rank ≤ 5
        H_list = [build_H_map_aided(r_r, r_a, r_vas, M_list)]
        results = rank_vs_steps(H_list, F)
        rank_l1 = results[0]["rank"]   # single snapshot
        assert rank_l1 < 8, (
            f"Map-aided single-snapshot rank should be <8, got {rank_l1}."
        )

    def test_mapless_rank_upper_bound(self, scenario):
        """Mapless rank after many epochs must not exceed 2N+5."""
        T, N, mm, r_r, r_a, r_vas, M_list = scenario
        F  = build_F_mapless(T, N)
        H_list = [
            build_H_mapless(r_r, r_a, r_vas) for _ in range(6)
        ]
        results = rank_vs_steps(H_list, F)
        max_rank = max(r["rank"] for r in results)
        assert max_rank <= 2 * N + 5, (
            f"Mapless rank {max_rank} exceeds theoretical bound {2*N+5}."
        )
