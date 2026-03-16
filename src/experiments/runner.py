"""Monte Carlo engine for the Radio SLAM Observability Benchmarking Suite.

Simulation scenario
-------------------
• Rectangular room (width W × height H metres) with 4 walls.
• One stationary anchor at a fixed off-centre position ra_true.
• N = cfg.N reflecting walls → N virtual anchors.
• Receiver follows a Lissajous figure-8 trajectory that covers both
  halves of the room, staying strictly inside the walls at all times.
• Clock bias b = c·δt and drift bd = c·δṫ are stored in **metres**
  (range-equivalent) throughout.  This eliminates the ~10⁸ scale mismatch
  between clock and position states that would otherwise make the EKF
  numerically degenerate.

Two EKFs run in parallel on every MC trial:
  • MaplessEKF   (10 + 2N states) — VAs as unknown static landmarks
  • MapAidedEKF  (10 states)      — VAs derived from floorplan via MapManager

Results dict keys
-----------------
  mapless / map_aided:
    errors     (n_steps, n_runs, dim)    true − estimated
    cov_diags  (n_steps, n_runs, dim)    diagonal of P
    nees       (n_steps, n_runs)         NEES scalar
    avg_nees   (n_steps,)               mean NEES over runs
    r1, r2     float                     99% Chi-squared bounds (obs. rank DoF)
    sigmas     (n_steps, dim)            sqrt of mean P diagonal
    rmse_pos   (n_steps,)               position RMSE

  obs_cond_ml/ma  (n_steps,)   condition number of O(t0, tk)
  obs_rank_ml/ma  (n_steps,)   rank of O(t0, tk)
  time            (n_steps,)   time axis [s]
  sample          dict         first-run data for 2-D trajectory plots
"""
from __future__ import annotations

import numpy as np

from src.core.dynamics import (
    build_F_map_aided,
    build_F_mapless,
    build_Q_clk,
)
from src.core.map_manager import MapManager
from src.core.measurement import build_H_map_aided, build_H_mapless, measure_mapless
from src.core.observability import build_observability_matrix, svd_rank_analysis
from src.estimators.map_aided_ekf import MapAidedEKF, build_initial_state_map_aided
from src.estimators.mapless_ekf import MaplessEKF, build_initial_state_mapless
from src.experiments.config import ExperimentConfig
from src.utils.stats import average_nees, chi2_bounds, compute_nees, rmse

_C = 3e8   # speed of light [m/s]

# Initial-uncertainty defaults (all in SI units consistent with metres-clock)
_SIGMA_POS = 0.8    # m   — receiver/anchor position cold-start.
                    #       Reduced from 2 m: large σ causes the EKF to get
                    #       trapped in local minima when N=4 (5 measurements/step
                    #       collapse P faster than coupled receiver-anchor errors
                    #       can converge). 0.8 m is still a meaningful "no prior"
                    #       scenario (e.g., coarse floor-map knowledge ≈ 1 m).
_SIGMA_VEL = 0.3    # m/s — receiver velocity cold-start
_SIGMA_B   = 12.0   # m   — clock bias cold-start (range-equivalent, ≈ 40 ns)
_SIGMA_BD  = 0.3    # m/s — clock drift cold-start
_SIGMA_VA  = 3.0    # m   — VA position warm-start (mapless only)


# ---------------------------------------------------------------------------
# Lissajous figure-8 trajectory generator
# ---------------------------------------------------------------------------

def _generate_true_trajectory(
    cfg: ExperimentConfig,
    r_a_true: np.ndarray,
    mm: MapManager,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one MC realisation of the ground-truth state trajectory.

    Receiver path: figure-8 Lissajous curve covering both halves of the room,
    with small Gaussian perturbations and hard clipping to room bounds.

    Clock states (x[4], x[5], x[8], x[9]) are in **metres** (range-equivalent).
    Anchor position and VA positions are held constant throughout.

    Returns:
        x_true: shape (n_steps + 1, 10 + 2*N), mapless-format full state.
    """
    N, T = cfg.N, cfg.T
    n    = cfg.n_steps
    margin = 1.2   # minimum distance from any wall [m]

    W, H = cfg.room_width, cfg.room_height
    cx, cy = W / 2.0, H / 2.0
    Ax = W / 2.0 - margin          # x amplitude
    Ay = H / 2.0 - margin          # y amplitude

    # Figure-8: x→1 full oscillation, y→2 full oscillations per simulation
    T_total = n * T
    t_arr   = np.arange(n + 1) * T
    omega   = 2.0 * np.pi / T_total

    x_nom = cx + Ax * np.sin(omega * t_arr)
    y_nom = cy + Ay * np.sin(2.0 * omega * t_arr)

    # Small perturbations (σ = 0.05 m) to break exact periodicity
    x_arr = np.clip(x_nom + rng.normal(0, 0.05, n + 1), margin, W - margin)
    y_arr = np.clip(y_nom + rng.normal(0, 0.05, n + 1), margin, H - margin)

    # Velocities via finite difference (forward difference at k=0)
    vx = np.empty(n + 1)
    vy = np.empty(n + 1)
    vx[1:] = (x_arr[1:] - x_arr[:-1]) / T
    vy[1:] = (y_arr[1:] - y_arr[:-1]) / T
    vx[0]  = vx[1]
    vy[0]  = vy[1]

    # ---- Clock trajectory in metres (b = c·δt) ----
    Q_clk_r = build_Q_clk(T, cfg.h0_r, cfg.h_2_r, c_light=_C)
    Q_clk_a = build_Q_clk(T, cfg.h0_a, cfg.h_2_a, c_light=_C)

    b_r  = np.zeros(n + 1)
    bd_r = np.zeros(n + 1)
    b_a  = np.zeros(n + 1)
    bd_a = np.zeros(n + 1)

    # Random initial clock biases (range-equivalent)
    b_r[0]  = rng.normal(0, 15.0)
    bd_r[0] = rng.normal(0,  0.2)
    b_a[0]  = rng.normal(0,  8.0)
    bd_a[0] = rng.normal(0,  0.05)

    sig_b_r  = np.sqrt(Q_clk_r[0, 0])
    sig_bd_r = np.sqrt(Q_clk_r[1, 1])
    sig_b_a  = np.sqrt(Q_clk_a[0, 0])
    sig_bd_a = np.sqrt(Q_clk_a[1, 1])

    for k in range(n):
        b_r[k+1]  = b_r[k]  + T * bd_r[k] + rng.normal(0, sig_b_r)
        bd_r[k+1] = bd_r[k]               + rng.normal(0, sig_bd_r)
        b_a[k+1]  = b_a[k]  + T * bd_a[k] + rng.normal(0, sig_b_a)
        bd_a[k+1] = bd_a[k]               + rng.normal(0, sig_bd_a)

    # ---- Assemble full mapless state ----
    r_vas_true = mm.virtual_anchor_positions(r_a_true)[:N]
    dim = 10 + 2 * N
    x_true = np.zeros((n + 1, dim))

    x_true[:, 0] = x_arr
    x_true[:, 1] = y_arr
    x_true[:, 2] = vx
    x_true[:, 3] = vy
    x_true[:, 4] = b_r          # clock bias  [m]
    x_true[:, 5] = bd_r         # clock drift [m/s]
    x_true[:, 6] = r_a_true[0]  # anchor x (fixed)
    x_true[:, 7] = r_a_true[1]  # anchor y (fixed)
    x_true[:, 8] = b_a          # anchor bias  [m]
    x_true[:, 9] = bd_a         # anchor drift [m/s]
    for i, rv in enumerate(r_vas_true):
        x_true[:, 10 + 2*i]     = rv[0]
        x_true[:, 10 + 2*i + 1] = rv[1]

    return x_true


# ---------------------------------------------------------------------------
# Measurement generator
# ---------------------------------------------------------------------------

def _generate_measurements(
    cfg: ExperimentConfig,
    x_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate noisy range measurements from the true state trajectory.

    Clock states are in metres, so c_light=1.0 is passed to measure_mapless.

    Returns:
        z_all: shape (n_steps + 1, N + 1).
    """
    N = cfg.N
    n_epochs = x_true.shape[0]
    z_all = np.zeros((n_epochs, N + 1))

    for k in range(n_epochs):
        r_r   = x_true[k, 0:2]
        r_a   = x_true[k, 6:8]
        r_vas = [x_true[k, 10 + 2*i: 12 + 2*i] for i in range(N)]
        b_r   = x_true[k, 4]
        b_a   = x_true[k, 8]
        z_true_k = measure_mapless(r_r, r_a, r_vas, b_r, b_a, c_light=1.0)
        z_all[k] = z_true_k + rng.normal(0, cfg.sigma_range, N + 1)

    return z_all


# ---------------------------------------------------------------------------
# Observability tracker
# ---------------------------------------------------------------------------

def _track_observability(
    cfg: ExperimentConfig,
    x_true: np.ndarray,
    mm: MapManager,
) -> dict:
    """Compute l-step observability rank and condition number along x_true.

    Jacobians use c_light=1.0 (consistent with metres-clock state) so that
    the H columns have uniform O(1) scale, giving meaningful condition numbers.
    """
    N = cfg.N
    T = cfg.T
    n_epochs = x_true.shape[0]

    F_ml = build_F_mapless(T, N)
    F_ma = build_F_map_aided(T)

    H_acc_ml: list[np.ndarray] = []
    H_acc_ma: list[np.ndarray] = []

    cond_ml = np.full(n_epochs, np.nan)
    rank_ml = np.zeros(n_epochs, dtype=int)
    cond_ma = np.full(n_epochs, np.nan)
    rank_ma = np.zeros(n_epochs, dtype=int)

    for k in range(n_epochs):
        r_r    = x_true[k, 0:2]
        r_a    = x_true[k, 6:8]
        r_vas  = [x_true[k, 10 + 2*i: 12 + 2*i] for i in range(N)]
        M_list = mm.householder_matrices()[:N]

        # c_light=1.0: H clock columns are ±1, consistent with metres-clock state
        Hk_ml = build_H_mapless(r_r, r_a, r_vas, c_light=1.0)
        Hk_ma = build_H_map_aided(r_r, r_a, r_vas, M_list, c_light=1.0)

        H_acc_ml.append(Hk_ml)
        H_acc_ma.append(Hk_ma)

        O_ml = build_observability_matrix(H_acc_ml, F_ml)
        O_ma = build_observability_matrix(H_acc_ma, F_ma)

        res_ml = svd_rank_analysis(O_ml)
        res_ma = svd_rank_analysis(O_ma)

        cond_ml[k] = res_ml["condition_number"]
        rank_ml[k] = res_ml["rank"]
        cond_ma[k] = res_ma["condition_number"]
        rank_ma[k] = res_ma["rank"]

    return {"cond_ml": cond_ml, "rank_ml": rank_ml,
            "cond_ma": cond_ma, "rank_ma": rank_ma}


# ---------------------------------------------------------------------------
# Monte Carlo main engine
# ---------------------------------------------------------------------------

def run_monte_carlo(cfg: ExperimentConfig) -> dict:
    """Execute full Monte Carlo comparison: Mapless vs Map-assisted EKF.

    Args:
        cfg: ExperimentConfig instance.

    Returns:
        Full results dict; see module docstring for key descriptions.
    """
    N, T       = cfg.N, cfg.T
    n_runs     = cfg.n_runs
    n_steps    = cfg.n_steps

    rng_main = np.random.default_rng(seed=2026)
    mm       = MapManager.rectangular_room(cfg.room_width, cfg.room_height)

    # Anchor at 30% × 40% of room (off-centre, well inside)
    r_a_true = np.array([cfg.room_width * 0.30, cfg.room_height * 0.40])

    dim_ml = 10 + 2 * N
    dim_ma = 10
    R_k    = cfg.sigma_range**2 * np.eye(N + 1)

    dyn_params = dict(
        q_x=cfg.q_x,   q_y=cfg.q_y,
        q_va=cfg.q_va,                   # VA regularisation noise (mapless only)
        h0_r=cfg.h0_r, h_2_r=cfg.h_2_r,
        h0_a=cfg.h0_a, h_2_a=cfg.h_2_a,
    )

    res_ml: dict = {
        "errors":    np.zeros((n_steps, n_runs, dim_ml)),
        "cov_diags": np.zeros((n_steps, n_runs, dim_ml)),
        "nees":      np.zeros((n_steps, n_runs)),
    }
    res_ma: dict = {
        "errors":    np.zeros((n_steps, n_runs, dim_ma)),
        "cov_diags": np.zeros((n_steps, n_runs, dim_ma)),
        "nees":      np.zeros((n_steps, n_runs)),
    }
    obs_info: dict | None = None

    # Sample (run 0) for 2-D trajectory plot
    x_true_sample:  np.ndarray | None = None
    x_hat_ml_sample = np.zeros((n_steps, dim_ml))
    x_hat_ma_sample = np.zeros((n_steps, dim_ma))

    # ---- Monte Carlo loop ----
    for run_idx in range(n_runs):
        rng = np.random.default_rng(rng_main.integers(0, 2**31))

        x_true = _generate_true_trajectory(cfg, r_a_true, mm, rng)
        z_all  = _generate_measurements(cfg, x_true, rng)

        if run_idx == 0:
            obs_info      = _track_observability(cfg, x_true, mm)
            x_true_sample = x_true.copy()

        # ---- EKF initialisation ----
        sig_b_r  = 0.1  if cfg.know_time     else _SIGMA_B
        sig_bd_r = 0.01 if cfg.know_time     else _SIGMA_BD
        sig_p_r  = 0.1  if cfg.know_position else _SIGMA_POS

        x0_ml, P0_ml = build_initial_state_mapless(
            x_true[0], N,
            sigma_pos=sig_p_r,
            sigma_vel=_SIGMA_VEL,
            sigma_b=sig_b_r,
            sigma_bd=sig_bd_r,
            sigma_va=_SIGMA_VA,
            know_position=cfg.know_position,
            know_time=cfg.know_time,
            rng=rng,
        )
        x0_ma, P0_ma = build_initial_state_map_aided(
            x_true[0],
            sigma_pos=sig_p_r,
            sigma_vel=_SIGMA_VEL,
            sigma_b=sig_b_r,
            sigma_bd=sig_bd_r,
            know_position=cfg.know_position,
            know_time=cfg.know_time,
            rng=rng,
        )

        ekf_ml = MaplessEKF(x0_ml, P0_ml, N, dyn_params, c_light=_C)
        ekf_ma = MapAidedEKF(x0_ma, P0_ma, mm, dyn_params, c_light=_C)

        # ---- Time loop ----
        for k in range(n_steps):
            z_k = z_all[k + 1]     # measurement at t_{k+1}

            ekf_ml.step(z_k, T, R_k)
            ekf_ma.step(z_k, T, R_k)

            err_ml = x_true[k + 1]        - ekf_ml.x
            err_ma = x_true[k + 1, :dim_ma] - ekf_ma.x

            res_ml["errors"][k, run_idx]    = err_ml
            res_ml["cov_diags"][k, run_idx] = np.maximum(np.diag(ekf_ml.P), 0.0)
            res_ml["nees"][k, run_idx]      = compute_nees(err_ml, ekf_ml.P)

            res_ma["errors"][k, run_idx]    = err_ma
            res_ma["cov_diags"][k, run_idx] = np.maximum(np.diag(ekf_ma.P), 0.0)
            res_ma["nees"][k, run_idx]      = compute_nees(err_ma, ekf_ma.P)

            if run_idx == 0:
                x_hat_ml_sample[k] = ekf_ml.x.copy()
                x_hat_ma_sample[k] = ekf_ma.x.copy()

    # ---- Post-process ----
    time_axis = np.arange(1, n_steps + 1) * T

    # Chi-squared bounds: use Table I observable DoF, accounting for initial priors.
    #
    # The SVD rank of O(t0,tl) captures measurement-driven observability.
    # A tight initial prior on a state effectively makes it "observable" from the
    # filter's perspective — pinv(P) will NOT zero it out because P[i,i] stays
    # small, so that direction contributes ~1 to NEES.
    #
    # Map-aided model (Table I):
    #   No prior        → rank 8  (2 clock DoF unobservable: abs bias + drift)
    #   Position prior  → rank 8  (clock still unobservable)
    #   Time prior      → rank 10 (strong prior constrains absolute clock DoF)
    #   Pos + Time      → rank 10 (full rank with both priors)
    #
    # Mapless model (Table I), state dim = 10 + 2N:
    #   No prior        → 2N+5  (unobs: trans×2, rot×1, clk_bias×1, clk_drift×1)
    #   Position prior  → 2N+7  (resolves 2D translation)
    #   Time prior      → 2N+7  (resolves clock bias + drift)
    #   Pos + Time      → 2N+9  (only rotation remains unobservable)
    eff_dim_ma = dim_ma if cfg.know_time else 8

    base_ml = 2 * N + 5
    if cfg.know_position:
        base_ml += 2   # position prior resolves global 2D translation
    if cfg.know_time:
        base_ml += 2   # time prior resolves absolute clock bias + drift
    eff_dim_ml = min(base_ml, dim_ml)

    r1_ml, r2_ml = chi2_bounds(eff_dim_ml, n_runs)
    r1_ma, r2_ma = chi2_bounds(eff_dim_ma, n_runs)

    sigmas_ml = np.sqrt(np.mean(res_ml["cov_diags"], axis=1))
    sigmas_ma = np.sqrt(np.mean(res_ma["cov_diags"], axis=1))

    return {
        "mapless": {
            **res_ml,
            "avg_nees": average_nees(res_ml["nees"]),
            "r1": r1_ml, "r2": r2_ml,
            "eff_dim": eff_dim_ml,
            "sigmas": sigmas_ml,
            "rmse_pos": rmse(res_ml["errors"][:, :, 0:2].mean(axis=2)),
        },
        "map_aided": {
            **res_ma,
            "avg_nees": average_nees(res_ma["nees"]),
            "r1": r1_ma, "r2": r2_ma,
            "eff_dim": eff_dim_ma,
            "sigmas": sigmas_ma,
            "rmse_pos": rmse(res_ma["errors"][:, :, 0:2].mean(axis=2)),
        },
        "obs_cond_ml": obs_info["cond_ml"][1:],
        "obs_rank_ml": obs_info["rank_ml"][1:],
        "obs_cond_ma": obs_info["cond_ma"][1:],
        "obs_rank_ma": obs_info["rank_ma"][1:],
        "time": time_axis,
        "cfg":  cfg,
        "sample": {
            "x_true":   x_true_sample,
            "x_hat_ml": x_hat_ml_sample,
            "x_hat_ma": x_hat_ma_sample,
            "r_a_true": r_a_true,
            "mm":       mm,
        },
    }
