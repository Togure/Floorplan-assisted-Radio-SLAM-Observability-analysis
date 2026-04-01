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
_SIGMA_POS = 1    # m   — receiver/anchor position cold-start.
                    #       Reduced from 2 m: large σ causes the EKF to get
                    #       trapped in local minima when N=4 (5 measurements/step
                    #       collapse P faster than coupled receiver-anchor errors
                    #       can converge). 0.8 m is still a meaningful "no prior"
                    #       scenario (e.g., coarse floor-map knowledge ≈ 1 m).
_SIGMA_VEL = 0.2    # m/s — receiver velocity cold-start
_SIGMA_B   = 3   # m   — clock bias cold-start (range-equivalent, ≈ 40 ns)
_SIGMA_BD  = 0.05    # m/s — clock drift cold-start
_SIGMA_VA  = 2    # m   — VA position warm-start (mapless only)



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

#     #Figure-8: x→1 full oscillation, y→2 full oscillations per simulation
#     T_total = n * T
#     t_arr   = np.arange(n + 1) * T
#     omega   = 2.0 * np.pi / T_total

#     x_nom = cx + Ax * np.sin(omega * t_arr)
#     y_nom = cy + Ay * np.sin(2.0 * omega * t_arr)

#     #Small perturbations (σ = 0.05 m) to break exact periodicity
#     x_arr = np.clip(x_nom + rng.normal(0, 0.01, n + 1), margin, W - margin)
#     y_arr = np.clip(y_nom + rng.normal(0, 0.01, n + 1), margin, H - margin)

#     #Velocities via finite difference (forward difference at k=0)
#     vx = np.empty(n + 1)
#     vy = np.empty(n + 1)
#     vx[1:] = (x_arr[1:] - x_arr[:-1]) / T
#     vy[1:] = (y_arr[1:] - y_arr[:-1]) / T
#     vx[0]  = vx[1]
#     vy[0]  = vy[1]


# # ---- 触壁反弹平滑轨迹 (Bouncing Unicycle Trajectory) ----
#     # 特点：在房间内部保持绝对直线，只有预测到快撞墙时才平滑转向，不随意转弯。
#    # ---- 触壁反弹平滑轨迹 (Bouncing Unicycle Trajectory) ----
#     T_total = n * T
#     t_arr   = np.arange(n + 1) * T

#     x_arr = np.zeros(n + 1)
#     y_arr = np.zeros(n + 1)
#     vx = np.zeros(n + 1)
#     vy = np.zeros(n + 1)

#     # 初始状态：从房间中心出发
#     W, H = cfg.room_width, cfg.room_height
#     margin = 1.5           # [m] 距离墙壁的最小安全距离
    
#     x, y = W / 2.0, H / 2.0
    
#     # 【核心修改 1】强制初始朝向向右
#     # 在极坐标中，0 弧度代表正 X 轴方向（向右）
#     theta = 0.0  
#     target_theta = theta

#     # 【核心修改 2】轨迹专属的固定随机数生成器 (解耦)
#     # 无论这是第几次 MC run，轨迹转弯的随机数序列都由这个固定的 seed(8888) 决定
#     traj_rng = np.random.default_rng(seed=8888)

#     speed = 1.0            # [m/s] 恒定线速度
#     omega_max = 0.8        # [rad/s] 最大角速度
#     lookahead = 3.0        # [m] 预警距离

#     for i in range(n + 1):
#         # 1. 记录当前时刻的真实状态
#         x_arr[i] = x
#         y_arr[i] = y
#         vx[i] = speed * np.cos(theta)
#         vy[i] = speed * np.sin(theta)

#         # 2. 预测前方位置 (探照灯逻辑)
#         pred_x = x + lookahead * np.cos(theta)
#         pred_y = y + lookahead * np.sin(theta)

#         # 3. 撞墙判定与目标航向分配 (将原来的 rng 替换为 traj_rng)
#         if pred_x < margin and np.cos(theta) < 0:
#             target_theta = traj_rng.uniform(-np.pi/3, np.pi/3)      # 撞左墙，向右看
#         elif pred_x > W - margin and np.cos(theta) > 0:
#             target_theta = traj_rng.uniform(2*np.pi/3, 4*np.pi/3)   # 撞右墙，向左看

#         if pred_y < margin and np.sin(theta) < 0:
#             target_theta = traj_rng.uniform(np.pi/6, 5*np.pi/6)     # 撞下墙，向上看
#         elif pred_y > H - margin and np.sin(theta) > 0:
#             target_theta = traj_rng.uniform(7*np.pi/6, 11*np.pi/6)  # 撞上墙，向下看

#         # 4. 平滑转弯执行 (运动学积分)
#         angle_diff = (target_theta - theta + np.pi) % (2 * np.pi) - np.pi
#         step_turn = np.clip(angle_diff, -omega_max * T, omega_max * T)
#         theta += step_turn

#         # 5. 更新下一时刻位置
#         x += speed * np.cos(theta) * T
#         y += speed * np.sin(theta) * T
        
#         x = np.clip(x, margin, W - margin)
#         y = np.clip(y, margin, H - margin)

#     # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    #complecated movement
    T_total = n * T
    t_arr   = np.arange(n + 1) * T

    x_arr = np.zeros(n + 1)
    y_arr = np.zeros(n + 1)
    vx = np.zeros(n + 1)
    vy = np.zeros(n + 1)

    #---- 平滑随机漫步模型 (Kinematic Unicycle) ----
    #初始状态：从房间中心附近出发，随机选择一个朝向
    x, y = W / 2.0, H / 2.0
    theta = rng.uniform(0, 2 * np.pi)
    target_theta = theta
    traj_rng = np.random.default_rng(seed=12) #231
    speed = 1.5           # [m/s] 常数线速度 (直线运动时的速度)
    omega_max = 0.5       # [rad/s] 最大角速度 (决定了转弯的平滑度，转弯半径 R = speed/omega_max = 3米)
    lookahead = 3.5       # [m] 墙壁防碰撞预警距离
    time_to_new_target = 0.0

    for i in range(n + 1):
        # 1. 记录当前时刻的真实状态
        x_arr[i] = x
        y_arr[i] = y
        vx[i] = speed * np.cos(theta)
        vy[i] = speed * np.sin(theta)

        #2. 预测前方是否会撞墙 (软边界排斥)
        pred_x = x + lookahead * np.cos(theta)
        pred_y = y + lookahead * np.sin(theta)

        hit_wall = False
        #如果预测会撞墙，立刻将目标航向角修改为看向房间内部，触发平滑转弯
        if pred_x < margin:
            target_theta = traj_rng.uniform(-np.pi/4, np.pi/4)      # 向右看
            hit_wall = True
        elif pred_x > W - margin:
            target_theta = traj_rng.uniform(3*np.pi/4, 5*np.pi/4)   # 向左看
            hit_wall = True

        if pred_y < margin:
            target_theta = traj_rng.uniform(np.pi/4, 3*np.pi/4)     # 向上看
            hit_wall = True
        elif pred_y > H - margin:
            target_theta = traj_rng.uniform(-3*np.pi/4, -np.pi/4)   # 向下看
            hit_wall = True

        #3. 如果绝对安全，则执行随机漫步逻辑
        if not hit_wall:
            time_to_new_target -= T
            if time_to_new_target <= 0:
                #随机转个弯 (-90度 到 90度之间)
                target_theta = theta + traj_rng.uniform(-np.pi/2, np.pi/2)
                #保持新航向一段时间，走出一段完美的直线 (持续 3~6 秒)
                time_to_new_target = traj_rng.uniform(3.0, 6.0)

        #4. 平滑转弯执行 (角速度截断)
        #计算当前朝向与目标朝向的最小夹角
        angle_diff = (target_theta - theta + np.pi) % (2 * np.pi) - np.pi
        #限制每步的最大转角，确保加速度物理有界
        step_turn = np.clip(angle_diff, -omega_max * T, omega_max * T)
        theta += step_turn

        #5. 运动学积分，更新下一时刻位置 (Euler Integration)
        x += speed * np.cos(theta) * T
        y += speed * np.sin(theta) * T
        
        #最后的保险机制，防止数值越界
        x = np.clip(x, margin, W - margin)
        y = np.clip(y, margin, H - margin)

    # T_total = n * T
    # t_arr   = np.arange(n + 1) * T

    # x_arr = np.zeros(n + 1)
    # y_arr = np.zeros(n + 1)
    # vx = np.zeros(n + 1)
    # vy = np.zeros(n + 1)

    # # ---- 跑道形轨迹 (Stadium Shape Trajectory) ----
    # # 完美的一阶可导轨迹：由两条水平直线和两个完美半圆组成
    # speed = 1.0  # [m/s] 恒定标量线速度
    
    # # 几何参数推导
    # cy = H / 2.0                            # 跑道中心线 y 坐标
    # R = (H - 2 * margin) / 2.0              # 半圆转弯半径 (贴着上下边界)
    # cx1 = margin + R                        # 左半圆的圆心 x 坐标
    # cx2 = W - margin - R                    # 右半圆的圆心 x 坐标
    # L = cx2 - cx1                           # 直线段的长度

    # perimeter = 2 * L + 2 * np.pi * R       # 跑道总周长

    # for i, t in enumerate(t_arr):
    #     d = (speed * t) % perimeter         # 当前在跑道上的一维投影距离

    #     if d < L:
    #         # 1. 底部直线 (匀速向右)
    #         x_arr[i] = cx1 + d
    #         y_arr[i] = cy - R
    #         vx[i] = speed
    #         vy[i] = 0.0
    #     elif d < L + np.pi * R:
    #         # 2. 右侧半圆 (向上转弯)
    #         delta_d = d - L
    #         phi = -np.pi / 2.0 + (delta_d / R)  # 当前圆心角
    #         x_arr[i] = cx2 + R * np.cos(phi)
    #         y_arr[i] = cy + R * np.sin(phi)
    #         # 速度是位置的精确解析求导: d(cos)/dt = -sin * d(phi)/dt, 其中 d(phi)/dt = speed/R
    #         vx[i] = -speed * np.sin(phi)
    #         vy[i] =  speed * np.cos(phi)
    #     elif d < 2 * L + np.pi * R:
    #         # 3. 顶部直线 (匀速向左)
    #         delta_d = d - (L + np.pi * R)
    #         x_arr[i] = cx2 - delta_d
    #         y_arr[i] = cy + R
    #         vx[i] = -speed
    #         vy[i] = 0.0
    #     else:
    #         # 4. 左侧半圆 (向下转弯)
    #         delta_d = d - (2 * L + np.pi * R)
    #         phi = np.pi / 2.0 + (delta_d / R)
    #         x_arr[i] = cx1 + R * np.cos(phi)
    #         y_arr[i] = cy + R * np.sin(phi)
    #         vx[i] = -speed * np.sin(phi)
    #         vy[i] =  speed * np.cos(phi)
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
        "errors":    np.zeros((n_steps + 1, n_runs, dim_ml)),
        "cov_diags": np.zeros((n_steps + 1, n_runs, dim_ml)),
        "nees":      np.zeros((n_steps + 1, n_runs)),
        "rel_clk_var": np.zeros((n_steps + 1, n_runs)),
    }
    res_ma: dict = {
        "errors":    np.zeros((n_steps + 1, n_runs, dim_ma)),
        "cov_diags": np.zeros((n_steps + 1, n_runs, dim_ma)),
        "nees":      np.zeros((n_steps + 1, n_runs)),
        "rel_clk_var": np.zeros((n_steps + 1, n_runs)),
    }
    obs_info: dict | None = None

    # Sample (run 0) for 2-D trajectory plot
    x_true_sample:  np.ndarray | None = None
    x_hat_ml_sample = np.zeros((n_steps, dim_ml))
    x_hat_ma_sample = np.zeros((n_steps, dim_ma))
    P_xy_ml_sample = np.zeros((n_steps, 2, 2))
    P_xy_ma_sample = np.zeros((n_steps, 2, 2))

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
            x_true[0].copy(), N,
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
            x_true[0].copy(),
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

        # ---- 记录 t=0 时刻的纯先验状态 (盲启动大漏斗的开口) ----
        err_ml_0 = x_true[0] - ekf_ml.x
        res_ml["errors"][0, run_idx] = err_ml_0
        res_ml["cov_diags"][0, run_idx] = np.maximum(np.diag(ekf_ml.P), 0.0)
        res_ml["nees"][0, run_idx] = compute_nees(err_ml_0, ekf_ml.P)
        res_ml["rel_clk_var"][0, run_idx] = ekf_ml.P[4, 4] + ekf_ml.P[8, 8] - 2 * ekf_ml.P[4, 8]

        err_ma_0 = x_true[0, :dim_ma] - ekf_ma.x
        res_ma["errors"][0, run_idx] = err_ma_0
        res_ma["cov_diags"][0, run_idx] = np.maximum(np.diag(ekf_ma.P), 0.0)
        res_ma["nees"][0, run_idx] = compute_nees(err_ma_0, ekf_ma.P)
        res_ma["rel_clk_var"][0, run_idx] = ekf_ma.P[4, 4] + ekf_ma.P[8, 8] - 2 * ekf_ma.P[4, 8]

        if run_idx == 0:
            x_hat_ml_sample[0] = ekf_ml.x.copy()
            x_hat_ma_sample[0] = ekf_ma.x.copy()


        # ---- Time loop ----
        for k in range(n_steps):
            z_k = z_all[k + 1]     # measurement at t_{k+1}

            ekf_ml.step(z_k, T, R_k)
            ekf_ma.step(z_k, T, R_k)

            err_ml = x_true[k + 1]        - ekf_ml.x
            err_ma = x_true[k + 1, :dim_ma] - ekf_ma.x

            res_ml["errors"][k + 1, run_idx]    = err_ml
            res_ml["cov_diags"][k + 1, run_idx] = np.maximum(np.diag(ekf_ml.P), 0.0)
            res_ml["nees"][k + 1, run_idx]      = compute_nees(err_ml, ekf_ml.P)
            res_ml["rel_clk_var"][k + 1, run_idx] = ekf_ml.P[4, 4] + ekf_ml.P[8, 8] - 2 * ekf_ml.P[4, 8]

            
            res_ma["errors"][k + 1, run_idx]    = err_ma
            res_ma["cov_diags"][k + 1, run_idx] = np.maximum(np.diag(ekf_ma.P), 0.0)
            res_ma["nees"][k + 1, run_idx]      = compute_nees(err_ma, ekf_ma.P)
            res_ma["rel_clk_var"][k + 1, run_idx] = ekf_ma.P[4, 4] + ekf_ma.P[8, 8] - 2 * ekf_ma.P[4, 8]

            if run_idx == 0:
                x_hat_ml_sample[k] = ekf_ml.x.copy()
                x_hat_ma_sample[k] = ekf_ma.x.copy()
                P_xy_ml_sample[k] = ekf_ml.P[0:2, 0:2].copy()
                P_xy_ma_sample[k] = ekf_ma.P[0:2, 0:2].copy()

    # ---- Post-process ----
    time_axis = np.arange(0, n_steps + 1) * T

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
            "sigmas_rel": np.sqrt(np.mean(res_ml["rel_clk_var"], axis=1)),
            "rmse_pos": rmse(res_ml["errors"][:, :, 0:2].mean(axis=2)),
        },
        "map_aided": {
            **res_ma,
            "sigmas_rel": np.sqrt(np.mean(res_ma["rel_clk_var"], axis=1)),
            "avg_nees": average_nees(res_ma["nees"]),
            "r1": r1_ma, "r2": r2_ma,
            "eff_dim": eff_dim_ma,
            "sigmas": sigmas_ma,
            "rmse_pos": rmse(res_ma["errors"][:, :, 0:2].mean(axis=2)),
        },
        "obs_cond_ml": obs_info["cond_ml"],
        "obs_rank_ml": obs_info["rank_ml"],
        "obs_cond_ma": obs_info["cond_ma"],
        "obs_rank_ma": obs_info["rank_ma"],
        "time": time_axis,
        "cfg":  cfg,
        "sample": {
            "x_true":   x_true_sample,
            "x_hat_ml": x_hat_ml_sample,
            "x_hat_ma": x_hat_ma_sample,
            "P_xy_ml":  P_xy_ml_sample,
            "P_xy_ma":  P_xy_ma_sample,
            "r_a_true": r_a_true,
            "mm":       mm,
        },
    }
