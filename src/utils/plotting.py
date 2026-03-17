"""Visualization utilities for the Radio SLAM Observability Benchmarking Suite.

Publication figures (matching IPIN 2026 paper style):

  figure1_error_trajectories — Fig. 1 style: per-state estimation error ± 2σ
                               4 rows × 2 cols (Mapless | Map-aided)
  figure4_nees               — Fig. 4 style: average NEES vs Chi-squared bounds
  figure_observability       — Rank evolution + condition number (log scale)
  figure_trajectory_2d       — 2D room: true path vs. EKF estimated paths
  figure_table1              — Graphical Table I from the paper

Legacy helpers (kept for backward compatibility):
  plot_error_with_sigma, plot_errors_comparison,
  plot_nees, plot_condition_number, plot_singular_values
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (consistent with paper figures)
# ---------------------------------------------------------------------------
_BLUE   = "#1f77b4"   # Mapless model
_ORANGE = "#ff7f0e"   # Map-aided model
_RED    = "#d62728"   # bounds / danger
_GREEN  = "#2ca02c"   # true trajectory
_GRAY   = "#7f7f7f"
_LGRAY  = "#cccccc"

_C_LIGHT = 3e8        # speed of light [m/s]


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

def apply_publication_style() -> None:
    """Set matplotlib rcParams to publication-quality defaults."""
    plt.rcParams.update({
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":    9,
        "legend.framealpha": 0.85,
        "figure.dpi":       120,
        "lines.linewidth":   1.5,
        "axes.grid":        True,
        "grid.alpha":        0.4,
        "grid.linewidth":    0.5,
        "axes.spines.top":  False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Figure 1 — Error trajectories ± 2σ  (paper Fig. 1 style)
# ---------------------------------------------------------------------------
def figure1_error_trajectories(results: dict, c_light: float = _C_LIGHT) -> plt.Figure:
    """
    修正版：包含所有坐标轴 Label，并集成方案 A 的相对时钟统计。
    """
    apply_publication_style()
    cfg  = results["cfg"]
    time = results["time"]
    N    = cfg.N

    ml = results["mapless"]
    ma = results["map_aided"]

    err_ml = ml["errors"]
    err_ma = ma["errors"]
    sig_ml = ml["sigmas"]
    sig_ma = ma["sigmas"]
    emp_ml = np.std(err_ml, axis=1)
    emp_ma = np.std(err_ma, axis=1)

    dim_ml = err_ml.shape[2]
    dim_ma = 10

    # --- 方案 A 内部函数：处理相对时钟 ---
    def _get_rel_clk_data(res_dict):
        err = res_dict["errors"]
        # 经验误差：先减再求标准差 (抵消共模发散)
        err_c = err[:, :, 4] - err[:, :, 8]
        emp_c = np.std(err_c, axis=1)
        # 理论边界：读取 runner.py 中记录的包含协方差项的 sigma
        # 如果 runner 还没改，这里会 fallback 到绝对相加(视觉上会很大)
        sig_c = res_dict.get("sigmas_rel", np.sqrt(res_dict["cov_diags"][:,:,4].mean(axis=1)))
        return err_c, sig_c, emp_c

    # 定义所有行的标签
    _CORE_LABELS = [
        (r"$\tilde{x}_r$ [m]", "x_r pos"),
        (r"$\tilde{y}_r$ [m]", "y_r pos"),
        (r"$\dot{\tilde{x}}_r$ [m/s]", "vx_r"),
        (r"$\dot{\tilde{y}}_r$ [m/s]", "vy_r"),
        (r"Rel. $c\delta\tilde{t}$ [m]", "relative clock"), # 第 4 行改为相对
        (r"$\dot{b}_r$ [m/s]", "bd_r drift"),
        (r"$\tilde{x}_a$ [m]", "x_a pos"),
        (r"$\tilde{y}_a$ [m]", "y_a pos"),
        (r"Excluded", "placeholder"),                   # 第 8 行原为 b_a，现排除
        (r"$\dot{b}_a$ [m/s]", "bd_a drift"),
    ]
    
    n_rows = dim_ml
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 1.8 * n_rows), sharex=True)

    def _draw_sub(ax, t, errors, filter_sig, emp_sig, color, label):
        mean_e = np.mean(errors, axis=1)
        ax.plot(t, mean_e, color=color, lw=1.4, label=f"{label} mean", zorder=3)
        ax.fill_between(t, -2 * filter_sig, 2 * filter_sig,
                        color=color, alpha=0.18, label=r"±2$\sigma$ filter", zorder=2)
        ax.plot(t,  2 * emp_sig, color=color, lw=0.8, ls="--", label=r"±2$\sigma$ emp.")
        ax.plot(t, -2 * emp_sig, color=color, lw=0.8, ls="--")
        ax.axhline(0, color=_GRAY, lw=0.6, ls=":")

    for row_idx in range(n_rows):
        ax_ml, ax_ma = axes[row_idx, 0], axes[row_idx, 1]
        
        # 获取标签
        if row_idx < 10:
            ylabel = _CORE_LABELS[row_idx][0]
        else:
            va_idx = (row_idx - 10) // 2 + 1
            coord = "x" if row_idx % 2 == 0 else "y"
            ylabel = rf"VA{va_idx} $\tilde{{{coord}}}$ [m]"

        # --- 特殊行处理 (方案 A) ---
        if row_idx == 4: # 第 5 行：展示相对时钟
            e_c_ml, s_c_ml, emp_c_ml = _get_rel_clk_data(ml)
            e_c_ma, s_c_ma, emp_c_ma = _get_rel_clk_data(ma)
            _draw_sub(ax_ml, time, e_c_ml, s_c_ml, emp_c_ml, _BLUE, "Mapless")
            _draw_sub(ax_ma, time, e_c_ma, s_c_ma, emp_c_ma, _ORANGE, "Map-aided")
        elif row_idx == 8: # 隐藏原有的绝对时钟 b_a 行
            ax_ml.set_visible(False)
            ax_ma.set_visible(False)
            continue
        else:
            # 普通行绘制
            _draw_sub(ax_ml, time, err_ml[:, :, row_idx], sig_ml[:, row_idx], emp_ml[:, row_idx], _BLUE, "Mapless")
            if row_idx < dim_ma:
                _draw_sub(ax_ma, time, err_ma[:, :, row_idx], sig_ma[:, row_idx], emp_ma[:, row_idx], _ORANGE, "Map-aided")

        # --- 显式设置 Y 轴标签 ---
        ax_ml.set_ylabel(ylabel, fontsize=10)
        ax_ma.set_ylabel(ylabel, fontsize=10)

        # 设置标题和图例 (仅第一行)
        if row_idx == 0:
            ax_ml.set_title("Mapless EKF", fontsize=11)
            ax_ma.set_title("Map-Aided EKF", fontsize=11)
            ax_ml.legend(fontsize=7, loc="upper right")
            ax_ma.legend(fontsize=7, loc="upper right")

    # --- 显式设置 X 轴标签 (仅最底部可见行) ---
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]", fontsize=10)

    fig.suptitle(f"Fig. 1 — Estimation Error ± 2σ | {cfg.case_id} (N={cfg.N})", fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Average NEES vs Chi-squared bounds  (paper Fig. 4 style)
# ---------------------------------------------------------------------------

def figure4_nees(results: dict) -> plt.Figure:
    """Publication Figure 4: average NEES with 99% Chi-squared acceptance region.

    A filter is *consistent* when the average NEES lies inside [r1, r2].
    Both models are shown in one figure with two panels for direct comparison.
    Individual MC run NEES traces are overlaid as thin gray lines to show spread.
    """
    apply_publication_style()
    cfg  = results["cfg"]
    time = results["time"]
    ml   = results["mapless"]
    ma   = results["map_aided"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, res, r1, r2, label, color in [
        (axes[0], ml, ml["r1"], ml["r2"], "Mapless EKF",    _BLUE),
        (axes[1], ma, ma["r1"], ma["r2"], "Map-Aided EKF",  _ORANGE),
    ]:
        nees_runs = res["nees"]   # (n_steps, n_runs)
        avg_nees  = res["avg_nees"]

        # Individual run traces (thin, transparent)
        for j in range(min(nees_runs.shape[1], 30)):   # cap at 30 for clarity
            ax.semilogy(time, nees_runs[:, j],
                        color=_LGRAY, lw=0.5, alpha=0.5, zorder=1)

        # Average NEES
        ax.semilogy(time, avg_nees, color=color, lw=2.2, label="Avg NEES", zorder=3)

        # Chi-squared bounds
        ax.axhline(r1, color=_RED, ls="--", lw=1.4,
                   label=f"$r_1$ = {r1:.2f}", zorder=4)
        ax.axhline(r2, color=_RED, ls=":",  lw=1.4,
                   label=f"$r_2$ = {r2:.2f}", zorder=4)
        ax.fill_between(time, r1, r2, color=_RED, alpha=0.07,
                        label="99% acceptance", zorder=2)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Average NEES  $\\bar{\\varepsilon}_k$")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=max(0.1, r1 * 0.5))

    case_label = _case_label(cfg)
    fig.suptitle(
        f"Fig. 4 — NEES Consistency Check  |  {case_label}  "
        f"({cfg.n_runs} MC runs, N={cfg.N})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure — Observability: rank evolution + condition number
# ---------------------------------------------------------------------------

def figure_observability(results: dict) -> plt.Figure:
    """Observability rank and condition number of O(t0, tk) vs. time.

    Left panel:  Rank of O vs. time (step function).
                 Dashed horizontal lines show theoretical maxima from Table I.
    Right panel: Condition number κ(O) on log scale.
                 A fast drop indicates rapidly improving estimability.
    """
    apply_publication_style()
    cfg     = results["cfg"]
    time    = results["time"]
    N       = cfg.N

    rank_ml = results["obs_rank_ml"]
    rank_ma = results["obs_rank_ma"]
    cond_ml = np.where(np.isfinite(results["obs_cond_ml"]),
                       results["obs_cond_ml"], np.nan)
    cond_ma = np.where(np.isfinite(results["obs_cond_ma"]),
                       results["obs_cond_ma"], np.nan)

    # Theoretical maximum ranks from Table I (measurement-only, no prior).
    # These lines mark the structural ceiling regardless of initial priors.
    th_max_ml = 2 * N + 5
    th_max_ma = 8

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ---- Left: rank ----
    ax = axes[0]
    ax.step(time, rank_ml, where="post", color=_BLUE,   lw=2.0, label="Mapless")
    ax.step(time, rank_ma, where="post", color=_ORANGE, lw=2.0, label="Map-aided")
    ax.axhline(th_max_ml, color=_BLUE,   ls="--", lw=1.0,
               label=f"Mapless max = {th_max_ml} (2N+5)")
    ax.axhline(th_max_ma, color=_ORANGE, ls="--", lw=1.0,
               label=f"Map-aided max = {th_max_ma}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Observable rank  $\\mathrm{rank}(\\mathcal{O})$")
    ax.set_title("Observability Rank vs. Time")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(th_max_ml, rank_ml.max() if rank_ml.size else th_max_ml) + 2)

    # ---- Right: condition number ----
    ax = axes[1]
    ax.semilogy(time, cond_ml, color=_BLUE,   lw=2.0, label="Mapless")
    ax.semilogy(time, cond_ma, color=_ORANGE, lw=2.0, label="Map-aided")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Condition number  $\\kappa(\\mathcal{O})$")
    ax.set_title("Estimability (Condition Number) vs. Time")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.grid(True, which="both", alpha=0.35)

    case_label = _case_label(cfg)
    fig.suptitle(
        f"Observability Analysis  |  {case_label}  (N={cfg.N})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure — 2D Position Trajectory
# ---------------------------------------------------------------------------

def figure_trajectory_2d(results: dict) -> plt.Figure:
    """2D top-down view: room geometry, true trajectory vs. EKF estimates.

    Uses the sample data (run 0) stored by the Monte Carlo runner.
    Shows the receiver path (true + estimated by both EKFs),
    the anchor position, and the four virtual anchor positions.
    """
    apply_publication_style()
    cfg    = results["cfg"]
    sample = results["sample"]

    x_true   = sample["x_true"]    # (n_steps+1, 10+2N)
    x_hat_ml = sample["x_hat_ml"]  # (n_steps, 10+2N)
    x_hat_ma = sample["x_hat_ma"]  # (n_steps, 10)
    r_a      = sample["r_a_true"]  # (2,)
    mm       = sample["mm"]

    N = cfg.N
    r_vas = mm.virtual_anchor_positions(r_a)[:N]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Room outline
    room = mpatches.FancyBboxPatch(
        (0, 0), cfg.room_width, cfg.room_height,
        boxstyle="square,pad=0", linewidth=2,
        edgecolor="black", facecolor="#f7f7f7", zorder=0,
    )
    ax.add_patch(room)

    # True receiver path
    ax.plot(x_true[:, 0], x_true[:, 1],
            color=_GREEN, lw=1.8, ls="-", label="True path", zorder=3)
    ax.plot(x_true[0, 0], x_true[0, 1], "o",
            color=_GREEN, ms=8, zorder=5, label="Start")
    ax.plot(x_true[-1, 0], x_true[-1, 1], "s",
            color=_GREEN, ms=8, zorder=5, label="End")

    # Mapless estimated path
    ax.plot(x_hat_ml[:, 0], x_hat_ml[:, 1],
            color=_BLUE, lw=1.4, ls="--", label="Mapless est.", zorder=3, alpha=0.85)

    # Map-aided estimated path
    ax.plot(x_hat_ma[:, 0], x_hat_ma[:, 1],
            color=_ORANGE, lw=1.4, ls="-.", label="Map-aided est.", zorder=3, alpha=0.85)

    # Physical anchor
    ax.plot(*r_a, marker="*", color=_RED, ms=14, zorder=6,
            label=f"Anchor $r_a$=({r_a[0]:.1f},{r_a[1]:.1f})")

    # Virtual anchors — all N labelled; they are outside the room by construction
    va_colors = ["#9467bd", "#8c564b", "#e377c2", "#bcbd22"]  # distinct per wall
    wall_names = ["Left wall", "Right wall", "Bottom wall", "Top wall"]
    for i, rv in enumerate(r_vas):
        col = va_colors[i % len(va_colors)]
        ax.plot(*rv, marker="^", color=col, ms=9, zorder=5,
                label=f"VA$_{{{i+1}}}$ ({wall_names[i] if i < len(wall_names) else ''})")
        # Dashed line from anchor to its virtual image
        ax.plot([r_a[0], rv[0]], [r_a[1], rv[1]],
                color=col, lw=0.8, ls=":", alpha=0.6, zorder=2)

    # Expand axes to include all VAs (they can be outside the room)
    all_x = [rv[0] for rv in r_vas] + [r_a[0], 0, cfg.room_width]
    all_y = [rv[1] for rv in r_vas] + [r_a[1], 0, cfg.room_height]
    pad = 2.0
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    case_label = _case_label(cfg)
    ax.set_title(f"2D Position Trajectory (run 0)  |  {case_label}  (N={cfg.N})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure — Table I (observability comparison)
# ---------------------------------------------------------------------------

def figure_table1(results: dict) -> plt.Figure:
    """Graphical rendering of Table I: observability metrics vs. initial priors.

    Displays both theoretical values (from the paper) and simulated final ranks.
    """
    apply_publication_style()
    cfg = results["cfg"]
    N   = cfg.N

    # Theoretical ranks from Table I (N >= 3)
    th_rows = [
        ("None",          f"≥ 3", f"2N+5 = {2*N+5}", "5", "≥ 2", "8", "2"),
        ("Position",      f"≥ 3", f"2N+7 = {2*N+7}", "3", "≥ 2", "8", "2"),
        ("Time",          f"≥ 3", f"2N+7 = {2*N+7}", "3", "≥ 2", "10","0"),
        ("Position+Time", f"≥ 3", f"2N+9 = {2*N+9}", "1", "≥ 2", "10","0"),
    ]
    col_labels = [
        "Prior", "ML steps", "ML rank", "ML unobs DoF",
        "MA steps", "MA rank", "MA unobs DoF",
    ]

    # Annotate current simulation row
    case_names = {1: "None", 2: "Position", 3: "Time", 4: "Position+Time"}
    sim_rank_ml = int(results["obs_rank_ml"][-1])
    sim_rank_ma = int(results["obs_rank_ma"][-1])

    fig, ax = plt.subplots(figsize=(13, 3.0))
    ax.axis("off")

    table_data = [list(r) for r in th_rows]
    row_colors = [["white"] * len(col_labels)] * 4

    # Highlight the current case row
    highlight_row = {1: 0, 2: 1, 3: 2, 4: 3}[cfg.case_id]
    row_colors[highlight_row] = ["#fffbe6"] * len(col_labels)

    # Add simulated values to the current row annotation
    table_data[highlight_row][2] += f"\n(sim={sim_rank_ml})"
    table_data[highlight_row][5] += f"\n(sim={sim_rank_ma})"

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.8)

    # Bold column headers
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#dce6f1")
        tbl[0, j].set_text_props(fontweight="bold")

    ax.set_title(
        f"Table I — System Observability Comparison (N={N})  "
        f"|  highlighted row = Case {cfg.case_id} (current run)",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _case_label(cfg) -> str:
    labels = {1: "No Prior", 2: "Position Prior",
              3: "Time Prior", 4: "Position + Time Prior"}
    return f"Case {cfg.case_id}: {labels.get(cfg.case_id, '')}"


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def plot_error_with_sigma(
    time: np.ndarray,
    errors: np.ndarray,
    sigmas: np.ndarray,
    title: str = "",
    state_label: str = "State",
    model_label: str = "EKF",
    color: str = _BLUE,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Single-panel error ± 2σ plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
    else:
        fig = ax.get_figure()
    mean_err = np.mean(errors, axis=1)
    bound    = 2.0 * sigmas
    ax.plot(time, mean_err, color=color, lw=1.5, label=f"{model_label} error")
    ax.fill_between(time, -bound, bound, color=color, alpha=0.20, label="±2σ bound")
    ax.axhline(0, color=_GRAY, lw=0.8, ls="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(state_label)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_errors_comparison(
    time, errors_ml, sigmas_ml, errors_ma, sigmas_ma,
    state_indices, state_labels, suptitle="",
) -> plt.Figure:
    """Side-by-side error plots for selected states."""
    n = len(state_indices)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.0 * n), sharex=True)
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, (idx, label) in enumerate(zip(state_indices, state_labels)):
        plot_error_with_sigma(time, errors_ml[:, :, idx], sigmas_ml[:, idx],
                              state_label=label, model_label="Mapless",
                              color=_BLUE, ax=axes[row, 0])
        axes[row, 0].set_title(f"Mapless — {label}")
        plot_error_with_sigma(time, errors_ma[:, :, idx], sigmas_ma[:, idx],
                              state_label=label, model_label="Map-aided",
                              color=_ORANGE, ax=axes[row, 1])
        axes[row, 1].set_title(f"Map-aided — {label}")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_nees(time, nees_ml, nees_ma, r1_ml, r2_ml, r1_ma, r2_ma,
              title="Average NEES Consistency Check") -> plt.Figure:
    """Dual-panel NEES vs Chi-squared bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, nees, r1, r2, label, color in [
        (axes[0], nees_ml, r1_ml, r2_ml, "Mapless",     _BLUE),
        (axes[1], nees_ma, r1_ma, r2_ma, "Map-assisted", _ORANGE),
    ]:
        ax.plot(time, nees, color=color, lw=1.5, label="Avg NEES")
        ax.axhline(r1, color=_RED, ls="--", lw=1.2, label=f"r1={r1:.2f}")
        ax.axhline(r2, color=_RED, ls="--", lw=1.2, label=f"r2={r2:.2f}")
        ax.fill_between(time, r1, r2, color=_RED, alpha=0.08, label="99% region")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Average NEES")
        ax.set_title(f"{label} — NEES")
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.set_ylim(bottom=0)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_condition_number(time, cond_ml, cond_ma,
                          title="Observability Condition Number") -> plt.Figure:
    """Condition number evolution on log scale."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(time, cond_ml, color=_BLUE,   lw=1.8, label="Mapless")
    ax.semilogy(time, cond_ma, color=_ORANGE, lw=1.8, label="Map-assisted")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Condition number κ(O)")
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", lw=0.4, alpha=0.5)
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    fig.tight_layout()
    return fig


def plot_singular_values(singular_values_ml, singular_values_ma,
                         step, title="") -> plt.Figure:
    """Singular value bar chart at a given l-step."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, svs, label, color in [
        (axes[0], singular_values_ml, "Mapless",     _BLUE),
        (axes[1], singular_values_ma, "Map-assisted", _ORANGE),
    ]:
        idx = np.arange(1, len(svs) + 1)
        ax.bar(idx, svs, color=color, alpha=0.75, edgecolor="black", lw=0.5)
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("σᵢ")
        ax.set_title(f"{label} — SVs at l={step}")
        ax.set_yscale("log")
        ax.grid(True, which="both", lw=0.4, alpha=0.5)
    fig.suptitle(title or f"Singular Value Distribution at l={step}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
