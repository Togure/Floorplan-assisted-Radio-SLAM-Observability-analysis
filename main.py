"""CLI entry point for the Radio SLAM Observability Benchmarking Suite.

Usage examples:
    # Run Case 1 (no prior) with defaults (50 runs, 100 steps)
    python main.py --case 1

    # Custom MC parameters
    python main.py --case 1 --runs 50 --steps 100 --N 3 --sigma 1.0

    # Run all four prior-cases sequentially
    python main.py --all-cases --runs 30 --steps 80

    # Interactive display + save to custom folder
    python main.py --case 4 --output results/ --show

    # Quick smoke-test (fast)
    python main.py --case 1 --runs 5 --steps 20 --output output_test/
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import textwrap
import time as _time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.experiments.config import ALL_CASES, ExperimentConfig
from src.experiments.runner import run_monte_carlo
from src.utils.plotting import (
    figure1_error_trajectories,
    figure4_nees,
    figure_observability,
    figure_table1,
    figure_trajectory_2d,
)

# Use non-interactive backend unless --show is passed
matplotlib.use("Agg")

_PRIOR_LABELS = {
    1: "No Prior",
    2: "Position Prior",
    3: "Time Prior",
    4: "Position + Time Prior",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Radio SLAM Observability Benchmarking Suite
            ============================================
            Compares Mapless EKF (10+2N states) vs Map-Aided EKF (10 states)
            across four initial-prior scenarios (Table I, IPIN 2026).
        """),
    )

    # ---- Case selection (mutually exclusive) ----
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--case", type=int, choices=[1, 2, 3, 4], metavar="CASE",
        help="Case ID (1=no prior | 2=position | 3=time | 4=pos+time).",
    )
    group.add_argument(
        "--all-cases", action="store_true",
        help="Run all four cases sequentially.",
    )

    # ---- Monte Carlo parameters ----
    mc = parser.add_argument_group("Monte Carlo parameters")
    mc.add_argument("--runs",  type=int,   default=50,  metavar="N",
                    help="Number of MC runs (default: 50).")
    mc.add_argument("--steps", type=int,   default=100, metavar="K",
                    help="Time steps per run (default: 100).")
    mc.add_argument("--T",     type=float, default=1.0, metavar="SEC",
                    help="Sampling period in seconds (default: 1.0).")

    # ---- Geometry / noise ----
    geo = parser.add_argument_group("Geometry and noise")
    geo.add_argument("--N",     type=int,   default=4,   metavar="N",
                     help="Number of virtual anchors / reflecting walls (default: 4, one per wall).")
    geo.add_argument("--sigma", type=float, default=1.0, metavar="M",
                     help="Range noise std-dev in metres (default: 1.0).")

    # ---- Output ----
    out = parser.add_argument_group("Output")
    out.add_argument("--output-dir", type=str, default="output",
                     metavar="DIR",
                     help="Root directory for saved figures (default: output/).")
    out.add_argument("--no-save", action="store_true",
                     help="Skip saving figures to disk.")
    out.add_argument("--show", action="store_true",
                     help="Display figures interactively after saving.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace, case_id: int) -> ExperimentConfig:
    base = ALL_CASES[case_id]
    return dataclasses.replace(
        base,
        n_runs  = args.runs,
        n_steps = args.steps,
        T       = args.T,
        N       = args.N,
        sigma_range = args.sigma,
    )


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def _make_output_dir(root: str, cfg: ExperimentConfig) -> Path:
    slug = _PRIOR_LABELS[cfg.case_id].lower().replace(" ", "_").replace("+", "and")
    out  = Path(root) / f"case_{cfg.case_id}_{slug}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def _print_banner(cfg: ExperimentConfig) -> None:
    w = 66
    label = _PRIOR_LABELS[cfg.case_id]
    print("╔" + "═" * w + "╗")
    print(f"║{'RADIO SLAM OBSERVABILITY BENCHMARKING SUITE':^{w}}║")
    print(f"║{f'Case {cfg.case_id} — {label}':^{w}}║")
    print(f"║{f'{cfg.n_runs} MC runs  ·  {cfg.n_steps} steps  ·  N={cfg.N}  ·  σ={cfg.sigma_range:.2f} m':^{w}}║")
    print("╚" + "═" * w + "╝")


def _print_summary(cfg: ExperimentConfig, results: dict) -> None:
    ml  = results["mapless"]
    ma  = results["map_aided"]
    N   = cfg.N
    w   = 66

    # State dimensions — defined first so they can be used as fallbacks below
    dim_ml = 10 + 2 * N
    dim_ma = 10

    final_rank_ml  = int(results["obs_rank_ml"][-1])
    final_rank_ma  = int(results["obs_rank_ma"][-1])
    final_cond_ml  = results["obs_cond_ml"][-1]
    final_cond_ma  = results["obs_cond_ma"][-1]
    final_nees_ml  = float(ml["avg_nees"][-1])
    final_nees_ma  = float(ma["avg_nees"][-1])
    final_rmse_ml  = float(ml["rmse_pos"][-1])
    final_rmse_ma  = float(ma["rmse_pos"][-1])
    consistent_ml  = ml["r1"] <= final_nees_ml <= ml["r2"]
    consistent_ma  = ma["r1"] <= final_nees_ma <= ma["r2"]
    eff_dim_ml     = ml.get("eff_dim", dim_ml)
    eff_dim_ma     = ma.get("eff_dim", dim_ma)

    sep  = "╠" + "═" * w + "╣"
    line = "║"

    print(sep)
    print(f"{line}{'':^{w}}{line}")
    print(f"{line}{'OBSERVABILITY ANALYSIS':^{w}}{line}")
    print(f"{line}{'':^{w}}{line}")

    # Dimension row
    print(f"{line}  {'State dimension':30s}  Mapless = {dim_ml:4d}  │  Map-aided = {dim_ma:4d}  {line}")
    print(f"{line}  {'Theoretical max rank':30s}  {2*N+5:4d}           │  {8:4d}           {line}")
    print(f"{line}  {'Final simulated rank':30s}  {final_rank_ml:4d}           │  {final_rank_ma:4d}           {line}")
    cond_str_ml = f"{final_cond_ml:.2e}" if np.isfinite(final_cond_ml) else "  Inf"
    cond_str_ma = f"{final_cond_ma:.2e}" if np.isfinite(final_cond_ma) else "  Inf"
    print(f"{line}  {'Condition number κ(O)':30s}  {cond_str_ml:>8s}       │  {cond_str_ma:>8s}       {line}")

    print(sep)
    print(f"{line}{'':^{w}}{line}")
    print(f"{line}{'FILTER CONSISTENCY (final epoch)':^{w}}{line}")
    print(f"{line}{'':^{w}}{line}")
    print(f"{line}  {'Avg NEES':30s}  {final_nees_ml:8.2f}       │  {final_nees_ma:8.2f}       {line}")
    bounds_ml = f"[{ml['r1']:.2f}, {ml['r2']:.2f}]"
    bounds_ma = f"[{ma['r1']:.2f}, {ma['r2']:.2f}]"
    print(f"{line}  {'Effective obs. DoF (pinv)':30s}  {eff_dim_ml:^14d}  │  {eff_dim_ma:^14d}  {line}")
    print(f"{line}  {'99% bounds [r1, r2]':30s}  {bounds_ml:>14s}  │  {bounds_ma:>14s}  {line}")
    yesno = lambda b: "✓ CONSISTENT" if b else "✗ NOT consistent"
    print(f"{line}  {'Filter consistency':30s}  {yesno(consistent_ml):^14s}  │  {yesno(consistent_ma):^14s}  {line}")

    print(sep)
    print(f"{line}{'':^{w}}{line}")
    print(f"{line}{'POSITION ACCURACY (final epoch, mean over runs)':^{w}}{line}")
    print(f"{line}{'':^{w}}{line}")
    print(f"{line}  {'Receiver RMSE (x,y)':30s}  {final_rmse_ml:8.3f} m      │  {final_rmse_ma:8.3f} m      {line}")
    print(f"{line}{'':^{w}}{line}")
    print("╚" + "═" * w + "╝")
    print()


def _save_summary_txt(cfg: ExperimentConfig, results: dict, out_dir: Path) -> None:
    """Write the console summary to a text file."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_banner(cfg)
        _print_summary(cfg, results)
    txt = buf.getvalue()
    path = out_dir / "summary.txt"
    path.write_text(txt, encoding="utf-8")
    print(f"  → summary.txt")


def _save_npz(results: dict, out_dir: Path) -> None:
    """Save raw numpy arrays for downstream analysis."""
    ml = results["mapless"]
    ma = results["map_aided"]
    np.savez_compressed(
        out_dir / "results.npz",
        time          = results["time"],
        err_ml        = ml["errors"],
        err_ma        = ma["errors"],
        nees_ml       = ml["nees"],
        nees_ma       = ma["nees"],
        avg_nees_ml   = ml["avg_nees"],
        avg_nees_ma   = ma["avg_nees"],
        obs_rank_ml   = results["obs_rank_ml"],
        obs_rank_ma   = results["obs_rank_ma"],
        obs_cond_ml   = results["obs_cond_ml"],
        obs_cond_ma   = results["obs_cond_ma"],
    )
    print(f"  → results.npz")


# ---------------------------------------------------------------------------
# Figure generation and saving
# ---------------------------------------------------------------------------

def _make_and_save(
    name: str,
    fig_fn,
    out_dir: Path | None,
    show: bool,
) -> None:
    """Generate a figure, optionally save it, optionally show it."""
    try:
        fig = fig_fn()
    except Exception as exc:
        print(f"  [WARN] {name} skipped: {exc}")
        return
    if out_dir is not None:
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  → {name}.png")
    if show:
        matplotlib.use("TkAgg")   # switch to interactive for display
        plt.show()
    plt.close(fig)


def _generate_all_figures(
    results: dict,
    cfg: ExperimentConfig,
    out_dir: Path | None,
    show: bool,
) -> None:
    print("\nGenerating figures …")
    _make_and_save(
        "fig1_errors", lambda: figure1_error_trajectories(results),
        out_dir, show,
    )
    _make_and_save(
        "fig4_nees", lambda: figure4_nees(results),
        out_dir, show,
    )
    _make_and_save(
        "fig_observability", lambda: figure_observability(results),
        out_dir, show,
    )
    _make_and_save(
        "fig_trajectory_2d", lambda: figure_trajectory_2d(results),
        out_dir, show,
    )
    _make_and_save(
        "fig_table1", lambda: figure_table1(results),
        out_dir, show,
    )


# ---------------------------------------------------------------------------
# Single-case runner
# ---------------------------------------------------------------------------

def _run_case(args: argparse.Namespace, case_id: int) -> None:
    cfg = _build_config(args, case_id)

    _print_banner(cfg)
    print(f"\nStarting Monte Carlo simulation …  ", end="", flush=True)
    t0 = _time.perf_counter()
    results = run_monte_carlo(cfg)
    elapsed = _time.perf_counter() - t0
    print(f"done in {elapsed:.1f} s\n")

    _print_summary(cfg, results)

    out_dir: Path | None = None
    if not args.no_save:
        out_dir = _make_output_dir(args.output_dir, cfg)
        print(f"Saving outputs to: {out_dir}/")
        _save_summary_txt(cfg, results, out_dir)
        _save_npz(results, out_dir)

    _generate_all_figures(results, cfg, out_dir, args.show)

    if out_dir is not None:
        print(f"\nAll outputs saved to: {out_dir.resolve()}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.show:
        # Allow interactive display
        matplotlib.use("TkAgg")

    if args.all_cases:
        cases = [1, 2, 3, 4]
    else:
        cases = [args.case]

    for case_id in cases:
        _run_case(args, case_id)
        if len(cases) > 1:
            print("─" * 68)


if __name__ == "__main__":
    main()
