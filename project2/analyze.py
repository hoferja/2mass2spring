"""Phase 6 analysis: build comparison table and plots from the sweep JSONs.

Consumes the unified schema written by ``run_sweep_mlp.py``,
``run_sweep_fno.py``, and ``run_sweep_pinn.py``::

    {
      "model": "mlp" | "pinn" | "fno",
      "lambda": float,                    # pinn only
      "by_n": {
        "<N>": {
          "N": int,
          "param_count": int,
          "train_wall_clock_s": float,
          "peak_gpu_memory_mb": float,
          "epochs_run": int,
          "best_val_loss": float,
          "inference_time_per_traj_ms": float,
          "flops_per_forward": int | null,
          "overall":     {"rmse_traj", "rmse_peak", "bias_peak", "count"},
          "per_stratum": {"r_0.0_1.0": {...}, "r_1.0_1.2": {...},
                          "r_1.2_1.5": {...}, "r_gt_1.5":  {...}},
          "overlay_sample": {"index", "amplification_ratio",
                              "t", "pred_sum", "true_sum"},
          "history": [{"epoch", "train_loss"/"train_total",
                        "val_loss"/"val_rmse", "lr"}, ...]
        }, ...
      }
    }

Usage::

    python analyze.py --results_dir results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from constants import STRATA

MODELS: tuple[str, ...] = ("mlp", "pinn", "fno")
OVERLAY_NS: tuple[int, ...] = (50, 1000, 5000)
COLORS: dict[str, str] = {"mlp": "tab:blue", "pinn": "tab:orange", "fno": "tab:green"}
STRATUM_NAMES: tuple[str, ...] = tuple(name for name, _, _ in STRATA)
STRATUM_BOUNDARIES: tuple[float, ...] = tuple(lo for _, lo, _ in STRATA if lo > 0)


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------
def load_metrics(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load per-model metric JSONs.

    Args:
        results_dir: Directory containing ``metrics_<model>.json`` files.

    Returns:
        Mapping from model name to its parsed metrics dict. Models with no
        file present are silently skipped.
    """
    out: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        path = results_dir / f"metrics_{model}.json"
        if not path.exists():
            print(f"[analyze] warning: {path} missing; skipping {model}")
            continue
        with path.open() as fh:
            out[model] = json.load(fh)
    return out


def _sorted_runs(model_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return run entries sorted by training set size N.

    Args:
        model_data: Metrics dict for a single model with a ``"by_n"`` key.

    Returns:
        List of per-N record dicts ordered by ``int(N)``.
    """
    return sorted(model_data["by_n"].values(), key=lambda r: r["N"])


# ---------------------------------------------------------------------------
# Comparison table.
# ---------------------------------------------------------------------------
def write_comparison_table(
    metrics: dict[str, dict[str, Any]], out_path: Path,
) -> None:
    """Write a flat ``(model, N, stratum)`` comparison table to CSV.

    Args:
        metrics: Mapping model name to metrics dict.
        out_path: Destination CSV path.
    """
    fields = [
        "model", "N", "stratum", "param_count",
        "train_wall_clock_s", "peak_gpu_memory_mb",
        "inference_time_per_traj_ms", "flops_per_forward",
        "rmse_traj", "rmse_peak", "bias_peak", "count",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for model, data in metrics.items():
            for run in _sorted_runs(data):
                base = {
                    "model": model,
                    "N": run["N"],
                    "param_count": run["param_count"],
                    "train_wall_clock_s": run["train_wall_clock_s"],
                    "peak_gpu_memory_mb": run["peak_gpu_memory_mb"],
                    "inference_time_per_traj_ms": run["inference_time_per_traj_ms"],
                    "flops_per_forward": run["flops_per_forward"],
                }
                # Overall row.
                ov = run["overall"]
                writer.writerow({
                    **base, "stratum": "overall",
                    "rmse_traj": ov["rmse_traj"], "rmse_peak": ov["rmse_peak"],
                    "bias_peak": ov["bias_peak"], "count": ov["count"],
                })
                # One row per stratum.
                for stratum in STRATUM_NAMES:
                    s = run["per_stratum"][stratum]
                    writer.writerow({
                        **base, "stratum": stratum,
                        "rmse_traj": s["rmse_traj"], "rmse_peak": s["rmse_peak"],
                        "bias_peak": s["bias_peak"], "count": s["count"],
                    })


# ---------------------------------------------------------------------------
# Curve extraction.
# ---------------------------------------------------------------------------
def _curve(
    metrics: dict[str, dict[str, Any]], section: str, field: str,
) -> dict[str, tuple[list[int], list[float]]]:
    """Extract sorted ``(Ns, values)`` per model.

    Args:
        metrics: Mapping model to metrics dict.
        section: Either ``'overall'`` or a stratum name from ``STRATUM_NAMES``.
        field: Field name inside that section (e.g. ``'rmse_traj'``).

    Returns:
        Mapping model to ``(Ns, values)`` sorted by ``N``.
    """
    out: dict[str, tuple[list[int], list[float]]] = {}
    for model, data in metrics.items():
        ns: list[int] = []
        vs: list[float] = []
        for run in _sorted_runs(data):
            if section == "overall":
                sec = run["overall"]
            else:
                sec = run["per_stratum"][section]
            ns.append(int(run["N"]))
            vs.append(float(sec[field]))
        out[model] = (ns, vs)
    return out


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------
def plot_rmse_vs_n(
    metrics: dict[str, dict[str, Any]], section: str, out_path: Path,
    title: str,
) -> None:
    """Plot trajectory RMSE vs ``N`` on log-log axes.

    Args:
        metrics: Mapping model to metrics dict.
        section: ``'overall'`` or a stratum name.
        out_path: Destination PNG path.
        title: Plot title.
    """
    curves = _curve(metrics, section, "rmse_traj")
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, (ns, vs) in curves.items():
        ax.loglog(ns, vs, marker="o", color=COLORS[model], label=model.upper())
    ax.set_xlabel("training set size N")
    ax.set_ylabel("trajectory RMSE on (x_r, x_b)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_max_elong_bias(
    metrics: dict[str, dict[str, Any]], out_path: Path,
) -> None:
    """Plot signed bias on max elongation vs ``N`` (strong stratum).

    Args:
        metrics: Mapping model to metrics dict.
        out_path: Destination PNG path.
    """
    curves = _curve(metrics, "r_gt_1.5", "bias_peak")
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, (ns, vs) in curves.items():
        ax.semilogx(ns, vs, marker="o", color=COLORS[model], label=model.upper())
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("training set size N")
    ax.set_ylabel("signed bias on max elongation [m]")
    ax.set_title("Max-elongation bias (strong stratum). Negative = under-prediction.")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto(
    metrics: dict[str, dict[str, Any]], out_path: Path,
) -> None:
    """Plot compute-vs-accuracy Pareto on the strong-amplification stratum.

    One marker per ``(model, N)``; marker size proportional to parameter
    count.

    Args:
        metrics: Mapping model to metrics dict.
        out_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, data in metrics.items():
        runs = _sorted_runs(data)
        x = [r["train_wall_clock_s"] for r in runs]
        y = [r["per_stratum"]["r_gt_1.5"]["rmse_traj"] for r in runs]
        sizes = [max(20.0, r["param_count"] / 1000.0) for r in runs]
        ax.scatter(
            x, y, s=sizes, c=COLORS[model], alpha=0.7,
            label=model.upper(), edgecolors="black",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training wall-clock [s]")
    ax.set_ylabel("trajectory RMSE on r > 1.5 stratum")
    ax.set_title("Compute vs accuracy (marker size proportional to params)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectory_overlays(
    metrics: dict[str, dict[str, Any]], out_path: Path,
) -> None:
    """Plot a 3x3 grid of predicted vs true ``x_r + x_b`` for the overlay sample.

    Rows are ``N in OVERLAY_NS``; columns are models. The true curve is
    taken from the first model's overlay sample (all sweeps share the same
    test set and the overlay index is computed deterministically).

    Args:
        metrics: Mapping model to metrics dict.
        out_path: Destination PNG path.
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    for col, model in enumerate(MODELS):
        if model not in metrics:
            for row in range(3):
                axes[row, col].set_visible(False)
            continue
        runs_by_n = {r["N"]: r for r in _sorted_runs(metrics[model])}
        for row, n in enumerate(OVERLAY_NS):
            ax = axes[row, col]
            if n not in runs_by_n:
                ax.set_visible(False)
                continue
            sample = runs_by_n[n]["overlay_sample"]
            t = np.asarray(sample["t"])
            true = np.asarray(sample["true_sum"])
            pred = np.asarray(sample["pred_sum"])
            ax.plot(t, true, color="black", linewidth=1.2, label="true")
            ax.plot(t, pred, color=COLORS[model], linewidth=1.0, label="pred")
            if row == 0:
                ax.set_title(model.upper())
            if col == 0:
                ax.set_ylabel(f"N={n}\n(x_r + x_b) [m]")
            if row == 2:
                ax.set_xlabel("time [s]")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Predicted vs true end-effector trajectory (highest-r test sample)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_amplification_histogram(
    metrics: dict[str, dict[str, Any]], out_path: Path,
) -> None:
    """Plot a placeholder note: the full ``r`` distribution is not in metrics.

    The full per-test-trajectory ``r`` array is not serialized in the unified
    schema (only one overlay sample is). This stub leaves a hint plot. To
    produce a real histogram, run a small script that loads
    ``data/dataset_test.npz`` and calls
    ``utils.amplification_ratio(traj, x_c)``.

    Args:
        metrics: Mapping model to metrics dict. Used only to annotate the
            figure with the overlay sample's ratio.
        out_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(
        0.5, 0.5,
        "Run histogram_test_amplification.py with dataset_test.npz\n"
        "to produce the full histogram.",
        ha="center", va="center", transform=ax.transAxes,
    )
    for boundary in STRATUM_BOUNDARIES:
        ax.axvline(boundary, color="tab:red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("amplification ratio r")
    ax.set_ylabel("count (placeholder)")
    ax.set_title("Amplification ratio distribution placeholder")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss-curve plots (one figure per model).
# ---------------------------------------------------------------------------
def plot_loss_curves(
    metrics: dict[str, dict[str, Any]], out_dir: Path,
) -> None:
    """Plot per-epoch training and validation loss curves, one figure per model.

    Each figure overlays curves for every ``N`` in the sweep on
    ``log_y`` axes. The training-loss field name is auto-detected
    (``train_loss`` for MLP/FNO, ``train_total`` for PINN), and likewise
    for the validation field (``val_loss`` vs ``val_rmse``).

    Args:
        metrics: Mapping model to metrics dict.
        out_dir: Destination directory for the PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, data in metrics.items():
        runs = _sorted_runs(data)
        if not runs:
            continue
        first_record = runs[0]["history"][0]
        train_key = "train_loss" if "train_loss" in first_record else "train_total"
        val_key = "val_loss" if "val_loss" in first_record else "val_rmse"

        fig, (ax_tr, ax_va) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(runs)))
        for color, run in zip(cmap, runs):
            history = run["history"]
            epochs = [rec["epoch"] for rec in history]
            ax_tr.plot(
                epochs, [rec[train_key] for rec in history],
                color=color, linewidth=1.0, label=f"N={run['N']}",
            )
            ax_va.plot(
                epochs, [rec[val_key] for rec in history],
                color=color, linewidth=1.0,
            )
        for ax in (ax_tr, ax_va):
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.grid(True, which="both", alpha=0.3)
        ax_tr.set_ylabel(train_key)
        ax_va.set_ylabel(val_key)
        ax_tr.set_title(f"{model.upper()}  training loss")
        ax_va.set_title(f"{model.upper()}  validation loss")
        ax_tr.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"loss_curves_{model}.png", dpi=150)
        plt.close(fig)


def plot_pinn_residual_curves(
    metrics: dict[str, dict[str, Any]], out_path: Path,
) -> None:
    """Plot the PINN data vs residual loss components across epochs.

    Only emitted if PINN metrics are present and contain the residual
    breakdown.

    Args:
        metrics: Mapping model to metrics dict.
        out_path: Destination PNG path.
    """
    if "pinn" not in metrics:
        return
    runs = _sorted_runs(metrics["pinn"])
    if not runs:
        return
    first_record = runs[0]["history"][0]
    if "train_data" not in first_record or "train_res" not in first_record:
        return

    fig, (ax_d, ax_r) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(runs)))
    for color, run in zip(cmap, runs):
        history = run["history"]
        epochs = [rec["epoch"] for rec in history]
        ax_d.plot(
            epochs, [rec["train_data"] for rec in history],
            color=color, linewidth=1.0, label=f"N={run['N']}",
        )
        ax_r.plot(
            epochs, [rec["train_res"] for rec in history],
            color=color, linewidth=1.0,
        )
    for ax in (ax_d, ax_r):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.grid(True, which="both", alpha=0.3)
    ax_d.set_ylabel("train_data (MSE)")
    ax_r.set_ylabel("train_res (residual MSE)")
    ax_d.set_title("PINN  data loss")
    ax_r.set_title("PINN  residual loss")
    ax_d.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main() -> None:
    """Build the comparison table and all Phase 6 plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    metrics = load_metrics(args.results_dir)
    if not metrics:
        raise SystemExit(f"No metrics_*.json files found in {args.results_dir}")

    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_comparison_table(metrics, args.results_dir / "comparison_table.csv")
    plot_rmse_vs_n(
        metrics, "overall", plots_dir / "rmse_vs_n.png",
        "Trajectory RMSE vs N (full test set)",
    )
    plot_rmse_vs_n(
        metrics, "r_gt_1.5", plots_dir / "rmse_vs_n_strong.png",
        "Trajectory RMSE vs N (strong amplification, r > 1.5)",
    )
    plot_max_elong_bias(metrics, plots_dir / "max_elong_bias_vs_n.png")
    plot_pareto(metrics, plots_dir / "pareto_compute_accuracy.png")
    plot_trajectory_overlays(metrics, plots_dir / "trajectory_overlays.png")
    plot_amplification_histogram(metrics, plots_dir / "amplification_histogram.png")
    plot_loss_curves(metrics, plots_dir)
    plot_pinn_residual_curves(metrics, plots_dir / "pinn_residual_curves.png")
    print(f"[analyze] wrote table and plots under {args.results_dir}")


if __name__ == "__main__":
    main()
