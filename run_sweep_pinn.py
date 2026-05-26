"""Train PINN with fixed lambda=1e-2 and sweep over N.

Stages:
1. **Final sweep.** Train one PINN per
   ``N in {50, 100, 200, 500, 1000, 2000, 5000}`` with
   ``lambda = 1e-2`` and evaluate on the held-out test set.

Outputs::
    results/metrics_pinn.json
    checkpoints/pinn_n{N}.pt

Usage::
    python run_sweep_pinn.py \\
        --train_npz data/dataset_train.npz \\
        --test_npz  data/dataset_test.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from constants import N_VALUES
from eval import evaluate
from train import train_one_run

FIXED_LAMBDA: float = 1.0e-2

# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Sweep PINN across N with fixed lambda=1e-2.")
    p.add_argument("--train_npz", type=str, default="data/dataset_train.npz")
    p.add_argument("--test_npz", type=str, default="data/dataset_test.npz")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument(
        "--out", type=str, default="results/metrics_pinn.json",
        help="Path for the per-N metrics aggregate.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--n_collocation", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _build_train_args(
    cli: argparse.Namespace, n: int, epochs: int, ckpt_path: Path,
) -> SimpleNamespace:
    """Assemble training arguments for one PINN run."""
    return SimpleNamespace(
        model="pinn",
        n=n,
        seed=cli.seed,
        epochs=epochs,
        lr=cli.lr,
        batch_size=cli.batch_size,
        weight_decay=cli.weight_decay,
        patience=cli.patience,
        val_frac=cli.val_frac,
        train_npz=cli.train_npz,
        out=str(ckpt_path),
        device=cli.device,
        lam=FIXED_LAMBDA,
        n_collocation=cli.n_collocation,
    )

# ---------------------------------------------------------------------------
# Final sweep.
# ---------------------------------------------------------------------------
def final_sweep(
    cli: argparse.Namespace, n_values: list[int] = N_VALUES,
) -> dict:
    """Train one PINN per N with fixed lambda and evaluate on test."""
    ckpt_dir = Path(cli.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate = {"model": "pinn", "lambda": FIXED_LAMBDA, "by_n": {}}
    for n in n_values:
        ckpt_path = ckpt_dir / f"pinn_n{n}.pt"
        train_args = _build_train_args(cli, n, cli.epochs, ckpt_path)

        print(f"[pinn N={n}  lam={FIXED_LAMBDA:.0e}] training...")
        train_summary = train_one_run(train_args)
        print(f"[pinn N={n}] eval...")
        eval_metrics = evaluate(str(ckpt_path), cli.test_npz, cli.device, batch_size=256)

        aggregate["by_n"][str(n)] = {
            "N": n,
            "lambda": FIXED_LAMBDA,
            "param_count": train_summary["param_count"],
            "train_wall_clock_s": train_summary["train_wall_clock_s"],
            "peak_gpu_memory_mb": train_summary["peak_gpu_memory_mb"],
            "epochs_run": train_summary["epochs_run"],
            "best_val_loss": train_summary["best_val_loss"],
            "best_val_peak_bias": train_summary["best_val_peak_bias"],
            "inference_time_per_traj_ms": eval_metrics["inference_time_per_traj_ms"],
            "flops_per_forward": eval_metrics["flops_per_forward"],
            "overall": eval_metrics["overall"],
            "per_stratum": eval_metrics["per_stratum"],
            "overlay_sample": eval_metrics["overlay_sample"],
            "history": train_summary["history"],
        }

        with out_path.open("w") as fh:
            json.dump(aggregate, fh, indent=2)
        print(f"[pinn N={n}] wrote {out_path}")

    return aggregate

def main() -> None:
    args = parse_args()
    final_sweep(args)

if __name__ == "__main__":
    main()