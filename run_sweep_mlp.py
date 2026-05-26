"""Sweep MLP training across ``N in {50, 100, 200, 500, 1000, 2000, 5000}``.

Writes ``results/metrics_mlp.json`` with the unified schema (see project
``README.md``). Per-N entries contain training summary, evaluation metric
set from concept Section 5, and the full per-epoch training ``history``.
Training data is the nested-prefix subset of ``dataset_train.npz`` so the
same trajectories are reused across N.

Usage::

    python run_sweep_mlp.py \\
        --train_npz data/dataset_train.npz \\
        --test_npz  data/dataset_test.npz \\
        --ckpt_dir  checkpoints \\
        --out       results/metrics_mlp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from constants import N_VALUES
from eval import evaluate
from train import train_one_run


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description="Sweep MLP across N.")
    p.add_argument("--train_npz", type=str, default="data/dataset_train.npz")
    p.add_argument("--test_npz", type=str, default="data/dataset_test.npz")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default="results/metrics_mlp.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def run(cli: argparse.Namespace) -> dict[str, Any]:
    """Run the full sweep, writing ``metrics_<model>.json`` incrementally.

    Args:
        cli: Parsed sweep arguments.

    Returns:
        Dict with the unified schema: ``{"model": "mlp", "by_n": {...}}``.
    """
    ckpt_dir = Path(cli.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, Any] = {"model": "mlp", "by_n": {}}
    for n in N_VALUES:
        ckpt_path = ckpt_dir / f"mlp_n{n}.pt"
        train_args = SimpleNamespace(
            model="mlp",
            n=n,
            seed=cli.seed,
            epochs=cli.epochs,
            lr=cli.lr,
            batch_size=cli.batch_size,
            weight_decay=cli.weight_decay,
            patience=cli.patience,
            val_frac=cli.val_frac,
            train_npz=cli.train_npz,
            out=str(ckpt_path),
            device=cli.device,
            lam=1e-2,
            n_collocation=2000,
        )

        print(f"[mlp N={n}] training...")
        train_summary = train_one_run(train_args)
        print(f"[mlp N={n}] eval...")
        eval_metrics = evaluate(
            str(ckpt_path), cli.test_npz, cli.device, batch_size=256
        )

        aggregate["by_n"][str(n)] = {
            "N": n,
            "param_count": train_summary["param_count"],
            "train_wall_clock_s": train_summary["train_wall_clock_s"],
            "peak_gpu_memory_mb": train_summary["peak_gpu_memory_mb"],
            "epochs_run": train_summary["epochs_run"],
            "best_val_loss": train_summary["best_val_loss"],
            "inference_time_per_traj_ms": eval_metrics["inference_time_per_traj_ms"],
            "flops_per_forward": eval_metrics["flops_per_forward"],
            "overall": eval_metrics["overall"],
            "per_stratum": eval_metrics["per_stratum"],
            "overlay_sample": eval_metrics["overlay_sample"],
            "history": train_summary["history"],
        }

        with out_path.open("w") as fh:
            json.dump(aggregate, fh, indent=2)
        print(f"[mlp N={n}] wrote {out_path}")

    return aggregate


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
