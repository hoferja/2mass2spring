"""Sweep FNO training across ``N in {50, 100, 200, 500, 1000, 2000, 5000}``.

Writes ``results/metrics_fno.json`` with the unified schema. Memory-aware:
if peak GPU memory exceeds the configured budget at any N, the script
retrains that N with ``modes=12`` instead of 16 and flags the fallback in
the metrics record for that N.

Usage::

    python run_sweep_fno.py \\
        --train_npz data/dataset_train.npz \\
        --test_npz  data/dataset_test.npz \\
        --ckpt_dir  checkpoints \\
        --out       results/metrics_fno.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from constants import N_VALUES
from eval import evaluate
from train import train_one_run

DEFAULT_MEM_BUDGET_MB: float = 2 # 6 * 1024.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description="Sweep FNO across N.")
    p.add_argument("--train_npz", type=str, default="data/dataset_train.npz")
    p.add_argument("--test_npz", type=str, default="data/dataset_test.npz")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--out", type=str, default="results/metrics_fno.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--mem_budget_mb", type=float, default=DEFAULT_MEM_BUDGET_MB,
        help="Trigger a modes=12 fallback when peak GPU memory exceeds this.",
    )
    return p.parse_args()


def _build_train_args(
    cli: argparse.Namespace, n: int, ckpt_path: Path,
) -> SimpleNamespace:
    """Assemble the per-N training argument namespace.

    Args:
        cli: Parsed sweep arguments.
        n: Training subset size.
        ckpt_path: Output checkpoint path.

    Returns:
        Namespace consumable by ``train.train_one_run``.
    """
    return SimpleNamespace(
        model="fno",
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


def run(cli: argparse.Namespace) -> dict[str, Any]:
    """Run the full sweep with a per-N memory fallback to ``modes=12``.

    Args:
        cli: Parsed sweep arguments.

    Returns:
        Dict with the unified schema.
    """
    ckpt_dir = Path(cli.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, Any] = {"model": "fno", "by_n": {}}
    for n in N_VALUES:
        ckpt_path = ckpt_dir / f"fno_n{n}.pt"
        train_args = _build_train_args(cli, n, ckpt_path)

        print(f"[fno N={n}] training...")
        train_summary = train_one_run(train_args)
        fallback = False

        if (
            torch.cuda.is_available()
            and train_summary["peak_gpu_memory_mb"] > cli.mem_budget_mb
        ):
            print(
                f"[fno N={n}] peak GPU memory "
                f"{train_summary['peak_gpu_memory_mb']:.0f} MB exceeds "
                f"budget {cli.mem_budget_mb:.0f} MB; retrying modes=12."
            )
            train_summary = _retrain_with_smaller_modes(train_args)
            fallback = True

        print(f"[fno N={n}] eval...")
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
            "fallback_modes_12": fallback,
        }

        with out_path.open("w") as fh:
            json.dump(aggregate, fh, indent=2)
        print(f"[fno N={n}] wrote {out_path}")

    return aggregate


def _retrain_with_smaller_modes(
    train_args: SimpleNamespace,
) -> dict[str, Any]:
    """Retrain the FNO with ``modes=12`` by monkey-patching the default.

    Args:
        train_args: Original training argument namespace.

    Returns:
        Updated training summary dict from the fallback run.
    """
    from models import fno as fno_module

    original = fno_module.FNO1d

    class _Smaller(original):  # type: ignore[misc, valid-type]
        """FNO variant defaulting to ``modes=12``."""

        def __init__(self, **kw: Any) -> None:
            kw.setdefault("modes", 12)
            super().__init__(**kw)

    fno_module.FNO1d = _Smaller  # type: ignore[assignment]
    try:
        return train_one_run(train_args)
    finally:
        fno_module.FNO1d = original  # type: ignore[assignment]


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
