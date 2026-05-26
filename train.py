"""Unified training entry point for the MLP, FNO, and PINN models.

Each model branch saves a checkpoint containing model weights, fitted
normalizer (where applicable), per-epoch training history, and the CLI
config. The training functions also return a summary dict containing the
full ``history`` so sweep orchestrators can serialize loss curves.

Usage::

    python train.py --model mlp  --n 1000 --out checkpoints/mlp_n1000.pt
    python train.py --model fno  --n 1000 --out checkpoints/fno_n1000.pt
    python train.py --model pinn --n 1000 --lam 1e-2 \\
        --batch_size 32 --out checkpoints/pinn_n1000.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split

from constants import N_OUT_CHANNELS, N_PARAMS, T_HORIZON
from models.fno import FNO1d, build_fno_input
from models.mlp import MLP
from models.pinn import PINN, ode_residual
from utils import (
    ChannelNormalizer,
    TrajectoryDataset,
    TrajectoryNormalizer,
    count_params,
    load_npz,
    nested_prefix,
    set_seed,
)


# ---------------------------------------------------------------------------
# Argparse.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description="Train MLP, FNO, or PINN.")
    p.add_argument("--model", choices=["mlp", "fno", "pinn"], required=True)
    p.add_argument(
        "--n", type=int, required=True,
        help="Training subset size; nested-prefix slice of dataset_train.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--train_npz", type=str, default="data/dataset_train.npz")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    # PINN-only flags.
    p.add_argument(
        "--lam", type=float, default=1e-2,
        help="PINN residual loss weight (ignored for MLP/FNO).",
    )
    p.add_argument(
        "--n_collocation", type=int, default=2000,
        help="Collocation points per trajectory per epoch (PINN only).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# MLP training.
# ---------------------------------------------------------------------------
def train_mlp_run(args: argparse.Namespace) -> dict[str, Any]:
    """Train an MLP on the nested-prefix subset of ``dataset_train.npz``.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary dict with keys ``best_val_loss``, ``epochs_run``,
        ``train_wall_clock_s``, ``peak_gpu_memory_mb``, ``param_count``,
        ``history``. ``history`` is a list of per-epoch records:
        ``{"epoch", "train_loss", "val_loss", "lr"}``.
    """
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    raw = load_npz(args.train_npz)
    params_all, traj_all = nested_prefix(raw["params"], raw["traj"], n=args.n)

    normalizer = TrajectoryNormalizer.from_arrays(params_all, traj_all).to(device)
    full_ds = TrajectoryDataset(params_all, traj_all, normalizer)

    n_val = max(1, int(round(args.val_frac * len(full_ds))))
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    bs = min(args.batch_size, n_train)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    model = MLP().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_losses: list[float] = []
        for p_batch, y_batch in train_loader:
            p_batch = p_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(p_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for p_batch, y_batch in val_loader:
                p_batch = p_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(p_batch)
                val_losses.append(loss_fn(y_pred, y_batch).item())
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": int(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                break

    train_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    _save_mlp_checkpoint(
        model=model, normalizer=normalizer, history=history, args=args,
        out_path=Path(args.out), extras={"best_val_loss": float(best_val)},
    )

    return {
        "best_val_loss": float(best_val),
        "epochs_run": int(len(history)),
        "train_wall_clock_s": float(train_seconds),
        "peak_gpu_memory_mb": float(peak_mb),
        "param_count": int(count_params(model)),
        "history": history,
    }


# ---------------------------------------------------------------------------
# FNO training.
# ---------------------------------------------------------------------------
def train_fno_run(args: argparse.Namespace) -> dict[str, Any]:
    """Train a 1D FNO on the nested-prefix subset of ``dataset_train.npz``.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary dict with keys ``best_val_loss``, ``epochs_run``,
        ``train_wall_clock_s``, ``peak_gpu_memory_mb``, ``param_count``,
        ``history``. History records have keys
        ``{"epoch", "train_loss", "val_loss", "lr"}``.
    """
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    raw = load_npz(args.train_npz)
    params_all, x_c_all, traj_all = nested_prefix(
        raw["params"], raw["x_c"], raw["traj"], n=args.n
    )
    params_t = torch.from_numpy(params_all).float()
    x_c_t = torch.from_numpy(x_c_all).float()
    traj_t = torch.from_numpy(traj_all).float()

    inputs = build_fno_input(x_c_t, params_t)        # (N, P+1, T)
    targets = traj_t.transpose(1, 2).contiguous()    # (N, C, T)

    n_total = inputs.shape[0]
    n_val = max(1, int(round(args.val_frac * n_total)))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    in_norm = ChannelNormalizer.from_tensor(inputs[tr_idx]).to(device)
    out_norm = ChannelNormalizer.from_tensor(targets[tr_idx]).to(device)

    bs = min(args.batch_size, len(tr_idx))
    train_ds = TensorDataset(inputs[tr_idx], targets[tr_idx])
    val_ds = TensorDataset(inputs[val_idx], targets[val_idx])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    model = FNO1d(
        in_channels=N_PARAMS + 1, out_channels=N_OUT_CHANNELS,
        width=32, modes=16, n_layers=4, proj_hidden=128,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_n = model(in_norm.encode(xb))
            loss = loss_fn(pred_n, out_norm.encode(yb))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred_n = model(in_norm.encode(xb))
                val_losses.append(loss_fn(pred_n, out_norm.encode(yb)).item())
        val_loss = float(np.mean(val_losses))

        history.append({
            "epoch": int(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                break

    train_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    _save_fno_checkpoint(
        model=model, in_norm=in_norm, out_norm=out_norm, history=history,
        args=args, out_path=Path(args.out),
        extras={"best_val_loss": float(best_val)},
    )

    return {
        "best_val_loss": float(best_val),
        "epochs_run": int(len(history)),
        "train_wall_clock_s": float(train_seconds),
        "peak_gpu_memory_mb": float(peak_mb),
        "param_count": int(count_params(model)),
        "history": history,
    }


# ---------------------------------------------------------------------------
# PINN training.
# ---------------------------------------------------------------------------
def train_pinn_run(args: argparse.Namespace) -> dict[str, Any]:
    """Train a PINN with data MSE + ``lam`` * residual MSE.

    Collocation points are resampled per minibatch (stronger stochasticity
    than per-epoch resampling at no extra cost). Validation uses RMSE on
    the dataset time grid; checkpoint selection is by lowest val RMSE.

    Args:
        args: Parsed CLI arguments. Uses ``args.lam`` and
            ``args.n_collocation`` in addition to common flags.

    Returns:
        Summary dict with keys ``best_val_loss`` (RMSE), ``best_val_peak_bias``,
        ``epochs_run``, ``train_wall_clock_s``, ``peak_gpu_memory_mb``,
        ``param_count``, ``lambda``, ``history``. History records have keys
        ``{"epoch", "train_data", "train_res", "train_total",
            "val_rmse", "val_peak_bias", "lr"}``.
    """
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    raw = load_npz(args.train_npz)
    params_all, traj_all = nested_prefix(raw["params"], raw["traj"], n=args.n)
    params_t = torch.from_numpy(params_all.astype(np.float32)).to(device)
    traj_t = torch.from_numpy(traj_all.astype(np.float32)).to(device)
    k_time = traj_t.shape[1]
    t_grid = torch.linspace(
        0.0, T_HORIZON, k_time, dtype=torch.float32, device=device
    )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(args.n)
    n_val = max(1, int(round(args.val_frac * args.n)))
    val_idx = torch.from_numpy(perm[:n_val]).long().to(device)
    train_idx = torch.from_numpy(perm[n_val:]).long().to(device)

    model = PINN().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val = float("inf")
    best_val_bias = float("nan")
    best_state: dict[str, Tensor] | None = None
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    bs = min(args.batch_size, int(train_idx.shape[0]))
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        order_perm = torch.randperm(train_idx.shape[0], device=device)
        order = train_idx[order_perm]
        running = {"data": 0.0, "res": 0.0, "total": 0.0, "n": 0}

        for start in range(0, order.shape[0], bs):
            batch = order[start:start + bs]
            n_batch = int(batch.shape[0])
            params_b = params_t[batch]
            traj_b = traj_t[batch]

            # Data branch: evaluate on the dataset time grid.
            p_d = params_b.unsqueeze(1).expand(n_batch, k_time, 6).reshape(-1, 6)
            t_d = t_grid.view(1, k_time, 1).expand(n_batch, k_time, 1).reshape(-1, 1)
            pred_d = model(p_d, t_d).view(n_batch, k_time, 2)
            data_loss = ((pred_d - traj_b) ** 2).mean()

            # Residual branch: fresh uniform collocation points.
            t_c = (
                torch.empty(n_batch * args.n_collocation, 1, device=device)
                .uniform_(0.0, T_HORIZON)
                .requires_grad_(True)
            )
            p_c = (
                params_b.unsqueeze(1)
                .expand(n_batch, args.n_collocation, 6)
                .reshape(-1, 6)
            )
            r1, r2 = ode_residual(model, p_c, t_c)
            res_loss = (r1 ** 2).mean() + (r2 ** 2).mean()

            total = data_loss + args.lam * res_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            running["data"] += float(data_loss.detach()) * n_batch
            running["res"] += float(res_loss.detach()) * n_batch
            running["total"] += float(total.detach()) * n_batch
            running["n"] += n_batch
        scheduler.step()

        # Validation: RMSE and signed peak bias on the held-out split.
        model.eval()
        with torch.no_grad():
            pred_val = model.predict_trajectory(params_t[val_idx], t_grid)
            traj_val = traj_t[val_idx]
            val_rmse = float(torch.sqrt(((pred_val - traj_val) ** 2).mean()).item())
            peak_pred = (pred_val[..., 0] + pred_val[..., 1]).amax(dim=1)
            peak_true = (traj_val[..., 0] + traj_val[..., 1]).amax(dim=1)
            val_peak_bias = float((peak_pred - peak_true).mean().item())

        denom = max(running["n"], 1)
        history.append({
            "epoch": int(epoch),
            "train_data": running["data"] / denom,
            "train_res": running["res"] / denom,
            "train_total": running["total"] / denom,
            "val_rmse": val_rmse,
            "val_peak_bias": val_peak_bias,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if val_rmse < best_val:
            best_val = val_rmse
            best_val_bias = val_peak_bias
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                break

    train_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    _save_pinn_checkpoint(
        model=model, history=history, args=args, out_path=Path(args.out),
        extras={
            "best_val_loss": float(best_val),
            "best_val_peak_bias": float(best_val_bias),
        },
    )

    return {
        "best_val_loss": float(best_val),
        "best_val_peak_bias": float(best_val_bias),
        "epochs_run": int(len(history)),
        "train_wall_clock_s": float(train_seconds),
        "peak_gpu_memory_mb": float(peak_mb),
        "param_count": int(count_params(model)),
        "lambda": float(args.lam),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Checkpoint savers.
# ---------------------------------------------------------------------------
def _save_mlp_checkpoint(
    model: nn.Module,
    normalizer: TrajectoryNormalizer,
    history: list[dict[str, float]],
    args: argparse.Namespace,
    out_path: Path,
    extras: dict[str, Any] | None = None,
) -> None:
    """Save an MLP checkpoint with model, normalizer, history, and config.

    Args:
        model: Trained MLP.
        normalizer: Fitted ``TrajectoryNormalizer``.
        history: Per-epoch metric dicts.
        args: Parsed CLI arguments.
        out_path: Destination ``.pt`` path.
        extras: Optional extra fields embedded at the top level.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": "mlp",
        "state_dict": model.state_dict(),
        "normalizer": {
            "param_mean": normalizer.param_mean.detach().cpu(),
            "param_std": normalizer.param_std.detach().cpu(),
            "traj_mean": normalizer.traj_mean.detach().cpu(),
            "traj_std": normalizer.traj_std.detach().cpu(),
        },
        "history": history,
        "args": vars(args),
    }
    if extras:
        payload.update(extras)
    torch.save(payload, out_path)


def _save_fno_checkpoint(
    model: nn.Module,
    in_norm: ChannelNormalizer,
    out_norm: ChannelNormalizer,
    history: list[dict[str, float]],
    args: argparse.Namespace,
    out_path: Path,
    extras: dict[str, Any] | None = None,
) -> None:
    """Save an FNO checkpoint with model, normalizers, history, and config.

    Args:
        model: Trained FNO.
        in_norm: Input ``ChannelNormalizer``.
        out_norm: Output ``ChannelNormalizer``.
        history: Per-epoch metric dicts.
        args: Parsed CLI arguments.
        out_path: Destination ``.pt`` path.
        extras: Optional extra fields embedded at the top level.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": "fno",
        "state_dict": model.state_dict(),
        "in_norm_mean": in_norm.mean.detach().cpu(),
        "in_norm_std": in_norm.std.detach().cpu(),
        "out_norm_mean": out_norm.mean.detach().cpu(),
        "out_norm_std": out_norm.std.detach().cpu(),
        "history": history,
        "args": vars(args),
    }
    if extras:
        payload.update(extras)
    torch.save(payload, out_path)


def _save_pinn_checkpoint(
    model: nn.Module,
    history: list[dict[str, float]],
    args: argparse.Namespace,
    out_path: Path,
    extras: dict[str, Any] | None = None,
) -> None:
    """Save a PINN checkpoint with model, history, and config.

    The PINN does not use a wrapping normalizer because the network
    normalizes its inputs internally (see :meth:`models.pinn.PINN._norm_inputs`)
    and outputs are produced in physical units to keep the ODE residual
    meaningful.

    Args:
        model: Trained PINN.
        history: Per-epoch metric dicts.
        args: Parsed CLI arguments.
        out_path: Destination ``.pt`` path.
        extras: Optional extra fields embedded at the top level.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": "pinn",
        "state_dict": model.state_dict(),
        "history": history,
        "args": vars(args),
        "lambda": float(args.lam),
    }
    if extras:
        payload.update(extras)
    torch.save(payload, out_path)


# ---------------------------------------------------------------------------
# CLI dispatch.
# ---------------------------------------------------------------------------
def train_one_run(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch to the per-model training function based on ``args.model``.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary dict returned by the per-model training function.

    Raises:
        ValueError: If ``args.model`` is not one of ``mlp``, ``fno``, ``pinn``.
    """
    if args.model == "mlp":
        return train_mlp_run(args)
    if args.model == "fno":
        return train_fno_run(args)
    if args.model == "pinn":
        return train_pinn_run(args)
    raise ValueError(f"Unknown model: {args.model}")


def main() -> None:
    """CLI entry point: parse args, dispatch, print brief summary."""
    args = parse_args()
    summary = train_one_run(args)
    brief = {k: v for k, v in summary.items() if k != "history"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
