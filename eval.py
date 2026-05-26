"""Evaluate a checkpoint (MLP, PINN, or FNO) on the test set.

Computes the full metric set from ``project2_concept.txt`` Section 5 and
returns a JSON-serializable dict. Also stores the prediction for one fixed
"overlay" test sample (argmax of amplification ratio) so Phase 6 can build
trajectory overlay plots without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from constants import N_OUT_CHANNELS, N_PARAMS, N_TIME, T_HORIZON
from models.fno import FNO1d, build_fno_input
from models.mlp import MLP
from models.pinn import PINN
from utils import (
    ChannelNormalizer,
    TrajectoryNormalizer,
    amplification_ratio,
    compute_metrics,
    count_params,
    load_npz,
    measure_flops,
    measure_inference_ms,
    stratum_indices,
)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_npz", type=str, default="data/dataset_test.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading.
# ---------------------------------------------------------------------------
def _load_checkpoint(
    ckpt_path: str, device: torch.device
) -> tuple[nn.Module, str, dict[str, Any]]:
    """Load a checkpoint and rebuild the model.

    Args:
        ckpt_path: Path to a ``.pt`` file produced by :mod:`train`.
        device: Target device.

    Returns:
        Tuple ``(model, model_name, payload)`` where ``payload`` is the raw
        checkpoint dict (for normalizer fields).

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    name = payload["model_name"]
    if name == "mlp":
        model: nn.Module = MLP().to(device)
    elif name == "pinn":
        model = PINN().to(device)
    elif name == "fno":
        model = FNO1d(
            in_channels=N_PARAMS + 1, out_channels=N_OUT_CHANNELS,
            width=32, modes=16, n_layers=4, proj_hidden=128,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name in checkpoint: {name}")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, name, payload


# ---------------------------------------------------------------------------
# Per-model prediction.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict_mlp(
    model: MLP, normalizer: TrajectoryNormalizer,
    params: np.ndarray, device: torch.device, batch_size: int,
) -> np.ndarray:
    """Run the MLP on all test parameters; return denormalized predictions.

    Args:
        model: Trained MLP.
        normalizer: Fitted ``TrajectoryNormalizer``.
        params: Test parameters of shape ``(N, 6)``.
        device: Inference device.
        batch_size: Batch size.

    Returns:
        Predictions of shape ``(N, T, 2)`` in physical units.
    """
    p = torch.tensor(params, dtype=torch.float32, device=device)
    p_n = normalizer.norm_params(p)
    outs: list[Tensor] = []
    for s in range(0, p_n.shape[0], batch_size):
        outs.append(model(p_n[s:s + batch_size]))
    y_n = torch.cat(outs, dim=0)
    return normalizer.denorm_traj(y_n).cpu().numpy()


@torch.no_grad()
def _predict_pinn(
    model: PINN, params: np.ndarray, device: torch.device, batch_size: int,
) -> np.ndarray:
    """Run the PINN on test params over the dataset time grid.

    Args:
        model: Trained PINN.
        params: Test parameters of shape ``(N, 6)``.
        device: Inference device.
        batch_size: Number of trajectories per chunk.

    Returns:
        Predictions of shape ``(N, T, 2)``.
    """
    p = torch.tensor(params, dtype=torch.float32, device=device)
    t_grid = torch.linspace(
        0.0, T_HORIZON, N_TIME, dtype=torch.float32, device=device
    )
    outs: list[Tensor] = []
    for s in range(0, p.shape[0], batch_size):
        outs.append(model.predict_trajectory(p[s:s + batch_size], t_grid))
    return torch.cat(outs, dim=0).cpu().numpy()


@torch.no_grad()
def _predict_fno(
    model: FNO1d, in_norm: ChannelNormalizer, out_norm: ChannelNormalizer,
    params: np.ndarray, x_c: np.ndarray, device: torch.device, batch_size: int,
) -> np.ndarray:
    """Run the FNO on test params + chirp; return denormalized predictions.

    Args:
        model: Trained FNO.
        in_norm: Input ``ChannelNormalizer``.
        out_norm: Output ``ChannelNormalizer``.
        params: Test parameters of shape ``(N, 6)``.
        x_c: Chirp signal of shape ``(N, T)``.
        device: Inference device.
        batch_size: Batch size.

    Returns:
        Predictions of shape ``(N, T, 2)`` in physical units.
    """
    p = torch.tensor(params, dtype=torch.float32, device=device)
    xc = torch.tensor(x_c, dtype=torch.float32, device=device)
    full = build_fno_input(xc, p)
    outs: list[Tensor] = []
    for s in range(0, full.shape[0], batch_size):
        x = full[s:s + batch_size]
        y_n = model(in_norm.encode(x))
        outs.append(out_norm.decode(y_n))
    y = torch.cat(outs, dim=0).transpose(1, 2).contiguous()
    return y.cpu().numpy()


# ---------------------------------------------------------------------------
# Inference timing / FLOPs.
# ---------------------------------------------------------------------------
def _measure_inference_ms_for_model(
    model: nn.Module, model_name: str, payload: dict[str, Any],
    params: np.ndarray, x_c: np.ndarray, device: torch.device,
) -> float:
    """Time one-sample inference for the given model type.

    Args:
        model: Trained model.
        model_name: ``'mlp'``, ``'pinn'``, or ``'fno'``.
        payload: Raw checkpoint payload (used for normalizer fields).
        params: Test parameters of shape ``(N, 6)``.
        x_c: Chirp signal of shape ``(N, T)``.
        device: Inference device.

    Returns:
        Mean wall-clock time per trajectory in milliseconds.

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    p1 = torch.tensor(params[:1], dtype=torch.float32, device=device)
    if model_name == "mlp":
        nz = payload["normalizer"]
        norm = TrajectoryNormalizer(
            nz["param_mean"], nz["param_std"], nz["traj_mean"], nz["traj_std"]
        ).to(device)
        p_n = norm.norm_params(p1)
        return measure_inference_ms(lambda x: model(x), p_n, device=device)
    if model_name == "pinn":
        t_grid = torch.linspace(
            0.0, T_HORIZON, N_TIME, dtype=torch.float32, device=device
        )
        return measure_inference_ms(
            lambda x, tg: model.predict_trajectory(x, tg),
            p1, t_grid, device=device,
        )
    if model_name == "fno":
        in_norm = ChannelNormalizer(
            payload["in_norm_mean"], payload["in_norm_std"]
        ).to(device)
        xc1 = torch.tensor(x_c[:1], dtype=torch.float32, device=device)
        x1 = build_fno_input(xc1, p1)
        x1_n = in_norm.encode(x1)
        return measure_inference_ms(lambda x: model(x), x1_n, device=device)
    raise ValueError(f"Unknown model: {model_name}")


def _measure_flops_for_model(
    model: nn.Module, model_name: str, device: torch.device,
) -> int | None:
    """Estimate FLOPs per forward pass at trajectory granularity.

    For PINN, point-wise FLOPs are multiplied by ``N_TIME`` so the figure
    matches the trajectory-level forward pass reported for MLP and FNO.

    Args:
        model: Trained model.
        model_name: ``'mlp'``, ``'pinn'``, or ``'fno'``.
        device: Inference device.

    Returns:
        FLOPs as ``int``, or ``None`` if ``fvcore`` is unavailable.

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    if model_name == "mlp":
        return measure_flops(model, (torch.zeros(1, N_PARAMS, device=device),))
    if model_name == "pinn":
        per_point = measure_flops(
            model,
            (
                torch.zeros(1, N_PARAMS, device=device),
                torch.zeros(1, 1, device=device),
            ),
        )
        return per_point * N_TIME if per_point is not None else None
    if model_name == "fno":
        return measure_flops(
            model, (torch.zeros(1, N_PARAMS + 1, N_TIME, device=device),)
        )
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Top-level evaluation.
# ---------------------------------------------------------------------------
def evaluate(
    ckpt_path: str, test_npz: str, device_str: str = "cuda",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Compute the full evaluation metric dict for one checkpoint.

    Args:
        ckpt_path: Path to a ``.pt`` checkpoint produced by :mod:`train`.
        test_npz: Path to test dataset (1000 trajectories).
        device_str: ``'cuda'`` or ``'cpu'``.
        batch_size: Inference batch size.

    Returns:
        JSON-serializable dict with overall metrics, per-stratum metrics,
        parameter count, inference time per trajectory, peak GPU memory at
        eval, FLOPs per forward, the index and prediction of the overlay
        sample (highest-r test trajectory), and the dataset time grid.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, name, payload = _load_checkpoint(ckpt_path, device)

    test = load_npz(test_npz)
    params = test["params"]
    traj = test["traj"]
    x_c = test["x_c"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if name == "mlp":
        nz = payload["normalizer"]
        norm = TrajectoryNormalizer(
            nz["param_mean"], nz["param_std"], nz["traj_mean"], nz["traj_std"]
        ).to(device)
        pred = _predict_mlp(model, norm, params, device, batch_size)
    elif name == "pinn":
        pred = _predict_pinn(model, params, device, batch_size)
    elif name == "fno":
        in_norm = ChannelNormalizer(
            payload["in_norm_mean"], payload["in_norm_std"]
        ).to(device)
        out_norm = ChannelNormalizer(
            payload["out_norm_mean"], payload["out_norm_std"]
        ).to(device)
        pred = _predict_fno(
            model, in_norm, out_norm, params, x_c, device, batch_size
        )
    else:
        raise ValueError(f"Unknown model_name: {name}")

    peak_mb_eval = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    overall = compute_metrics(pred, traj, indices=None)
    r = amplification_ratio(traj, x_c)
    strata = stratum_indices(r)
    per_stratum = {
        stratum_name: compute_metrics(pred, traj, indices=idx)
        for stratum_name, idx in strata.items()
    }

    inf_ms = _measure_inference_ms_for_model(
        model, name, payload, params, x_c, device
    )
    flops = _measure_flops_for_model(model, name, device)

    # Overlay sample: fixed for the whole sweep, picked from ground truth.
    overlay_idx = int(np.argmax(r))
    overlay_pred_sum = (pred[overlay_idx, :, 0] + pred[overlay_idx, :, 1]).tolist()
    overlay_true_sum = (traj[overlay_idx, :, 0] + traj[overlay_idx, :, 1]).tolist()
    t_grid_list = np.linspace(0.0, T_HORIZON, N_TIME, endpoint=False).tolist()

    return {
        "model": name,
        "ckpt": str(ckpt_path),
        "overall": overall,
        "per_stratum": per_stratum,
        "param_count": int(count_params(model)),
        "inference_time_per_traj_ms": float(inf_ms),
        "peak_gpu_memory_mb_eval": float(peak_mb_eval),
        "flops_per_forward": flops,
        "overlay_sample": {
            "index": overlay_idx,
            "amplification_ratio": float(r[overlay_idx]),
            "t": t_grid_list,
            "pred_sum": overlay_pred_sum,
            "true_sum": overlay_true_sum,
        },
    }


def main() -> None:
    """CLI entry point: evaluate a single checkpoint and print the metrics."""
    args = parse_args()
    result = evaluate(args.ckpt, args.test_npz, args.device, args.batch_size)
    # Drop the long overlay arrays from stdout.
    brief = {k: v for k, v in result.items() if k != "overlay_sample"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
