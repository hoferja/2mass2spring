"""Shared utilities: dataset I/O, normalizers, metrics, timing helpers.

All training and eval scripts import from this module. Two normalizer
classes are provided because the MLP/PINN-style layout (B, T, C) and the
FNO-style layout (B, C, T) require different broadcasting:

* ``TrajectoryNormalizer``: scalar mean/std per parameter feature and per
  trajectory channel. Used by MLP and (optionally) PINN.
* ``ChannelNormalizer``: per-channel mean/std over time, shape ``(1, C, 1)``.
  Used by FNO on inputs and outputs.

Both inherit from ``nn.Module`` so ``.to(device)`` works uniformly.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from constants import STRATA


# ---------------------------------------------------------------------------
# Dataset I/O.
# ---------------------------------------------------------------------------
def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a dataset .npz with required fields ``params``, ``x_c``, ``traj``.

    Args:
        path: Path to the .npz file.

    Returns:
        Dict mapping field name to numpy array (includes optional fields
        such as ``vel`` if present).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If a required key is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    for k in ("params", "x_c", "traj"):
        if k not in data.files:
            raise KeyError(f"Missing key '{k}' in {path}")
    return {k: data[k] for k in data.files}


def nested_prefix(*arrays: np.ndarray, n: int) -> Tuple[np.ndarray, ...]:
    """Return the first ``n`` rows of each array (nested-prefix slice).

    Args:
        *arrays: Arrays sharing the same leading dimension.
        n: Prefix size.

    Returns:
        Tuple of arrays sliced to ``[:n]``.

    Raises:
        ValueError: If ``n`` exceeds the leading dimension of any input.
    """
    for a in arrays:
        if n > a.shape[0]:
            raise ValueError(f"n={n} exceeds dataset size {a.shape[0]}")
    return tuple(a[:n] for a in arrays)


# ---------------------------------------------------------------------------
# Normalizers.
# ---------------------------------------------------------------------------
class TrajectoryNormalizer(nn.Module):
    """Per-feature / per-channel z-score normalizer for MLP-style layouts.

    Stores statistics for parameters of shape ``(B, P)`` and trajectories
    of shape ``(B, T, C)``. All statistics are registered as buffers so
    ``.to(device)`` moves them appropriately.

    Attributes:
        param_mean: Buffer of shape ``(P,)``.
        param_std:  Buffer of shape ``(P,)``.
        traj_mean:  Buffer of shape ``(C,)``.
        traj_std:   Buffer of shape ``(C,)``.
    """

    def __init__(
        self,
        param_mean: Tensor,
        param_std: Tensor,
        traj_mean: Tensor,
        traj_std: Tensor,
        eps: float = 1e-8,
    ) -> None:
        """Initialize with precomputed statistics.

        Args:
            param_mean: Per-feature parameter mean, shape ``(P,)``.
            param_std: Per-feature parameter std, shape ``(P,)``.
            traj_mean: Per-channel trajectory mean, shape ``(C,)``.
            traj_std: Per-channel trajectory std, shape ``(C,)``.
            eps: Numerical floor for std values.
        """
        super().__init__()
        self.register_buffer("param_mean", param_mean.float())
        self.register_buffer("param_std", param_std.float().clamp_min(eps))
        self.register_buffer("traj_mean", traj_mean.float())
        self.register_buffer("traj_std", traj_std.float().clamp_min(eps))

    @classmethod
    def from_arrays(
        cls, params: np.ndarray, traj: np.ndarray
    ) -> "TrajectoryNormalizer":
        """Fit normalizer statistics from numpy training arrays.

        Args:
            params: Array of shape ``(N, P)``.
            traj: Array of shape ``(N, T, C)``.

        Returns:
            A fitted ``TrajectoryNormalizer`` (CPU buffers).
        """
        p_mean = torch.tensor(params.mean(axis=0), dtype=torch.float32)
        p_std = torch.tensor(params.std(axis=0), dtype=torch.float32)
        t_flat = traj.reshape(-1, traj.shape[-1])
        t_mean = torch.tensor(t_flat.mean(axis=0), dtype=torch.float32)
        t_std = torch.tensor(t_flat.std(axis=0), dtype=torch.float32)
        return cls(p_mean, p_std, t_mean, t_std)

    def norm_params(self, x: Tensor) -> Tensor:
        """Standardize parameter tensor of shape ``(..., P)``.

        Args:
            x: Parameter tensor.

        Returns:
            ``(x - param_mean) / param_std``.
        """
        return (x - self.param_mean) / self.param_std

    def norm_traj(self, x: Tensor) -> Tensor:
        """Standardize trajectory tensor of shape ``(..., C)``.

        Args:
            x: Trajectory tensor.

        Returns:
            ``(x - traj_mean) / traj_std``.
        """
        return (x - self.traj_mean) / self.traj_std

    def denorm_traj(self, x: Tensor) -> Tensor:
        """Inverse-standardize trajectory tensor of shape ``(..., C)``.

        Args:
            x: Normalized trajectory tensor.

        Returns:
            ``x * traj_std + traj_mean``.
        """
        return x * self.traj_std + self.traj_mean


class ChannelNormalizer(nn.Module):
    """Per-channel z-score normalizer for channel-first tensors ``(B, C, T)``.

    Mean and std are stored with shape ``(1, C, 1)`` so they broadcast over
    batch and time.

    Attributes:
        mean: Buffer of shape ``(1, C, 1)``.
        std:  Buffer of shape ``(1, C, 1)``.
    """

    def __init__(self, mean: Tensor, std: Tensor, eps: float = 1e-8) -> None:
        """Initialize with precomputed statistics.

        Args:
            mean: Per-channel mean of shape ``(C,)``, ``(1, C, 1)``, or
                another shape broadcastable with ``(B, C, T)``.
            std: Per-channel std with the same shape as ``mean``.
            eps: Numerical floor for std values.
        """
        super().__init__()
        if mean.ndim == 1:
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(eps))

    @classmethod
    def from_tensor(cls, x: Tensor) -> "ChannelNormalizer":
        """Fit per-channel statistics from a tensor of shape ``(B, C, T)``.

        Args:
            x: Channel-first tensor.

        Returns:
            A fitted ``ChannelNormalizer``.
        """
        m = x.mean(dim=(0, 2))
        s = x.std(dim=(0, 2))
        return cls(m, s)

    def encode(self, x: Tensor) -> Tensor:
        """Standardize ``x``: ``(x - mean) / std``.

        Args:
            x: Tensor of shape ``(B, C, T)``.

        Returns:
            Standardized tensor of the same shape.
        """
        return (x - self.mean) / self.std

    def decode(self, x: Tensor) -> Tensor:
        """Invert the standardization: ``x * std + mean``.

        Args:
            x: Normalized tensor of shape ``(B, C, T)``.

        Returns:
            Tensor in original units.
        """
        return x * self.std + self.mean


# ---------------------------------------------------------------------------
# Dataset wrapper (used by MLP training).
# ---------------------------------------------------------------------------
class TrajectoryDataset(Dataset):
    """Torch dataset returning normalized ``(params, traj)`` pairs.

    Layout matches the MLP convention: params shape ``(P,)``, trajectory
    shape ``(T, C)`` per sample.

    Attributes:
        params: Tensor of shape ``(N, P)``.
        traj: Tensor of shape ``(N, T, C)``.
        normalizer: Fitted ``TrajectoryNormalizer``.
    """

    def __init__(
        self,
        params: np.ndarray,
        traj: np.ndarray,
        normalizer: TrajectoryNormalizer,
    ) -> None:
        """Initialize the dataset.

        Args:
            params: Array of shape ``(N, P)``.
            traj: Array of shape ``(N, T, C)``.
            normalizer: Fitted ``TrajectoryNormalizer``.
        """
        self.params = torch.tensor(params, dtype=torch.float32)
        self.traj = torch.tensor(traj, dtype=torch.float32)
        self.normalizer = normalizer

    def __len__(self) -> int:
        """Return the number of trajectories.

        Returns:
            Integer length.
        """
        return self.params.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return normalized ``(params, traj)`` at index ``idx``.

        Args:
            idx: Sample index.

        Returns:
            Tuple of normalized parameter and trajectory tensors.
        """
        p = self.normalizer.norm_params(self.params[idx])
        y = self.normalizer.norm_traj(self.traj[idx])
        return p, y


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def amplification_ratio(traj: np.ndarray, x_c: np.ndarray) -> np.ndarray:
    """Compute per-trajectory amplification ratio.

    ``r = max_t (x_r(t) + x_b(t)) / max_t |x_c(t)|`` per concept Section 5.

    Args:
        traj: Trajectories of shape ``(N, T, 2)`` with columns
            ``(x_r, x_b)``.
        x_c: Chirp signal of shape ``(N, T)``.

    Returns:
        Array of shape ``(N,)`` with the amplification ratios. Guarded
        against zero denominators with a floor of ``1e-12``.
    """
    peak_sum = (traj[..., 0] + traj[..., 1]).max(axis=-1)
    peak_xc = np.abs(x_c).max(axis=-1)
    return peak_sum / np.maximum(peak_xc, 1e-12)


def stratum_indices(r: np.ndarray) -> dict[str, np.ndarray]:
    """Group sample indices by amplification stratum.

    Strata follow ``constants.STRATA``.

    Args:
        r: Per-sample amplification ratios of shape ``(N,)``.

    Returns:
        Dict mapping stratum name to integer index array.
    """
    out: dict[str, np.ndarray] = {}
    for name, lo, hi in STRATA:
        mask = (r >= lo) & (r < hi)
        out[name] = np.nonzero(mask)[0]
    return out


def compute_metrics(
    pred_traj: np.ndarray,
    true_traj: np.ndarray,
    indices: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute trajectory RMSE, peak RMSE, signed peak bias, and sample count.

    Peak is taken on the end-effector position ``x_r + x_b``. Signed bias
    is ``mean(pred_peak - true_peak)``; negative values indicate systematic
    under-prediction of peaks.

    Args:
        pred_traj: Predicted trajectories of shape ``(N, T, 2)``.
        true_traj: Ground-truth trajectories of the same shape.
        indices: Optional integer indices selecting a subset. ``None``
            selects all rows.

    Returns:
        Dict with keys ``rmse_traj``, ``rmse_peak``, ``bias_peak``,
        ``count``. Returns zeros and ``count=0`` if the selection is
        empty.
    """
    if indices is not None:
        pred = pred_traj[indices]
        true = true_traj[indices]
    else:
        pred = pred_traj
        true = true_traj

    n = pred.shape[0]
    if n == 0:
        return {"rmse_traj": 0.0, "rmse_peak": 0.0, "bias_peak": 0.0, "count": 0}

    rmse_traj = float(np.sqrt(np.mean((pred - true) ** 2)))
    pred_peak = (pred[..., 0] + pred[..., 1]).max(axis=-1)
    true_peak = (true[..., 0] + true[..., 1]).max(axis=-1)
    rmse_peak = float(np.sqrt(np.mean((pred_peak - true_peak) ** 2)))
    bias_peak = float(np.mean(pred_peak - true_peak))
    return {
        "rmse_traj": rmse_traj,
        "rmse_peak": rmse_peak,
        "bias_peak": bias_peak,
        "count": int(n),
    }


# ---------------------------------------------------------------------------
# Miscellaneous helpers.
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch RNG seeds.

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in a module.

    Args:
        model: A ``torch.nn.Module``.

    Returns:
        Integer count of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_ms(
    forward_fn: Callable[..., Any],
    *args: Any,
    n_warmup: int = 5,
    n_trials: int = 50,
    device: Optional[torch.device] = None,
) -> float:
    """Time a single forward pass averaged over multiple trials.

    Args:
        forward_fn: Callable producing one forward pass when invoked as
            ``forward_fn(*args)``.
        *args: Positional arguments passed to ``forward_fn`` each call.
        n_warmup: Warmup calls discarded before timing.
        n_trials: Number of timed calls.
        device: Optional torch device. If CUDA, ``torch.cuda.synchronize``
            is used before and after the timed block.

    Returns:
        Mean wall-clock time per call in milliseconds.
    """
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = forward_fn(*args)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            _ = forward_fn(*args)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / n_trials


def measure_flops(model: nn.Module, sample_args: tuple) -> Optional[int]:
    """Estimate FLOPs per forward pass via ``fvcore``, if available.

    Args:
        model: Model to profile.
        sample_args: Tuple of positional inputs forwarded to ``model``.

    Returns:
        Integer FLOP count, or ``None`` if ``fvcore`` is unavailable or
        fails.
    """
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        flops = FlopCountAnalysis(model, sample_args)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return int(flops.total())
    except Exception:
        return None
