"""Generate trajectory dataset for the 2-DoF mass-spring-damper system.

Implements the data generation pipeline of ``project2_concept.txt``
Sections 2 and 3: linear-in-time chirp excitation of a coupled robot/base
mass-spring-damper system with no external force, integrated with
``scipy.integrate.solve_ivp`` (LSODA, rtol=1e-7, atol=1e-9). Parameters
are drawn by Latin-hypercube sampling.

Run as a script from the project root:

    python data/generate.py --n_samples 5000 --seed 0 \\
        --out data/dataset_train.npz
    python data/generate.py --n_samples 1000 --seed 1 \\
        --out data/dataset_test.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Make the project root importable when running this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc

from constants import (
    D_B,
    FS,
    K_B,
    M_B,
    M_R,
    N_TIME,
    PARAM_BOUNDS,
    T_HORIZON,
)


# ---------------------------------------------------------------------------
# Chirp signal.
# ---------------------------------------------------------------------------
def chirp_signal(
    t: np.ndarray,
    A: float,
    f0: float,
    f1: float,
    offset: float,
    T: float = T_HORIZON,
) -> np.ndarray:
    """Compute a linear-in-time chirp signal.

    Args:
        t: Time array in seconds.
        A: Amplitude in metres.
        f0: Start frequency in Hz.
        f1: End frequency in Hz.
        offset: Constant offset in metres.
        T: Total horizon in seconds.

    Returns:
        Chirp signal samples evaluated at ``t``.
    """
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t**2)
    return offset + A * np.sin(phase)


def _chirp_scalar(
    t: float,
    A: float,
    f0: float,
    f1: float,
    offset: float,
    T: float,
) -> float:
    """Evaluate the chirp signal at a single time (used inside the ODE RHS).

    Args:
        t: Time in seconds.
        A: Amplitude in metres.
        f0: Start frequency in Hz.
        f1: End frequency in Hz.
        offset: Constant offset in metres.
        T: Total horizon in seconds.

    Returns:
        Chirp value at time ``t``.
    """
    return offset + A * np.sin(2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t * t))


# ---------------------------------------------------------------------------
# 2-DoF ODE right-hand side.
# ---------------------------------------------------------------------------
def ode_rhs(
    t: float,
    state: np.ndarray,
    A: float,
    f0: float,
    f1: float,
    offset: float,
    k_r: float,
    d_r: float,
    T: float,
) -> np.ndarray:
    """Compute the time derivative of the 4-D state.

    Equations (``project2_concept`` Section 2, no environment, no external
    force)::

        (m_r + m_b) x_b'' + m_r x_r'' + d_b x_b' + k_b x_b = 0
        m_r x_b''         + m_r x_r'' + d_r x_r' + k_r (x_r - x_c) = 0

    Subtracting the second equation from the first decouples the
    accelerations::

        x_b'' = (d_r x_r' + k_r (x_r - x_c) - d_b x_b' - k_b x_b) / m_b
        x_r'' = -x_b'' - (d_r x_r' + k_r (x_r - x_c)) / m_r

    State ordering is ``[x_b, x_r, x_b_dot, x_r_dot]``.

    Args:
        t: Current time in seconds.
        state: State vector ``[x_b, x_r, x_b_dot, x_r_dot]``.
        A: Chirp amplitude in metres.
        f0: Chirp start frequency in Hz.
        f1: Chirp end frequency in Hz.
        offset: Chirp offset in metres.
        k_r: Robot stiffness in N/m.
        d_r: Robot damping in Ns/m.
        T: Total time horizon in seconds.

    Returns:
        Time derivative of the state, shape ``(4,)``.
    """
    x_b, x_r, x_b_dot, x_r_dot = state
    x_c = _chirp_scalar(t, A, f0, f1, offset, T)
    spring_robot = k_r * (x_r - x_c)
    damp_robot = d_r * x_r_dot
    x_b_ddot = (damp_robot + spring_robot - D_B * x_b_dot - K_B * x_b) / M_B
    x_r_ddot = -x_b_ddot - (damp_robot + spring_robot) / M_R
    return np.array([x_b_dot, x_r_dot, x_b_ddot, x_r_ddot], dtype=np.float64)


# ---------------------------------------------------------------------------
# Single-trajectory simulation.
# ---------------------------------------------------------------------------
def simulate_trajectory(
    params: np.ndarray,
    T: float = T_HORIZON,
    fs: float = FS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one trajectory of the 2-DoF system under chirp excitation.

    Args:
        params: 6-vector ``[A, f0, f1, offset, k_r, zeta_r]``.
        T: Total horizon in seconds.
        fs: Sampling frequency in Hz.

    Returns:
        Tuple ``(x_c, traj, vel)`` where ``x_c`` has shape ``(N,)``,
        ``traj`` has shape ``(N, 2)`` and columns ``(x_r, x_b)``, and
        ``vel`` has shape ``(N, 2)`` and columns ``(x_r_dot, x_b_dot)``.

    Raises:
        RuntimeError: If the ODE integration fails.
    """
    A, f0, f1, offset, k_r, zeta_r = params
    d_r = 2.0 * zeta_r * np.sqrt(k_r * M_R)
    n = int(T * fs)
    t_eval = np.linspace(0.0, T, n, endpoint=False)
    state0 = np.zeros(4, dtype=np.float64)
    sol = solve_ivp(
        fun=ode_rhs,
        t_span=(0.0, T),
        y0=state0,
        method="LSODA",
        t_eval=t_eval,
        args=(A, f0, f1, offset, k_r, d_r, T),
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    x_b, x_r, x_b_dot, x_r_dot = sol.y
    x_c = chirp_signal(t_eval, A, f0, f1, offset, T)
    traj = np.stack([x_r, x_b], axis=1)
    vel = np.stack([x_r_dot, x_b_dot], axis=1)
    return x_c, traj, vel


# ---------------------------------------------------------------------------
# Parameter sampling.
# ---------------------------------------------------------------------------
def sample_parameters(n_samples: int, seed: int) -> np.ndarray:
    """Latin-hypercube sample of the 6 parameters within their bounds.

    Args:
        n_samples: Number of parameter tuples to draw.
        seed: Random seed for the LHS engine.

    Returns:
        Sampled parameters of shape ``(n_samples, 6)``.
    """
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    u = sampler.random(n=n_samples)
    return qmc.scale(u, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


# ---------------------------------------------------------------------------
# Full dataset generation.
# ---------------------------------------------------------------------------
def generate_dataset(
    n_samples: int,
    seed: int,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dataset of trajectories.

    Args:
        n_samples: Number of trajectories.
        seed: Random seed for LHS and noise.
        noise_std: Standard deviation of Gaussian noise added to
            ``(x_r, x_b)`` after simulation. Set to 0 for a clean dataset.

    Returns:
        Tuple ``(params, x_c_all, traj_all, vel_all)`` with shapes
        ``(N, 6)``, ``(N, 1000)``, ``(N, 1000, 2)``, ``(N, 1000, 2)``.
    """
    params = sample_parameters(n_samples, seed)
    x_c_all = np.zeros((n_samples, N_TIME), dtype=np.float64)
    traj_all = np.zeros((n_samples, N_TIME, 2), dtype=np.float64)
    vel_all = np.zeros((n_samples, N_TIME, 2), dtype=np.float64)

    for i in range(n_samples):
        x_c, traj, vel = simulate_trajectory(params[i])
        x_c_all[i] = x_c
        traj_all[i] = traj
        vel_all[i] = vel
        if (i + 1) % 500 == 0:
            print(f"  ... simulated {i + 1}/{n_samples}")

    if noise_std > 0.0:
        rng = np.random.default_rng(seed + 99991)
        traj_all = traj_all + rng.normal(0.0, noise_std, size=traj_all.shape)

    return params, x_c_all, traj_all, vel_all


# ---------------------------------------------------------------------------
# Coverage check.
# ---------------------------------------------------------------------------
def coverage_check(traj_all: np.ndarray, x_c_all: np.ndarray) -> Dict[str, float]:
    """Compute amplification-ratio statistics over a generated dataset.

    The amplification ratio per trajectory is
    ``r = max_t (x_r + x_b) / max_t |x_c|`` (concept Section 5).

    Args:
        traj_all: Trajectories of shape ``(N, T, 2)`` with channels
            ``(x_r, x_b)``.
        x_c_all: Chirp signals of shape ``(N, T)``.

    Returns:
        Mapping with keys ``r_min``, ``r_mean``, ``r_max``, ``r_p95``,
        ``frac_r_gt_1.0``, ``frac_r_gt_1.2``, ``frac_r_gt_1.5``.
    """
    end_eff_max = np.max(traj_all[..., 0] + traj_all[..., 1], axis=1)
    xc_max = np.max(np.abs(x_c_all), axis=1)
    safe = xc_max > 1e-12
    r = np.full(len(traj_all), np.nan, dtype=np.float64)
    r[safe] = end_eff_max[safe] / xc_max[safe]
    r_f = r[np.isfinite(r)]
    return {
        "r_min": float(r_f.min()),
        "r_mean": float(r_f.mean()),
        "r_max": float(r_f.max()),
        "r_p95": float(np.percentile(r_f, 95)),
        "frac_r_gt_1.0": float((r_f > 1.0).mean()),
        "frac_r_gt_1.2": float((r_f > 1.2).mean()),
        "frac_r_gt_1.5": float((r_f > 1.5).mean()),
    }


# ---------------------------------------------------------------------------
# Verification block (single trajectory).
# ---------------------------------------------------------------------------
def _verify_single_trajectory() -> None:
    """Run one trajectory and assert basic sanity properties.

    Raises:
        AssertionError: If shape, finiteness, or energy-boundedness fails.
    """
    test_params = np.array([0.05, 0.5, 3.0, 0.0, 1000.0, 0.30], dtype=np.float64)
    x_c, traj, vel = simulate_trajectory(test_params)
    state = np.concatenate([traj, vel], axis=1)
    assert state.shape == (N_TIME, 4), f"state.shape={state.shape}, expected ({N_TIME}, 4)"

    end_eff = traj[:, 0] + traj[:, 1]
    assert np.all(np.isfinite(end_eff)), "non-finite end-effector trajectory"

    A, f0, f1, offset, k_r, zeta_r = test_params
    d_r = 2.0 * zeta_r * np.sqrt(k_r * M_R)
    x_r = traj[:, 0]
    x_b = traj[:, 1]
    x_r_dot = vel[:, 0]
    x_b_dot = vel[:, 1]
    t_eval = np.linspace(0.0, T_HORIZON, N_TIME, endpoint=False)
    xc = chirp_signal(t_eval, A, f0, f1, offset)
    kinetic = 0.5 * M_R * (x_b_dot + x_r_dot) ** 2 + 0.5 * M_B * x_b_dot**2
    potential = 0.5 * k_r * (x_r - xc) ** 2 + 0.5 * K_B * x_b**2
    energy = kinetic + potential
    assert np.all(np.isfinite(energy)), "energy contains non-finite values"
    assert energy.max() < 1.0e6, f"energy unbounded: max={energy.max():.3e} J"
    print(
        f"[verify] OK  state.shape={state.shape}  "
        f"end_eff range=[{end_eff.min():.4f}, {end_eff.max():.4f}] m  "
        f"energy.max={energy.max():.3e} J"
    )


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_samples", type=int, required=True, help="number of trajectories")
    p.add_argument("--seed", type=int, required=True, help="RNG seed for LHS/noise")
    p.add_argument(
        "--noise_std", type=float, default=0.0,
        help="Gaussian noise std on (x_r, x_b)",
    )
    p.add_argument("--out", type=Path, required=True, help="output .npz path")
    return p.parse_args()


def main() -> None:
    """Generate a dataset and run the post-generation coverage check.

    Raises:
        AssertionError: If fewer than 10 percent of trajectories have
            ``r > 1.2``.
    """
    args = _parse_args()
    print(
        f"[generate] n_samples={args.n_samples} seed={args.seed} "
        f"noise_std={args.noise_std} out={args.out}"
    )

    print("[generate] single-trajectory verification...")
    _verify_single_trajectory()

    t0 = time.perf_counter()
    params, x_c_all, traj_all, vel_all = generate_dataset(
        args.n_samples, args.seed, args.noise_std
    )
    wall = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        params=params,
        x_c=x_c_all,
        traj=traj_all,
        vel=vel_all,
    )
    size_mb = args.out.stat().st_size / 1.0e6
    print(
        f"[generate] params={params.shape} x_c={x_c_all.shape} "
        f"traj={traj_all.shape} vel={vel_all.shape}  "
        f"wall={wall:.1f}s  file={size_mb:.1f} MB"
    )

    print("[generate] coverage check (amplification ratio r):")
    stats = coverage_check(traj_all, x_c_all)
    for k, v in stats.items():
        print(f"    {k:>16s} = {v:.4f}")

    assert stats["frac_r_gt_1.2"] >= 0.10, (
        f"only {stats['frac_r_gt_1.2'] * 100:.1f}% of trajectories have r > 1.2; "
        "widen sampling ranges and regenerate"
    )
    print("[generate] done.")


if __name__ == "__main__":
    main()
