"""PINN for the 2-DoF mass-spring-damper system.

The network is a point-wise regressor ``f(params, t) -> (x_r, x_b)``. The
governing ODE from ``project2_concept.txt`` Section 2 is enforced through
an autograd-based residual loss at collocation points:

    R1 = (m_r + m_b) * x_b_ddot + m_r * x_r_ddot
         + d_b * x_b_dot + k_b * x_b
    R2 = m_r * x_b_ddot + m_r * x_r_ddot
         + d_r * x_r_dot + k_r * (x_r - x_c(t))

with ``d_r = 2 * zeta_r * sqrt(k_r * m_r)`` per trajectory and base
parameters fixed in :mod:`constants`. The chirp signal
``x_c(t) = offset + A * sin(2*pi * (f0*t + 0.5*(f1-f0)/T*t^2))`` is
recomputed from per-trajectory parameters because collocation points
generally do not lie on the dataset time grid.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn

from constants import D_B, K_B, M_B, M_R, PARAM_HIGH, PARAM_LOW, T_HORIZON


def chirp(
    t: Tensor,
    amp: Tensor,
    f0: Tensor,
    f1: Tensor,
    offset: Tensor,
    horizon: float = T_HORIZON,
) -> Tensor:
    """Evaluate the linear-in-time chirp signal ``x_c(t)``.

    Args:
        t: Time tensor of any shape broadcastable with the chirp parameters.
        amp: Chirp amplitude tensor, broadcastable with ``t``.
        f0: Start frequency [Hz], broadcastable with ``t``.
        f1: End frequency [Hz], broadcastable with ``t``.
        offset: Static offset, broadcastable with ``t``.
        horizon: Total simulation horizon ``T`` [s]. Defaults to
            :data:`constants.T_HORIZON`.

    Returns:
        Chirp signal ``offset + amp * sin(2*pi*(f0*t + 0.5*(f1-f0)/T*t**2))``
        with the broadcast shape.
    """
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) / horizon * t * t)
    return offset + amp * torch.sin(phase)


class PINN(nn.Module):
    """Feed-forward PINN mapping ``(params, t)`` to ``(x_r, x_b)``.

    Inputs are normalized internally: the 6 physical parameters are scaled
    to roughly ``[0, 1]`` using the sampling ranges in :mod:`constants`,
    and time using ``t / T_HORIZON``. The architecture follows the concept
    Section 4 spec: hidden ``[256, 256, 256]`` with GELU activations.

    Attributes:
        net: Backbone ``nn.Sequential``.
        hidden: Tuple of hidden layer widths.
        p_low: Buffer of parameter lower bounds.
        p_high: Buffer of parameter upper bounds.
    """

    def __init__(self, hidden: Tuple[int, ...] =(128, 128, 128)): #(256, 256, 256)) -> None:
        """Initialize the PINN.

        Args:
            hidden: Hidden layer widths. Defaults to ``(256, 256, 256)``.
        """
        super().__init__()
        self.hidden = tuple(hidden)
        layers: list[nn.Module] = []
        in_dim = 7
        for width in self.hidden:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.GELU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

        self.register_buffer(
            "p_low", torch.tensor(PARAM_LOW, dtype=torch.float32)
        )
        self.register_buffer(
            "p_high", torch.tensor(PARAM_HIGH, dtype=torch.float32)
        )

    def _norm_inputs(self, params: Tensor, t: Tensor) -> Tensor:
        """Concatenate normalized params and time into one feature vector.

        Args:
            params: Tensor of shape ``(N, 6)``.
            t: Tensor of shape ``(N, 1)``.

        Returns:
            Tensor of shape ``(N, 7)`` ready for ``self.net``.
        """
        p = (params - self.p_low) / (self.p_high - self.p_low)
        tn = t / T_HORIZON
        return torch.cat([p, tn], dim=-1)

    def forward(self, params: Tensor, t: Tensor) -> Tensor:
        """Predict ``(x_r, x_b)`` at query points.

        Args:
            params: Tensor of shape ``(N, 6)`` with columns
                ``[A, f0, f1, offset, k_r, zeta_r]``.
            t: Tensor of shape ``(N, 1)`` with query times in seconds.

        Returns:
            Tensor of shape ``(N, 2)`` containing ``(x_r, x_b)``.
        """
        return self.net(self._norm_inputs(params, t))

    def predict_trajectory(
        self, params: Tensor, t_grid: Tensor, chunk: int = 256
    ) -> Tensor:
        """Evaluate the PINN on a time grid for a batch of trajectories.

        Args:
            params: Tensor of shape ``(B, 6)``.
            t_grid: Tensor of shape ``(K,)`` with the query time grid.
            chunk: Maximum trajectories evaluated jointly to bound memory.
                Defaults to 256.

        Returns:
            Tensor of shape ``(B, K, 2)`` with ``(x_r, x_b)`` per time.
        """
        batch, k = params.shape[0], t_grid.shape[0]
        out = params.new_empty((batch, k, 2))
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            n = end - start
            p_rep = params[start:end].unsqueeze(1).expand(n, k, 6).reshape(-1, 6)
            t_rep = t_grid.view(1, k, 1).expand(n, k, 1).reshape(-1, 1)
            out[start:end] = self.forward(p_rep, t_rep).view(n, k, 2)
        return out


def ode_residual(
    model: PINN, params: Tensor, t: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute the two ODE residuals at collocation points via autograd.

    Both residuals follow ``project2_concept.txt`` Section 2:

        R1 = (m_r + m_b) * x_b_ddot + m_r * x_r_ddot
             + d_b * x_b_dot + k_b * x_b
        R2 = m_r * x_b_ddot + m_r * x_r_ddot
             + d_r * x_r_dot + k_r * (x_r - x_c(t))

    with ``d_r = 2 * zeta_r * sqrt(k_r * m_r)``.

    Args:
        model: PINN instance.
        params: Per-collocation-point parameters of shape ``(N, 6)``
            (typically broadcast from per-trajectory params).
        t: Tensor of shape ``(N, 1)`` with ``requires_grad=True``.

    Returns:
        Tuple ``(r1, r2)`` of tensors of shape ``(N, 1)`` with the two
        residuals.

    Raises:
        ValueError: If ``t.requires_grad`` is ``False``.
    """
    if not t.requires_grad:
        raise ValueError("t must have requires_grad=True for autograd residual")

    out = model(params, t)
    x_r = out[:, 0:1]
    x_b = out[:, 1:2]

    ones_r = torch.ones_like(x_r)
    ones_b = torch.ones_like(x_b)
    x_r_dot = torch.autograd.grad(
        x_r, t, grad_outputs=ones_r, create_graph=True, retain_graph=True
    )[0]
    x_b_dot = torch.autograd.grad(
        x_b, t, grad_outputs=ones_b, create_graph=True, retain_graph=True
    )[0]
    x_r_ddot = torch.autograd.grad(
        x_r_dot, t, grad_outputs=ones_r, create_graph=True, retain_graph=True
    )[0]
    x_b_ddot = torch.autograd.grad(
        x_b_dot, t, grad_outputs=ones_b, create_graph=True, retain_graph=True
    )[0]

    amp = params[:, 0:1]
    f0 = params[:, 1:2]
    f1 = params[:, 2:3]
    offset = params[:, 3:4]
    k_r = params[:, 4:5]
    zeta_r = params[:, 5:6]
    d_r = 2.0 * zeta_r * torch.sqrt(k_r * M_R)

    x_c = chirp(t, amp, f0, f1, offset)

    r1 = (M_R + M_B) * x_b_ddot + M_R * x_r_ddot + D_B * x_b_dot + K_B * x_b
    r2 = M_R * x_b_ddot + M_R * x_r_ddot + d_r * x_r_dot + k_r * (x_r - x_c)
    return r1, r2
