"""MLP baseline for trajectory state estimation.

Maps a 6-dim parameter vector ``[A, f0, f1, offset, k_r, zeta_r]`` to a
``(N_TIME, N_OUT_CHANNELS)`` trajectory of ``(x_r, x_b)``. Dimensions and
default architecture follow ``project2_concept.txt`` Section 4.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from constants import N_OUT_CHANNELS, N_PARAMS, N_TIME


class MLP(nn.Module):
    """Feed-forward network mapping parameters to a full trajectory.

    The architecture is three GELU-activated hidden layers of width 256.
    The output layer produces a ``N_TIME * N_OUT_CHANNELS``-dim vector
    which is reshaped to ``(N_TIME, N_OUT_CHANNELS)``.

    Attributes:
        backbone: ``nn.Sequential`` of Linear + GELU layers.
        head: Final Linear layer producing the flat trajectory vector.
        out_steps: Number of time steps in the output trajectory.
        out_channels: Number of output channels per time step.
    """

    def __init__(
        self,
        in_dim: int = N_PARAMS,
        hidden: Tuple[int, ...] = (256, 256, 256),
        out_steps: int = N_TIME,
        out_channels: int = N_OUT_CHANNELS,
    ) -> None:
        """Initialize the MLP.

        Args:
            in_dim: Number of input parameters.
            hidden: Widths of the hidden layers.
            out_steps: Number of time steps in the output trajectory.
            out_channels: Number of output channels per time step.
        """
        super().__init__()
        self.out_steps = out_steps
        self.out_channels = out_channels

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_steps * out_channels)

    def forward(self, params: Tensor) -> Tensor:
        """Forward pass.

        Args:
            params: Input parameter tensor of shape ``(B, in_dim)``.

        Returns:
            Trajectory tensor of shape ``(B, out_steps, out_channels)``.
        """
        h = self.backbone(params)
        y = self.head(h)
        return y.view(-1, self.out_steps, self.out_channels)
