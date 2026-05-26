"""1D Fourier Neural Operator for two-mass-spring-damper trajectory prediction.

Architecture per ``project2_concept.txt`` Section 4:

- 4 spectral layers, modes = 16, width = 32, GELU activations.
- Input:  7 channels (chirp signal + 6 broadcast parameters), length 1000.
- Output: 2 channels (``x_r``, ``x_b``), length 1000.
- Conditioning: parameter broadcasting (no FiLM).

Inline implementation; no ``neuraloperator`` dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from constants import N_OUT_CHANNELS, N_PARAMS


class SpectralConv1d(nn.Module):
    """1D spectral convolution: complex linear map on the lowest Fourier modes.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes retained.
        weight: Complex weights stored as real ``(Cin, Cout, modes, 2)``.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        """Initialize spectral conv weights.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes: Number of Fourier modes retained
                (must be ``<= length // 2 + 1``).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``rFFT -> complex linear on low modes -> irFFT``.

        Args:
            x: Input tensor of shape ``(batch, in_channels, length)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, length)``.
        """
        batch_size, _, length = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                  # (B, Cin, L//2 + 1)
        weight = torch.view_as_complex(self.weight)       # (Cin, Cout, modes)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.size(-1),
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, : self.modes], weight
        )
        return torch.fft.irfft(out_ft, n=length, dim=-1)


class FNOBlock(nn.Module):
    """One FNO block: spectral path + 1x1 conv skip, summed and GELU-activated.

    Attributes:
        spectral: ``SpectralConv1d`` instance.
        pointwise: ``nn.Conv1d`` 1x1 skip.
    """

    def __init__(self, width: int, modes: int) -> None:
        """Initialize block.

        Args:
            width: Channel width.
            modes: Number of Fourier modes for the spectral path.
        """
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, width, length)``.

        Returns:
            Tensor of shape ``(batch, width, length)``.
        """
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FNO1d(nn.Module):
    """1D FNO mapping ``(chirp, broadcast params)`` to ``(x_r, x_b)`` curves.

    Attributes:
        lift: Channel-lifting 1x1 conv.
        blocks: List of ``FNOBlock`` instances.
        project: Projection MLP back to output channels.
    """

    def __init__(
        self,
        in_channels: int = N_PARAMS + 1,
        out_channels: int = N_OUT_CHANNELS,
        width: int = 32,
        modes: int = 16,
        n_layers: int = 4,
        proj_hidden: int = 128,
    ) -> None:
        """Initialize FNO.

        Args:
            in_channels: Number of input channels (chirp + broadcast params).
            out_channels: Number of output channels (``x_r``, ``x_b``).
            width: Channel width within spectral blocks.
            modes: Number of Fourier modes retained per spectral conv.
            n_layers: Number of stacked FNO blocks.
            proj_hidden: Hidden width of the projection MLP.
        """
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.Conv1d(width, proj_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(proj_hidden, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map input fields to trajectories.

        Args:
            x: Input tensor of shape ``(batch, in_channels, length)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, length)``.
        """
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


def build_fno_input(x_c: Tensor, params: Tensor) -> Tensor:
    """Concatenate the chirp signal with parameter channels broadcast over time.

    Args:
        x_c: Chirp signal of shape ``(batch, length)``.
        params: Per-trajectory parameters of shape ``(batch, P)``.

    Returns:
        Tensor of shape ``(batch, 1 + P, length)`` suitable as input to
        :class:`FNO1d`.
    """
    batch, length = x_c.shape
    p = params.unsqueeze(-1).expand(batch, params.size(-1), length)
    return torch.cat([x_c.unsqueeze(1), p], dim=1)
