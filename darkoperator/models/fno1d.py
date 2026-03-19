"""
Fourier Neural Operator — 1-D

Implements FNO-1D as described in:
  Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations"
  ICLR 2021.  https://arxiv.org/abs/2010.08895

Key idea
--------
An FNO layer combines:
  1. A spectral (global) path  – multiply chosen Fourier modes by learned complex weights
  2. A local (pointwise) path  – ordinary W x residual connection
  3. Activation (GELU)

The 1-D variant is ideal for learning solution operators of 1-D PDEs
(Burgers, heat equation, wave equation, …).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level building block
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """
    Spectral convolution in 1-D Fourier space.

    Truncates to *modes* lowest-frequency components, applies a learned
    complex-valued linear map in that truncated space, then pads back.

    Args:
        in_channels:  number of feature channels entering the layer
        out_channels: number of feature channels leaving the layer
        modes:        number of Fourier modes to retain (≤ n // 2 + 1)
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    # complex multiplication: (batch, in, modes) × (in, out, modes) → (batch, out, modes)
    @staticmethod
    def _compl_mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bim,iom->bom", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: real tensor of shape (batch, in_channels, n)

        Returns:
            real tensor of shape (batch, out_channels, n)
        """
        n = x.size(-1)

        # Forward FFT – only positive frequencies needed (rfft)
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, n//2+1)

        # Allocate output in frequency domain
        out_ft = torch.zeros(
            x.size(0), self.out_channels, n // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # Keep only the lowest *modes* frequencies
        out_ft[:, :, : self.modes] = self._compl_mul(
            x_ft[:, :, : self.modes], self.weights
        )

        # Inverse FFT back to physical space
        return torch.fft.irfft(out_ft, n=n, dim=-1)  # (batch, out_channels, n)


# ---------------------------------------------------------------------------
# FNO layer (one FNO "block")
# ---------------------------------------------------------------------------

class FNOBlock1d(nn.Module):
    """One FNO layer = spectral path + local path + activation + optional dropout."""

    def __init__(self, width: int, modes: int, dropout: float = 0.1):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm1d(width)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.gelu(self.norm(self.spectral(x) + self.local(x))))


# ---------------------------------------------------------------------------
# Full FNO-1D model
# ---------------------------------------------------------------------------

class FourierNeuralOperator1d(nn.Module):
    """
    1-D Fourier Neural Operator.

    Learns the solution operator  u₀(x)  →  u(x, T)  for 1-D PDEs.

    Architecture
    ------------
    Input lifting  →  N × FNO blocks  →  Output projection

    The input is a discretised function sampled at *n* points.
    The output is also a discretised function at the same *n* points.

    Args:
        modes:      number of Fourier modes to keep per layer
        width:      channel dimension inside the FNO layers
        n_layers:   number of FNO blocks
        in_channels:  channels of the input function
                      (e.g. 1 for a scalar PDE initial condition, or
                       2 if you append the spatial coordinate x)
        out_channels: channels of the predicted solution
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        in_channels: int = 2,   # function value + grid position
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lift input to channel dimension
        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)

        # FNO blocks
        self.blocks = nn.ModuleList(
            [FNOBlock1d(width, modes, dropout=dropout) for _ in range(n_layers)]
        )

        # Project back to output channels
        self.project = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width // 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (batch, in_channels, n)
               where n is the number of spatial grid points

        Returns:
            tensor of shape (batch, out_channels, n)
        """
        x = self.lift(x)          # (batch, width, n)

        for block in self.blocks:
            x = block(x)          # (batch, width, n)

        return self.project(x)    # (batch, out_channels, n)
