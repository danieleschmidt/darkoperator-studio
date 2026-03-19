"""
Synthetic PDE data generators for 1-D physics surrogate training.

Two classic problems are included:

1. **Heat equation** (linear, diffusive)
     ∂u/∂t = ν ∂²u/∂x²,   x∈[0,1],  t∈[0,T]
   With periodic BCs and a random Fourier-mode initial condition.
   Exact solution via FFT.

2. **Burgers' equation** (nonlinear, convection-diffusion)
     ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²,   x∈[0,1], t∈[0,T]
   With periodic BCs.  Solved numerically using a pseudo-spectral
   Crank–Nicolson / Adams-Bashforth integrator.

Both generators return:
  (u0, uT)  – initial and final fields, shape (N, n)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_fourier_ic(
    n: int,
    n_modes: int = 10,
    amplitude: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Random initial condition as a sum of low-frequency Fourier modes.

    Returns a real-valued array of shape (n,) on the grid x = [0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    k = np.arange(1, n_modes + 1)
    a = rng.uniform(-amplitude, amplitude, size=n_modes)
    b = rng.uniform(-amplitude, amplitude, size=n_modes)

    x = np.linspace(0, 1, n, endpoint=False)
    u = np.zeros(n)
    for i, ki in enumerate(k):
        u += a[i] * np.cos(2 * np.pi * ki * x) + b[i] * np.sin(2 * np.pi * ki * x)

    return u / (n_modes)  # normalise


# ---------------------------------------------------------------------------
# Heat equation (exact)
# ---------------------------------------------------------------------------

def generate_heat_data(
    n_samples: int = 1000,
    n_grid: int = 128,
    nu: float = 0.01,
    T: float = 1.0,
    n_modes_ic: int = 10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (u0, uT) pairs for the 1-D heat equation.

    The exact solution is  û_k(T) = û_k(0) · exp(-ν (2πk)² T),
    where ^  denotes the Fourier coefficient.

    Args:
        n_samples:    number of (u0, uT) pairs to generate
        n_grid:       spatial resolution (number of grid points)
        nu:           diffusivity coefficient
        T:            final time
        n_modes_ic:   number of Fourier modes in the random IC
        seed:         random seed

    Returns:
        u0: (n_samples, n_grid) initial conditions
        uT: (n_samples, n_grid) solutions at time T
    """
    rng = np.random.default_rng(seed)

    # Wavenumbers for n_grid periodic grid
    k = np.fft.rfftfreq(n_grid, d=1.0 / n_grid)  # [0, 1, 2, …, n//2]

    decay = np.exp(-nu * (2 * np.pi * k) ** 2 * T)

    u0_all = np.empty((n_samples, n_grid), dtype=np.float32)
    uT_all = np.empty((n_samples, n_grid), dtype=np.float32)

    for i in range(n_samples):
        u0 = _random_fourier_ic(n_grid, n_modes=n_modes_ic, rng=rng)
        u0_hat = np.fft.rfft(u0)
        uT = np.fft.irfft(u0_hat * decay, n=n_grid)

        u0_all[i] = u0.astype(np.float32)
        uT_all[i] = uT.astype(np.float32)

    return torch.tensor(u0_all), torch.tensor(uT_all)


# ---------------------------------------------------------------------------
# Burgers' equation (pseudo-spectral, 2nd-order in time)
# ---------------------------------------------------------------------------

def _burgers_nonlinear(u_hat: np.ndarray, kk: np.ndarray) -> np.ndarray:
    """
    Compute the nonlinear advection term  -u ∂u/∂x  in spectral space.

    Uses a pseudo-spectral approach with 3/2 de-aliasing.
    kk: physical wavenumbers (2π * [0,1,...,n/2]).
    """
    n_modes = len(u_hat)
    n = (n_modes - 1) * 2  # original grid size

    # 3/2 zero-padding for de-aliasing
    n_pad = 3 * n // 2
    u_hat_pad = np.zeros(n_pad // 2 + 1, dtype=complex)
    u_hat_pad[:n_modes] = u_hat

    # ∂u/∂x in padded physical space
    du_hat_pad = np.zeros_like(u_hat_pad)
    kk_pad = 2 * np.pi * np.fft.rfftfreq(n_pad) * n_pad
    kk_pad_trunc = kk_pad[:n_modes]
    du_hat_pad[:n_modes] = 1j * kk_pad_trunc * u_hat   # derivative of u

    u_phys   = np.fft.irfft(u_hat_pad, n=n_pad)
    du_phys  = np.fft.irfft(du_hat_pad, n=n_pad)

    # product in physical space, transform back, truncate, scale
    udu_hat_pad = np.fft.rfft(u_phys * du_phys)
    udu_hat     = udu_hat_pad[:n_modes] * (n / n_pad)

    return -udu_hat   # advection: -u ∂u/∂x


def _burgers_step(
    u_hat: np.ndarray,
    kk: np.ndarray,
    nu: float,
    dt: float,
) -> np.ndarray:
    """
    Semi-implicit (IMEX) step for Burgers' equation.

    Diffusion treated implicitly (Crank–Nicolson), advection explicitly.
    This is unconditionally stable for the diffusive part.

    kk: physical wavenumbers array of length n//2+1.
    """
    D = nu * kk ** 2  # diffusion operator (real, non-negative)

    N = _burgers_nonlinear(u_hat, kk)  # explicit nonlinear term

    # Crank–Nicolson: (1 + dt/2 * D) u^{n+1} = (1 - dt/2 * D) u^n + dt * N
    num   = (1.0 - 0.5 * dt * D) * u_hat + dt * N
    denom = 1.0 + 0.5 * dt * D
    return num / denom


def generate_burgers_data(
    n_samples: int = 1000,
    n_grid: int = 128,
    nu: float = 0.01,
    T: float = 1.0,
    n_steps: int = 200,
    n_modes_ic: int = 8,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (u0, uT) pairs for the viscous Burgers' equation.

    Args:
        n_samples:  number of pairs
        n_grid:     spatial resolution
        nu:         viscosity
        T:          final time
        n_steps:    number of time-integration steps
        n_modes_ic: number of Fourier modes in IC
        seed:       random seed

    Returns:
        u0: (n_samples, n_grid) initial conditions
        uT: (n_samples, n_grid) solutions at time T
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Physical wavenumbers: 2π × [0, 1, …, n//2]
    kk = 2 * np.pi * np.fft.rfftfreq(n_grid, d=1.0 / n_grid)

    u0_all = np.empty((n_samples, n_grid), dtype=np.float32)
    uT_all = np.empty((n_samples, n_grid), dtype=np.float32)

    for i in range(n_samples):
        u0 = _random_fourier_ic(n_grid, n_modes=n_modes_ic, rng=rng)
        u_hat = np.fft.rfft(u0)

        for _ in range(n_steps):
            u_hat = _burgers_step(u_hat, kk, nu, dt)

        uT = np.fft.irfft(u_hat, n=n_grid).real

        u0_all[i] = u0.astype(np.float32)
        uT_all[i] = uT.astype(np.float32)

    return torch.tensor(u0_all), torch.tensor(uT_all)
