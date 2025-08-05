"""Calorimeter operator for electromagnetic and hadronic shower simulation."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np

from .base import PhysicsOperator
from ..models.fno import SpectralConv3d


class CalorimeterOperator(PhysicsOperator):
    """
    Neural operator for calorimeter shower simulation.
    
    Learns the mapping from particle 4-vectors to 3D energy deposits
    in electromagnetic and hadronic calorimeters.
    """
    
    def __init__(
        self,
        input_dim: int = 4,  # 4-momentum
        modes: int = 32,
        width: int = 64,
        output_shape: Tuple[int, int, int] = (50, 50, 25), # (eta, phi, depth)
        n_layers: int = 4,
        **kwargs
    ):
        super().__init__(input_dim, output_shape, **kwargs)
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        
        # Input embedding for 4-vectors
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, width)
        )
        
        # Spectral convolution layers
        self.spectral_convs = nn.ModuleList([
            SpectralConv3d(width, width, modes, modes, modes//2) 
            for _ in range(n_layers)
        ])
        
        # Local convolutions for non-linear interactions
        self.local_convs = nn.ModuleList([
            nn.Conv3d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv3d(width, width // 2, 1),
            nn.GELU(),
            nn.Conv3d(width // 2, 1, 1),  # Single channel for energy density
            nn.ReLU()  # Ensure positive energy deposits
        )
        
        # Positional encoding for detector geometry
        self.register_buffer('pos_encoding', self._create_positional_encoding())
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create 3D positional encoding for detector geometry."""
        eta_dim, phi_dim, depth_dim = self.output_shape
        
        # Create coordinate grids
        eta = torch.linspace(-2.5, 2.5, eta_dim)  # Pseudorapidity range
        phi = torch.linspace(-np.pi, np.pi, phi_dim)  # Azimuthal angle
        depth = torch.linspace(0, 1, depth_dim)  # Normalized depth
        
        eta_grid, phi_grid, depth_grid = torch.meshgrid(eta, phi, depth, indexing='ij')
        
        # Stack coordinates
        pos_encoding = torch.stack([eta_grid, phi_grid, depth_grid], dim=0)
        return pos_encoding.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through calorimeter operator.
        
        Args:
            x: Input 4-vectors (batch, n_particles, 4) - (E, px, py, pz)
            
        Returns:
            Energy deposits (batch, eta, phi, depth)
        """
        batch_size = x.shape[0]
        
        # Convert particles to initial spatial distribution
        x_embedded = self.input_embedding(x)  # (batch, n_particles, width)
        
        # Project particles onto detector grid
        x_spatial = self._project_to_grid(x, x_embedded)  # (batch, width, eta, phi, depth)
        
        # Add positional encoding
        pos_enc = self.pos_encoding.expand(batch_size, -1, -1, -1, -1)
        x_spatial = torch.cat([x_spatial, pos_enc], dim=1)
        
        # Pad to maintain width dimension
        if x_spatial.shape[1] < self.width:
            padding = torch.zeros(
                batch_size, self.width - x_spatial.shape[1], *self.output_shape,
                device=x_spatial.device
            )
            x_spatial = torch.cat([x_spatial, padding], dim=1)
        
        # Apply spectral convolutions
        for i in range(self.n_layers):
            residual = x_spatial
            x_spatial = self.spectral_convs[i](x_spatial)
            x_spatial = x_spatial + self.local_convs[i](residual)
            x_spatial = torch.nn.functional.gelu(x_spatial)
        
        # Output projection
        output = self.output_projection(x_spatial)  # (batch, 1, eta, phi, depth)
        output = output.squeeze(1)  # Remove channel dimension
        
        return output
    
    def _project_to_grid(self, particles: torch.Tensor, embedded: torch.Tensor) -> torch.Tensor:
        """Project particles onto detector grid using their kinematics."""
        batch_size, n_particles, _ = particles.shape
        eta_dim, phi_dim, depth_dim = self.output_shape
        
        # Extract kinematics
        energy = particles[:, :, 0]  # (batch, n_particles)
        px = particles[:, :, 1]
        py = particles[:, :, 2]
        pz = particles[:, :, 3]
        
        # Compute pseudorapidity and azimuthal angle
        pt = torch.sqrt(px**2 + py**2)
        eta = torch.asinh(pz / (pt + 1e-8))  # Avoid division by zero
        phi = torch.atan2(py, px)
        
        # Normalize to grid coordinates
        eta_normalized = (eta + 2.5) / 5.0  # Map [-2.5, 2.5] to [0, 1]
        phi_normalized = (phi + np.pi) / (2 * np.pi)  # Map [-pi, pi] to [0, 1]
        
        # Initialize output tensor
        x_spatial = torch.zeros(
            batch_size, self.width, eta_dim, phi_dim, depth_dim,
            device=particles.device
        )
        
        # Deposit energy at grid points (simplified deposition)
        for b in range(batch_size):
            for p in range(n_particles):
                if energy[b, p] > 0:  # Only process particles with positive energy
                    # Find nearest grid points
                    eta_idx = torch.clamp(
                        (eta_normalized[b, p] * eta_dim).long(), 0, eta_dim - 1
                    )
                    phi_idx = torch.clamp(
                        (phi_normalized[b, p] * phi_dim).long(), 0, phi_dim - 1
                    )
                    
                    # Deposit energy (simplified - in reality would use shower profiles)
                    energy_weight = energy[b, p] / torch.sum(energy[b] + 1e-8)
                    for d in range(depth_dim):
                        depth_weight = torch.exp(-d / 10.0)  # Exponential shower profile
                        x_spatial[b, :, eta_idx, phi_idx, d] += (
                            embedded[b, p, :] * energy_weight * depth_weight
                        )
        
        return x_spatial