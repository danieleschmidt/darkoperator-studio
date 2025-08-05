"""Base classes for physics-informed neural operators."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class PhysicsOperator(nn.Module, ABC):
    """
    Base class for physics-informed neural operators.
    
    Enforces conservation laws and symmetries in detector simulations.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_shape: Tuple[int, ...],
        preserve_energy: bool = True,
        preserve_momentum: bool = True,
        device: str = "auto"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.preserve_energy = preserve_energy
        self.preserve_momentum = preserve_momentum
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.to(self.device)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the operator."""
        pass
    
    def physics_loss(self, input_4vec: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss enforcing conservation laws.
        
        Args:
            input_4vec: Input 4-vectors (batch, n_particles, 4) - (E, px, py, pz)
            output: Output detector response (batch, *output_shape)
            
        Returns:
            Physics loss combining energy and momentum conservation
        """
        loss = torch.tensor(0.0, device=self.device)
        
        if self.preserve_energy:
            # Energy conservation: sum of input energies should match total output energy
            input_energy = input_4vec[:, :, 0].sum(dim=1)  # Sum over particles
            output_energy = output.sum(dim=tuple(range(1, len(output.shape))))  # Sum over spatial dims
            energy_loss = torch.mean((input_energy - output_energy) ** 2)
            loss += energy_loss
            
        if self.preserve_momentum:
            # Momentum conservation in transverse plane
            input_px = input_4vec[:, :, 1].sum(dim=1)
            input_py = input_4vec[:, :, 2].sum(dim=1)
            
            # Compute center of mass of output distribution
            batch_size = output.shape[0]
            output_px, output_py = self._compute_momentum_from_output(output)
            
            px_loss = torch.mean((input_px - output_px) ** 2)
            py_loss = torch.mean((input_py - output_py) ** 2)
            loss += px_loss + py_loss
            
        return loss
    
    def _compute_momentum_from_output(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute momentum components from spatial energy distribution."""
        batch_size = output.shape[0]
        
        if len(self.output_shape) == 3:  # 3D calorimeter (eta, phi, depth)
            eta_dim, phi_dim, depth_dim = self.output_shape
            
            # Create coordinate grids
            eta_coords = torch.linspace(-2.5, 2.5, eta_dim, device=self.device)
            phi_coords = torch.linspace(-np.pi, np.pi, phi_dim, device=self.device)
            
            eta_grid, phi_grid = torch.meshgrid(eta_coords, phi_coords, indexing='ij')
            eta_grid = eta_grid.unsqueeze(0).expand(batch_size, -1, -1, depth_dim)
            phi_grid = phi_grid.unsqueeze(0).expand(batch_size, -1, -1, depth_dim)
            
            # Convert to px, py using eta, phi
            pt = output.sum(dim=-1)  # Sum over depth
            px = pt * torch.cos(phi_grid[:, :, :, 0])
            py = pt * torch.sin(phi_grid[:, :, :, 0])
            
            output_px = (px * output.sum(dim=-1)).sum(dim=(1, 2))
            output_py = (py * output.sum(dim=-1)).sum(dim=(1, 2))
            
        else:  # 2D case or other geometries
            # Simplified momentum calculation
            output_px = torch.zeros(batch_size, device=self.device)
            output_py = torch.zeros(batch_size, device=self.device)
            
        return output_px, output_py
    
    @classmethod
    def from_pretrained(cls, model_name: str, cache_dir: Optional[str] = None) -> "PhysicsOperator":
        """Load a pre-trained operator from checkpoint."""
        # Implementation would download from model hub
        raise NotImplementedError("Pre-trained model loading not yet implemented")
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save operator checkpoint with metadata."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_shape': self.output_shape,
            'preserve_energy': self.preserve_energy,
            'preserve_momentum': self.preserve_momentum,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load operator from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('metadata', {})