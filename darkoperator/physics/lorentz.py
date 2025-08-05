"""Lorentz-invariant embeddings for 4-vectors."""

import torch
import torch.nn as nn
import numpy as np


class LorentzEmbedding(nn.Module):
    """Lorentz-invariant embedding for 4-momentum vectors."""
    
    def __init__(self, input_dim: int = 4, embed_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Invariant features
        self.invariant_net = nn.Sequential(
            nn.Linear(3, embed_dim // 2),  # m^2, pt, eta
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2)
        )
        
        # Directional features
        self.directional_net = nn.Sequential(
            nn.Linear(2, embed_dim // 2),  # cos(phi), sin(phi)
            nn.ReLU(), 
            nn.Linear(embed_dim // 2, embed_dim // 2)
        )
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed 4-vectors preserving Lorentz invariance.
        
        Args:
            x: 4-vectors (batch, ..., 4) - (E, px, py, pz)
            
        Returns:
            Embedded vectors (batch, ..., embed_dim)
        """
        original_shape = x.shape
        x = x.reshape(-1, 4)
        
        E, px, py, pz = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # Compute invariant quantities
        m_squared = E**2 - px**2 - py**2 - pz**2
        pt = torch.sqrt(px**2 + py**2 + 1e-8)
        eta = torch.asinh(pz / (pt + 1e-8))
        
        invariants = torch.stack([m_squared, pt, eta], dim=1)
        
        # Compute directional quantities
        phi = torch.atan2(py, px)
        directions = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)
        
        # Embed features
        inv_features = self.invariant_net(invariants)
        dir_features = self.directional_net(directions)
        
        # Combine
        combined = torch.cat([inv_features, dir_features], dim=1)
        embedded = self.output_proj(combined)
        
        # Restore original shape
        embedded = embedded.reshape(*original_shape[:-1], self.embed_dim)
        
        return embedded