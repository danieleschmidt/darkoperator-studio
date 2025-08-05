"""Muon system operator for muon detection simulation."""

import torch
import torch.nn as nn
from typing import Tuple
from .base import PhysicsOperator


class MuonOperator(PhysicsOperator):
    """Neural operator for muon system simulation."""
    
    def __init__(self, input_dim: int = 4, output_shape: Tuple[int, int, int] = (40, 40, 10), **kwargs):
        super().__init__(input_dim, output_shape, **kwargs)
        
        self.embedding = nn.Linear(input_dim, 32)
        self.conv_layers = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 8, 3, padding=1), 
            nn.ReLU(),
            nn.Conv3d(8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_emb = self.embedding(x.reshape(-1, self.input_dim))
        x_spatial = x_emb.reshape(batch_size, 32, *self.output_shape)
        return self.conv_layers(x_spatial).squeeze(1)