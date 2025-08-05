"""Fourier Neural Operator implementation for physics applications."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution layer for Fourier Neural Operators."""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep in x direction
        self.modes2 = modes2  # Number of Fourier modes to keep in y direction  
        self.modes3 = modes3  # Number of Fourier modes to keep in z direction
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Fourier weights for different mode combinations
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication for 3D tensors."""
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.
        
        Args:
            x: Input tensor (batch, channels, x, y, z)
            
        Returns:
            Convolved tensor in spectral domain
        """
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[-3], x.shape[-2], x.shape[-1]

        # Apply 3D FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, size_x, size_y, size_z//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        
        # Mode 1: Low frequencies in all dimensions
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        
        # Mode 2: Low x,y and high z frequencies  
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:], self.weights2)
        
        # Mode 3: Low x, high y,z frequencies
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        
        # Mode 4: Low x, high y, low z frequencies  
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:], self.weights4)

        # Apply inverse 3D FFT
        x = torch.fft.irfftn(out_ft, s=(size_x, size_y, size_z), dim=[-3, -2, -1])
        
        return x


class FourierNeuralOperator(nn.Module):
    """
    Fourier Neural Operator for learning operators between function spaces.
    
    Particularly suited for PDEs and physics simulations where the solution
    operator maps between infinite-dimensional function spaces.
    """
    
    def __init__(
        self,
        modes: int = 32,
        width: int = 64,
        input_dim: int = 4,
        output_shape: Tuple[int, int, int] = (50, 50, 25),
        n_layers: int = 4,
        input_embedding = None,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_shape = output_shape
        
        # Input embedding
        if input_embedding is not None:
            self.input_embedding = input_embedding
        else:
            self.input_embedding = nn.Linear(input_dim, width)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv3d(width, width, modes, modes, modes//2)
            for _ in range(n_layers)
        ])
        
        # Local (pointwise) convolutions  
        self.local_layers = nn.ModuleList([
            nn.Conv3d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu
            
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv3d(width, width // 2, 1),
            nn.GELU(),
            nn.Conv3d(width // 2, 1, 1)
        )
        
        # Normalization layers
        self.norms = nn.ModuleList([
            nn.GroupNorm(min(32, width//4), width) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FNO.
        
        Args:
            x: Input tensor, shape depends on problem
            
        Returns:
            Output tensor of shape (*batch_dims, *output_shape)
        """
        # Handle different input formats
        if len(x.shape) == 3:  # (batch, n_particles, features)
            # Convert particle list to grid representation
            x = self._particles_to_grid(x)
        
        # Ensure proper dimensions for 3D convolution
        if len(x.shape) == 4:  # (batch, height, width, depth)
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Input embedding
        if x.shape[1] != self.width:
            # Apply embedding if needed
            original_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            x = self.input_embedding(x)
            x = x.reshape(*original_shape[:-1], self.width)
            
            # Rearrange to (batch, channels, spatial_dims)
            x = x.permute(0, -1, *range(1, len(x.shape)-1))
        
        # Fourier neural operator layers
        for i in range(self.n_layers):
            # Spectral convolution path
            x1 = self.fourier_layers[i](x)
            
            # Local convolution path  
            x2 = self.local_layers[i](x)
            
            # Combine paths
            x = x1 + x2
            
            # Activation and normalization
            x = self.activation(x)
            x = self.norms[i](x)
        
        # Output projection
        output = self.output_projection(x)
        output = output.squeeze(1)  # Remove channel dimension
        
        return output
    
    def _particles_to_grid(self, particles: torch.Tensor) -> torch.Tensor:
        """Convert particle 4-vectors to grid representation."""
        batch_size, n_particles, features = particles.shape
        
        # Create empty grid
        grid = torch.zeros(batch_size, *self.output_shape, features, device=particles.device)
        
        # Simple projection (could be more sophisticated)
        for b in range(batch_size):
            for p in range(min(n_particles, np.prod(self.output_shape))):
                # Simple indexing - in practice would use proper eta/phi mapping
                idx = np.unravel_index(p, self.output_shape)
                grid[b, idx[0], idx[1], idx[2], :] = particles[b, p, :]
        
        return grid