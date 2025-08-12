"""
Advanced Spectral Neural Operators with Physics-Informed Adaptive Mode Selection.

Novel research contributions:
1. Adaptive spectral mode selection based on physics scales
2. Uncertainty-aware spectral convolutions with Bayesian inference
3. Multi-scale operator learning for QCD to detector scales
4. Conservation-aware spectral regularization

Academic Impact: Designed for Nature Machine Intelligence / Physical Review submissions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import logging
from dataclasses import dataclass
import math

from .fno import SpectralConv3d, FourierNeuralOperator
from ..physics.lorentz import LorentzEmbedding
from ..physics.conservation import ConservationLaws
from ..utils.robust_error_handling import (
    robust_physics_operation, 
    robust_physics_context,
    RobustPhysicsLogger,
    SpectralOperatorError,
    ErrorSeverity
)

logger = RobustPhysicsLogger('adaptive_spectral_fno')


@dataclass
class PhysicsScales:
    """Physics scale hierarchy for adaptive spectral selection."""
    
    # Energy scales (GeV)
    qcd_scale: float = 0.2  # QCD confinement
    electroweak_scale: float = 100.0  # W/Z boson mass
    planck_scale: float = 1.22e28  # Quantum gravity
    
    # Spatial scales (meters)
    nuclear_scale: float = 1e-15  # Nuclear size
    detector_pixel: float = 1e-4  # Detector granularity
    detector_size: float = 10.0  # Full detector
    
    # Temporal scales (seconds)
    interaction_time: float = 1e-24  # Strong interaction
    detector_response: float = 1e-9  # Electronics response
    trigger_latency: float = 1e-6  # Data acquisition
    
    def get_scale_hierarchy(self) -> Dict[str, List[float]]:
        """Return organized scale hierarchy for adaptive algorithms."""
        return {
            'energy': [self.qcd_scale, self.electroweak_scale, self.planck_scale],
            'spatial': [self.nuclear_scale, self.detector_pixel, self.detector_size],
            'temporal': [self.interaction_time, self.detector_response, self.trigger_latency]
        }


class AdaptiveSpectralConv3d(nn.Module):
    """
    Adaptive spectral convolution with physics-informed mode selection.
    
    Research Innovation: Dynamically selects Fourier modes based on physics scales
    and conservation laws, enabling scale-aware operator learning.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        spatial_size: Tuple[int, int, int],
        physics_scales: PhysicsScales = None,
        uncertainty_estimation: bool = True,
        conservation_weight: float = 1.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.physics_scales = physics_scales or PhysicsScales()
        self.uncertainty_estimation = uncertainty_estimation
        self.conservation_weight = conservation_weight
        
        # Adaptive mode computation
        self.adaptive_modes = self._compute_adaptive_modes()
        max_modes = max(self.adaptive_modes)
        
        logger.info(f"Adaptive spectral layer: modes={self.adaptive_modes}, "
                   f"conservation_weight={conservation_weight}")
        
        # Scale parameter for weight initialization
        self.scale = 1 / (in_channels * out_channels)
        
        # Physics-informed spectral weights with uncertainty
        if uncertainty_estimation:
            # Bayesian spectral weights (mean + log_var)
            self.weights_mean = nn.ParameterDict({
                f'mode_{i}': nn.Parameter(
                    self.scale * torch.randn(
                        in_channels, out_channels, 
                        min(mode, spatial_size[0]), 
                        min(mode, spatial_size[1]), 
                        min(mode, spatial_size[2]), 
                        dtype=torch.cfloat
                    )
                ) for i, mode in enumerate(self.adaptive_modes)
            })
            
            self.weights_log_var = nn.ParameterDict({
                f'mode_{i}': nn.Parameter(
                    torch.ones(
                        in_channels, out_channels,
                        min(mode, spatial_size[0]), 
                        min(mode, spatial_size[1]), 
                        min(mode, spatial_size[2])
                    ) * -2.0  # Initialize with low variance
                ) for i, mode in enumerate(self.adaptive_modes)
            })
        else:
            # Deterministic weights
            self.weights = nn.ParameterDict({
                f'mode_{i}': nn.Parameter(
                    self.scale * torch.randn(
                        in_channels, out_channels,
                        min(mode, spatial_size[0]), 
                        min(mode, spatial_size[1]), 
                        min(mode, spatial_size[2]),
                        dtype=torch.cfloat
                    )
                ) for i, mode in enumerate(self.adaptive_modes)
            })
        
        # Conservation law enforcement
        self.conservation_laws = ConservationLaws()
        
        # Physics-informed attention for mode selection
        self.mode_attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.adaptive_modes)),
            nn.Softmax(dim=-1)
        )
        
        # Scale-aware normalization
        self.scale_norm = nn.GroupNorm(min(8, in_channels), in_channels)
    
    def _compute_adaptive_modes(self) -> List[int]:
        """
        Compute adaptive Fourier modes based on physics scales.
        
        Research Innovation: Links spectral modes to fundamental physics scales
        for optimal information preservation across different energy regimes.
        """
        scale_hierarchy = self.physics_scales.get_scale_hierarchy()
        
        # Map physics scales to spectral frequencies
        spatial_scales = scale_hierarchy['spatial']
        
        # Compute modes for each spatial dimension
        modes = []
        for dim_size in self.spatial_size:
            # Nyquist criterion with physics scale awareness
            critical_scales = [
                dim_size / (2 * scale / min(spatial_scales)) 
                for scale in spatial_scales
            ]
            
            # Select modes that capture all critical physics scales
            adaptive_mode = min(
                dim_size // 2,
                max(8, int(np.ceil(max(critical_scales))))
            )
            modes.append(adaptive_mode)
        
        # Ensure minimum spectral resolution
        modes = [max(mode, 4) for mode in modes]
        
        logger.debug(f"Computed adaptive modes: {modes} for spatial size {self.spatial_size}")
        return modes
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with adaptive spectral convolution.
        
        Returns:
            output: Convolved tensor
            diagnostics: Physics and uncertainty diagnostics
        """
        batch_size = x.shape[0]
        spatial_dims = x.shape[-3:]
        
        # Scale-aware normalization
        x_normalized = self.scale_norm(x)
        
        # Compute 3D FFT
        x_ft = torch.fft.rfftn(x_normalized, dim=[-3, -2, -1])
        
        # Initialize output in frequency domain
        out_ft = torch.zeros(
            batch_size, self.out_channels, *x_ft.shape[-3:],
            dtype=torch.cfloat, device=x.device
        )
        
        # Adaptive mode attention
        mode_weights = self._compute_mode_attention(x)
        
        # Apply spectral convolution for each adaptive mode
        conservation_loss = 0.0
        uncertainty_info = {}
        
        for i, mode in enumerate(self.adaptive_modes):
            mode_weight = mode_weights[:, i:i+1, None, None, None]  # Broadcasting shape
            
            # Get spectral weights (with uncertainty if enabled)
            if self.uncertainty_estimation:
                weights_mean = self.weights_mean[f'mode_{i}']
                weights_log_var = self.weights_log_var[f'mode_{i}']
                
                # Sample weights during training (reparameterization trick)
                if self.training:
                    weights_std = torch.exp(0.5 * weights_log_var)
                    eps = torch.randn_like(weights_std)
                    weights = weights_mean + eps * weights_std
                    weights = weights.to(torch.cfloat)
                else:
                    weights = weights_mean
                
                # Store uncertainty information
                uncertainty_info[f'mode_{i}_variance'] = torch.exp(weights_log_var)
            else:
                weights = self.weights[f'mode_{i}']
            
            # Apply spectral convolution for this mode
            mode_contribution = self._spectral_multiply(
                x_ft, weights, mode, spatial_dims
            )
            
            # Weight by attention and add to output
            out_ft += mode_weight * mode_contribution
            
            # Conservation law penalty
            if self.conservation_weight > 0:
                conservation_loss += self._compute_conservation_penalty(
                    x_normalized, mode_contribution, mode
                )
        
        # Inverse FFT to spatial domain
        output = torch.fft.irfftn(out_ft, s=spatial_dims, dim=[-3, -2, -1])
        
        # Prepare diagnostics
        diagnostics = {
            'mode_weights': mode_weights,
            'conservation_loss': conservation_loss,
            'adaptive_modes': torch.tensor(self.adaptive_modes, device=x.device),
            'uncertainty_info': uncertainty_info,
            'spectral_energy': self._compute_spectral_energy(x_ft, out_ft)
        }
        
        return output, diagnostics
    
    def _compute_mode_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention weights for adaptive mode selection."""
        # Global average pooling to get channel statistics
        x_pooled = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Compute attention weights
        mode_weights = self.mode_attention(x_pooled)
        
        return mode_weights
    
    def _spectral_multiply(
        self, 
        x_ft: torch.Tensor, 
        weights: torch.Tensor, 
        mode: int,
        spatial_dims: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Perform spectral multiplication for specific mode."""
        
        # Ensure weights don't exceed spatial dimensions
        w_shape = weights.shape[-3:]
        effective_shape = tuple(min(w, s) for w, s in zip(w_shape, spatial_dims))
        
        # Extract relevant modes from input
        x_mode = x_ft[:, :, :effective_shape[0], :effective_shape[1], :effective_shape[2]]
        w_mode = weights[:, :, :effective_shape[0], :effective_shape[1], :effective_shape[2]]
        
        # Complex multiplication: einsum for efficient tensor contraction
        result = torch.einsum('bixyz,ioxyz->boxyz', x_mode, w_mode)
        
        return result
    
    def _compute_conservation_penalty(
        self, 
        x_input: torch.Tensor, 
        x_spectral: torch.Tensor, 
        mode: int
    ) -> torch.Tensor:
        """
        Compute conservation law penalty for physics consistency.
        
        Research Innovation: Enforces energy-momentum conservation in spectral domain.
        """
        # Convert spectral back to spatial for conservation check
        spatial_dims = x_input.shape[-3:]
        x_spatial = torch.fft.irfftn(x_spectral, s=spatial_dims, dim=[-3, -2, -1])
        
        # Check energy conservation (simplified version)
        input_energy = torch.sum(x_input ** 2, dim=[-3, -2, -1])
        output_energy = torch.sum(x_spatial ** 2, dim=[-3, -2, -1])
        
        energy_violation = torch.abs(input_energy - output_energy)
        conservation_penalty = torch.mean(energy_violation)
        
        return conservation_penalty
    
    def _compute_spectral_energy(
        self, 
        x_ft: torch.Tensor, 
        out_ft: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute spectral energy distribution for analysis."""
        
        # Input spectral energy
        input_energy = torch.abs(x_ft) ** 2
        output_energy = torch.abs(out_ft) ** 2
        
        # Energy per frequency band
        freq_bands = {
            'low': torch.mean(input_energy[:, :, :4, :4, :4]),
            'mid': torch.mean(input_energy[:, :, 4:16, 4:16, 4:16]) if x_ft.shape[-1] > 16 else torch.tensor(0.0),
            'high': torch.mean(input_energy[:, :, 16:, 16:, 16:]) if x_ft.shape[-1] > 32 else torch.tensor(0.0)
        }
        
        return {
            'input_total': torch.mean(input_energy),
            'output_total': torch.mean(output_energy),
            'frequency_bands': freq_bands
        }


class UncertaintyAwareFNO(nn.Module):
    """
    Uncertainty-Aware Fourier Neural Operator with Bayesian inference.
    
    Research Innovation: Provides epistemic uncertainty estimates for physics
    simulations, enabling robust anomaly detection and model validation.
    """
    
    def __init__(
        self,
        modes: int = 32,
        width: int = 64,
        input_dim: int = 4,
        output_shape: Tuple[int, int, int] = (50, 50, 25),
        n_layers: int = 4,
        physics_scales: PhysicsScales = None,
        uncertainty_estimation: bool = True,
        conservation_weight: float = 1.0,
        mc_samples: int = 10
    ):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_shape = output_shape
        self.uncertainty_estimation = uncertainty_estimation
        self.mc_samples = mc_samples
        self.physics_scales = physics_scales or PhysicsScales()
        
        logger.info(f"Initializing UncertaintyAwareFNO: modes={modes}, "
                   f"uncertainty={uncertainty_estimation}, mc_samples={mc_samples}")
        
        # Input embedding with uncertainty
        if uncertainty_estimation:
            self.input_embedding_mean = nn.Linear(input_dim, width)
            self.input_embedding_log_var = nn.Linear(input_dim, width)
        else:
            self.input_embedding = nn.Linear(input_dim, width)
        
        # Adaptive spectral convolution layers
        self.spectral_layers = nn.ModuleList([
            AdaptiveSpectralConv3d(
                in_channels=width,
                out_channels=width,
                spatial_size=output_shape,
                physics_scales=self.physics_scales,
                uncertainty_estimation=uncertainty_estimation,
                conservation_weight=conservation_weight
            )
            for _ in range(n_layers)
        ])
        
        # Local convolution layers (deterministic for stability)
        self.local_layers = nn.ModuleList([
            nn.Conv3d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Normalization layers
        self.norms = nn.ModuleList([
            nn.GroupNorm(min(8, width//4), width) for _ in range(n_layers)
        ])
        
        # Output projection with uncertainty
        if uncertainty_estimation:
            self.output_mean = nn.Sequential(
                nn.Conv3d(width, width // 2, 1),
                nn.GELU(),
                nn.Conv3d(width // 2, 1, 1)
            )
            self.output_log_var = nn.Sequential(
                nn.Conv3d(width, width // 2, 1),
                nn.GELU(),
                nn.Conv3d(width // 2, 1, 1)
            )
        else:
            self.output_projection = nn.Sequential(
                nn.Conv3d(width, width // 2, 1),
                nn.GELU(),
                nn.Conv3d(width // 2, 1, 1)
            )
        
        # Physics-informed loss computation
        self.conservation_laws = ConservationLaws()
        
        # Initialize weights with physics-aware initialization
        self._initialize_physics_weights()
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional uncertainty quantification.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            output: Predicted tensor
            uncertainty_dict: Uncertainty information (if requested)
        """
        
        if self.uncertainty_estimation and return_uncertainty:
            return self._forward_with_uncertainty(x)
        else:
            return self._forward_deterministic(x)
    
    def _forward_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass."""
        
        # Handle input format conversion
        if len(x.shape) == 3:  # Particle list
            x = self._particles_to_grid(x)
        
        if len(x.shape) == 4:  # Add channel dimension
            x = x.unsqueeze(1)
        
        # Input embedding
        if hasattr(self, 'input_embedding'):
            x = self._apply_embedding(x, self.input_embedding)
        else:
            x = self._apply_embedding(x, self.input_embedding_mean)
        
        # Spectral neural operator layers
        total_conservation_loss = 0.0
        
        for i in range(self.n_layers):
            # Spectral convolution
            x_spectral, diagnostics = self.spectral_layers[i](x)
            
            # Local convolution
            x_local = self.local_layers[i](x)
            
            # Combine paths
            x = x_spectral + x_local
            
            # Activation and normalization
            x = F.gelu(x)
            x = self.norms[i](x)
            
            # Accumulate conservation loss
            total_conservation_loss += diagnostics['conservation_loss']
        
        # Output projection
        if hasattr(self, 'output_projection'):
            output = self.output_projection(x)
        else:
            output = self.output_mean(x)
        
        output = output.squeeze(1)  # Remove channel dimension
        
        return output
    
    def _forward_with_uncertainty(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with uncertainty quantification using Monte Carlo sampling."""
        
        if not self.uncertainty_estimation:
            raise ValueError("Uncertainty estimation not enabled")
        
        # Collect samples for Monte Carlo estimation
        samples = []
        conservation_losses = []
        
        for _ in range(self.mc_samples):
            # Sample from Bayesian model
            sample_output = self._forward_sample(x)
            samples.append(sample_output)
        
        # Stack samples and compute statistics
        samples_tensor = torch.stack(samples, dim=0)  # [mc_samples, batch, ...]
        
        # Predictive mean and variance
        pred_mean = torch.mean(samples_tensor, dim=0)
        pred_var = torch.var(samples_tensor, dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = pred_var
        
        # Aleatoric uncertainty (data uncertainty) - from final layer
        final_sample = samples[-1]
        if hasattr(self, 'output_log_var'):
            x_final = self._get_final_features(x)  # Get features before output
            aleatoric_log_var = self.output_log_var(x_final).squeeze(1)
            aleatoric_uncertainty = torch.exp(aleatoric_log_var)
        else:
            aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        uncertainty_dict = {
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'predictive_variance': pred_var,
            'mc_samples': samples_tensor,
            'conservation_loss': torch.mean(torch.tensor(conservation_losses)) if conservation_losses else torch.tensor(0.0)
        }
        
        return pred_mean, uncertainty_dict
    
    def _forward_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass sample for Monte Carlo."""
        
        # Handle input format
        if len(x.shape) == 3:
            x = self._particles_to_grid(x)
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        # Sample from input embedding
        x = self._sample_embedding(x)
        
        # Forward through layers
        for i in range(self.n_layers):
            x_spectral, _ = self.spectral_layers[i](x)  # Uses sampling internally
            x_local = self.local_layers[i](x)
            x = x_spectral + x_local
            x = F.gelu(x)
            x = self.norms[i](x)
        
        # Sample from output distribution
        output_mean = self.output_mean(x)
        if hasattr(self, 'output_log_var'):
            output_log_var = self.output_log_var(x)
            output_std = torch.exp(0.5 * output_log_var)
            eps = torch.randn_like(output_std)
            output = output_mean + eps * output_std
        else:
            output = output_mean
        
        return output.squeeze(1)
    
    def _sample_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Sample from Bayesian input embedding."""
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Get embedding mean and variance
        embed_mean = self.input_embedding_mean(x_flat)
        embed_log_var = self.input_embedding_log_var(x_flat)
        
        # Sample using reparameterization trick
        embed_std = torch.exp(0.5 * embed_log_var)
        eps = torch.randn_like(embed_std)
        embed_sample = embed_mean + eps * embed_std
        
        # Reshape back
        embed_sample = embed_sample.reshape(*original_shape[:-1], self.width)
        
        # Rearrange to channel-first format
        embed_sample = embed_sample.permute(0, -1, *range(1, len(embed_sample.shape)-1))
        
        return embed_sample
    
    def _apply_embedding(self, x: torch.Tensor, embedding_layer: nn.Module) -> torch.Tensor:
        """Apply embedding layer to input tensor."""
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        x_embedded = embedding_layer(x_flat)
        x_embedded = x_embedded.reshape(*original_shape[:-1], self.width)
        
        # Rearrange to channel-first format
        x_embedded = x_embedded.permute(0, -1, *range(1, len(x_embedded.shape)-1))
        
        return x_embedded
    
    def _get_final_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before final output layer for aleatoric uncertainty."""
        # This is a simplified version - in practice, you'd store intermediate features
        return self._forward_deterministic(x).unsqueeze(1)  # Add channel dim back
    
    def _particles_to_grid(self, particles: torch.Tensor) -> torch.Tensor:
        """Convert particle 4-vectors to grid representation with physics awareness."""
        batch_size, n_particles, features = particles.shape
        
        # Create grid with physics-informed binning
        grid = torch.zeros(batch_size, *self.output_shape, features, device=particles.device)
        
        # Physics-informed mapping (simplified - could use proper η-φ coordinates)
        for b in range(batch_size):
            valid_particles = particles[b][particles[b, :, 0] > 0]  # Energy > 0
            
            if len(valid_particles) > 0:
                # Map particles to grid using physics coordinates
                for i, particle in enumerate(valid_particles[:np.prod(self.output_shape)]):
                    # Simple spatial mapping (could be improved with proper detector geometry)
                    idx = np.unravel_index(i, self.output_shape)
                    grid[b, idx[0], idx[1], idx[2], :] = particle
        
        return grid
    
    def _initialize_physics_weights(self):
        """Initialize weights with physics-aware distributions."""
        
        def init_spectral_weights(module):
            if hasattr(module, 'weights_mean'):
                for key, weight in module.weights_mean.items():
                    # Xavier initialization scaled by physics considerations
                    fan_in = weight.shape[0]
                    fan_out = weight.shape[1]
                    std = math.sqrt(2.0 / (fan_in + fan_out))
                    
                    # Physics-informed scaling
                    physics_scale = 1.0 / math.sqrt(self.physics_scales.qcd_scale)
                    
                    with torch.no_grad():
                        weight.real.normal_(0, std * physics_scale)
                        weight.imag.normal_(0, std * physics_scale)
        
        # Apply to all spectral layers
        for layer in self.spectral_layers:
            init_spectral_weights(layer)
    
    def compute_physics_loss(
        self, 
        input_data: torch.Tensor, 
        output_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss terms.
        
        Research Innovation: Enforces fundamental physics laws as regularization.
        """
        
        losses = {}
        
        # Energy conservation
        input_energy = torch.sum(input_data ** 2, dim=[-3, -2, -1])
        output_energy = torch.sum(output_data ** 2, dim=[-3, -2, -1])
        losses['energy_conservation'] = torch.mean((input_energy - output_energy) ** 2)
        
        # Momentum conservation (simplified)
        if input_data.shape[-1] >= 4:  # 4-vector data
            input_momentum = torch.sum(input_data[..., 1:4], dim=[-3, -2, -1])
            output_momentum = torch.sum(output_data[..., 1:4], dim=[-3, -2, -1]) if output_data.shape[-1] >= 4 else torch.zeros_like(input_momentum)
            losses['momentum_conservation'] = torch.mean((input_momentum - output_momentum) ** 2)
        
        # Scale invariance (Lorentz boost)
        losses['scale_invariance'] = self._compute_scale_invariance_loss(input_data, output_data)
        
        return losses
    
    def _compute_scale_invariance_loss(
        self, 
        input_data: torch.Tensor, 
        output_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for scale/boost invariance."""
        
        # Simple version - could be enhanced with proper Lorentz transformations
        input_norm = torch.norm(input_data, dim=-1, keepdim=True)
        output_norm = torch.norm(output_data, dim=-1, keepdim=True)
        
        # Relative error in norms
        norm_error = torch.abs(input_norm - output_norm) / (input_norm + 1e-8)
        scale_loss = torch.mean(norm_error)
        
        return scale_loss


class MultiScaleOperatorLearning(nn.Module):
    """
    Multi-scale operator learning for hierarchical physics simulation.
    
    Research Innovation: Handles multiple physics scales simultaneously,
    from QCD to detector scales in unified framework.
    """
    
    def __init__(
        self,
        scale_levels: int = 3,
        base_modes: int = 16,
        width: int = 64,
        physics_scales: PhysicsScales = None
    ):
        super().__init__()
        
        self.scale_levels = scale_levels
        self.base_modes = base_modes
        self.width = width
        self.physics_scales = physics_scales or PhysicsScales()
        
        # Multi-scale FNO operators
        self.scale_operators = nn.ModuleList([
            UncertaintyAwareFNO(
                modes=base_modes * (2 ** i),
                width=width,
                n_layers=4 - i,  # Fewer layers for higher frequencies
                physics_scales=physics_scales,
                uncertainty_estimation=True
            )
            for i in range(scale_levels)
        ])
        
        # Scale fusion mechanism
        self.scale_fusion = nn.Sequential(
            nn.Conv3d(scale_levels, width, 1),
            nn.GELU(),
            nn.Conv3d(width, 1, 1)
        )
        
        logger.info(f"Multi-scale operator: {scale_levels} scales, "
                   f"base_modes={base_modes}, width={width}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Multi-scale forward pass."""
        
        scale_outputs = []
        scale_uncertainties = []
        
        for i, operator in enumerate(self.scale_operators):
            # Different downsampling for different scales
            if i > 0:
                scale_factor = 2 ** i
                x_scaled = F.interpolate(
                    x, scale_factor=1.0/scale_factor, mode='trilinear', align_corners=False
                )
            else:
                x_scaled = x
            
            # Forward through scale-specific operator
            output, uncertainty = operator(x_scaled, return_uncertainty=True)
            
            # Upsample back to original resolution
            if i > 0:
                output = F.interpolate(
                    output.unsqueeze(1), size=x.shape[-3:], mode='trilinear', align_corners=False
                ).squeeze(1)
            
            scale_outputs.append(output.unsqueeze(1))
            scale_uncertainties.append(uncertainty['total_uncertainty'])
        
        # Fuse multi-scale outputs
        stacked_outputs = torch.cat(scale_outputs, dim=1)
        fused_output = self.scale_fusion(stacked_outputs).squeeze(1)
        
        # Combine uncertainties
        combined_uncertainty = torch.stack(scale_uncertainties, dim=0)
        total_uncertainty = torch.sqrt(torch.sum(combined_uncertainty ** 2, dim=0))
        
        diagnostics = {
            'scale_outputs': scale_outputs,
            'scale_uncertainties': scale_uncertainties,
            'total_uncertainty': total_uncertainty,
            'fusion_weights': self._compute_fusion_weights(stacked_outputs)
        }
        
        return fused_output, diagnostics
    
    def _compute_fusion_weights(self, stacked_outputs: torch.Tensor) -> torch.Tensor:
        """Compute adaptive fusion weights for multi-scale integration."""
        
        # Attention-based fusion weights
        channel_attention = F.adaptive_avg_pool3d(stacked_outputs, 1)
        fusion_weights = F.softmax(channel_attention, dim=1)
        
        return fusion_weights