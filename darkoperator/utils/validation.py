"""Input validation and sanitization utilities."""

import torch
import numpy as np
from typing import Union, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_4vectors(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Validate and sanitize 4-vector inputs.
    
    Args:
        x: Input 4-vectors (batch, n_particles, 4) - (E, px, py, pz)
        
    Returns:
        Validated tensor with proper shape and physical constraints
        
    Raises:
        ValidationError: If input is invalid
    """
    # Convert to tensor if needed
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    elif not isinstance(x, torch.Tensor):
        raise ValidationError(f"Input must be torch.Tensor or numpy.ndarray, got {type(x)}")
    
    # Check dimensions
    if x.dim() < 2:
        raise ValidationError(f"Input must have at least 2 dimensions, got {x.dim()}")
    
    if x.shape[-1] != 4:
        raise ValidationError(f"Last dimension must be 4 (E, px, py, pz), got {x.shape[-1]}")
    
    # Check for NaN/Inf values
    if torch.isnan(x).any():
        logger.warning("NaN values detected in input, replacing with zeros")
        x = torch.nan_to_num(x, nan=0.0)
    
    if torch.isinf(x).any():
        logger.warning("Infinite values detected in input, clipping")
        x = torch.clamp(x, -1e6, 1e6)
    
    # Physical constraints
    energy = x[..., 0]
    px, py, pz = x[..., 1], x[..., 2], x[..., 3]
    
    # Energy must be positive
    if torch.any(energy < 0):
        logger.warning("Negative energies detected, taking absolute value")
        x[..., 0] = torch.abs(energy)
        energy = x[..., 0]
    
    # Check energy-momentum relation (E^2 >= p^2 for massive particles)
    p_squared = px**2 + py**2 + pz**2
    e_squared = energy**2
    
    # Allow small violations due to numerical precision
    violation_mask = e_squared < p_squared - 1e-6
    if torch.any(violation_mask):
        n_violations = violation_mask.sum().item()
        logger.warning(f"Found {n_violations} violations of E^2 >= p^2, correcting energies")
        
        # Correct energy to satisfy constraint
        x[violation_mask, 0] = torch.sqrt(p_squared[violation_mask] + 1e-6)
    
    return x


def validate_detector_geometry(
    eta_range: Tuple[float, float] = (-5.0, 5.0),
    phi_range: Tuple[float, float] = (-np.pi, np.pi),
    energy_range: Tuple[float, float] = (0.1, 10000.0)
) -> callable:
    """
    Create validator for detector geometry constraints.
    
    Args:
        eta_range: Valid pseudorapidity range
        phi_range: Valid azimuthal angle range  
        energy_range: Valid energy range in GeV
        
    Returns:
        Validation function
    """
    def validator(x: torch.Tensor) -> torch.Tensor:
        """Apply geometry-based validation."""
        x = validate_4vectors(x)
        
        energy = x[..., 0]
        px, py, pz = x[..., 1], x[..., 2], x[..., 3]
        
        # Compute kinematic variables
        pt = torch.sqrt(px**2 + py**2 + 1e-8)
        eta = torch.asinh(pz / pt)
        phi = torch.atan2(py, px)
        
        # Energy range check
        energy_mask = (energy >= energy_range[0]) & (energy <= energy_range[1])
        if not torch.all(energy_mask):
            n_violations = (~energy_mask).sum().item()
            logger.warning(f"Found {n_violations} particles outside energy range [{energy_range[0]}, {energy_range[1]}] GeV")
            
            # Clip energies to valid range
            x[..., 0] = torch.clamp(energy, energy_range[0], energy_range[1])
        
        # Pseudorapidity range check  
        eta_mask = (eta >= eta_range[0]) & (eta <= eta_range[1])
        if not torch.all(eta_mask):
            n_violations = (~eta_mask).sum().item()
            logger.warning(f"Found {n_violations} particles outside eta range [{eta_range[0]}, {eta_range[1]}]")
        
        return x
    
    return validator


def sanitize_file_path(path: str) -> str:
    """
    Sanitize file paths to prevent directory traversal attacks.
    
    Args:
        path: Input file path
        
    Returns:
        Sanitized path
        
    Raises:
        ValidationError: If path is unsafe
    """
    import os.path
    
    # Remove null bytes
    path = path.replace('\x00', '')
    
    # Check for directory traversal attempts
    if '..' in path or path.startswith('/'):
        raise ValidationError(f"Unsafe path detected: {path}")
    
    # Normalize path
    path = os.path.normpath(path)
    
    # Additional checks
    dangerous_patterns = ['~', '$', '`', '|', ';', '&', '>', '<']
    for pattern in dangerous_patterns:
        if pattern in path:
            raise ValidationError(f"Dangerous character '{pattern}' in path: {path}")
    
    return path


def validate_model_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate model checkpoint file for security.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        True if checkpoint is safe to load
        
    Raises:
        ValidationError: If checkpoint is unsafe
    """
    import os
    import pickle
    
    # Sanitize path
    checkpoint_path = sanitize_file_path(checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        raise ValidationError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Check file size (prevent extremely large files)
    max_size_mb = 1000  # 1GB limit
    file_size = os.path.getsize(checkpoint_path)
    if file_size > max_size_mb * 1024 * 1024:
        raise ValidationError(f"Checkpoint file too large: {file_size / (1024*1024):.1f} MB > {max_size_mb} MB")
    
    # Basic checkpoint structure validation
    try:
        # Use safe loading for pickle-based checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # Verify expected structure
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                raise ValidationError(f"Missing required key in checkpoint: {key}")
        
        logger.info(f"Checkpoint validation passed: {checkpoint_path}")
        return True
        
    except Exception as e:
        raise ValidationError(f"Failed to validate checkpoint: {e}")


class SecureModelLoader:
    """Secure model loading with validation."""
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str, model: torch.nn.Module) -> dict:
        """
        Securely load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model instance to load into
            
        Returns:
            Checkpoint metadata
        """
        # Validate checkpoint
        validate_model_checkpoint(checkpoint_path)
        
        # Load with security restrictions
        checkpoint = torch.load(
            checkpoint_path, 
            map_location='cpu',
            weights_only=True  # Prevent arbitrary code execution
        )
        
        # Load state dict with strict=False to handle missing keys gracefully
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            raise ValidationError(f"Incompatible checkpoint: {e}")
        
        return checkpoint.get('metadata', {})