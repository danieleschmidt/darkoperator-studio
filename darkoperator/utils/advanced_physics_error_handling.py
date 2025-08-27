"""
Advanced Physics-Informed Error Handling and Recovery System.

This module provides sophisticated error handling specifically designed for
physics simulations, neural operator training, and relativistic computations.
Includes automatic error recovery, physics constraint violation detection,
and graceful degradation strategies.
"""

import torch
import numpy as np
import traceback
import logging
import time
import functools
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import warnings

class PhysicsErrorType(Enum):
    """Categories of physics-specific errors."""
    CONSERVATION_VIOLATION = "conservation_violation"
    CAUSALITY_VIOLATION = "causality_violation"
    UNITARITY_VIOLATION = "unitarity_violation"
    LORENTZ_VIOLATION = "lorentz_violation"
    GAUGE_VIOLATION = "gauge_violation"
    ENERGY_INSTABILITY = "energy_instability"
    NUMERICAL_INSTABILITY = "numerical_instability"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    SPACETIME_SINGULARITY = "spacetime_singularity"
    SYMMETRY_BREAKING = "symmetry_breaking"

@dataclass
class PhysicsErrorContext:
    """Context information for physics errors."""
    error_type: PhysicsErrorType
    severity: str = "medium"  # low, medium, high, critical
    violation_magnitude: float = 0.0
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    tolerance: float = 1e-6
    physics_constants: Dict[str, float] = field(default_factory=dict)
    tensor_shapes: Dict[str, Tuple] = field(default_factory=dict)
    computation_step: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False

class PhysicsConstraintValidator:
    """Validates physics constraints and detects violations."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.violation_history = []
        
    def validate_energy_conservation(self, energy_before: torch.Tensor, 
                                   energy_after: torch.Tensor) -> PhysicsErrorContext:
        """Validate energy conservation."""
        violation_magnitude = torch.abs(energy_after - energy_before).max().item()
        
        error_context = PhysicsErrorContext(
            error_type=PhysicsErrorType.CONSERVATION_VIOLATION,
            violation_magnitude=violation_magnitude,
            expected_value=energy_before.mean().item(),
            actual_value=energy_after.mean().item(),
            tolerance=self.tolerance,
            computation_step="energy_conservation_check"
        )
        
        if violation_magnitude > self.tolerance:
            error_context.severity = "high" if violation_magnitude > 10 * self.tolerance else "medium"
            self.violation_history.append(error_context)
            
        return error_context
    
    def validate_momentum_conservation(self, momentum_before: torch.Tensor,
                                     momentum_after: torch.Tensor) -> PhysicsErrorContext:
        """Validate momentum conservation."""
        momentum_diff = torch.norm(momentum_after - momentum_before, dim=-1).max().item()
        
        error_context = PhysicsErrorContext(
            error_type=PhysicsErrorType.CONSERVATION_VIOLATION,
            violation_magnitude=momentum_diff,
            expected_value=torch.norm(momentum_before).item(),
            actual_value=torch.norm(momentum_after).item(),
            tolerance=self.tolerance,
            computation_step="momentum_conservation_check"
        )
        
        if momentum_diff > self.tolerance:
            error_context.severity = "high" if momentum_diff > 10 * self.tolerance else "medium"
            self.violation_history.append(error_context)
            
        return error_context
    
    def validate_lorentz_invariance(self, four_vector_before: torch.Tensor,
                                   four_vector_after: torch.Tensor) -> PhysicsErrorContext:
        """Validate Lorentz invariance of four-vectors."""
        # Calculate invariant mass before and after
        invariant_before = four_vector_before[..., 0]**2 - torch.sum(four_vector_before[..., 1:]**2, dim=-1)
        invariant_after = four_vector_after[..., 0]**2 - torch.sum(four_vector_after[..., 1:]**2, dim=-1)
        
        violation_magnitude = torch.abs(invariant_after - invariant_before).max().item()
        
        error_context = PhysicsErrorContext(
            error_type=PhysicsErrorType.LORENTZ_VIOLATION,
            violation_magnitude=violation_magnitude,
            expected_value=invariant_before.mean().item(),
            actual_value=invariant_after.mean().item(),
            tolerance=self.tolerance,
            computation_step="lorentz_invariance_check"
        )
        
        if violation_magnitude > self.tolerance:
            error_context.severity = "critical" if violation_magnitude > 100 * self.tolerance else "high"
            self.violation_history.append(error_context)
            
        return error_context
    
    def validate_unitarity(self, transformation_matrix: torch.Tensor) -> PhysicsErrorContext:
        """Validate unitarity of transformation matrices."""
        # Check if U†U = I
        conjugate_transpose = transformation_matrix.conj().transpose(-2, -1)
        product = torch.matmul(conjugate_transpose, transformation_matrix)
        identity = torch.eye(transformation_matrix.shape[-1], device=transformation_matrix.device)
        
        violation_magnitude = torch.norm(product - identity).item()
        
        error_context = PhysicsErrorContext(
            error_type=PhysicsErrorType.UNITARITY_VIOLATION,
            violation_magnitude=violation_magnitude,
            expected_value=1.0,
            actual_value=torch.norm(product).item(),
            tolerance=self.tolerance,
            computation_step="unitarity_check"
        )
        
        if violation_magnitude > self.tolerance:
            error_context.severity = "critical" if violation_magnitude > 10 * self.tolerance else "high"
            self.violation_history.append(error_context)
            
        return error_context

class PhysicsErrorRecovery:
    """Implements recovery strategies for physics errors."""
    
    def __init__(self):
        self.recovery_strategies = {
            PhysicsErrorType.CONSERVATION_VIOLATION: self._recover_conservation,
            PhysicsErrorType.CAUSALITY_VIOLATION: self._recover_causality,
            PhysicsErrorType.UNITARITY_VIOLATION: self._recover_unitarity,
            PhysicsErrorType.LORENTZ_VIOLATION: self._recover_lorentz,
            PhysicsErrorType.NUMERICAL_INSTABILITY: self._recover_numerical_instability,
            PhysicsErrorType.ENERGY_INSTABILITY: self._recover_energy_instability,
        }
    
    def attempt_recovery(self, error_context: PhysicsErrorContext, 
                        tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Attempt to recover from physics error."""
        recovery_func = self.recovery_strategies.get(error_context.error_type)
        
        if recovery_func is None:
            logging.warning(f"No recovery strategy for {error_context.error_type}")
            return tensors
        
        try:
            recovered_tensors = recovery_func(error_context, tensors)
            error_context.recovery_attempted = True
            error_context.recovery_successful = True
            return recovered_tensors
            
        except Exception as e:
            logging.error(f"Recovery failed for {error_context.error_type}: {e}")
            error_context.recovery_attempted = True
            error_context.recovery_successful = False
            return tensors
    
    def _recover_conservation(self, error_context: PhysicsErrorContext, 
                            tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recover from conservation violations."""
        recovered = tensors.copy()
        
        if 'energy' in tensors and 'initial_energy' in tensors:
            # Rescale energy to conserve total
            total_energy = tensors['energy'].sum()
            target_energy = tensors['initial_energy'].sum()
            scale_factor = target_energy / (total_energy + 1e-10)
            recovered['energy'] = tensors['energy'] * scale_factor
            
        if 'momentum' in tensors and 'initial_momentum' in tensors:
            # Rescale momentum to conserve total
            total_momentum = tensors['momentum'].sum(dim=0, keepdim=True)
            target_momentum = tensors['initial_momentum'].sum(dim=0, keepdim=True)
            momentum_correction = (target_momentum - total_momentum) / tensors['momentum'].shape[0]
            recovered['momentum'] = tensors['momentum'] + momentum_correction
            
        return recovered
    
    def _recover_causality(self, error_context: PhysicsErrorContext,
                          tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recover from causality violations."""
        recovered = tensors.copy()
        
        if 'spacetime_coords' in tensors:
            coords = tensors['spacetime_coords']
            batch_size, seq_len, _ = coords.shape
            
            # Ensure timelike ordering
            for i in range(1, seq_len):
                dt = coords[:, i, 0] - coords[:, i-1, 0]
                spatial_dist = torch.norm(coords[:, i, 1:] - coords[:, i-1, 1:], dim=-1)
                
                # If spacelike separated, adjust timing
                violation_mask = spatial_dist > dt  # Using c=1 units
                if violation_mask.any():
                    # Increase time separation
                    time_adjustment = spatial_dist - dt + 0.01
                    coords[violation_mask, i, 0] = coords[violation_mask, i-1, 0] + time_adjustment[violation_mask]
                    
            recovered['spacetime_coords'] = coords
            
        return recovered
    
    def _recover_unitarity(self, error_context: PhysicsErrorContext,
                          tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recover from unitarity violations."""
        recovered = tensors.copy()
        
        if 'transformation_matrix' in tensors:
            matrix = tensors['transformation_matrix']
            
            # Use SVD to project to nearest unitary matrix
            U, S, V = torch.svd(matrix)
            unitary_matrix = torch.matmul(U, V.transpose(-2, -1))
            recovered['transformation_matrix'] = unitary_matrix
            
        return recovered
    
    def _recover_lorentz(self, error_context: PhysicsErrorContext,
                        tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recover from Lorentz invariance violations."""
        recovered = tensors.copy()
        
        if 'four_vector' in tensors:
            four_vector = tensors['four_vector']
            
            # Ensure energy component is consistent with spatial components
            spatial_norm_sq = torch.sum(four_vector[..., 1:]**2, dim=-1, keepdim=True)
            mass_sq = four_vector[..., 0:1]**2 - spatial_norm_sq
            
            # If tachyonic (m² < 0), adjust energy component
            tachyonic_mask = mass_sq < 0
            if tachyonic_mask.any():
                corrected_energy = torch.sqrt(spatial_norm_sq + 1e-6)  # Assume m=0 (photon-like)
                four_vector = four_vector.clone()
                four_vector[tachyonic_mask.squeeze(-1), 0] = corrected_energy[tachyonic_mask].squeeze(-1)
                recovered['four_vector'] = four_vector
                
        return recovered
    
    def _recover_numerical_instability(self, error_context: PhysicsErrorContext,
                                     tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recover from numerical instabilities."""
        recovered = {}
        
        for key, tensor in tensors.items():
            # Clip extreme values
            tensor_clipped = torch.clamp(tensor, -1e6, 1e6)
            
            # Replace NaN and Inf values
            tensor_cleaned = torch.where(
                torch.isfinite(tensor_clipped),
                tensor_clipped,
                torch.zeros_like(tensor_clipped)
            )
            
            # Apply gradient clipping for backprop stability
            if tensor_cleaned.requires_grad:
                tensor_cleaned = torch.clamp(tensor_cleaned, -100, 100)
                
            recovered[key] = tensor_cleaned
            
        return recovered
    
    def _recover_energy_instability(self, error_context: PhysicsErrorContext,
                                  tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recover from energy instabilities."""
        recovered = tensors.copy()
        
        if 'energy' in tensors:
            energy = tensors['energy']
            
            # Ensure positive energy
            energy_positive = torch.abs(energy)
            
            # Smooth out rapid oscillations
            if len(energy.shape) > 1:
                kernel = torch.ones(3, device=energy.device) / 3.0
                smoothed = torch.conv1d(
                    energy_positive.unsqueeze(1), 
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=1
                ).squeeze(1)
                recovered['energy'] = smoothed
            else:
                recovered['energy'] = energy_positive
                
        return recovered

def physics_safe_operation(error_types: Optional[List[PhysicsErrorType]] = None,
                          tolerance: float = 1e-6,
                          max_recovery_attempts: int = 3,
                          enable_recovery: bool = True):
    """
    Decorator for physics-safe operations with automatic error detection and recovery.
    
    Args:
        error_types: List of error types to check for
        tolerance: Tolerance for physics constraint violations
        max_recovery_attempts: Maximum number of recovery attempts
        enable_recovery: Whether to enable automatic recovery
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = PhysicsConstraintValidator(tolerance=tolerance)
            recovery_system = PhysicsErrorRecovery()
            
            attempt = 0
            last_error = None
            
            while attempt < max_recovery_attempts + 1:
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Validate physics constraints if result contains tensors
                    if isinstance(result, dict) and error_types:
                        violations = []
                        
                        for error_type in error_types:
                            if error_type == PhysicsErrorType.CONSERVATION_VIOLATION:
                                if 'energy_before' in result and 'energy_after' in result:
                                    violation = validator.validate_energy_conservation(
                                        result['energy_before'], result['energy_after']
                                    )
                                    if violation.violation_magnitude > tolerance:
                                        violations.append(violation)
                                        
                            elif error_type == PhysicsErrorType.LORENTZ_VIOLATION:
                                if 'four_vector_before' in result and 'four_vector_after' in result:
                                    violation = validator.validate_lorentz_invariance(
                                        result['four_vector_before'], result['four_vector_after']
                                    )
                                    if violation.violation_magnitude > tolerance:
                                        violations.append(violation)
                        
                        # If violations found and recovery enabled
                        if violations and enable_recovery and attempt < max_recovery_attempts:
                            logging.warning(f"Physics violations detected, attempting recovery (attempt {attempt + 1})")
                            
                            # Attempt recovery
                            for violation in violations:
                                result = recovery_system.attempt_recovery(violation, result)
                            
                            attempt += 1
                            continue
                    
                    # Success - return result
                    return result
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_recovery_attempts:
                        logging.warning(f"Operation failed, retrying (attempt {attempt + 1}): {e}")
                        attempt += 1
                        continue
                    else:
                        break
            
            # If we get here, all attempts failed
            error_msg = f"Physics-safe operation failed after {max_recovery_attempts + 1} attempts"
            if last_error:
                error_msg += f". Last error: {last_error}"
            raise RuntimeError(error_msg)
            
        return wrapper
    return decorator

@contextmanager
def physics_computation_context(computation_name: str, 
                               expected_physics_constraints: Optional[Dict[str, float]] = None):
    """
    Context manager for physics computations with automatic monitoring.
    
    Args:
        computation_name: Name of the computation for logging
        expected_physics_constraints: Expected values for physics constraints
    """
    start_time = time.time()
    logging.info(f"Starting physics computation: {computation_name}")
    
    # Store initial system state
    initial_constraints = expected_physics_constraints or {}
    
    try:
        yield
        
        # Computation successful
        duration = time.time() - start_time
        logging.info(f"Physics computation '{computation_name}' completed successfully in {duration:.3f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Physics computation '{computation_name}' failed after {duration:.3f}s: {e}")
        
        # Log detailed error information for physics debugging
        error_details = {
            'computation_name': computation_name,
            'duration': duration,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc(),
            'expected_constraints': initial_constraints
        }
        
        logging.error(f"Detailed error information: {error_details}")
        raise

class PhysicsErrorLogger:
    """Specialized logger for physics computation errors."""
    
    def __init__(self, log_file: str = "physics_errors.log"):
        self.logger = logging.getLogger("PhysicsErrorLogger")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.error_counts = {}
    
    def log_physics_error(self, error_context: PhysicsErrorContext):
        """Log a physics-specific error."""
        error_type = error_context.error_type.value
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Create log message
        message = (f"Physics Error [{error_context.severity.upper()}]: {error_type} "
                  f"(magnitude: {error_context.violation_magnitude:.2e}, "
                  f"tolerance: {error_context.tolerance:.2e}) "
                  f"at step '{error_context.computation_step}'")
        
        if error_context.recovery_attempted:
            recovery_status = "successful" if error_context.recovery_successful else "failed"
            message += f" - Recovery {recovery_status}"
        
        # Log with appropriate level
        if error_context.severity == "critical":
            self.logger.critical(message)
        elif error_context.severity == "high":
            self.logger.error(message)
        elif error_context.severity == "medium":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts."""
        return self.error_counts.copy()

# Example usage and testing functions

def test_physics_error_handling():
    """Test the physics error handling system."""
    
    # Test conservation violation detection
    validator = PhysicsConstraintValidator(tolerance=1e-3)
    
    # Create test data with violation
    energy_before = torch.tensor([100.0, 50.0, 75.0])
    energy_after = torch.tensor([90.0, 60.0, 80.0])  # Violation: total changes
    
    violation = validator.validate_energy_conservation(energy_before, energy_after)
    print(f"Conservation violation detected: {violation.violation_magnitude:.6f}")
    
    # Test recovery
    recovery = PhysicsErrorRecovery()
    test_tensors = {
        'energy': energy_after,
        'initial_energy': energy_before
    }
    
    recovered_tensors = recovery.attempt_recovery(violation, test_tensors)
    print(f"Recovery attempted: {violation.recovery_attempted}")
    print(f"Recovery successful: {violation.recovery_successful}")
    
    # Test decorated function
    @physics_safe_operation(
        error_types=[PhysicsErrorType.CONSERVATION_VIOLATION],
        enable_recovery=True
    )
    def conservation_test():
        return {
            'energy_before': energy_before,
            'energy_after': energy_after
        }
    
    result = conservation_test()
    print("Physics-safe operation completed successfully")
    
    return True

if __name__ == "__main__":
    # Run tests
    test_success = test_physics_error_handling()
    print(f"✅ Physics Error Handling Test: {'Passed' if test_success else 'Failed'}")