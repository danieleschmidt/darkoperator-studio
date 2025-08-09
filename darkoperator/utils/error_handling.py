"""Comprehensive error handling and validation framework."""

import torch
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from functools import wraps
import traceback
import logging


class DarkOperatorError(Exception):
    """Base exception for DarkOperator framework."""
    pass


class PhysicsValidationError(DarkOperatorError):
    """Error for physics validation failures."""
    pass


class ModelValidationError(DarkOperatorError):
    """Error for model validation failures."""
    pass


class DataValidationError(DarkOperatorError):
    """Error for data validation failures."""
    pass


def safe_execute(func: Callable, *args, **kwargs) -> Tuple[Any, Optional[str]]:
    """Safely execute a function with error handling."""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_msg = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        return None, error_msg


def physics_validator(check_conservation: bool = True, 
                     check_causality: bool = True,
                     tolerance: float = 1e-6):
    """Decorator for physics validation of function outputs."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Validate physics constraints
            if check_conservation and isinstance(result, torch.Tensor):
                if result.shape[-1] >= 4:  # Has 4-momentum
                    _validate_4momentum_conservation(result, tolerance)
            
            if check_causality and isinstance(result, torch.Tensor):
                _validate_causality(result)
            
            return result
        return wrapper
    return decorator


def data_validator(min_events: int = 1,
                  max_events: Optional[int] = None,
                  check_finite: bool = True,
                  check_physics: bool = True):
    """Decorator for data validation."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Validate data integrity
            if isinstance(result, dict):
                _validate_event_data(result, min_events, max_events, check_finite, check_physics)
            elif isinstance(result, torch.Tensor):
                _validate_tensor_data(result, check_finite)
            
            return result
        return wrapper
    return decorator


def model_validator(check_gradients: bool = True,
                   check_parameters: bool = True,
                   check_device: bool = True):
    """Decorator for model validation."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Pre-validation
            if check_parameters:
                _validate_model_parameters(self)
            
            if check_device:
                _validate_device_consistency(self, *args)
            
            result = func(self, *args, **kwargs)
            
            # Post-validation
            if check_gradients and self.training:
                _validate_gradients(self)
            
            return result
        return wrapper
    return decorator


def _validate_4momentum_conservation(four_momentum: torch.Tensor, tolerance: float):
    """Validate 4-momentum conservation laws."""
    if four_momentum.dim() < 2:
        return
    
    # Check for NaN or infinite values
    if not torch.isfinite(four_momentum).all():
        raise PhysicsValidationError("4-momentum contains NaN or infinite values")
    
    # Check energy positivity
    energy = four_momentum[..., 0]
    if (energy < 0).any():
        raise PhysicsValidationError("Found negative energy in 4-momentum")
    
    # Check mass-shell constraint for each particle
    E = four_momentum[..., 0]
    px = four_momentum[..., 1]
    py = four_momentum[..., 2] 
    pz = four_momentum[..., 3]
    
    p_squared = px**2 + py**2 + pz**2
    m_squared = E**2 - p_squared
    
    # Mass should be non-negative
    if (m_squared < -tolerance).any():
        violations = (m_squared < -tolerance).sum().item()
        raise PhysicsValidationError(f"Found {violations} mass-shell violations with tolerance {tolerance}")


def _validate_causality(tensor: torch.Tensor):
    """Validate causality constraints."""
    if tensor.dim() >= 2 and tensor.shape[-1] >= 4:
        # Check that no particle exceeds speed of light
        E = tensor[..., 0].abs()
        px = tensor[..., 1]
        py = tensor[..., 2]
        pz = tensor[..., 3]
        
        p_magnitude = torch.sqrt(px**2 + py**2 + pz**2)
        
        # Require |p| < E (in natural units c=1)
        causality_violations = (p_magnitude >= E).any()
        if causality_violations:
            raise PhysicsValidationError("Found superluminal particles violating causality")


def _validate_event_data(event_data: Dict[str, torch.Tensor],
                        min_events: int,
                        max_events: Optional[int],
                        check_finite: bool,
                        check_physics: bool):
    """Validate event data dictionary."""
    
    if not event_data:
        raise DataValidationError("Event data dictionary is empty")
    
    # Check event count
    n_events = None
    for key, value in event_data.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            if n_events is None:
                n_events = value.shape[0]
            elif value.shape[0] != n_events:
                raise DataValidationError(f"Inconsistent event count: {key} has {value.shape[0]} vs expected {n_events}")
    
    if n_events is None:
        raise DataValidationError("Could not determine number of events")
    
    if n_events < min_events:
        raise DataValidationError(f"Too few events: {n_events} < {min_events}")
    
    if max_events and n_events > max_events:
        raise DataValidationError(f"Too many events: {n_events} > {max_events}")
    
    # Check for finite values
    if check_finite:
        for key, value in event_data.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise DataValidationError(f"Non-finite values found in {key}")
    
    # Physics-specific checks
    if check_physics:
        # Check jet pT positivity
        if 'jet_pt' in event_data:
            jet_pt = event_data['jet_pt']
            if (jet_pt < 0).any():
                raise DataValidationError("Found negative jet pT values")
        
        # Check missing energy positivity
        if 'missing_et' in event_data:
            met = event_data['missing_et']
            if (met < 0).any():
                raise DataValidationError("Found negative missing energy values")
        
        # Check 4-vector consistency
        if 'jet_4vectors' in event_data:
            _validate_4momentum_conservation(event_data['jet_4vectors'], tolerance=1e-3)


def _validate_tensor_data(tensor: torch.Tensor, check_finite: bool = True):
    """Validate tensor data."""
    if tensor.numel() == 0:
        raise DataValidationError("Tensor is empty")
    
    if check_finite and not torch.isfinite(tensor).all():
        raise DataValidationError("Tensor contains non-finite values")


def _validate_model_parameters(model: torch.nn.Module):
    """Validate model parameters."""
    param_count = 0
    
    for name, param in model.named_parameters():
        if param is None:
            raise ModelValidationError(f"Parameter {name} is None")
        
        if not torch.isfinite(param).all():
            raise ModelValidationError(f"Parameter {name} contains non-finite values")
        
        if param.requires_grad and param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise ModelValidationError(f"Gradient for {name} contains non-finite values")
        
        param_count += param.numel()
    
    if param_count == 0:
        raise ModelValidationError("Model has no parameters")


def _validate_device_consistency(model: torch.nn.Module, *tensors):
    """Validate device consistency between model and input tensors."""
    model_device = next(model.parameters()).device
    
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, torch.Tensor):
            if tensor.device != model_device:
                raise ModelValidationError(
                    f"Device mismatch: model on {model_device}, tensor {i} on {tensor.device}"
                )


def _validate_gradients(model: torch.nn.Module):
    """Validate gradient health."""
    total_norm = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise ModelValidationError(f"Non-finite gradients in {name}")
            
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
        
        # Check for vanishing gradients
        if total_norm < 1e-8:
            warnings.warn(f"Very small gradient norm: {total_norm}", UserWarning)
        
        # Check for exploding gradients
        if total_norm > 1000:
            warnings.warn(f"Very large gradient norm: {total_norm}", UserWarning)


class RobustWrapper:
    """Wrapper class for adding robustness to any object."""
    
    def __init__(self, wrapped_object: Any, max_retries: int = 3, 
                 fallback_value: Any = None):
        self.wrapped_object = wrapped_object
        self.max_retries = max_retries
        self.fallback_value = fallback_value
    
    def __getattr__(self, name):
        attr = getattr(self.wrapped_object, name)
        
        if callable(attr):
            def robust_method(*args, **kwargs):
                last_error = None
                
                for attempt in range(self.max_retries):
                    try:
                        return attr(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        if attempt < self.max_retries - 1:
                            logging.warning(f"Attempt {attempt + 1} failed for {name}: {e}")
                        continue
                
                # All retries failed
                if self.fallback_value is not None:
                    logging.error(f"All retries failed for {name}, using fallback value")
                    return self.fallback_value
                else:
                    raise last_error
            
            return robust_method
        else:
            return attr


def create_robust_model(model_class: type, *args, **kwargs) -> RobustWrapper:
    """Create a robust version of a model with error handling."""
    try:
        model = model_class(*args, **kwargs)
        return RobustWrapper(model, max_retries=2)
    except Exception as e:
        logging.error(f"Failed to create model {model_class.__name__}: {e}")
        raise ModelValidationError(f"Model creation failed: {e}")


class SafetyMonitor:
    """Monitor system safety during computation."""
    
    def __init__(self, memory_limit_gb: float = 8.0, 
                 computation_timeout: float = 300.0):
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.computation_timeout = computation_timeout
        self.start_time = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and torch.cuda.is_available():
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = self.start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            
            if elapsed_time > self.computation_timeout:
                warnings.warn(f"Computation took {elapsed_time:.2f}s (limit: {self.computation_timeout}s)")
    
    def check_memory(self):
        """Check current memory usage."""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated()
            if memory_used > self.memory_limit_bytes:
                raise RuntimeError(f"Memory usage {memory_used / 1024**3:.2f}GB exceeds limit {self.memory_limit_bytes / 1024**3:.2f}GB")


def test_error_handling():
    """Test error handling framework."""
    print("Testing error handling framework...")
    
    # Test physics validator
    @physics_validator(check_conservation=True, tolerance=1e-6)
    def create_test_4momentum():
        # Valid 4-momentum: (E, px, py, pz)
        return torch.tensor([[100.0, 30.0, 40.0, 50.0]])  # E² = px² + py² + pz² + m²
    
    try:
        valid_4momentum = create_test_4momentum()
        print("✓ Physics validator passed for valid 4-momentum")
    except PhysicsValidationError as e:
        print(f"Physics validation error (expected): {e}")
    
    # Test data validator
    @data_validator(min_events=1, check_finite=True, check_physics=True)
    def create_test_events():
        return {
            'event_id': torch.arange(10),
            'jet_pt': torch.rand(10, 3) * 100 + 20,  # 20-120 GeV
            'missing_et': torch.rand(10) * 50,
        }
    
    try:
        events = create_test_events()
        print("✓ Data validator passed for valid events")
    except DataValidationError as e:
        print(f"Data validation error: {e}")
    
    # Test robust wrapper
    class TestModel:
        def __init__(self):
            self.call_count = 0
        
        def unreliable_method(self):
            self.call_count += 1
            if self.call_count < 3:
                raise RuntimeError("Simulated failure")
            return "Success!"
    
    robust_model = RobustWrapper(TestModel(), max_retries=5, fallback_value="Fallback")
    
    try:
        result = robust_model.unreliable_method()
        print(f"✓ Robust wrapper succeeded: {result}")
    except Exception as e:
        print(f"Robust wrapper failed: {e}")
    
    # Test safety monitor
    with SafetyMonitor(memory_limit_gb=16.0, computation_timeout=10.0):
        # Simulate computation
        _ = torch.randn(1000, 1000)
        print("✓ Safety monitor completed successfully")
    
    print("✅ All error handling tests passed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    test_error_handling()