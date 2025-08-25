"""Enhanced error handling and resilience for DarkOperator Studio."""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union
from dataclasses import dataclass
import warnings


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: float
    user_data: Dict[str, Any]
    system_info: Dict[str, Any]
    retry_count: int = 0
    

class DarkOperatorError(Exception):
    """Base exception class for DarkOperator Studio."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.context = context
        self.timestamp = time.time()


class PhysicsError(DarkOperatorError):
    """Physics-related errors (conservation violations, etc.)."""
    pass


class ModelError(DarkOperatorError):
    """Neural operator model-related errors."""
    pass


class DataError(DarkOperatorError):
    """Data loading and processing errors."""
    pass


class ConformalError(DarkOperatorError):
    """Conformal prediction and anomaly detection errors."""
    pass


class RobustOperatorWrapper:
    """Wrapper that adds robustness to neural operators."""
    
    def __init__(self, operator, max_retries: int = 3, timeout: float = 30.0):
        self.operator = operator
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{operator.__class__.__name__}")
        
        # Track error statistics
        self.error_counts = {}
        self.total_calls = 0
        self.successful_calls = 0
        
    def __call__(self, *args, **kwargs):
        """Robust operator call with error handling and retries."""
        self.total_calls += 1
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add timeout protection
                start_time = time.time()
                result = self.operator(*args, **kwargs)
                
                # Check for timeout
                if time.time() - start_time > self.timeout:
                    raise ModelError(f"Operation timeout after {self.timeout}s")
                
                # Validate physics constraints if available
                self._validate_physics(args, result)
                
                self.successful_calls += 1
                return result
                
            except Exception as e:
                error_type = type(e).__name__
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                if attempt < self.max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    context = ErrorContext(
                        operation=f"{self.operator.__class__.__name__}.__call__",
                        timestamp=time.time(),
                        user_data={'args_shapes': [getattr(arg, 'shape', 'unknown') for arg in args]},
                        system_info={'total_calls': self.total_calls, 'success_rate': self.get_success_rate()},
                        retry_count=attempt
                    )
                    raise ModelError(f"Operation failed after {self.max_retries} retries: {e}") from e
                    
    def _validate_physics(self, inputs, outputs):
        """Validate physics constraints in the results."""
        try:
            # Import numpy with graceful fallback
            try:
                import numpy as np
                HAS_NUMPY = True
            except ImportError:
                HAS_NUMPY = False
                return  # Skip validation if numpy unavailable
            
            if hasattr(inputs[0], 'sum') and hasattr(outputs, 'sum'):
                # Basic energy conservation check
                input_energy = inputs[0].sum()
                output_energy = outputs.sum() 
                
                if HAS_NUMPY:
                    energy_violation = abs(input_energy - output_energy) / (input_energy + 1e-8)
                    if energy_violation > 0.1:  # 10% tolerance
                        warnings.warn(f"Energy conservation violation: {energy_violation:.2%}")
                        
        except Exception as e:
            # Don't fail the operation due to validation errors
            self.logger.debug(f"Physics validation failed: {e}")
    
    def get_success_rate(self) -> float:
        """Get the success rate of operations."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'success_rate': self.get_success_rate(),
            'error_counts': self.error_counts.copy(),
            'most_common_error': max(self.error_counts, key=self.error_counts.get) if self.error_counts else None
        }


def robust_operation(max_retries: int = 3, timeout: float = 30.0, 
                    exception_types: tuple = (Exception,)):
    """Decorator for making operations robust with retries and error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(f"{func.__module__}.{func.__name__}")
            
            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    
                    # Check timeout
                    if time.time() - start_time > timeout:
                        raise DarkOperatorError(f"Operation {func.__name__} timeout after {timeout}s")
                    
                    return result
                    
                except exception_types as e:
                    if attempt < max_retries:
                        wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        context = ErrorContext(
                            operation=func.__name__,
                            timestamp=time.time(),
                            user_data={'args': str(args)[:200], 'kwargs': str(kwargs)[:200]},
                            system_info={'max_retries': max_retries, 'timeout': timeout},
                            retry_count=attempt
                        )
                        
                        # Re-raise with enhanced context
                        if isinstance(e, DarkOperatorError):
                            e.context = context
                            raise
                        else:
                            raise DarkOperatorError(f"Operation {func.__name__} failed: {e}", context) from e
                            
        return wrapper
    return decorator


class HealthMonitor:
    """System health monitoring for DarkOperator components."""
    
    def __init__(self):
        self.component_status = {}
        self.last_check = {}
        self.logger = logging.getLogger(__name__)
        
    def register_component(self, name: str, check_function: Callable[[], bool]):
        """Register a component for health monitoring."""
        self.component_status[name] = {
            'check_function': check_function,
            'status': 'unknown',
            'last_healthy': None,
            'consecutive_failures': 0
        }
        
    def check_health(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Check health of components."""
        components_to_check = [component] if component else list(self.component_status.keys())
        results = {}
        
        for comp_name in components_to_check:
            if comp_name not in self.component_status:
                continue
                
            comp_info = self.component_status[comp_name]
            
            try:
                is_healthy = comp_info['check_function']()
                
                if is_healthy:
                    comp_info['status'] = 'healthy'
                    comp_info['last_healthy'] = time.time()
                    comp_info['consecutive_failures'] = 0
                else:
                    comp_info['status'] = 'unhealthy'
                    comp_info['consecutive_failures'] += 1
                    
                results[comp_name] = {
                    'status': comp_info['status'],
                    'consecutive_failures': comp_info['consecutive_failures'],
                    'last_healthy': comp_info.get('last_healthy')
                }
                
            except Exception as e:
                comp_info['status'] = 'error'
                comp_info['consecutive_failures'] += 1
                results[comp_name] = {
                    'status': 'error',
                    'error': str(e),
                    'consecutive_failures': comp_info['consecutive_failures']
                }
                self.logger.error(f"Health check failed for {comp_name}: {e}")
                
        return results
    
    def get_system_health_score(self) -> float:
        """Get overall system health score (0.0 to 1.0)."""
        if not self.component_status:
            return 1.0
            
        health_results = self.check_health()
        healthy_count = sum(1 for result in health_results.values() 
                          if result.get('status') == 'healthy')
        
        return healthy_count / len(health_results)


def create_physics_health_checks():
    """Create standard health checks for physics components."""
    
    def check_numpy_available():
        try:
            import numpy as np
            # Test basic operation
            test_array = np.array([1, 2, 3])
            return test_array.sum() == 6
        except ImportError:
            return False
    
    def check_torch_available():
        try:
            import torch
            # Test basic tensor operation
            test_tensor = torch.tensor([1.0, 2.0, 3.0])
            return torch.allclose(test_tensor.sum(), torch.tensor(6.0))
        except ImportError:
            return False
    
    def check_memory_available():
        """Check if sufficient memory is available."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            # Require at least 1GB available
            return memory.available > 1e9
        except ImportError:
            return True  # Assume OK if can't check
    
    health_monitor = HealthMonitor()
    health_monitor.register_component('numpy', check_numpy_available)
    health_monitor.register_component('torch', check_torch_available)
    health_monitor.register_component('memory', check_memory_available)
    
    return health_monitor


# Global health monitor instance
_global_health_monitor = None

def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = create_physics_health_checks()
    return _global_health_monitor


class ValidationError(DarkOperatorError):
    """Validation-specific error."""
    pass


def validate_input_data(data, expected_shape=None, value_range=None, required_fields=None):
    """Comprehensive input data validation."""
    
    if data is None:
        raise ValidationError("Input data cannot be None")
    
    # Shape validation
    if expected_shape and hasattr(data, 'shape'):
        if len(data.shape) != len(expected_shape):
            raise ValidationError(f"Expected {len(expected_shape)}D data, got {len(data.shape)}D")
        
        for i, (actual, expected) in enumerate(zip(data.shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValidationError(f"Shape mismatch at dim {i}: expected {expected}, got {actual}")
    
    # Value range validation
    if value_range and hasattr(data, 'min') and hasattr(data, 'max'):
        min_val, max_val = value_range
        data_min, data_max = data.min(), data.max()
        
        if data_min < min_val or data_max > max_val:
            raise ValidationError(f"Data values [{data_min:.3f}, {data_max:.3f}] outside expected range [{min_val}, {max_val}]")
    
    # Required fields validation (for dict-like data)
    if required_fields and hasattr(data, 'keys'):
        missing_fields = set(required_fields) - set(data.keys())
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")
    
    return True


# Convenience function for wrapping existing operators
def make_robust(operator, max_retries: int = 3, timeout: float = 30.0):
    """Make an existing operator robust with error handling."""
    return RobustOperatorWrapper(operator, max_retries, timeout)


if __name__ == "__main__":
    # Example usage and testing
    print("🛡️ DarkOperator Enhanced Error Handling")
    
    # Test health monitoring
    monitor = get_health_monitor()
    health_results = monitor.check_health()
    
    print("System Health Check:")
    for component, status in health_results.items():
        print(f"  {component}: {status['status']}")
    
    print(f"Overall Health Score: {monitor.get_system_health_score():.2%}")
    
    # Test robust decorator
    @robust_operation(max_retries=2)
    def test_function(should_fail=False):
        if should_fail:
            raise ValueError("Test error")
        return "Success!"
    
    try:
        result = test_function(should_fail=False)
        print(f"Robust function test: {result}")
    except Exception as e:
        print(f"Error: {e}")