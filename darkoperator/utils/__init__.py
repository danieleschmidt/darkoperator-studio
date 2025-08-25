"""Utility functions and training helpers."""

try:
    from .training import OperatorTrainer
    from .analysis import PhysicsInterpreter
    # from .metrics import physics_metrics
    ML_COMPONENTS_AVAILABLE = True
except ImportError:
    # Create stub classes when ML dependencies unavailable
    class OperatorTrainer:
        pass
    class PhysicsInterpreter:
        pass
    ML_COMPONENTS_AVAILABLE = False

# Enhanced error handling and monitoring (lightweight)
try:
    from .enhanced_error_handling import (
        RobustOperatorWrapper, robust_operation, HealthMonitor,
        get_health_monitor, validate_input_data, make_robust
    )
    ENHANCED_ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ENHANCED_ERROR_HANDLING_AVAILABLE = False

__all__ = ["OperatorTrainer", "PhysicsInterpreter"]

if ENHANCED_ERROR_HANDLING_AVAILABLE:
    __all__.extend([
        "RobustOperatorWrapper", "robust_operation", "HealthMonitor",
        "get_health_monitor", "validate_input_data", "make_robust"
    ])