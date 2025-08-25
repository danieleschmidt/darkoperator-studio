"""
Distributed Training System for DarkOperator Studio.

Provides advanced distributed training capabilities across multiple GPUs
with physics-informed optimizations and fault tolerance.
"""

# Legacy distributed components
try:
    from .gpu_trainer import DistributedGPUTrainer, GPUCluster
    GPU_TRAINING_AVAILABLE = True
except ImportError:
    class DistributedGPUTrainer:
        pass
    class GPUCluster:
        pass
    GPU_TRAINING_AVAILABLE = False

try:
    from .auto_scaling import ResourceMonitor, AutoScaler, DistributedCoordinator
    LEGACY_SCALING_AVAILABLE = True
except ImportError:
    class ResourceMonitor:
        pass
    class AutoScaler:
        pass
    class DistributedCoordinator:
        pass
    LEGACY_SCALING_AVAILABLE = False

# New Generation 3 intelligent scaling (lightweight)
try:
    from .intelligent_scaling import (
        IntelligentAutoscaler, PredictiveLoadForecaster, ResourcePool,
        ScalingMetrics, ResourceNode, get_autoscaler, create_sample_resource_pool
    )
    INTELLIGENT_SCALING_AVAILABLE = True
except ImportError:
    INTELLIGENT_SCALING_AVAILABLE = False

__all__ = [
    'DistributedGPUTrainer',
    'GPUCluster', 
    'ResourceMonitor',
    'AutoScaler', 
    'DistributedCoordinator',
]

if INTELLIGENT_SCALING_AVAILABLE:
    __all__.extend([
        'IntelligentAutoscaler', 'PredictiveLoadForecaster', 'ResourcePool',
        'ScalingMetrics', 'ResourceNode', 'get_autoscaler', 'create_sample_resource_pool'
    ])