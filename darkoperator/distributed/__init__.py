"""
Distributed Training System for DarkOperator Studio.

Provides advanced distributed training capabilities across multiple GPUs
with physics-informed optimizations and fault tolerance.
"""

from .gpu_trainer import DistributedGPUTrainer, GPUCluster
# from .data_parallel import PhysicsDataParallel, QuantumDataParallel
# from .model_parallel import ModelParallelManager, PhysicsModelShard
# from .communication import QuantumAllReduce, PhysicsGradientSync
from .auto_scaling import ResourceMonitor, AutoScaler, DistributedCoordinator

__all__ = [
    'DistributedGPUTrainer',
    'GPUCluster', 
    'ResourceMonitor',
    'AutoScaler', 
    'DistributedCoordinator',
]