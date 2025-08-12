"""
Research Module for Novel Algorithmic Contributions.

Contains breakthrough research implementations:
- Advanced Spectral Neural Operators with Physics-Informed Adaptive Mode Selection
- Physics-Informed Quantum Circuits for Conservation-Aware Optimization  
- Conservation-Aware Conformal Prediction with Multi-Modal Fusion
- Comprehensive Validation and Statistical Significance Testing

Academic Impact: Designed for Nature Machine Intelligence, ICML, NeurIPS, and Physical Review publications.
"""

from .adaptive_spectral_fno import (
    UncertaintyAwareFNO,
    MultiScaleOperatorLearning,
    AdaptiveSpectralConv3d,
    PhysicsScales
)

from .physics_informed_quantum_circuits import (
    VariationalQuantumPhysicsOptimizer,
    PhysicsInformedQuantumCircuit,
    GaugeInvariantQuantumState,
    QuantumPhysicsConfig
)

from .conservation_aware_conformal import (
    ConservationAwareConformalDetector,
    MultiModalPhysicsConformal,
    PhysicsConformityConfig,
    ConservationAwareConformityScore
)

from .validation_benchmarks import (
    ResearchValidationSuite,
    BenchmarkConfig
)

__all__ = [
    # Advanced Spectral Neural Operators
    'UncertaintyAwareFNO',
    'MultiScaleOperatorLearning', 
    'AdaptiveSpectralConv3d',
    'PhysicsScales',
    
    # Physics-Informed Quantum Circuits
    'VariationalQuantumPhysicsOptimizer',
    'PhysicsInformedQuantumCircuit',
    'GaugeInvariantQuantumState', 
    'QuantumPhysicsConfig',
    
    # Conservation-Aware Conformal Prediction
    'ConservationAwareConformalDetector',
    'MultiModalPhysicsConformal',
    'PhysicsConformityConfig',
    'ConservationAwareConformityScore',
    
    # Validation and Benchmarking
    'ResearchValidationSuite',
    'BenchmarkConfig'
]