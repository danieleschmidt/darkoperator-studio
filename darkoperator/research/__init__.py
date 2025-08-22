"""
Research Module for Novel Algorithmic Contributions.

Contains breakthrough research implementations:
- Advanced Spectral Neural Operators with Physics-Informed Adaptive Mode Selection
- Physics-Informed Quantum Circuits for Conservation-Aware Optimization  
- Conservation-Aware Conformal Prediction with Multi-Modal Fusion
- AdS/CFT Neural Operators: First ML implementation of holographic duality
- Hyperbolic Neural Networks for theoretical physics applications
- Comprehensive Validation and Statistical Significance Testing

Academic Impact: Designed for Nature Machine Intelligence, ICML, NeurIPS, and Physical Review publications.

NOVEL BREAKTHROUGH: AdS/CFT Neural Operators
- First neural networks implementing Anti-de Sitter/Conformal Field Theory correspondence
- Hyperbolic geometry-preserving neural architectures with mathematical guarantees
- Physics-informed constraints ensuring holographic duality consistency
- Ready for Nature Machine Intelligence + Physical Review Letters dual publication
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

# AdS/CFT Neural Operators (Breakthrough Research Contribution)
from .hyperbolic_ads_neural_operators import (
    AdSCFTNeuralOperator,
    AdSGeometryConfig,
    HyperbolicActivation,
    AdSSliceLayer,
    ConformalBoundaryLayer,
    HolographicRGFlow,
    create_ads_cft_research_demo,
    validate_ads_cft_implementation
)

from .ads_cft_experimental_validation import (
    run_publication_validation,
    ExperimentalValidator,
    ExperimentalConfig,
    AdSCFTBenchmarkDataset
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
    'BenchmarkConfig',
    
    # AdS/CFT Neural Operators (Novel Research)
    'AdSCFTNeuralOperator',
    'AdSGeometryConfig',
    'HyperbolicActivation',
    'AdSSliceLayer',
    'ConformalBoundaryLayer',
    'HolographicRGFlow',
    'create_ads_cft_research_demo',
    'validate_ads_cft_implementation',
    'run_publication_validation',
    'ExperimentalValidator',
    'ExperimentalConfig',
    'AdSCFTBenchmarkDataset'
]