"""
DarkOperator Studio: Neural Operators for Ultra-Rare Dark Matter Detection.

A framework for accelerating Large Hadron Collider (LHC) simulations using neural operators
and detecting dark matter signatures with conformal anomaly detection.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Graceful dependency handling for production environments
_OPTIONAL_DEPENDENCIES = {
    'torch': 'PyTorch neural networks',
    'numpy': 'Numerical operations',
    'scipy': 'Scientific computing',
    'matplotlib': 'Plotting and visualization',
    'pandas': 'Data manipulation',
    'h5py': 'HDF5 data format support'
}

def _check_dependencies():
    """Check and warn about missing optional dependencies."""
    missing = []
    for dep, desc in _OPTIONAL_DEPENDENCIES.items():
        try:
            __import__(dep)
        except ImportError:
            missing.append(f"{dep} ({desc})")
    
    if missing:
        import warnings
        warnings.warn(
            f"Optional dependencies missing: {', '.join(missing)}. "
            "Some functionality may be limited. Install with: pip install darkoperator[full]",
            UserWarning
        )

_check_dependencies()

# Core imports with graceful fallbacks
try:
    from .operators import CalorimeterOperator, TrackerOperator, MuonOperator
    from .anomaly import ConformalDetector, MultiModalAnomalyDetector
    from .models import FourierNeuralOperator
    from .physics import LorentzEmbedding
except ImportError as e:
    import warnings
    warnings.warn(f"Core ML components unavailable: {e}. Install ML dependencies.", ImportWarning)
    # Provide stub classes for documentation/testing
    class CalorimeterOperator: pass
    class TrackerOperator: pass 
    class MuonOperator: pass
    class ConformalDetector: pass
    class MultiModalAnomalyDetector: pass
    class FourierNeuralOperator: pass
    class LorentzEmbedding: pass

# Data and visualization imports (less dependencies)
try:
    from .data import load_opendata, list_opendata_datasets, download_dataset
except ImportError:
    def load_opendata(*args, **kwargs):
        raise ImportError("Data utilities require additional dependencies")
    def list_opendata_datasets(*args, **kwargs):
        raise ImportError("Data utilities require additional dependencies")
    def download_dataset(*args, **kwargs):
        raise ImportError("Data utilities require additional dependencies")

try:
    from .visualization import visualize_event, visualize_3d, plot_operator_kernels
except ImportError:
    def visualize_event(*args, **kwargs):
        raise ImportError("Visualization requires matplotlib and plotting dependencies")
    def visualize_3d(*args, **kwargs):
        raise ImportError("Visualization requires matplotlib and plotting dependencies")  
    def plot_operator_kernels(*args, **kwargs):
        raise ImportError("Visualization requires matplotlib and plotting dependencies")

try:
    from .utils import OperatorTrainer, PhysicsInterpreter
except ImportError:
    class OperatorTrainer: pass
    class PhysicsInterpreter: pass

__all__ = [
    # Core components
    "CalorimeterOperator",
    "TrackerOperator", 
    "MuonOperator",
    "ConformalDetector",
    "MultiModalAnomalyDetector",
    "FourierNeuralOperator",
    "LorentzEmbedding",
    
    # Data utilities
    "load_opendata",
    "list_opendata_datasets", 
    "download_dataset",
    
    # Visualization
    "visualize_event",
    "visualize_3d",
    "plot_operator_kernels",
    
    # Training and analysis
    "OperatorTrainer",
    "PhysicsInterpreter",
]