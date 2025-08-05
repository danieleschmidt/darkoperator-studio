"""
DarkOperator Studio: Neural Operators for Ultra-Rare Dark Matter Detection.

A framework for accelerating Large Hadron Collider (LHC) simulations using neural operators
and detecting dark matter signatures with conformal anomaly detection.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Core imports
from .operators import CalorimeterOperator, TrackerOperator, MuonOperator
from .anomaly import ConformalDetector, MultiModalAnomalyDetector
from .models import FourierNeuralOperator
from .physics import LorentzEmbedding
from .data import load_opendata, list_opendata_datasets, download_dataset
from .visualization import visualize_event, visualize_3d, plot_operator_kernels
from .utils import OperatorTrainer, PhysicsInterpreter

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