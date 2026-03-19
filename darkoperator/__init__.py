"""
DarkOperator Studio
===================

Neural operator surrogates for physics simulations with conformal anomaly detection.

Quick start::

    from darkoperator.models.fno1d import FourierNeuralOperator1d
    from darkoperator.anomaly.conformal_detector import ConformalAnomalyDetector
    from darkoperator.physics.pde_data import generate_heat_data
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from darkoperator.models.fno1d import FourierNeuralOperator1d
from darkoperator.anomaly.conformal_detector import ConformalAnomalyDetector
from darkoperator.physics.pde_data import generate_heat_data, generate_burgers_data

__all__ = [
    "FourierNeuralOperator1d",
    "ConformalAnomalyDetector",
    "generate_heat_data",
    "generate_burgers_data",
]
