"""
Minimal test suite for core operators functionality.
"""

import pytest
import numpy as np
import torch


def test_import_operators():
    """Test basic imports work."""
    from darkoperator import CalorimeterOperator, TrackerOperator, MuonOperator
    from darkoperator.anomaly import ConformalDetector
    # These should import without errors


def test_core_functionality():
    """Test basic darkoperator functionality."""
    import darkoperator as do
    
    # Test version
    assert hasattr(do, '__version__')
    assert do.__version__ == "0.1.0"
    
    # Test basic imports exist
    assert hasattr(do, 'CalorimeterOperator')
    assert hasattr(do, 'ConformalDetector')


def test_numpy_torch_integration():
    """Test numpy and torch integration."""
    # Test basic numpy functionality
    arr = np.array([1, 2, 3, 4])
    assert arr.sum() == 10
    
    # Test basic torch functionality
    tensor = torch.tensor([1.0, 2.0, 3.0])
    assert tensor.sum().item() == 6.0


def test_physics_4vector_basic():
    """Test basic 4-vector physics."""
    # Energy-momentum 4-vector: (E, px, py, pz)
    # For massless particle: E^2 = px^2 + py^2 + pz^2
    energy = 5.0
    px, py, pz = 3.0, 4.0, 0.0
    
    # Check energy-momentum relation
    momentum_squared = px**2 + py**2 + pz**2
    assert abs(energy**2 - momentum_squared) < 1e-6  # Should be ~0 for massless


if __name__ == '__main__':
    pytest.main([__file__, '-v'])