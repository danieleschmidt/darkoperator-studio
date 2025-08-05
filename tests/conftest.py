"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_4vectors():
    """Generate sample 4-vector data for testing."""
    n_events = 100
    n_particles = 5
    
    # Generate physically reasonable 4-vectors
    events = torch.randn(n_events, n_particles, 4)
    
    # Ensure positive energies
    events[:, :, 0] = torch.abs(events[:, :, 0]) + 1.0
    
    # Adjust to satisfy E^2 >= p^2 (approximately)
    px, py, pz = events[:, :, 1], events[:, :, 2], events[:, :, 3]
    p_squared = px**2 + py**2 + pz**2
    
    # For events where E^2 < p^2, adjust energy
    mask = events[:, :, 0]**2 < p_squared
    events[mask, 0] = torch.sqrt(p_squared[mask]) + 0.1
    
    return events


@pytest.fixture
def sample_jet_events():
    """Generate sample jet events for testing."""
    n_events = 200
    n_jets = 4
    
    # Realistic jet energy distribution (log-normal)
    energies = torch.distributions.LogNormal(4.0, 1.0).sample((n_events, n_jets))
    energies = torch.clamp(energies, 10, 1000)  # 10 GeV to 1 TeV
    
    # Random directions
    eta = torch.distributions.Normal(0, 2.0).sample((n_events, n_jets))
    phi = torch.distributions.Uniform(-np.pi, np.pi).sample((n_events, n_jets))
    
    # Convert to 4-vectors
    pt = energies * torch.rand_like(energies) * 0.8 + 0.2  # pT < E
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    
    # Adjust energy to be physical
    p_mag = torch.sqrt(px**2 + py**2 + pz**2)
    energies = torch.maximum(energies, p_mag + 0.1)
    
    jets = torch.stack([energies, px, py, pz], dim=-1)
    return jets


@pytest.fixture
def sample_background_events():
    """Generate background events for anomaly detection testing."""
    return torch.randn(1000, 6, 4) * torch.tensor([50, 30, 30, 30]) + torch.tensor([100, 0, 0, 0])


@pytest.fixture
def sample_signal_events():
    """Generate signal events for anomaly detection testing."""
    # Higher energy, different topology
    return torch.randn(100, 6, 4) * torch.tensor([100, 60, 60, 60]) + torch.tensor([300, 0, 0, 0])


@pytest.fixture
def mock_opendata_response():
    """Mock response for LHC Open Data API."""
    return [
        {
            "name": "test-dataset-2016",
            "experiment": "CMS",
            "year": 2016,
            "energy": "13 TeV",
            "data_type": "jets",
            "size_gb": 1.0,
            "n_events": 10000,
            "url": "http://test.opendata.cern.ch/test",
            "description": "Test dataset"
        }
    ]


@pytest.fixture(autouse=True)
def cleanup_cuda_memory():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests
        if ("performance" in item.name.lower() or 
            "scaling" in item.name.lower() or
            "large" in item.name.lower()):
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if ("integration" in item.name.lower() or
            "end_to_end" in item.name.lower()):
            item.add_marker(pytest.mark.integration)


# Skip GPU tests if CUDA not available
def pytest_runtest_setup(item):
    """Setup hook to skip GPU tests when CUDA is not available."""
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")