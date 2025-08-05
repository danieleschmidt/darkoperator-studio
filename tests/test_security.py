"""Security tests for DarkOperator Studio."""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from darkoperator.security.model_security import SecureModelLoader, SecurityError, validate_checkpoint
from darkoperator.utils.validation import validate_4vectors, ValidationError, sanitize_file_path


class TestSecureModelLoader:
    """Test secure model loading functionality."""
    
    def test_trusted_url_validation(self):
        """Test URL validation for trusted domains."""
        loader = SecureModelLoader()
        
        # Trusted URLs
        assert loader._is_trusted_url("https://huggingface.co/model.pt")
        assert loader._is_trusted_url("https://github.com/user/repo/model.pt")
        assert loader._is_trusted_url("https://opendata.cern.ch/data/model.pt")
        
        # Untrusted URLs
        assert not loader._is_trusted_url("https://malicious-site.com/model.pt")
        assert not loader._is_trusted_url("http://random-domain.net/model.pt")
    
    def test_checkpoint_validation(self):
        """Test checkpoint file validation."""
        # Create a valid temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint = {'model_state_dict': {'weight': torch.randn(3, 3)}}
            torch.save(checkpoint, f.name)
            temp_path = f.name
        
        try:
            # Should pass validation
            assert validate_checkpoint(temp_path) == True
        finally:
            Path(temp_path).unlink()
        
        # Test non-existent file
        with pytest.raises(SecurityError):
            validate_checkpoint("nonexistent.pt")


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_4vector_validation_basic(self):
        """Test basic 4-vector validation."""
        # Valid input
        valid_4vec = torch.tensor([[[100.0, 50.0, 30.0, 20.0]]])
        result = validate_4vectors(valid_4vec)
        assert result.shape == valid_4vec.shape
        assert torch.allclose(result, valid_4vec)
    
    def test_4vector_energy_correction(self):
        """Test energy conservation correction."""
        # Invalid 4-vector (E < |p|)
        invalid_4vec = torch.tensor([[[10.0, 50.0, 30.0, 20.0]]])  # E=10 < |p|≈62
        result = validate_4vectors(invalid_4vec)
        
        # Energy should be corrected
        energy = result[0, 0, 0]
        px, py, pz = result[0, 0, 1], result[0, 0, 2], result[0, 0, 3]
        p_mag = torch.sqrt(px**2 + py**2 + pz**2)
        
        assert energy >= p_mag  # Energy should now satisfy E >= |p|
    
    def test_negative_energy_correction(self):
        """Test negative energy correction."""
        negative_energy = torch.tensor([[[-50.0, 30.0, 20.0, 10.0]]])
        result = validate_4vectors(negative_energy)
        
        # Energy should be positive
        assert torch.all(result[:, :, 0] > 0)
    
    def test_nan_inf_handling(self):
        """Test NaN and infinity handling."""
        # Input with NaN
        nan_input = torch.tensor([[[torch.nan, 30.0, 20.0, 10.0]]])
        result = validate_4vectors(nan_input)
        assert not torch.isnan(result).any()
        
        # Input with infinity
        inf_input = torch.tensor([[[torch.inf, 30.0, 20.0, 10.0]]])
        result = validate_4vectors(inf_input)
        assert not torch.isinf(result).any()
    
    def test_invalid_dimensions(self):
        """Test validation of invalid input dimensions."""
        # Wrong last dimension
        with pytest.raises(ValidationError):
            validate_4vectors(torch.randn(2, 3))  # Should be (..., 4)
        
        # Too few dimensions
        with pytest.raises(ValidationError):
            validate_4vectors(torch.randn(4))  # Should be at least 2D
    
    def test_file_path_sanitization(self):
        """Test file path sanitization."""
        # Safe paths
        assert sanitize_file_path("model.pt") == "model.pt"
        assert sanitize_file_path("data/events.npz") == "data/events.npz"
        
        # Dangerous paths
        with pytest.raises(ValidationError):
            sanitize_file_path("../../../etc/passwd")
        
        with pytest.raises(ValidationError):
            sanitize_file_path("/absolute/path")
        
        with pytest.raises(ValidationError):
            sanitize_file_path("file; rm -rf /")
        
        with pytest.raises(ValidationError):
            sanitize_file_path("file`malicious`")


class TestPhysicsConstraints:
    """Test physics constraint validation."""
    
    def test_energy_momentum_relation(self):
        """Test E^2 >= p^2 constraint enforcement."""
        # Create batch with various constraint violations
        batch = torch.tensor([
            [[100.0, 50.0, 30.0, 20.0]],  # Valid: E=100, |p|≈62
            [[10.0, 50.0, 30.0, 20.0]],   # Invalid: E=10, |p|≈62
            [[200.0, 100.0, 100.0, 50.0]], # Valid: E=200, |p|≈150
        ])
        
        result = validate_4vectors(batch)
        
        # Check constraint satisfaction
        for i in range(batch.shape[0]):
            energy = result[i, 0, 0]
            px, py, pz = result[i, 0, 1], result[i, 0, 2], result[i, 0, 3]
            p_squared = px**2 + py**2 + pz**2
            
            # Allow small numerical errors
            assert energy**2 >= p_squared - 1e-6
    
    def test_massless_particles(self):
        """Test handling of massless particles (photons, gluons)."""
        # Photon-like 4-vector: E = |p|
        photon_energy = 100.0
        photon = torch.tensor([[[photon_energy, photon_energy, 0.0, 0.0]]])
        
        result = validate_4vectors(photon)
        
        # Should remain unchanged for valid massless particle
        assert torch.allclose(result, photon, atol=1e-6)
    
    def test_massive_particles(self):
        """Test handling of massive particles."""
        # Electron-like 4-vector with rest mass
        mass = 0.511  # MeV
        momentum = 100.0
        energy = np.sqrt(momentum**2 + mass**2)
        
        electron = torch.tensor([[[energy, momentum, 0.0, 0.0]]]))
        result = validate_4vectors(electron)
        
        # Should remain approximately unchanged for valid massive particle
        assert torch.allclose(result, electron, atol=1e-3)