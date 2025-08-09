#!/usr/bin/env python3
"""
Quick test to verify the fixed components work correctly.
"""

import torch
import darkoperator as do
from darkoperator.models import ConvolutionalAutoencoder, VariationalAutoencoder
from darkoperator.physics import ConservationLoss
from darkoperator.anomaly import ConformalDetector


def test_autoencoder_fix():
    """Test that autoencoder shape issues are fixed."""
    print("Testing autoencoder fix...")
    
    # Test Convolutional Autoencoder
    conv_ae = ConvolutionalAutoencoder(input_channels=1, latent_dim=64)
    test_input = torch.randn(4, 1, 50, 50)
    
    # Test forward pass
    output = conv_ae(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} vs {test_input.shape}"
    print("✅ Autoencoder shape test passed")


def test_fno_fix():
    """Test FourierNeuralOperator initialization."""
    print("Testing FourierNeuralOperator...")
    
    # Check what parameters FNO actually accepts
    try:
        fno = do.FourierNeuralOperator(modes=16, width=32)
        print("✅ FourierNeuralOperator created with basic parameters")
        
        # Test with input
        test_input = torch.randn(2, 4, 32, 32)
        output = fno(test_input) 
        print(f"FNO: {test_input.shape} -> {output.shape}")
        
    except Exception as e:
        print(f"FNO error: {e}")
        # Try with different parameters
        try:
            fno = do.FourierNeuralOperator(4, 4, 16, 32)  # Positional args
            test_input = torch.randn(2, 4, 32, 32)
            output = fno(test_input)
            print("✅ FourierNeuralOperator works with positional args")
        except Exception as e2:
            print(f"FNO still failing: {e2}")


def test_conformal_detector_fix():
    """Test ConformalDetector initialization."""
    print("Testing ConformalDetector...")
    
    # Create a simple autoencoder
    autoencoder = ConvolutionalAutoencoder(input_channels=1, latent_dim=32)
    
    try:
        # Try with model parameter
        detector = do.ConformalDetector(model=autoencoder, calibration_alpha=0.1)
        print("✅ ConformalDetector created with model parameter")
    except Exception as e:
        print(f"ConformalDetector with model failed: {e}")
        
        try:
            # Try without model parameter
            detector = do.ConformalDetector(calibration_alpha=0.1)
            print("✅ ConformalDetector created without model")
        except Exception as e2:
            print(f"ConformalDetector still failing: {e2}")


def main():
    """Run all fix tests."""
    print("🔧 Testing Quality Gate Fixes")
    print("=" * 40)
    
    test_autoencoder_fix()
    print()
    test_fno_fix()
    print()  
    test_conformal_detector_fix()
    
    print("\n🎯 Fix testing completed!")


if __name__ == "__main__":
    main()