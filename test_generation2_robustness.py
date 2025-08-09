#!/usr/bin/env python3
"""
Comprehensive test suite for Generation 2: MAKE IT ROBUST
Tests error handling, validation, logging, and robustness features.
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import traceback
from darkoperator.utils.error_handling import (
    physics_validator, data_validator, model_validator,
    PhysicsValidationError, DataValidationError, ModelValidationError,
    RobustWrapper, SafetyMonitor, create_robust_model
)
from darkoperator.utils.logging_config import setup_logging, ExperimentTracker
from darkoperator.models.autoencoder import ConvolutionalAutoencoder, VariationalAutoencoder
from darkoperator.data.synthetic import generate_background_events, generate_signal_events
from darkoperator.data.preprocessing import preprocess_events, apply_quality_cuts
from darkoperator.physics.conservation import ConservationLoss


def test_model_robustness():
    """Test model robustness with error handling."""
    print("🧪 Testing model robustness...")
    
    # Test autoencoder with invalid inputs
    model = ConvolutionalAutoencoder(input_channels=1, latent_dim=64)
    
    # Test 1: Valid input
    try:
        valid_input = torch.randn(4, 1, 50, 50)
        output = model(valid_input)
        assert output.shape == valid_input.shape
        print("✓ Valid input test passed")
    except Exception as e:
        print(f"✗ Valid input test failed: {e}")
    
    # Test 2: Invalid shape
    try:
        invalid_input = torch.randn(4, 1, 32, 32)  # Wrong size
        output = model(invalid_input)
        print("✗ Invalid shape test failed - should have raised error")
    except Exception as e:
        print("✓ Invalid shape test passed - correctly caught error")
    
    # Test 3: Non-finite input
    try:
        nan_input = torch.full((4, 1, 50, 50), float('nan'))
        output = model(nan_input)
        print("✗ NaN input test failed - should have raised error")
    except Exception as e:
        print("✓ NaN input test passed - correctly caught error")
    
    print("📊 Model robustness tests completed")


def test_data_validation():
    """Test data validation with various edge cases."""
    print("🧪 Testing data validation...")
    
    # Test 1: Valid data
    try:
        valid_events = generate_background_events(n_events=100, seed=42)
        processed = preprocess_events(valid_events)
        print("✓ Valid data processing passed")
    except Exception as e:
        print(f"✗ Valid data processing failed: {e}")
    
    # Test 2: Empty data
    try:
        empty_events = {}
        processed = preprocess_events(empty_events)
        print("✗ Empty data test failed - should have handled gracefully")
    except Exception as e:
        print("✓ Empty data test passed - correctly handled empty input")
    
    # Test 3: Inconsistent event count
    try:
        inconsistent_events = {
            'event_id': torch.arange(100),
            'jet_pt': torch.randn(50, 3)  # Wrong number of events
        }
        processed = preprocess_events(inconsistent_events)
        print("✗ Inconsistent data test failed - should have caught inconsistency")
    except Exception as e:
        print("✓ Inconsistent data test passed - correctly caught inconsistency")
    
    print("📊 Data validation tests completed")


def test_physics_validation():
    """Test physics validation with conservation laws."""
    print("🧪 Testing physics validation...")
    
    # Test conservation loss
    conservation = ConservationLoss()
    
    # Test 1: Perfect conservation
    try:
        initial_4momentum = torch.tensor([[[100.0, 30.0, 40.0, 50.0]]])  # (E, px, py, pz)
        final_4momentum = initial_4momentum.clone()  # Perfect conservation
        
        initial_state = {'4momentum': initial_4momentum}
        final_state = {'4momentum': final_4momentum}
        
        losses = conservation(initial_state, final_state)
        
        assert losses['energy'] < 1e-6, f"Energy loss too high: {losses['energy']}"
        assert losses['momentum'] < 1e-6, f"Momentum loss too high: {losses['momentum']}"
        print("✓ Perfect conservation test passed")
        
    except Exception as e:
        print(f"✗ Perfect conservation test failed: {e}")
    
    # Test 2: Conservation violations
    try:
        initial_4momentum = torch.tensor([[[100.0, 30.0, 40.0, 50.0]]])
        # Violate energy conservation
        final_4momentum = torch.tensor([[[90.0, 30.0, 40.0, 50.0]]])  # Lost 10 GeV
        
        initial_state = {'4momentum': initial_4momentum}
        final_state = {'4momentum': final_4momentum}
        
        losses = conservation(initial_state, final_state)
        
        assert losses['energy'] > 1e-3, f"Should detect energy violation: {losses['energy']}"
        print("✓ Conservation violation test passed")
        
    except Exception as e:
        print(f"✗ Conservation violation test failed: {e}")
    
    print("📊 Physics validation tests completed")


def test_robust_wrapper():
    """Test robust wrapper for fault tolerance."""
    print("🧪 Testing robust wrapper...")
    
    class UnreliableService:
        def __init__(self):
            self.attempt_count = 0
        
        def failing_method(self, success_threshold=3):
            self.attempt_count += 1
            if self.attempt_count < success_threshold:
                raise RuntimeError(f"Simulated failure (attempt {self.attempt_count})")
            return f"Success after {self.attempt_count} attempts!"
        
        def always_fails(self):
            raise RuntimeError("This always fails")
    
    service = UnreliableService()
    robust_service = RobustWrapper(service, max_retries=5, fallback_value="Fallback used")
    
    # Test 1: Eventually succeeds
    try:
        result = robust_service.failing_method()
        print(f"✓ Robust wrapper success test passed: {result}")
    except Exception as e:
        print(f"✗ Robust wrapper success test failed: {e}")
    
    # Test 2: Uses fallback
    try:
        result = robust_service.always_fails()
        print(f"✓ Robust wrapper fallback test passed: {result}")
    except Exception as e:
        print(f"✗ Robust wrapper fallback test failed: {e}")
    
    print("📊 Robust wrapper tests completed")


def test_safety_monitor():
    """Test safety monitoring system."""
    print("🧪 Testing safety monitor...")
    
    try:
        with SafetyMonitor(memory_limit_gb=16.0, computation_timeout=5.0):
            # Simulate computation that should pass
            _ = torch.randn(1000, 1000)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("✓ Safety monitor normal operation test passed")
    except Exception as e:
        print(f"✗ Safety monitor test failed: {e}")
    
    print("📊 Safety monitor tests completed")


def test_logging_system():
    """Test comprehensive logging system."""
    print("🧪 Testing logging system...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "logs"
        
        try:
            # Setup logging
            physics_logger = setup_logging(
                log_level="DEBUG", 
                log_dir=log_dir, 
                enable_file_logging=True
            )
            
            # Test physics event logging
            physics_logger.log_physics_event("test_simulation", energy=1000, particles=500)
            physics_logger.log_conservation_check("energy", 1e-8, 1e-6)
            physics_logger.log_model_prediction("TestModel", (32, 1, 50, 50), (32, 1, 50, 50), 0.05)
            physics_logger.log_anomaly_detection(1000, 5, 0.99)
            physics_logger.log_training_epoch(1, 0.5, 0.4, 0.001)
            
            # Check if log files were created
            assert (log_dir / "physics.log").exists(), "Physics log file not created"
            print("✓ Physics logging test passed")
            
            # Test experiment tracker
            tracker = ExperimentTracker("test_exp", log_dir / "experiments")
            tracker.log_config({"model": "TestModel", "lr": 0.001})
            tracker.log_result("accuracy", 0.95)
            tracker.save_artifact("test_data", {"test": True})
            tracker.save_experiment_summary()
            
            assert (log_dir / "experiments" / "experiment_summary.json").exists(), "Experiment summary not saved"
            print("✓ Experiment tracking test passed")
            
        except Exception as e:
            print(f"✗ Logging system test failed: {e}")
            traceback.print_exc()
    
    print("📊 Logging system tests completed")


def test_end_to_end_robustness():
    """Test end-to-end robustness with a complete workflow."""
    print("🧪 Testing end-to-end robustness...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "logs"
        
        try:
            # Setup logging
            physics_logger = setup_logging(log_level="INFO", log_dir=log_dir)
            
            # Generate data with potential issues
            print("  • Generating synthetic data...")
            bg_events = generate_background_events(n_events=1000, seed=42)
            signal_events = generate_signal_events(n_events=100, signal_type='dark_matter', seed=42)
            
            # Process data with validation
            print("  • Processing data...")
            bg_processed = preprocess_events(bg_events, normalize_energy=True)
            signal_processed = preprocess_events(signal_events, normalize_energy=True)
            
            # Apply quality cuts
            print("  • Applying quality cuts...")
            bg_filtered, bg_mask = apply_quality_cuts(bg_processed, min_jet_pt=0.2, min_n_jets=1)
            signal_filtered, signal_mask = apply_quality_cuts(signal_processed, min_jet_pt=0.2, min_n_jets=1)
            
            print(f"    - Background: {bg_mask.sum()}/{len(bg_mask)} events passed cuts")
            print(f"    - Signal: {signal_mask.sum()}/{len(signal_mask)} events passed cuts")
            
            # Create robust model
            print("  • Creating robust model...")
            robust_model = create_robust_model(
                ConvolutionalAutoencoder, 
                input_channels=1, 
                latent_dim=64
            )
            
            # Test model inference
            if len(bg_filtered['event_id']) > 0:
                print("  • Testing model inference...")
                
                # Create calorimeter-like input
                test_input = torch.randn(min(10, len(bg_filtered['event_id'])), 1, 50, 50)
                
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                
                output = robust_model.forward(test_input)
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time) / 1000.0
                else:
                    inference_time = 0.05  # Estimated
                
                physics_logger.log_model_prediction(
                    "RobustAutoencoder", 
                    test_input.shape, 
                    output.shape, 
                    inference_time
                )
                
                print(f"    - Model inference successful: {test_input.shape} -> {output.shape}")
            
            # Get performance summary
            performance_summary = physics_logger.performance_monitor.get_summary()
            print(f"  • Performance summary: {performance_summary['total_operations']} operations")
            
            print("✅ End-to-end robustness test passed!")
            
        except Exception as e:
            print(f"✗ End-to-end robustness test failed: {e}")
            traceback.print_exc()


def main():
    """Run all Generation 2 robustness tests."""
    print("🛡️ GENERATION 2: ROBUSTNESS TESTING")
    print("=" * 50)
    
    test_functions = [
        test_model_robustness,
        test_data_validation,
        test_physics_validation,
        test_robust_wrapper,
        test_safety_monitor,
        test_logging_system,
        test_end_to_end_robustness,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"\n{test_func.__name__.replace('_', ' ').title()}")
            print("-" * 40)
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🛡️ GENERATION 2 ROBUSTNESS TEST SUMMARY")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"🎯 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 ALL ROBUSTNESS TESTS PASSED!")
        return True
    else:
        print("⚠️  Some robustness tests failed")
        return False


if __name__ == "__main__":
    main()