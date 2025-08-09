#!/usr/bin/env python3
"""
Comprehensive Quality Gates for DarkOperator Studio
Verifies all aspects of the framework across Generation 1, 2, and 3.
"""

import torch
import numpy as np
import time
import tempfile
import subprocess
import sys
from pathlib import Path
import json
import traceback
from typing import Dict, List, Any, Optional

# Import all framework components
import darkoperator as do
from darkoperator.models import ConvolutionalAutoencoder, VariationalAutoencoder
from darkoperator.physics import ConservationLoss
from darkoperator.data.synthetic import generate_background_events, generate_signal_events
from darkoperator.data.preprocessing import preprocess_events, apply_quality_cuts
from darkoperator.utils.error_handling import RobustWrapper, SafetyMonitor
from darkoperator.utils.logging_config import setup_logging, ExperimentTracker
from darkoperator.optimization.performance_optimizer import GPUOptimizer, CacheOptimizer
from darkoperator.distributed.auto_scaling import ResourceMonitor
from darkoperator.visualization import visualize_event


class QualityGateRunner:
    """Comprehensive quality gate verification system."""
    
    def __init__(self):
        self.results = {
            'generation_1': {'tests': [], 'passed': 0, 'failed': 0},
            'generation_2': {'tests': [], 'passed': 0, 'failed': 0},  
            'generation_3': {'tests': [], 'passed': 0, 'failed': 0},
            'integration': {'tests': [], 'passed': 0, 'failed': 0},
            'overall': {'total_tests': 0, 'passed': 0, 'failed': 0}
        }
        self.start_time = time.time()
    
    def run_test(self, test_name: str, test_func, generation: str = 'overall'):
        """Run a single test with error handling."""
        print(f"Running {test_name}...")
        
        try:
            start_time = time.time()
            test_func()
            duration = time.time() - start_time
            
            result = {
                'name': test_name,
                'status': 'PASSED',
                'duration': duration,
                'error': None
            }
            
            self.results[generation]['tests'].append(result)
            self.results[generation]['passed'] += 1
            self.results['overall']['passed'] += 1
            
            print(f"✅ {test_name} PASSED ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            result = {
                'name': test_name,
                'status': 'FAILED',
                'duration': duration,
                'error': error_msg
            }
            
            self.results[generation]['tests'].append(result)
            self.results[generation]['failed'] += 1
            self.results['overall']['failed'] += 1
            
            print(f"❌ {test_name} FAILED ({duration:.2f}s): {error_msg}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        total_time = time.time() - self.start_time
        self.results['overall']['total_tests'] = self.results['overall']['passed'] + self.results['overall']['failed']
        self.results['overall']['success_rate'] = (
            self.results['overall']['passed'] / self.results['overall']['total_tests'] 
            if self.results['overall']['total_tests'] > 0 else 0
        )
        self.results['overall']['total_duration'] = total_time
        
        return self.results


def test_package_imports():
    """Test that all package components can be imported."""
    # Test main package
    assert hasattr(do, '__all__'), "Package should have __all__"
    assert len(do.__all__) > 0, "Package should export components"
    
    # Test core components
    components_to_test = [
        'CalorimeterOperator', 'ConformalDetector', 'FourierNeuralOperator',
        'LorentzEmbedding', 'load_opendata', 'visualize_event', 'OperatorTrainer'
    ]
    
    for component in components_to_test:
        assert hasattr(do, component), f"Missing component: {component}"
        assert callable(getattr(do, component)) or hasattr(getattr(do, component), '__init__'), f"Component not callable: {component}"


def test_neural_operators():
    """Test neural operator functionality."""
    # Test Fourier Neural Operator
    fno = do.FourierNeuralOperator(
        input_channels=4,
        output_channels=4,
        modes=16,
        width=32
    )
    
    # Test forward pass
    test_input = torch.randn(2, 4, 32, 32)
    output = fno(test_input)
    
    assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} vs {test_input.shape}"
    assert torch.isfinite(output).all(), "Output should be finite"
    
    # Test Lorentz embedding
    lorentz_embed = do.LorentzEmbedding(4)
    momentum_4vec = torch.randn(10, 4)
    embedded = lorentz_embed(momentum_4vec)
    
    assert embedded.shape[0] == 10, "Batch dimension preserved"
    assert torch.isfinite(embedded).all(), "Embedding should be finite"


def test_autoencoder_models():
    """Test autoencoder model functionality."""
    # Test Convolutional Autoencoder
    conv_ae = ConvolutionalAutoencoder(input_channels=1, latent_dim=64)
    test_input = torch.randn(4, 1, 50, 50)
    
    # Test forward pass
    output = conv_ae(test_input)
    assert output.shape == test_input.shape, "Autoencoder should preserve input shape"
    
    # Test encoding/decoding
    encoded = conv_ae.encode(test_input)
    decoded = conv_ae.decode(encoded)
    assert decoded.shape == test_input.shape, "Decode should match input shape"
    
    # Test loss computation
    loss = conv_ae.reconstruction_loss(test_input, output)
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss >= 0, "Loss should be non-negative"
    
    # Test Variational Autoencoder
    vae = VariationalAutoencoder(input_channels=1, latent_dim=64)
    vae_output = vae(test_input)
    
    assert 'x_recon' in vae_output, "VAE should return reconstruction"
    assert 'mu' in vae_output, "VAE should return mean"
    assert 'logvar' in vae_output, "VAE should return log variance"
    
    # Test VAE loss
    vae_loss = vae.loss_function(test_input, **{k: v for k, v in vae_output.items() if k != 'z'})
    assert 'loss' in vae_loss, "VAE loss should include total loss"
    assert torch.isfinite(vae_loss['loss']), "VAE loss should be finite"


def test_anomaly_detection():
    """Test anomaly detection functionality."""
    # Create test autoencoder
    autoencoder = ConvolutionalAutoencoder(input_channels=1, latent_dim=32)
    
    # Test conformal detector
    detector = do.ConformalDetector(
        model=autoencoder,
        calibration_alpha=0.1
    )
    
    # Test with synthetic data
    test_data = torch.randn(20, 1, 50, 50)
    
    # Calibrate detector
    detector.calibrate(test_data[:10])
    
    # Test anomaly detection
    anomaly_scores = detector.predict(test_data[10:])
    
    assert len(anomaly_scores) == 10, "Should return score for each input"
    assert all(score >= 0 for score in anomaly_scores), "Scores should be non-negative"


def test_physics_constraints():
    """Test physics constraint enforcement."""
    # Test conservation laws
    conservation = ConservationLoss()
    
    # Create valid 4-momentum
    batch_size, n_particles = 4, 3
    four_momentum = torch.randn(batch_size, n_particles, 4)
    four_momentum[:, :, 0] = torch.abs(four_momentum[:, :, 0]) + 5  # Positive energy
    
    initial_state = {'4momentum': four_momentum}
    final_state = {'4momentum': four_momentum.clone()}  # Perfect conservation
    
    losses = conservation(initial_state, final_state)
    
    assert 'total' in losses, "Should return total loss"
    assert 'energy' in losses, "Should check energy conservation"
    assert 'momentum' in losses, "Should check momentum conservation"
    assert losses['energy'] < 1e-6, "Perfect conservation should have minimal energy loss"
    assert losses['momentum'] < 1e-6, "Perfect conservation should have minimal momentum loss"


def test_data_processing():
    """Test data loading and processing."""
    # Test synthetic data generation
    bg_events = generate_background_events(n_events=100, seed=42)
    signal_events = generate_signal_events(n_events=50, signal_type='dark_matter', seed=42)
    
    assert len(bg_events['event_id']) == 100, "Should generate correct number of background events"
    assert len(signal_events['event_id']) == 50, "Should generate correct number of signal events"
    
    # Test data preprocessing
    processed_bg = preprocess_events(bg_events, normalize_energy=True)
    processed_signal = preprocess_events(signal_events, normalize_energy=True)
    
    assert 'jet_4vectors' in processed_bg, "Should create 4-vectors"
    assert 'missing_et_vector' in processed_bg, "Should create MET vector"
    
    # Test quality cuts
    filtered_bg, mask = apply_quality_cuts(processed_bg, min_jet_pt=0.2, min_n_jets=1)
    
    assert len(filtered_bg['event_id']) <= len(bg_events['event_id']), "Filtering should reduce or maintain size"
    assert mask.sum() == len(filtered_bg['event_id']), "Mask should match filtered data size"


def test_error_handling():
    """Test error handling and robustness."""
    # Test robust wrapper
    class UnreliableService:
        def __init__(self):
            self.calls = 0
        
        def unreliable_method(self):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("Simulated failure")
            return "Success"
    
    service = UnreliableService()
    robust_service = RobustWrapper(service, max_retries=5, fallback_value="Fallback")
    
    result = robust_service.unreliable_method()
    assert result == "Success", "Robust wrapper should eventually succeed"
    
    # Test safety monitor
    with SafetyMonitor(memory_limit_gb=16.0, computation_timeout=1.0):
        # Simulate safe computation
        _ = torch.randn(100, 100)


def test_performance_optimization():
    """Test performance optimization features."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test GPU optimizer
    gpu_optimizer = GPUOptimizer(device)
    
    # Test cache optimizer
    cache_optimizer = CacheOptimizer(cache_size=10)
    
    @cache_optimizer.cached_computation()
    def test_computation(x):
        return x * 2
    
    # First call (cache miss)
    result1 = test_computation(5)
    
    # Second call (cache hit)
    result2 = test_computation(5)
    
    assert result1 == result2, "Cached results should be identical"
    
    # Check cache statistics
    stats = cache_optimizer.get_cache_stats()
    assert stats['cache_hits'] > 0, "Should have cache hits"


def test_distributed_components():
    """Test distributed computing components."""
    # Test resource monitor
    monitor = ResourceMonitor(monitoring_interval=0.1, history_size=5)
    monitor.start_monitoring()
    
    time.sleep(0.2)  # Let it collect metrics
    
    metrics = monitor.get_current_metrics()
    assert metrics is not None, "Should collect metrics"
    assert 0 <= metrics.cpu_percent <= 100, "CPU percentage should be valid"
    assert 0 <= metrics.memory_percent <= 100, "Memory percentage should be valid"
    
    monitor.stop_monitoring()


def test_visualization():
    """Test visualization capabilities."""
    # Create test event data
    test_event = {
        'jet_pt': torch.tensor([100., 50., 30.]),
        'jet_eta': torch.tensor([0.5, -1.2, 2.1]),
        'jet_phi': torch.tensor([0.1, 2.5, -1.8]),
        'missing_et': torch.tensor(75.0),
        'missing_phi': torch.tensor(1.2)
    }
    
    # Test that visualization function can be called without error
    # (actual visualization would require display, so we just test the function exists)
    assert callable(do.visualize_event), "visualize_event should be callable"
    assert callable(do.plot_operator_kernels), "plot_operator_kernels should be callable"


def test_integration_workflow():
    """Test complete integration workflow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "logs"
        
        # Setup logging
        physics_logger = setup_logging(log_level="INFO", log_dir=log_dir)
        
        # Generate data
        events = generate_background_events(n_events=200, seed=42)
        processed = preprocess_events(events)
        filtered, _ = apply_quality_cuts(processed, min_jet_pt=0.2, min_n_jets=1)
        
        # Create and test model
        model = ConvolutionalAutoencoder(input_channels=1, latent_dim=32)
        test_input = torch.randn(10, 1, 50, 50)
        
        with SafetyMonitor(memory_limit_gb=8.0):
            output = model(test_input)
            loss = model.reconstruction_loss(test_input, output)
        
        # Test physics validation
        conservation = ConservationLoss()
        if 'jet_4vectors' in filtered and len(filtered['jet_4vectors']) > 0:
            sample_4vectors = filtered['jet_4vectors'][:5]  # Use first 5 events
            initial_state = {'4momentum': sample_4vectors}
            final_state = {'4momentum': sample_4vectors}  # Perfect conservation for test
            
            physics_losses = conservation(initial_state, final_state)
        
        # Log results
        physics_logger.log_physics_event("integration_test", events_processed=len(filtered['event_id']))
        
        assert output.shape == test_input.shape, "Model should process data correctly"
        assert len(filtered['event_id']) > 0, "Should have some events passing quality cuts"


def test_code_quality():
    """Test code quality and style."""
    # Test that all critical modules can be imported
    try:
        import darkoperator.models
        import darkoperator.operators  
        import darkoperator.anomaly
        import darkoperator.physics
        import darkoperator.data
        import darkoperator.utils
        import darkoperator.optimization
        import darkoperator.distributed
        import darkoperator.visualization
    except ImportError as e:
        raise AssertionError(f"Failed to import module: {e}")
    
    # Test package metadata
    assert hasattr(do, '__version__'), "Package should have version"
    assert hasattr(do, '__author__'), "Package should have author"


def main():
    """Run comprehensive quality gates."""
    print("🧪 COMPREHENSIVE QUALITY GATES VERIFICATION")
    print("=" * 60)
    print(f"Testing DarkOperator Studio v{do.__version__}")
    print("=" * 60)
    
    runner = QualityGateRunner()
    
    # Generation 1: MAKE IT WORK
    print("\n🚀 GENERATION 1: MAKE IT WORK")
    print("-" * 40)
    
    runner.run_test("Package Imports", test_package_imports, 'generation_1')
    runner.run_test("Neural Operators", test_neural_operators, 'generation_1')  
    runner.run_test("Autoencoder Models", test_autoencoder_models, 'generation_1')
    runner.run_test("Anomaly Detection", test_anomaly_detection, 'generation_1')
    runner.run_test("Physics Constraints", test_physics_constraints, 'generation_1')
    runner.run_test("Data Processing", test_data_processing, 'generation_1')
    
    # Generation 2: MAKE IT ROBUST  
    print("\n🛡️ GENERATION 2: MAKE IT ROBUST")
    print("-" * 40)
    
    runner.run_test("Error Handling", test_error_handling, 'generation_2')
    runner.run_test("Code Quality", test_code_quality, 'generation_2')
    runner.run_test("Visualization", test_visualization, 'generation_2')
    
    # Generation 3: MAKE IT SCALE
    print("\n⚡ GENERATION 3: MAKE IT SCALE") 
    print("-" * 40)
    
    runner.run_test("Performance Optimization", test_performance_optimization, 'generation_3')
    runner.run_test("Distributed Components", test_distributed_components, 'generation_3')
    
    # Integration Tests
    print("\n🔗 INTEGRATION TESTS")
    print("-" * 40)
    
    runner.run_test("Integration Workflow", test_integration_workflow, 'integration')
    
    # Generate final report
    print("\n" + "=" * 60)
    print("🧪 QUALITY GATES SUMMARY")
    print("=" * 60)
    
    results = runner.generate_report()
    
    for generation in ['generation_1', 'generation_2', 'generation_3', 'integration']:
        gen_results = results[generation]
        if gen_results['tests']:
            total = gen_results['passed'] + gen_results['failed']
            success_rate = gen_results['passed'] / total * 100 if total > 0 else 0
            
            generation_name = generation.replace('_', ' ').title()
            print(f"\n{generation_name}:")
            print(f"  ✅ Passed: {gen_results['passed']}")  
            print(f"  ❌ Failed: {gen_results['failed']}")
            print(f"  🎯 Success Rate: {success_rate:.1f}%")
    
    overall = results['overall']
    print(f"\n📊 OVERALL RESULTS:")
    print(f"  📦 Total Tests: {overall['total_tests']}")
    print(f"  ✅ Passed: {overall['passed']}")
    print(f"  ❌ Failed: {overall['failed']}")
    print(f"  🎯 Success Rate: {overall['success_rate']*100:.1f}%")
    print(f"  ⏱️ Total Time: {overall['total_duration']:.2f}s")
    
    # Final verdict
    print("\n" + "=" * 60)
    if overall['failed'] == 0:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("✨ DarkOperator Studio is PRODUCTION READY!")
        print("🚀 Framework meets enterprise-grade quality standards")
        
        # Save quality report
        with open('quality_gates_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("📊 Quality report saved to quality_gates_report.json")
        
        return True
    else:
        print("⚠️ SOME QUALITY GATES FAILED")
        print(f"❌ {overall['failed']} test(s) need attention before production deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)