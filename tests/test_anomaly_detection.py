"""Tests for anomaly detection functionality."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from darkoperator.anomaly import ConformalDetector
from darkoperator.operators import CalorimeterOperator


class TestConformalDetector:
    """Test conformal anomaly detection."""
    
    @pytest.fixture
    def sample_operator(self):
        """Create sample operator for testing."""
        return CalorimeterOperator(output_shape=(10, 10, 5))
    
    @pytest.fixture
    def sample_background_data(self):
        """Create sample background data."""
        # Generate realistic background events
        n_events = 1000
        n_particles = 4
        
        # Standard Model background (QCD jets)
        events = torch.randn(n_events, n_particles, 4)
        events[:, :, 0] = torch.abs(events[:, :, 0]) + 10  # Positive energy
        
        return events
    
    @pytest.fixture
    def sample_signal_data(self):
        """Create sample signal data (anomalous)."""
        n_events = 100
        n_particles = 4
        
        # Anomalous events with higher energy
        events = torch.randn(n_events, n_particles, 4) * 2  # Higher variance
        events[:, :, 0] = torch.abs(events[:, :, 0]) + 50  # Much higher energy
        
        return events
    
    def test_detector_initialization(self, sample_operator):
        """Test detector initialization."""
        detector = ConformalDetector(
            operator=sample_operator,
            alpha=1e-6
        )
        
        assert detector.alpha == 1e-6
        assert detector.operator == sample_operator
        assert not detector.is_calibrated
    
    def test_calibration_process(self, sample_operator, sample_background_data):
        """Test detector calibration."""
        detector = ConformalDetector(
            operator=sample_operator,
            alpha=1e-3  # Less stringent for testing
        )
        
        # Calibrate detector
        detector.calibrate(sample_background_data)
        
        assert detector.is_calibrated
        assert detector.calibration_scores is not None
        assert detector.quantile_threshold is not None
        assert len(detector.calibration_scores) > 0
    
    def test_p_value_computation(self, sample_operator, sample_background_data):
        """Test p-value computation."""
        detector = ConformalDetector(
            operator=sample_operator,
            alpha=1e-3
        )
        
        # Calibrate
        detector.calibrate(sample_background_data)
        
        # Test on small sample
        test_events = sample_background_data[:10]
        p_values = detector.compute_p_values(test_events)
        
        assert len(p_values) == 10
        assert all(0 <= p <= 1 for p in p_values)
    
    def test_anomaly_detection(self, sample_operator, sample_background_data, sample_signal_data):
        """Test anomaly detection with background and signal."""
        detector = ConformalDetector(
            operator=sample_operator,
            alpha=0.1  # Less stringent for testing
        )
        
        # Calibrate on background
        detector.calibrate(sample_background_data)
        
        # Test on mixed data (background + signal)
        mixed_data = torch.cat([sample_background_data[:50], sample_signal_data[:20]])
        anomalies, p_values = detector.find_anomalies(mixed_data, return_scores=True)
        
        # Should find some anomalies
        assert len(anomalies) > 0
        assert len(p_values) == len(anomalies)
        
        # Anomalies should have low p-values
        assert all(p < detector.alpha for p in p_values)
    
    def test_false_discovery_rate_control(self, sample_operator, sample_background_data):
        """Test that FDR is controlled at nominal level."""
        detector = ConformalDetector(
            operator=sample_operator,
            alpha=0.05  # 5% FDR
        )
        
        detector.calibrate(sample_background_data)
        
        # Test on pure background (should have ~5% discoveries)
        test_background = sample_background_data[-200:]  # Independent test set
        anomalies = detector.find_anomalies(test_background)
        
        empirical_fdr = len(anomalies) / len(test_background)
        
        # Should be close to nominal level (within reasonable bounds for testing)
        assert empirical_fdr <= 0.15  # Allow some variation for small sample
    
    def test_power_estimation(self, sample_operator, sample_background_data, sample_signal_data):
        """Test statistical power estimation."""
        detector = ConformalDetector(
            operator=sample_operator,
            alpha=0.1
        )
        
        detector.calibrate(sample_background_data)
        
        # Estimate power on signal
        mean_power, std_power = detector.estimate_discovery_power(
            sample_signal_data,
            n_bootstrap=50  # Reduced for testing speed
        )
        
        assert 0 <= mean_power <= 1
        assert std_power >= 0
        
        # Should have reasonable power for strong signal
        assert mean_power > 0.1  # At least 10% power


class TestAnomalyDetectionIntegration:
    """Integration tests for anomaly detection pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete anomaly detection pipeline."""
        # Create operator
        operator = CalorimeterOperator(output_shape=(8, 8, 4))
        
        # Generate synthetic data
        background = torch.randn(500, 5, 4)
        background[:, :, 0] = torch.abs(background[:, :, 0]) + 5
        
        signal = torch.randn(50, 5, 4) * 1.5
        signal[:, :, 0] = torch.abs(signal[:, :, 0]) + 20
        
        # Create detector
        detector = ConformalDetector(operator=operator, alpha=0.05)
        
        # Calibrate
        detector.calibrate(background[:400])  # Use part for calibration
        
        # Test on remaining data + signal
        test_data = torch.cat([background[400:], signal])
        anomalies = detector.find_anomalies(test_data)
        
        # Should detect some anomalies
        assert len(anomalies) > 0
        
        # Most detections should be from signal region
        signal_start_idx = len(background[400:])
        signal_detections = sum(1 for idx in anomalies if idx >= signal_start_idx)
        background_detections = len(anomalies) - signal_detections
        
        # Signal should be detected more often than background
        signal_rate = signal_detections / len(signal)
        background_rate = background_detections / len(background[400:])
        
        assert signal_rate > background_rate
    
    def test_performance_metrics(self, sample_operator=None):
        """Test performance and timing of anomaly detection."""
        if sample_operator is None:
            sample_operator = CalorimeterOperator(output_shape=(10, 10, 5))
        
        detector = ConformalDetector(operator=sample_operator, alpha=0.01)
        
        # Large dataset for performance testing
        large_background = torch.randn(2000, 6, 4)
        large_background[:, :, 0] = torch.abs(large_background[:, :, 0]) + 8
        
        # Measure calibration time
        import time
        start_time = time.time()
        detector.calibrate(large_background[:1500])
        calibration_time = time.time() - start_time
        
        # Should complete calibration in reasonable time
        assert calibration_time < 30  # 30 seconds max
        
        # Measure inference time
        test_events = large_background[1500:1600]  # 100 events
        start_time = time.time()
        anomalies = detector.find_anomalies(test_events)
        inference_time = time.time() - start_time
        
        # Should process 100 events quickly
        assert inference_time < 10  # 10 seconds max
        events_per_second = len(test_events) / inference_time
        assert events_per_second > 5  # At least 5 events/sec
    
    def test_memory_usage(self):
        """Test memory usage during anomaly detection."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Create large-scale test
        operator = CalorimeterOperator(output_shape=(20, 20, 10))
        detector = ConformalDetector(operator=operator, alpha=0.001)
        
        # Large dataset
        large_data = torch.randn(5000, 8, 4)
        large_data[:, :, 0] = torch.abs(large_data[:, :, 0]) + 10
        
        # Process data
        detector.calibrate(large_data[:4000])
        detector.find_anomalies(large_data[4000:])
        
        # Check memory usage
        final_memory = process.memory_info().rss / (1024**2)  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory (allow 2GB increase)
        assert memory_increase < 2000  # 2GB limit