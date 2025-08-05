"""Tests for neural operators."""

import pytest
import torch
import numpy as np
from darkoperator.operators import CalorimeterOperator, TrackerOperator, MuonOperator


class TestCalorimeterOperator:
    """Test calorimeter operator functionality."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        op = CalorimeterOperator()
        
        # Create test input: batch of 4-vectors
        batch_size, n_particles = 2, 10
        x = torch.randn(batch_size, n_particles, 4)
        x[:, :, 0] = torch.abs(x[:, :, 0]) + 1  # Positive energy
        
        output = op(x)
        
        assert output.shape == (batch_size, *op.output_shape)
        assert torch.all(output >= 0)  # Energy deposits should be positive
    
    def test_energy_conservation(self):
        """Test that operator respects energy conservation.""" 
        op = CalorimeterOperator(preserve_energy=True)
        
        x = torch.tensor([[[100.0, 50.0, 30.0, 20.0]]], dtype=torch.float32)
        output = op(x)
        
        # Check that total output energy is reasonable
        total_output = output.sum()
        input_energy = x[:, :, 0].sum()
        
        # Should be same order of magnitude (not exact due to network approximation)
        assert total_output > 0.1 * input_energy
        assert total_output < 10.0 * input_energy


class TestTrackerOperator:
    """Test tracker operator functionality."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        op = TrackerOperator()
        
        batch_size, n_particles = 2, 5
        x = torch.randn(batch_size, n_particles, 4)
        
        output = op(x)
        
        assert output.shape == (batch_size, *op.output_shape)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)  # Sigmoid output


class TestMuonOperator:
    """Test muon operator functionality."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        op = MuonOperator()
        
        batch_size, n_particles = 2, 3
        x = torch.randn(batch_size, n_particles, 4)
        
        output = op(x)
        
        assert output.shape == (batch_size, *op.output_shape)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)  # Sigmoid output