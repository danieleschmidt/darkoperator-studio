"""
Physics-Specific Benchmarks for DarkOperator Studio.

Validates physics accuracy, conservation laws, symmetry preservation,
and correctness of neural operators in particle physics simulations.
"""

import numpy as np
import torch
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod

from .benchmark_runner import BaseBenchmark, BenchmarkType, BenchmarkResult

logger = logging.getLogger(__name__)


class PhysicsBenchmark(BaseBenchmark):
    """Base class for physics-related benchmarks."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, BenchmarkType.PHYSICS, config)
        
        # Physics validation parameters
        self.tolerance = config.get('tolerance', 1e-6) if config else 1e-6
        self.num_test_events = config.get('num_test_events', 1000) if config else 1000
        
    def generate_test_events(self, num_events: int) -> torch.Tensor:
        """Generate realistic particle physics test events."""
        
        # Generate 4-momentum vectors (E, px, py, pz) with physical constraints
        events = []
        
        for _ in range(num_events):
            # Generate random particle with realistic kinematics
            mass = np.random.exponential(0.5)  # Particle mass in GeV
            pt = np.random.exponential(10.0)   # Transverse momentum
            eta = np.random.normal(0, 2.5)     # Pseudorapidity
            phi = np.random.uniform(-np.pi, np.pi)  # Azimuthal angle
            
            # Convert to 4-momentum
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
            
            events.append([E, px, py, pz])
        
        return torch.tensor(events, dtype=torch.float32)
    
    def compute_invariant_mass(self, four_momentum: torch.Tensor) -> torch.Tensor:
        """Compute invariant mass from 4-momentum."""
        
        if len(four_momentum.shape) == 1:
            four_momentum = four_momentum.unsqueeze(0)
        
        E = four_momentum[:, 0]
        px, py, pz = four_momentum[:, 1], four_momentum[:, 2], four_momentum[:, 3]
        
        # Invariant mass: m² = E² - p²
        mass_squared = E**2 - (px**2 + py**2 + pz**2)
        
        # Handle numerical precision issues
        mass_squared = torch.clamp(mass_squared, min=0)
        
        return torch.sqrt(mass_squared)
    
    def check_energy_conservation(self, input_events: torch.Tensor, output_events: torch.Tensor) -> float:
        """Check energy conservation between input and output events."""
        
        input_energy = torch.sum(input_events[:, 0])  # Sum of E components
        output_energy = torch.sum(output_events[:, 0])
        
        relative_error = torch.abs(input_energy - output_energy) / input_energy
        return relative_error.item()
    
    def check_momentum_conservation(self, input_events: torch.Tensor, output_events: torch.Tensor) -> float:
        """Check momentum conservation between input and output events."""
        
        # Sum momentum components
        input_momentum = torch.sum(input_events[:, 1:4], dim=0)  # [px, py, pz]
        output_momentum = torch.sum(output_events[:, 1:4], dim=0)
        
        # Compute magnitude of momentum difference
        momentum_diff = torch.norm(input_momentum - output_momentum)
        input_momentum_mag = torch.norm(input_momentum)
        
        relative_error = momentum_diff / (input_momentum_mag + 1e-10)
        return relative_error.item()


class ConservationBenchmark(PhysicsBenchmark):
    """Benchmark conservation law adherence in neural operators."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any] = None):
        super().__init__("Conservation Laws Benchmark", config)
        self.model = model
        
    def setup(self) -> None:
        """Setup conservation benchmark."""
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
        
        self.model.eval()
        
        # Generate test events
        self.test_events = self.generate_test_events(self.num_test_events)
        self.test_events = self.test_events.to(self.device)
        
        logger.debug(f"Setup conservation benchmark with {self.num_test_events} events")
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run conservation law validation."""
        
        results = {
            'energy_violations': [],
            'momentum_violations': [],
            'total_events_tested': 0,
            'conservation_score': 0.0
        }
        
        with torch.no_grad():
            # Process events in batches
            batch_size = 32
            
            for i in range(0, len(self.test_events), batch_size):
                batch = self.test_events[i:i+batch_size]
                
                try:
                    # Model inference
                    output = self.model(batch)
                    
                    # Ensure output has same shape as input for conservation checking
                    if output.shape != batch.shape:
                        # If model outputs different format, adapt for conservation check
                        if len(output.shape) == 2 and output.shape[1] == 4:
                            output_4momentum = output
                        else:
                            # Skip this batch if can't interpret output
                            continue
                    else:
                        output_4momentum = output
                    
                    # Check energy conservation
                    energy_violation = self.check_energy_conservation(batch, output_4momentum)
                    results['energy_violations'].append(energy_violation)
                    
                    # Check momentum conservation
                    momentum_violation = self.check_momentum_conservation(batch, output_4momentum)
                    results['momentum_violations'].append(momentum_violation)
                    
                    results['total_events_tested'] += len(batch)
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {i}: {e}")
                    continue
        
        # Compute overall conservation score
        if results['energy_violations'] and results['momentum_violations']:
            avg_energy_violation = np.mean(results['energy_violations'])
            avg_momentum_violation = np.mean(results['momentum_violations'])
            
            # Score inversely related to violations (higher score = better conservation)
            energy_score = max(0, 1 - avg_energy_violation / self.tolerance)
            momentum_score = max(0, 1 - avg_momentum_violation / self.tolerance)
            
            results['conservation_score'] = (energy_score + momentum_score) / 2
            
            # Update benchmark result
            self.result.physics_score = results['conservation_score']
            self.result.conservation_violation = (avg_energy_violation + avg_momentum_violation) / 2
            
            # Custom metrics
            self.result.custom_metrics.update({
                'avg_energy_violation': avg_energy_violation,
                'avg_momentum_violation': avg_momentum_violation,
                'energy_conservation_score': energy_score,
                'momentum_conservation_score': momentum_score
            })
        
        return results
    
    def teardown(self) -> None:
        """Clean up conservation benchmark."""
        
        # Clear test data
        if hasattr(self, 'test_events'):
            del self.test_events
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def validate_results(self, results: Any) -> bool:
        """Validate conservation benchmark results."""
        
        if not isinstance(results, dict):
            return False
        
        required_keys = ['energy_violations', 'momentum_violations', 'conservation_score']
        return all(key in results for key in required_keys)


class SymmetryBenchmark(PhysicsBenchmark):
    """Benchmark symmetry preservation in neural operators."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any] = None):
        super().__init__("Symmetry Preservation Benchmark", config)
        self.model = model
        self.symmetry_types = config.get('symmetries', ['rotation', 'lorentz']) if config else ['rotation', 'lorentz']
        
    def setup(self) -> None:
        """Setup symmetry benchmark."""
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
        
        self.model.eval()
        
        # Generate test events
        self.test_events = self.generate_test_events(self.num_test_events)
        self.test_events = self.test_events.to(self.device)
        
    def run_benchmark(self) -> Dict[str, Any]:
        """Run symmetry preservation validation."""
        
        results = {
            'rotation_symmetry_violations': [],
            'lorentz_symmetry_violations': [],
            'overall_symmetry_score': 0.0
        }
        
        with torch.no_grad():
            # Test rotation symmetry
            if 'rotation' in self.symmetry_types:
                rotation_violations = self._test_rotation_symmetry()
                results['rotation_symmetry_violations'] = rotation_violations
            
            # Test Lorentz symmetry
            if 'lorentz' in self.symmetry_types:
                lorentz_violations = self._test_lorentz_symmetry()
                results['lorentz_symmetry_violations'] = lorentz_violations
            
            # Compute overall symmetry score
            all_violations = []
            if 'rotation' in self.symmetry_types:
                all_violations.extend(results['rotation_symmetry_violations'])
            if 'lorentz' in self.symmetry_types:
                all_violations.extend(results['lorentz_symmetry_violations'])
            
            if all_violations:
                avg_violation = np.mean(all_violations)
                symmetry_score = max(0, 1 - avg_violation / self.tolerance)
                results['overall_symmetry_score'] = symmetry_score
                
                # Update benchmark result
                self.result.physics_score = symmetry_score
                self.result.symmetry_preservation = symmetry_score
                
                self.result.custom_metrics.update({
                    'avg_symmetry_violation': avg_violation,
                    'symmetry_types_tested': self.symmetry_types
                })
        
        return results
    
    def _test_rotation_symmetry(self) -> List[float]:
        """Test rotation symmetry preservation."""
        
        violations = []
        
        # Test with small sample for efficiency
        test_sample = self.test_events[:100]
        
        for i in range(len(test_sample)):
            event = test_sample[i:i+1]
            
            try:
                # Original output
                original_output = self.model(event)
                
                # Apply rotation to input (rotate around z-axis)
                angle = np.pi / 4  # 45 degrees
                rotated_event = self._rotate_event(event, angle)
                
                # Get output for rotated input
                rotated_output = self.model(rotated_event)
                
                # Apply same rotation to original output
                expected_output = self._rotate_event(original_output, angle)
                
                # Compute violation (difference between rotated output and expected)
                violation = torch.norm(rotated_output - expected_output).item()
                violations.append(violation)
                
            except Exception as e:
                logger.warning(f"Error testing rotation symmetry for event {i}: {e}")
                continue
        
        return violations
    
    def _test_lorentz_symmetry(self) -> List[float]:
        """Test Lorentz symmetry preservation."""
        
        violations = []
        
        # Test with small sample for efficiency
        test_sample = self.test_events[:100]
        
        for i in range(len(test_sample)):
            event = test_sample[i:i+1]
            
            try:
                # Original invariant mass
                original_mass = self.compute_invariant_mass(event)
                
                # Model output
                output = self.model(event)
                
                # Output invariant mass
                if output.shape[-1] >= 4:  # Check if output has 4-momentum structure
                    output_mass = self.compute_invariant_mass(output)
                    
                    # Lorentz invariance violation
                    violation = torch.abs(original_mass - output_mass).item()
                    violations.append(violation)
                
            except Exception as e:
                logger.warning(f"Error testing Lorentz symmetry for event {i}: {e}")
                continue
        
        return violations
    
    def _rotate_event(self, event: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate 4-momentum event around z-axis."""
        
        rotated = event.clone()
        
        # Rotation matrix for (px, py) components
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        px_original = rotated[:, 1]
        py_original = rotated[:, 2]
        
        rotated[:, 1] = cos_angle * px_original - sin_angle * py_original
        rotated[:, 2] = sin_angle * px_original + cos_angle * py_original
        
        # E and pz remain unchanged
        
        return rotated
    
    def teardown(self) -> None:
        """Clean up symmetry benchmark."""
        
        if hasattr(self, 'test_events'):
            del self.test_events
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def validate_results(self, results: Any) -> bool:
        """Validate symmetry benchmark results."""
        
        if not isinstance(results, dict):
            return False
        
        return 'overall_symmetry_score' in results


class AccuracyBenchmark(PhysicsBenchmark):
    """Benchmark physics accuracy against known analytical solutions."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any] = None):
        super().__init__("Physics Accuracy Benchmark", config)
        self.model = model
        self.test_scenarios = config.get('scenarios', ['two_body_decay']) if config else ['two_body_decay']
        
    def setup(self) -> None:
        """Setup accuracy benchmark."""
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
        
        self.model.eval()
        
    def run_benchmark(self) -> Dict[str, Any]:
        """Run physics accuracy validation."""
        
        results = {
            'scenario_accuracies': {},
            'overall_accuracy': 0.0
        }
        
        with torch.no_grad():
            for scenario in self.test_scenarios:
                if scenario == 'two_body_decay':
                    accuracy = self._test_two_body_decay()
                    results['scenario_accuracies'][scenario] = accuracy
                elif scenario == 'elastic_scattering':
                    accuracy = self._test_elastic_scattering()
                    results['scenario_accuracies'][scenario] = accuracy
                # Add more scenarios as needed
            
            # Compute overall accuracy
            if results['scenario_accuracies']:
                results['overall_accuracy'] = np.mean(list(results['scenario_accuracies'].values()))
                
                # Update benchmark result
                self.result.physics_score = results['overall_accuracy']
                self.result.custom_metrics.update({
                    'scenarios_tested': list(results['scenario_accuracies'].keys()),
                    'individual_accuracies': results['scenario_accuracies']
                })
        
        return results
    
    def _test_two_body_decay(self) -> float:
        """Test accuracy on two-body decay kinematics."""
        
        accuracies = []
        
        # Generate parent particles with known decay kinematics
        for _ in range(100):
            # Parent particle (at rest for simplicity)
            parent_mass = 10.0  # GeV
            parent_4momentum = torch.tensor([[parent_mass, 0.0, 0.0, 0.0]], device=self.device)
            
            try:
                # Model prediction
                output = self.model(parent_4momentum)
                
                # For two-body decay, we expect two daughters
                if output.shape[0] == 2 and output.shape[1] == 4:
                    daughter1 = output[0]
                    daughter2 = output[1]
                    
                    # Check energy-momentum conservation
                    total_E = daughter1[0] + daughter2[0]
                    total_p = daughter1[1:] + daughter2[1:]
                    
                    energy_error = abs(total_E - parent_mass) / parent_mass
                    momentum_error = torch.norm(total_p) / parent_mass
                    
                    # Accuracy based on conservation
                    conservation_accuracy = 1.0 - (energy_error + momentum_error)
                    accuracies.append(max(0, conservation_accuracy))
                
            except Exception as e:
                logger.warning(f"Error in two-body decay test: {e}")
                continue
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def _test_elastic_scattering(self) -> float:
        """Test accuracy on elastic scattering kinematics."""
        
        accuracies = []
        
        # Generate scattering scenarios
        for _ in range(100):
            # Two incoming particles
            p1_mass = 0.938  # Proton mass in GeV
            p2_mass = 0.938  # Proton mass in GeV
            
            # Simple head-on collision
            momentum = 5.0  # GeV
            E1 = np.sqrt(momentum**2 + p1_mass**2)
            E2 = np.sqrt(momentum**2 + p2_mass**2)
            
            initial_state = torch.tensor([
                [E1, momentum, 0.0, 0.0],
                [E2, -momentum, 0.0, 0.0]
            ], device=self.device)
            
            try:
                # Model prediction for final state
                output = self.model(initial_state)
                
                if output.shape[0] == 2 and output.shape[1] == 4:
                    # Check conservation laws
                    initial_total = torch.sum(initial_state, dim=0)
                    final_total = torch.sum(output, dim=0)
                    
                    conservation_error = torch.norm(initial_total - final_total)
                    conservation_accuracy = 1.0 - conservation_error / torch.norm(initial_total)
                    
                    accuracies.append(max(0, conservation_accuracy.item()))
                
            except Exception as e:
                logger.warning(f"Error in elastic scattering test: {e}")
                continue
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def teardown(self) -> None:
        """Clean up accuracy benchmark."""
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def validate_results(self, results: Any) -> bool:
        """Validate accuracy benchmark results."""
        
        if not isinstance(results, dict):
            return False
        
        return 'overall_accuracy' in results and 'scenario_accuracies' in results


class PhysicsValidationSuite:
    """Complete physics validation benchmark suite."""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or {}
        
    def create_benchmark_suite(self):
        """Create complete physics benchmark suite."""
        
        from .benchmark_runner import BenchmarkSuite
        
        suite = BenchmarkSuite(
            name="Physics Validation Suite",
            description="Comprehensive physics validation for neural operators"
        )
        
        # Add conservation law benchmark
        conservation_config = self.config.get('conservation', {})
        conservation_benchmark = ConservationBenchmark(self.model, conservation_config)
        suite.add_benchmark(conservation_benchmark)
        
        # Add symmetry benchmark
        symmetry_config = self.config.get('symmetry', {})
        symmetry_benchmark = SymmetryBenchmark(self.model, symmetry_config)
        suite.add_benchmark(symmetry_benchmark)
        
        # Add accuracy benchmark
        accuracy_config = self.config.get('accuracy', {})
        accuracy_benchmark = AccuracyBenchmark(self.model, accuracy_config)
        suite.add_benchmark(accuracy_benchmark)
        
        return suite