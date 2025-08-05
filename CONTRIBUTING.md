# Contributing to DarkOperator Studio

Thank you for your interest in contributing to DarkOperator Studio! This project aims to revolutionize dark matter detection through neural operators and we welcome contributions from the physics, machine learning, and software engineering communities.

## 🌟 How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Physics Models**: New neural operator architectures for particle physics
2. **Anomaly Detection**: Novel algorithms for BSM discovery
3. **Performance**: Optimizations for speed and memory efficiency
4. **Documentation**: Tutorials, examples, and improved documentation
5. **Bug Reports**: Issues with existing functionality
6. **Feature Requests**: Ideas for new capabilities

### Priority Areas

The following areas are particularly important for the project:

#### 🔬 Physics-Informed Architectures
- Gauge-equivariant neural networks
- Lorentz-invariant embeddings
- Conservation law enforcement in operators
- Multi-scale calorimeter modeling

#### 🔍 Advanced Anomaly Detection
- Conformal prediction improvements
- Quantum anomaly detection algorithms
- Unsupervised domain adaptation
- Real-time trigger integration

#### ⚡ Performance & Scaling
- Multi-GPU optimization
- Distributed training frameworks
- Memory-efficient attention mechanisms
- FPGA deployment for L1 triggers

#### 🛡️ Robustness & Validation
- Physics benchmark datasets
- Cross-experiment validation
- Systematic uncertainty quantification
- Statistical significance testing

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/darkoperator-studio.git
   cd darkoperator-studio
   ```

2. **Create Development Environment**
   ```bash
   conda env create -f environment.yml
   conda activate darkoperator
   pip install -e .[dev]
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v
   python quality_gates_final.py
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow the coding standards below
   - Write tests for new functionality
   - Update documentation as needed

3. **Test Changes**
   ```bash
   # Run full test suite
   pytest tests/ -v --cov=darkoperator
   
   # Run quality gates
   python quality_gates_final.py
   
   # Check code style
   black darkoperator/ tests/
   flake8 darkoperator/ tests/
   mypy darkoperator/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new calorimeter operator architecture"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

## 📝 Coding Standards

### Python Style

We follow PEP 8 with some modifications:

```python
# Use black for formatting
max_line_length = 88

# Import organization
import standard_library
import third_party
import darkoperator_modules

# Type hints required for public APIs
def process_events(events: torch.Tensor) -> List[AnomalyResult]:
    """Process physics events for anomaly detection."""
    pass

# Docstring format (Google style)
def compute_invariant_mass(particles: torch.Tensor) -> torch.Tensor:
    """
    Compute invariant mass of particle system.
    
    Args:
        particles: 4-vectors of particles (batch, n_particles, 4)
        
    Returns:
        Invariant masses (batch,)
        
    Example:
        >>> particles = torch.tensor([[[100, 50, 30, 20]]])
        >>> mass = compute_invariant_mass(particles)
    """
    pass
```

### Physics Code Guidelines

```python
# Always validate 4-vector inputs
def validate_4vectors(x: torch.Tensor) -> torch.Tensor:
    """Ensure E^2 >= p^2 for all particles."""
    pass

# Use physics units consistently (GeV, cm, ns)
MUON_MASS = 0.106  # GeV
DETECTOR_RADIUS = 120  # cm

# Preserve symmetries in neural architectures
class LorentzEquivariantLayer(nn.Module):
    """Layer that preserves Lorentz symmetry."""
    pass
```

### Testing Requirements

- **Unit tests** for all public functions
- **Integration tests** for complete workflows
- **Physics validation** tests with known results
- **Performance benchmarks** for critical paths

```python
def test_energy_conservation():
    """Test that operator conserves energy."""
    operator = CalorimeterOperator()
    input_4vec = torch.tensor([[[100, 50, 30, 20]]])
    
    output = operator(input_4vec)
    
    input_energy = input_4vec[:, :, 0].sum()
    output_energy = output.sum()
    
    assert torch.isclose(input_energy, output_energy, rtol=0.1)
```

## 🔬 Physics Contributions

### Adding New Operators

1. **Inherit from PhysicsOperator**
   ```python
   class MyOperator(PhysicsOperator):
       def __init__(self, ...):
           super().__init__(...)
           # Initialize architecture
           
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           # Implement forward pass
           pass
           
       def physics_loss(self, input_4vec, output):
           # Implement physics constraints
           pass
   ```

2. **Add Comprehensive Tests**
   ```python
   class TestMyOperator:
       def test_forward_pass(self):
           # Test basic functionality
           pass
           
       def test_energy_conservation(self):
           # Test physics constraints
           pass
           
       def test_lorentz_invariance(self):
           # Test symmetry preservation
           pass
   ```

3. **Provide Physics Benchmarks**
   ```python
   def benchmark_against_geant4():
       """Compare operator output to Geant4 simulation."""
       pass
   ```

### Anomaly Detection Algorithms

When contributing new anomaly detection methods:

1. **Inherit from AnomalyDetector**
2. **Provide statistical guarantees** (p-values, confidence intervals)
3. **Include benchmarks** on standard datasets
4. **Document statistical properties**

```python
class MyAnomalyDetector(AnomalyDetector):
    def find_anomalies(self, events: torch.Tensor) -> List[int]:
        """Find anomalous events with statistical guarantees."""
        pass
        
    def compute_p_values(self, events: torch.Tensor) -> np.ndarray:
        """Compute rigorous p-values for each event."""
        pass
```

## 📊 Data Contributions

### LHC Open Data Integration

When adding support for new datasets:

1. **Follow LHC Open Data standards**
2. **Implement proper caching**
3. **Add metadata validation**
4. **Include usage examples**

```python
def load_new_dataset(
    dataset_name: str,
    cache_dir: str = "./data"
) -> torch.Tensor:
    """
    Load new physics dataset.
    
    Args:
        dataset_name: Name of dataset in LHC Open Data
        cache_dir: Local cache directory
        
    Returns:
        Events tensor (n_events, n_particles, 4)
    """
    pass
```

### Synthetic Data Generators

For testing and validation:

```python
def generate_signal_events(
    process: str,
    n_events: int,
    energy: float = 13000  # GeV
) -> torch.Tensor:
    """Generate Monte Carlo events for specific BSM process."""
    pass
```

## 🧪 Benchmarking

### Performance Benchmarks

Include benchmarks for new features:

```python
@pytest.mark.benchmark
def test_operator_throughput():
    """Benchmark operator throughput."""
    operator = MyOperator()
    events = generate_test_events(1000)
    
    start_time = time.time()
    with torch.no_grad():
        output = operator(events)
    elapsed = time.time() - start_time
    
    throughput = len(events) / elapsed
    assert throughput > 100  # events/sec
```

### Physics Benchmarks

Validate against known physics results:

```python
def test_z_boson_reconstruction():
    """Test Z boson invariant mass reconstruction."""
    # Generate Z → μμ events
    muon_pairs = generate_z_muon_pairs(1000)
    
    # Reconstruct invariant mass
    masses = compute_invariant_mass(muon_pairs)
    
    # Should peak at Z mass (91.2 GeV)
    peak_mass = masses[torch.argmax(torch.histc(masses, 100))]
    assert abs(peak_mass - 91.2) < 1.0  # GeV
```

## 📖 Documentation

### Code Documentation

- **Docstrings** for all public functions
- **Type hints** for all parameters
- **Physics explanations** for complex algorithms
- **Usage examples** in docstrings

### Tutorials

When adding tutorials:

1. **Jupyter notebooks** in `examples/` directory
2. **Step-by-step explanations**
3. **Physics context** and motivation
4. **Complete working examples**

```python
# Example: examples/anomaly_detection_tutorial.ipynb
"""
# Dark Matter Search Tutorial

This tutorial demonstrates how to search for dark matter
signals in LHC data using conformal anomaly detection.

## Physics Background
...

## Implementation
...
"""
```

### API Documentation

Update API docs for new features:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Short description of function.
    
    Longer description with physics context and mathematical
    details if needed. Include relevant equations:
    
    .. math::
        E^2 = p^2 + m^2
    
    Args:
        param1: Description of parameter with units
        param2: Description with allowed values
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input validation fails
        PhysicsError: When physics constraints violated
        
    Example:
        >>> result = my_function(42, "test")
        >>> assert result is True
        
    Note:
        Important implementation details or caveats.
        
    References:
        [1] Relevant physics paper or documentation
    """
    pass
```

## 🔍 Review Process

### Pull Request Guidelines

Your PR should include:

1. **Clear description** of changes and motivation
2. **Tests** for new functionality
3. **Documentation** updates
4. **Benchmark results** if applicable
5. **Physics validation** for new algorithms

### Review Criteria

PRs are reviewed for:

- **Code quality** and style compliance
- **Test coverage** and correctness
- **Physics validity** and accuracy
- **Performance** impact
- **Documentation** completeness
- **Security** considerations

### Physics Review

For physics contributions, we also check:

- **Theoretical correctness**
- **Numerical stability**
- **Symmetry preservation**
- **Units and dimensional analysis**
- **Comparison with established results**

## 🏆 Recognition

Contributors will be recognized through:

- **GitHub contributors page**
- **Academic paper acknowledgments**
- **Conference presentations**
- **Community showcase**

## 📞 Getting Help

If you need help contributing:

1. **GitHub Discussions** for general questions
2. **Issues** for bug reports and feature requests
3. **Discord** for real-time chat
4. **Email** for sensitive topics

### Physics Questions

For physics-specific questions:

- **arXiv discussions** for theoretical issues
- **INSPIRE-HEP** for literature searches
- **Physics Stack Exchange** for general physics

### Technical Questions

For technical implementation:

- **Stack Overflow** for Python/PyTorch issues
- **GitHub Issues** for project-specific problems
- **Documentation** for API questions

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Our Standards

- **Respectful** communication
- **Constructive** feedback
- **Collaborative** problem-solving
- **Educational** approach to mistakes
- **Open** to different perspectives

### Physics Community Values

- **Reproducible** research
- **Open** science practices
- **Rigorous** peer review
- **Ethical** use of data and methods

---

## 🎯 Conclusion

Contributing to DarkOperator Studio means joining the effort to revolutionize fundamental physics through AI. Whether you're a physicist, machine learning engineer, or software developer, your contributions can help unlock the mysteries of dark matter and beyond.

Thank you for being part of this exciting journey! 🌌