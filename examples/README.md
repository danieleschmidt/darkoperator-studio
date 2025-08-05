# DarkOperator Studio Examples

This directory contains examples and tutorials demonstrating the capabilities of DarkOperator Studio for neural operator-based dark matter detection.

## 📚 Available Examples

### 1. Quick Start Tutorial
- **File**: `quickstart.py`
- **Description**: Basic anomaly detection on LHC Open Data
- **Concepts**: Data loading, operator setup, anomaly detection
- **Runtime**: ~5 minutes

### 2. Advanced Anomaly Detection
- **File**: `advanced_anomaly_detection.py`
- **Description**: Multi-modal anomaly detection with conformal prediction
- **Concepts**: Statistical guarantees, p-values, discovery significance
- **Runtime**: ~15 minutes

### 3. Custom Operator Development
- **File**: `custom_operator.py`
- **Description**: Building physics-informed neural operators
- **Concepts**: Conservation laws, symmetries, operator training
- **Runtime**: ~30 minutes

### 4. Performance Optimization
- **File**: `performance_optimization.py`
- **Description**: Scaling to large datasets with parallel processing
- **Concepts**: Caching, batching, memory optimization
- **Runtime**: ~20 minutes

### 5. Real-Time Processing
- **File**: `realtime_processing.py`
- **Description**: Streaming event processing for trigger systems
- **Concepts**: Async processing, latency optimization
- **Runtime**: ~10 minutes

## 🚀 Getting Started

### Prerequisites

Make sure DarkOperator Studio is installed:

```bash
pip install darkoperator
# or
conda install -c conda-forge darkoperator
```

### Running Examples

Each example is self-contained and can be run independently:

```bash
python examples/quickstart.py
```

For interactive exploration, use Jupyter notebooks:

```bash
jupyter notebook examples/
```

## 📖 Example Descriptions

### Quick Start Tutorial

This example demonstrates the basic workflow for dark matter detection:

1. **Data Loading**: Download CMS jet events from LHC Open Data
2. **Operator Setup**: Initialize calorimeter neural operator
3. **Anomaly Detection**: Run conformal anomaly detection
4. **Results Analysis**: Examine discovered anomalies

```python
import darkoperator as do

# Load data
events = do.load_opendata('cms-jets-13tev-2016', max_events=10000)

# Setup operator and detector
operator = do.CalorimeterOperator.from_pretrained('cms-ecal-2024')
detector = do.ConformalDetector(operator=operator, alpha=1e-6)

# Run anomaly detection
anomalies = detector.find_anomalies(events)
print(f"Found {len(anomalies)} potential dark matter events!")
```

### Advanced Anomaly Detection

Demonstrates sophisticated statistical methods:

- **Conformal Prediction**: Rigorous p-value computation
- **Multi-Modal Analysis**: Combining calorimeter, tracker, and muon data
- **Systematic Uncertainties**: Handling experimental uncertainties
- **Discovery Significance**: Computing statistical significance

### Custom Operator Development

Shows how to implement new physics-informed operators:

- **Architecture Design**: Building gauge-equivariant networks
- **Physics Constraints**: Enforcing conservation laws
- **Training Procedures**: Optimizing with physics losses
- **Validation Methods**: Testing against Monte Carlo truth

### Performance Optimization

Covers scaling to production workloads:

- **Batch Processing**: Efficient GPU utilization
- **Parallel Computing**: Multi-core CPU processing
- **Memory Management**: Handling large datasets
- **Caching Strategies**: Optimizing repeated computations

### Real-Time Processing

Demonstrates integration with experimental systems:

- **Streaming Data**: Processing events as they arrive
- **Latency Optimization**: Meeting trigger timing requirements
- **Throughput Scaling**: Handling high event rates
- **Quality Monitoring**: Real-time performance tracking

## 🔬 Physics Use Cases

### Dark Matter Searches

Examples focused on beyond-Standard-Model physics:

- **Mono-jet + MET**: Missing energy signatures
- **Displaced Vertices**: Long-lived particle decays
- **Soft Unclustered Energy**: Low-energy anomalies
- **Resonance Searches**: New particle discovery

### Detector Studies

Examples for detector development:

- **Calorimeter Optimization**: Energy resolution studies
- **Trigger Efficiency**: Optimizing selection criteria
- **Background Modeling**: Understanding SM processes
- **Systematic Studies**: Uncertainty quantification

## 🎯 Learning Objectives

After working through these examples, you will understand:

1. **Neural Operators**: How to use neural operators for physics simulations
2. **Anomaly Detection**: Statistical methods for BSM discovery
3. **Performance Optimization**: Scaling to large datasets
4. **Physics Validation**: Ensuring scientific rigor
5. **Production Deployment**: Moving from research to operations

## 📊 Expected Outputs

### Quickstart Example Output
```
Loading CMS jets dataset: 10,000 events
Calibrating conformal detector...
Running anomaly detection...
Found 23 anomalies with p < 1e-6
Most significant event: p = 2.3e-8 (4.8σ)
Results saved to: results/quickstart_anomalies.json
```

### Performance Benchmark Output
```
Baseline processing: 2.3 events/sec
With optimization: 847 events/sec (368x speedup)
Memory usage: 4.2 GB → 1.8 GB (57% reduction)
Latency: 435ms → 1.2ms per event
```

## 🔧 Customization

### Modifying Examples

Each example includes configuration sections for easy customization:

```python
# Configuration
CONFIG = {
    'dataset': 'cms-jets-13tev-2016',
    'max_events': 50000,
    'alpha': 1e-6,
    'batch_size': 64,
    'device': 'cuda'
}
```

### Adding New Examples

To contribute new examples:

1. Follow the template structure
2. Include clear documentation
3. Add physics validation
4. Test on multiple datasets

## 🐛 Troubleshooting

### Common Issues

**OutOfMemoryError**: Reduce batch size or max_events
```python
CONFIG['batch_size'] = 16  # Reduce from 64
CONFIG['max_events'] = 10000  # Reduce from 50000
```

**Slow Performance**: Enable GPU acceleration
```python
CONFIG['device'] = 'cuda'  # Use GPU instead of 'cpu'
```

**Network Timeouts**: Use cached data
```python
events = do.load_opendata(dataset_name, use_cache=True)
```

### Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check the full API documentation
- **Community**: Join our Discord for discussions
- **Papers**: Read the associated physics publications

## 📚 Additional Resources

### Physics Background
- [Neural Operators for PDEs](https://arxiv.org/abs/2010.08895)
- [Dark Matter at the LHC](https://arxiv.org/abs/1705.04582)
- [Anomaly Detection in Physics](https://arxiv.org/abs/2006.01137)

### Technical References
- [PyTorch Documentation](https://pytorch.org/docs/)
- [LHC Open Data Portal](http://opendata.cern.ch/)
- [Conformal Prediction](https://en.wikipedia.org/wiki/Conformal_prediction)

### Related Projects
- [Particle Physics ML](https://github.com/matthewfeickert/particle-physics-ml)
- [DeepJet](https://github.com/mstoye/DeepJet)
- [ATLAS ML](https://github.com/atlas-ml)

---

**Happy Dark Matter Hunting!** 🌌