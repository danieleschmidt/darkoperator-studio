# DarkOperator Studio - Deployment Guide

## 🚀 Production Deployment

This guide covers deploying DarkOperator Studio for production use in dark matter detection and high-energy physics analysis.

## Prerequisites

### System Requirements

- **CPU**: 16+ cores (Intel Xeon or AMD EPYC recommended)
- **RAM**: 64GB+ (128GB recommended for large datasets)
- **GPU**: NVIDIA GPU with 24GB+ VRAM (A100/H100 recommended)
- **Storage**: 1TB+ NVMe SSD for data caching
- **Network**: High-bandwidth connection for LHC Open Data access

### Software Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- Docker (optional, for containerized deployment)
- Kubernetes (optional, for large-scale deployment)

## Installation Methods

### 1. Standard Installation

```bash
# Clone repository
git clone https://github.com/danieleschmidt/darkoperator-studio.git
cd darkoperator-studio

# Create conda environment
conda env create -f environment.yml
conda activate darkoperator

# Install package
pip install -e .

# Verify installation
darkoperator --version
```

### 2. Docker Deployment

```bash
# Build Docker image
docker build -t darkoperator:latest .

# Run container with GPU support
docker run --gpus all -v /data:/data -p 8888:8888 darkoperator:latest

# Run interactive analysis
docker run -it --gpus all darkoperator:latest bash
```

### 3. Kubernetes Deployment

```yaml
# darkoperator-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: darkoperator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: darkoperator
  template:
    metadata:
      labels:
        app: darkoperator
    spec:
      containers:
      - name: darkoperator
        image: darkoperator:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          limits:
            nvidia.com/gpu: 1
            memory: "64Gi"
            cpu: "16"
        volumeMounts:
        - name: data-volume
          mountPath: /data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: darkoperator-data
```

## Configuration

### Environment Variables

```bash
# Core settings
export DARKOPERATOR_DEVICE=cuda
export DARKOPERATOR_MAX_BATCH_SIZE=64
export DARKOPERATOR_MAX_MEMORY_GB=32

# Data paths
export DARKOPERATOR_CACHE_DIR=/data/cache
export DARKOPERATOR_MODEL_CACHE=/data/models
export DARKOPERATOR_LOG_DIR=/logs

# Security
export DARKOPERATOR_ENABLE_VALIDATION=true
export DARKOPERATOR_REQUIRE_CHECKSUMS=true

# Performance
export DARKOPERATOR_WORKERS=8
export DARKOPERATOR_ASYNC_PROCESSING=true
```

### Configuration File

Create `darkoperator.yml`:

```yaml
# Production configuration
debug: false
verbose: false
config_version: "1.0"

security:
  max_file_size_gb: 10.0
  trusted_domains:
    - "opendata.cern.ch"
    - "zenodo.org"
    - "huggingface.co"
  enable_model_validation: true
  require_checksums: true
  max_memory_usage_gb: 32.0

model:
  default_device: "cuda"
  max_batch_size: 64
  inference_timeout_s: 600.0
  enable_mixed_precision: true
  cache_dir: "/data/models"

data:
  cache_dir: "/data/cache"
  max_events_per_file: 10_000_000
  validation_split: 0.2
  random_seed: 42
  preprocessing_workers: 8

monitoring:
  log_level: "INFO"
  log_dir: "/logs"
  enable_performance_monitoring: true
  metrics_retention_days: 30
  alert_thresholds:
    memory_usage_gb: 28.0
    inference_time_ms: 2000.0
    error_rate: 0.01
```

## Performance Tuning

### GPU Optimization

```python
# Optimize for your specific GPU
import darkoperator as do

# Auto-detect optimal batch size
operator = do.CalorimeterOperator.from_pretrained('atlas-ecal-2024')
optimal_batch = do.optimization.optimize_batch_size(operator, sample_input)

# Enable memory-efficient attention
do.optimization.enable_memory_efficient_attention(operator)

# Use mixed precision
processor = do.BatchProcessor(
    model=operator,
    batch_size=optimal_batch,
    mixed_precision=True
)
```

### Distributed Processing

```python
# Multi-GPU setup
import torch.distributed as dist

# Initialize distributed backend
dist.init_process_group(backend='nccl')

# Create distributed detector
detector = do.ConformalDetector(
    operator=operator,
    distributed=True,
    world_size=torch.cuda.device_count()
)
```

## Monitoring and Alerting

### Prometheus Metrics

DarkOperator exposes metrics for monitoring:

```
# HELP darkoperator_inference_time_seconds Time taken for inference
# TYPE darkoperator_inference_time_seconds histogram
darkoperator_inference_time_seconds_bucket{model="calorimeter",le="0.1"} 1234

# HELP darkoperator_memory_usage_bytes Current memory usage
# TYPE darkoperator_memory_usage_bytes gauge
darkoperator_memory_usage_bytes{device="gpu_0"} 8589934592

# HELP darkoperator_anomalies_detected_total Total anomalies detected
# TYPE darkoperator_anomalies_detected_total counter
darkoperator_anomalies_detected_total{significance="5sigma"} 42
```

### Grafana Dashboard

Import the provided dashboard: `grafana-dashboard.json`

Key metrics to monitor:
- Inference latency and throughput
- Memory usage (CPU/GPU)
- Anomaly detection rates
- Data processing pipeline health
- System resource utilization

### Alerting Rules

```yaml
# Alert on high inference latency
- alert: HighInferenceLatency
  expr: darkoperator_inference_time_seconds > 2.0
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High inference latency detected"

# Alert on memory pressure
- alert: HighMemoryUsage
  expr: darkoperator_memory_usage_bytes / darkoperator_memory_total_bytes > 0.9
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Memory usage above 90%"
```

## Security Considerations

### Network Security

- Use TLS encryption for all API endpoints
- Implement proper authentication/authorization
- Restrict access to sensitive data endpoints
- Use VPN or private networks for production traffic

### Data Security

- Encrypt data at rest and in transit
- Implement proper access controls for LHC data
- Regular security audits of model checkpoints
- Secure handling of temporary files

### Model Security

```python
# Verify model checksums
from darkoperator.security import SecureModelLoader

loader = SecureModelLoader()
model = loader.load_model_safe(
    "https://trusted-source.com/model.pt",
    expected_hash="sha256:abc123..."
)
```

## Scaling Guidelines

### Horizontal Scaling

For large-scale deployments:

1. **Load Balancer**: Use nginx or HAProxy
2. **Multiple Workers**: Deploy multiple inference workers
3. **Data Partitioning**: Partition datasets across workers  
4. **Result Aggregation**: Collect and merge results

### Auto-scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: darkoperator-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: darkoperator
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Backup and Recovery

### Data Backup

```bash
# Backup model cache
rsync -av /data/models/ backup-server:/backup/models/

# Backup analysis results
rsync -av /data/results/ backup-server:/backup/results/

# Database backup (if using database)
pg_dump darkoperator_db > backup/darkoperator_$(date +%Y%m%d).sql
```

### Disaster Recovery

1. **Regular snapshots** of data volumes
2. **Multi-region deployment** for critical workloads
3. **Automated backups** with tested restore procedures
4. **Documentation** of recovery procedures

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

```python
# Reduce batch size
processor = do.BatchProcessor(batch_size=16)  # Reduce from 64

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use memory optimization
do.optimization.optimize_memory_usage(model)
```

#### Slow Inference

```python
# Profile inference
with do.monitoring.time_operation("inference"):
    results = detector.find_anomalies(events)

# Check GPU utilization
nvidia-smi

# Optimize model
traced_model = do.optimization.optimize_for_inference(model, sample_input)
```

#### Network Timeouts

```yaml
# Increase timeouts in configuration
model:
  inference_timeout_s: 1200.0  # 20 minutes
  
data:
  download_timeout_s: 3600.0   # 1 hour
```

### Log Analysis

```bash
# Check application logs
tail -f /logs/darkoperator.log

# Check for anomaly patterns
grep "anomaly" /logs/darkoperator.log | tail -100

# Monitor performance metrics
grep "performance" /logs/darkoperator.log | jq .
```

## Maintenance

### Regular Tasks

1. **Update model checkpoints** monthly
2. **Clean old cache files** weekly
3. **Rotate logs** daily
4. **Update dependencies** quarterly
5. **Security patches** as needed

### Health Checks

```bash
# Application health check
darkoperator health-check

# System resource check
darkoperator system-status

# Model validation
darkoperator validate-models
```

## Support

For production support:

- **Documentation**: [https://darkoperator.readthedocs.io](https://darkoperator.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/danieleschmidt/darkoperator-studio/issues)
- **Community**: [Discord Server](https://discord.gg/darkoperator)
- **Enterprise Support**: enterprise@darkoperator.ai

---

**Note**: This deployment guide assumes access to significant computational resources. For smaller-scale deployments, reduce the resource requirements accordingly.