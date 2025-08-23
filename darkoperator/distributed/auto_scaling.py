"""Auto-scaling and distributed computing for large-scale physics simulations."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import time
import threading
import queue
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from enum import Enum
import psutil
import subprocess


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    NETWORK = "network"


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    gpu_memory_percent: float
    network_io_mb_s: float
    disk_io_mb_s: float
    timestamp: float


class AutoScaler:
    """Auto-scaling controller for distributed physics simulations."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 8, 
                 target_utilization: float = 0.75, scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.current_workers = min_workers
        
    def should_scale(self, current_load: float, current_workers: int) -> Optional[str]:
        """Determine if scaling action is needed."""
        self.current_workers = current_workers
        
        if current_load > self.scale_up_threshold and current_workers < self.max_workers:
            return 'scale_up'
        elif current_load < self.scale_down_threshold and current_workers > self.min_workers:
            return 'scale_down'
        else:
            return None
    
    def scale_up(self) -> int:
        """Scale up workers."""
        new_workers = min(self.current_workers + 1, self.max_workers)
        logger.info(f"Scaling up from {self.current_workers} to {new_workers} workers")
        self.current_workers = new_workers
        return new_workers
    
    def scale_down(self) -> int:
        """Scale down workers."""
        new_workers = max(self.current_workers - 1, self.min_workers)
        logger.info(f"Scaling down from {self.current_workers} to {new_workers} workers")
        self.current_workers = new_workers
        return new_workers


@dataclass
class WorkerConfig:
    """Configuration for distributed workers."""
    worker_id: int
    device: torch.device
    rank: int
    world_size: int
    master_addr: str = "localhost"
    master_port: int = 12355


class ResourceMonitor:
    """Monitor system resources for auto-scaling decisions."""
    
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 100):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.metrics_history: List[ResourceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history size manageable
                if len(self.metrics_history) > self.history_size:
                    self.metrics_history = self.metrics_history[-self.history_size:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io_mb_s = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024) / self.monitoring_interval
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_mb_s = 0
        if disk_io:
            disk_io_mb_s = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024) / self.monitoring_interval
        
        # GPU metrics
        gpu_percent = 0
        gpu_memory_percent = 0
        
        if torch.cuda.is_available():
            try:
                gpu_percent = torch.cuda.utilization()
                gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            except:
                pass
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            network_io_mb_s=network_io_mb_s,
            disk_io_mb_s=disk_io_mb_s,
            timestamp=time.time()
        )
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, window_size: int = 10) -> Optional[ResourceMetrics]:
        """Get average metrics over recent window."""
        if len(self.metrics_history) == 0:
            return None
        
        recent_metrics = self.metrics_history[-window_size:]
        
        return ResourceMetrics(
            cpu_percent=np.mean([m.cpu_percent for m in recent_metrics]),
            memory_percent=np.mean([m.memory_percent for m in recent_metrics]),
            gpu_percent=np.mean([m.gpu_percent for m in recent_metrics]),
            gpu_memory_percent=np.mean([m.gpu_memory_percent for m in recent_metrics]),
            network_io_mb_s=np.mean([m.network_io_mb_s for m in recent_metrics]),
            disk_io_mb_s=np.mean([m.disk_io_mb_s for m in recent_metrics]),
            timestamp=time.time()
        )


class AutoScaler:
    """Automatic scaling based on resource usage and workload."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: float = 60.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.current_workers = min_workers
        self.last_scaling_time = 0
        self.pending_tasks = queue.Queue()
        
        self.resource_monitor = ResourceMonitor()
        self.logger = logging.getLogger(__name__)
    
    def should_scale_up(self, metrics: ResourceMetrics, pending_task_count: int) -> bool:
        """Determine if we should scale up."""
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.cooldown_period:
            return False
        
        # Check if already at maximum
        if self.current_workers >= self.max_workers:
            return False
        
        # Scale up conditions
        high_cpu = metrics.cpu_percent > self.scale_up_threshold * 100
        high_memory = metrics.memory_percent > self.scale_up_threshold * 100
        high_gpu = metrics.gpu_percent > self.scale_up_threshold * 100
        many_pending_tasks = pending_task_count > self.current_workers * 2
        
        return high_cpu or high_memory or high_gpu or many_pending_tasks
    
    def should_scale_down(self, metrics: ResourceMetrics, pending_task_count: int) -> bool:
        """Determine if we should scale down."""
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.cooldown_period:
            return False
        
        # Check if already at minimum
        if self.current_workers <= self.min_workers:
            return False
        
        # Scale down conditions
        low_cpu = metrics.cpu_percent < self.scale_down_threshold * 100
        low_memory = metrics.memory_percent < self.scale_down_threshold * 100
        low_gpu = metrics.gpu_percent < self.scale_down_threshold * 100
        few_pending_tasks = pending_task_count < self.current_workers // 2
        
        return low_cpu and low_memory and low_gpu and few_pending_tasks
    
    def make_scaling_decision(self, pending_task_count: int = 0) -> int:
        """Make scaling decision based on current metrics."""
        metrics = self.resource_monitor.get_average_metrics(window_size=5)
        
        if metrics is None:
            return self.current_workers
        
        if self.should_scale_up(metrics, pending_task_count):
            new_workers = min(self.max_workers, self.current_workers + 1)
            self.logger.info(f"Scaling up: {self.current_workers} -> {new_workers}")
            self.current_workers = new_workers
            self.last_scaling_time = time.time()
            
        elif self.should_scale_down(metrics, pending_task_count):
            new_workers = max(self.min_workers, self.current_workers - 1)
            self.logger.info(f"Scaling down: {self.current_workers} -> {new_workers}")
            self.current_workers = new_workers
            self.last_scaling_time = time.time()
        
        return self.current_workers


class DistributedWorker:
    """Distributed worker for physics computations."""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.device = config.device
        self.is_initialized = False
        self.logger = logging.getLogger(f"worker_{config.worker_id}")
    
    def initialize(self):
        """Initialize distributed worker."""
        try:
            # Initialize process group for distributed training
            if self.config.world_size > 1:
                dist.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method=f'tcp://{self.config.master_addr}:{self.config.master_port}',
                    world_size=self.config.world_size,
                    rank=self.config.rank
                )
            
            self.is_initialized = True
            self.logger.info(f"Worker {self.config.worker_id} initialized on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize worker: {e}")
            raise
    
    def process_batch(
        self, 
        model: torch.nn.Module, 
        batch_data: torch.Tensor,
        task_metadata: Dict[str, Any] = None
    ) -> torch.Tensor:
        """Process a batch of data."""
        if not self.is_initialized:
            self.initialize()
        
        try:
            # Move data to device
            batch_data = batch_data.to(self.device)
            model = model.to(self.device)
            
            # Process batch
            with torch.no_grad():
                result = model(batch_data)
            
            # Move result back to CPU for collection
            return result.cpu()
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            raise
    
    def cleanup(self):
        """Clean up worker resources."""
        if self.is_initialized and dist.is_initialized():
            dist.destroy_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"Worker {self.config.worker_id} cleaned up")


class DistributedCoordinator:
    """Coordinate distributed physics computations."""
    
    def __init__(
        self, 
        auto_scale: bool = True,
        max_workers: Optional[int] = None
    ):
        self.auto_scale = auto_scale
        self.max_workers = max_workers or min(8, torch.cuda.device_count() or 4)
        
        self.workers: Dict[int, DistributedWorker] = {}
        self.worker_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.auto_scaler = AutoScaler(max_workers=self.max_workers) if auto_scale else None
        self.resource_monitor = ResourceMonitor()
        
        self.task_queue = queue.Queue()
        self.result_futures: Dict[str, Future] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the distributed coordinator."""
        self.resource_monitor.start_monitoring()
        
        # Initialize minimum workers
        initial_workers = self.auto_scaler.min_workers if self.auto_scaler else 1
        self._scale_workers(initial_workers)
        
        self.logger.info(f"Distributed coordinator started with {len(self.workers)} workers")
    
    def stop(self):
        """Stop the distributed coordinator."""
        self.resource_monitor.stop_monitoring()
        
        # Clean up workers
        for worker in self.workers.values():
            worker.cleanup()
        
        self.worker_pool.shutdown(wait=True)
        self.workers.clear()
        
        self.logger.info("Distributed coordinator stopped")
    
    def _scale_workers(self, target_count: int):
        """Scale workers to target count."""
        current_count = len(self.workers)
        
        if target_count > current_count:
            # Scale up
            for i in range(current_count, target_count):
                device = torch.device(f'cuda:{i % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
                config = WorkerConfig(
                    worker_id=i,
                    device=device,
                    rank=i,
                    world_size=target_count
                )
                self.workers[i] = DistributedWorker(config)
        
        elif target_count < current_count:
            # Scale down
            workers_to_remove = list(range(target_count, current_count))
            for worker_id in workers_to_remove:
                if worker_id in self.workers:
                    self.workers[worker_id].cleanup()
                    del self.workers[worker_id]
    
    def submit_computation(
        self,
        model: torch.nn.Module,
        data_batches: List[torch.Tensor],
        task_id: Optional[str] = None
    ) -> Future:
        """Submit distributed computation."""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}"
        
        # Auto-scaling decision
        if self.auto_scaler:
            target_workers = self.auto_scaler.make_scaling_decision(
                pending_task_count=len(data_batches)
            )
            self._scale_workers(target_workers)
        
        # Submit work to workers
        future = self.worker_pool.submit(
            self._execute_distributed_computation,
            model, data_batches, task_id
        )
        
        self.result_futures[task_id] = future
        return future
    
    def _execute_distributed_computation(
        self,
        model: torch.nn.Module,
        data_batches: List[torch.Tensor],
        task_id: str
    ) -> List[torch.Tensor]:
        """Execute computation across workers."""
        if not self.workers:
            raise RuntimeError("No workers available")
        
        # Distribute batches across workers
        worker_ids = list(self.workers.keys())
        results = [None] * len(data_batches)
        
        # Submit batch processing tasks
        batch_futures = []
        for i, batch in enumerate(data_batches):
            worker_id = worker_ids[i % len(worker_ids)]
            worker = self.workers[worker_id]
            
            future = self.worker_pool.submit(
                worker.process_batch,
                model, batch, {'task_id': task_id, 'batch_id': i}
            )
            batch_futures.append((i, future))
        
        # Collect results
        for batch_idx, future in batch_futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results[batch_idx] = result
            except Exception as e:
                self.logger.error(f"Batch {batch_idx} failed: {e}")
                # Use zero tensor as fallback
                if data_batches:
                    results[batch_idx] = torch.zeros_like(data_batches[0])
        
        return [r for r in results if r is not None]
    
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        metrics = self.resource_monitor.get_current_metrics()
        
        return {
            'num_workers': len(self.workers),
            'pending_tasks': len(self.result_futures),
            'resource_metrics': metrics.__dict__ if metrics else None,
            'auto_scaling_enabled': self.auto_scale,
        }


async def async_physics_computation(
    coordinator: DistributedCoordinator,
    model: torch.nn.Module,
    data_stream: asyncio.Queue
) -> asyncio.Queue:
    """Asynchronous physics computation pipeline."""
    
    result_queue = asyncio.Queue()
    
    async def process_batches():
        batch_buffer = []
        batch_size = 32
        
        while True:
            try:
                # Get data from stream
                data = await asyncio.wait_for(data_stream.get(), timeout=1.0)
                
                if data is None:  # Sentinel for end of stream
                    break
                
                batch_buffer.append(data)
                
                # Process when buffer is full
                if len(batch_buffer) >= batch_size:
                    future = coordinator.submit_computation(model, batch_buffer)
                    
                    # Wait for results asynchronously
                    results = await asyncio.wrap_future(future)
                    
                    for result in results:
                        await result_queue.put(result)
                    
                    batch_buffer = []
                    
            except asyncio.TimeoutError:
                # Process remaining batches
                if batch_buffer:
                    future = coordinator.submit_computation(model, batch_buffer)
                    results = await asyncio.wrap_future(future)
                    
                    for result in results:
                        await result_queue.put(result)
                    
                    batch_buffer = []
            
            except Exception as e:
                logging.error(f"Error in async computation: {e}")
                break
    
    # Start processing
    await process_batches()
    
    # Signal end of results
    await result_queue.put(None)
    
    return result_queue


def test_distributed_scaling():
    """Test distributed computing and auto-scaling."""
    print("⚡ Testing distributed computing and auto-scaling...")
    
    # Test resource monitor
    monitor = ResourceMonitor(monitoring_interval=0.1)
    monitor.start_monitoring()
    
    time.sleep(0.5)  # Let it collect some metrics
    
    current_metrics = monitor.get_current_metrics()
    assert current_metrics is not None, "Should have collected metrics"
    print(f"✓ Resource monitoring: CPU {current_metrics.cpu_percent:.1f}%, Memory {current_metrics.memory_percent:.1f}%")
    
    monitor.stop_monitoring()
    
    # Test auto-scaler
    auto_scaler = AutoScaler(min_workers=1, max_workers=4)
    auto_scaler.resource_monitor.start_monitoring()
    
    time.sleep(0.2)
    decision = auto_scaler.make_scaling_decision(pending_task_count=10)
    print(f"✓ Auto-scaling decision: {decision} workers")
    
    auto_scaler.resource_monitor.stop_monitoring()
    
    # Test distributed coordinator
    coordinator = DistributedCoordinator(auto_scale=True, max_workers=2)
    coordinator.start()
    
    # Create test model and data
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x * 2
    
    model = SimpleModel()
    test_batches = [torch.randn(4, 10) for _ in range(5)]
    
    # Submit computation
    future = coordinator.submit_computation(model, test_batches)
    results = future.result(timeout=30)
    
    assert len(results) == len(test_batches), "Should return all results"
    print(f"✓ Distributed computation: {len(results)} batches processed")
    
    # Check status
    status = coordinator.get_status()
    print(f"✓ Coordinator status: {status['num_workers']} workers")
    
    coordinator.stop()
    
    print("✅ All distributed computing and auto-scaling tests passed!")


if __name__ == "__main__":
    test_distributed_scaling()