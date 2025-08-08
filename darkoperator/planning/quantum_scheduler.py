"""
Quantum-inspired task scheduler using superposition and entanglement principles.

This module implements quantum-inspired algorithms for optimal task scheduling
in physics simulations and anomaly detection pipelines.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from ..operators.base import PhysicsOperator


logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels inspired by quantum energy levels."""
    GROUND_STATE = 0      # Highest priority (like ground state)
    EXCITED_1 = 1         # High priority
    EXCITED_2 = 2         # Medium priority  
    EXCITED_3 = 3         # Low priority
    METASTABLE = 4        # Background tasks


@dataclass
class QuantumTask:
    """
    Task representation with quantum-inspired properties.
    
    Uses superposition principle where tasks can exist in multiple states
    until observed/executed.
    """
    
    task_id: str
    name: str
    operation: callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Quantum-inspired properties
    priority: TaskPriority = TaskPriority.EXCITED_2
    entangled_tasks: List[str] = field(default_factory=list)  # Dependencies
    superposition_weight: float = 1.0  # Probability amplitude
    coherence_time: float = 3600.0     # How long task stays valid (seconds)
    
    # Physics constraints
    energy_requirement: float = 1.0     # Computational cost
    momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Resource vector
    spin: float = 0.0                   # Task coupling strength
    
    # Runtime properties
    created_time: float = field(default_factory=time.time)
    state: str = "superposition"        # superposition, collapsed, executed, failed
    result: Any = None
    execution_time: Optional[float] = None
    
    def is_coherent(self) -> bool:
        """Check if task is still within coherence time."""
        return (time.time() - self.created_time) < self.coherence_time
    
    def collapse_wavefunction(self) -> None:
        """Collapse task from superposition to definite state."""
        if self.state == "superposition":
            self.state = "collapsed"
            logger.debug(f"Task {self.task_id} wavefunction collapsed")


class QuantumTaskGraph:
    """
    Task dependency graph with quantum entanglement properties.
    
    Uses networkx for graph operations with quantum-inspired scheduling.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entangled_pairs: Dict[Tuple[str, str], float] = {}
        
    def add_task(self, task: QuantumTask) -> None:
        """Add task node to quantum graph."""
        self.graph.add_node(
            task.task_id,
            task=task,
            priority=task.priority.value,
            energy=task.energy_requirement,
            spin=task.spin
        )
        
    def entangle_tasks(self, task1_id: str, task2_id: str, coupling_strength: float = 1.0) -> None:
        """Create entanglement (dependency) between tasks."""
        self.graph.add_edge(task1_id, task2_id, weight=coupling_strength)
        self.entangled_pairs[(task1_id, task2_id)] = coupling_strength
        
        # Update task entanglement lists
        if task1_id in self.graph.nodes:
            task1 = self.graph.nodes[task1_id]['task']
            if task2_id not in task1.entangled_tasks:
                task1.entangled_tasks.append(task2_id)
    
    def get_ready_tasks(self) -> List[QuantumTask]:
        """Get tasks ready for execution (no unfulfilled dependencies)."""
        ready_tasks = []
        
        for node_id in self.graph.nodes():
            # Check if all predecessors are completed
            predecessors = list(self.graph.predecessors(node_id))
            if not predecessors or all(
                self.graph.nodes[pred]['task'].state == "executed" 
                for pred in predecessors
            ):
                task = self.graph.nodes[node_id]['task']
                if task.state in ["superposition", "collapsed"] and task.is_coherent():
                    ready_tasks.append(task)
        
        return ready_tasks
    
    def compute_interference_pattern(self) -> Dict[str, float]:
        """
        Compute quantum interference effects on task priorities.
        
        Tasks can interfere constructively (higher priority) or destructively.
        """
        interference = {}
        
        for node_id in self.graph.nodes():
            task = self.graph.nodes[node_id]['task']
            base_priority = task.priority.value
            
            # Consider entangled task effects
            interference_factor = 0.0
            for entangled_id in task.entangled_tasks:
                if entangled_id in self.graph.nodes:
                    other_task = self.graph.nodes[entangled_id]['task']
                    coupling = self.entangled_pairs.get((task.task_id, entangled_id), 1.0)
                    
                    # Constructive interference if tasks have similar priorities
                    priority_diff = abs(base_priority - other_task.priority.value)
                    if priority_diff <= 1:
                        interference_factor += coupling * 0.1  # Boost priority
                    else:
                        interference_factor -= coupling * 0.05  # Reduce priority
            
            # Apply superposition weight
            final_priority = base_priority - interference_factor * task.superposition_weight
            interference[node_id] = max(0, final_priority)  # Ensure non-negative
            
        return interference


class QuantumScheduler:
    """
    Quantum-inspired task scheduler for physics simulations with large-scale optimization.
    
    Uses quantum principles like superposition, entanglement, and interference
    to optimize task execution order and resource allocation for >1000 concurrent tasks.
    
    Enhanced Features:
    - Hierarchical task clustering for scalability
    - Adaptive load balancing with quantum interference
    - Memory-efficient sparse matrix operations
    - Dynamic worker pool scaling
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        quantum_annealing_steps: int = 100,
        temperature: float = 1.0,
        physics_operator: Optional[PhysicsOperator] = None,
        enable_clustering: bool = True,
        cluster_size_limit: int = 50,
        adaptive_scaling: bool = True,
        max_concurrent_tasks: int = 2000
    ):
        self.max_workers = max_workers
        self.quantum_annealing_steps = quantum_annealing_steps
        self.temperature = temperature
        self.physics_operator = physics_operator
        self.enable_clustering = enable_clustering
        self.cluster_size_limit = cluster_size_limit
        self.adaptive_scaling = adaptive_scaling
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self.task_graph = QuantumTaskGraph()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_tasks: Dict[str, Any] = {}
        
        # Large-scale optimization features
        self.task_clusters: Dict[int, List[str]] = {}
        self.cluster_heads: Dict[int, str] = {}
        self.sparse_adjacency_matrix = None
        self.adaptive_worker_pool = None
        self.load_balancer = None
        
        # Quantum state tracking
        self.global_phase = 0.0
        self.entanglement_entropy = 0.0
        
        # Performance monitoring
        self.execution_stats = {
            'tasks_processed': 0,
            'total_execution_time': 0.0,
            'quantum_efficiency': 0.0,
            'cluster_rebalance_count': 0,
            'worker_scaling_events': 0
        }
        
        logger.info(f"Quantum scheduler initialized with enhanced scaling: max_concurrent={max_concurrent_tasks}")
        
        if self.adaptive_scaling:
            self._initialize_adaptive_components()
        
        if self.enable_clustering:
            self._initialize_clustering_system()
    
    def _initialize_adaptive_components(self) -> None:
        """Initialize components for adaptive scaling."""
        
        try:
            from concurrent.futures import ProcessPoolExecutor
            from queue import Queue
            import threading
            
            # Adaptive worker pool for CPU-intensive tasks
            self.adaptive_worker_pool = {
                'thread_pool': ThreadPoolExecutor(max_workers=self.max_workers),
                'process_pool': ProcessPoolExecutor(max_workers=max(1, self.max_workers // 2)),
                'current_load': 0,
                'max_load': self.max_concurrent_tasks
            }
            
            # Load balancer for task distribution
            self.load_balancer = {
                'worker_queues': [Queue() for _ in range(self.max_workers)],
                'worker_loads': [0] * self.max_workers,
                'round_robin_counter': 0,
                'load_lock': threading.Lock()
            }
            
            logger.info("Adaptive scaling components initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize adaptive components: {e}")
            self.adaptive_scaling = False
    
    def _initialize_clustering_system(self) -> None:
        """Initialize hierarchical task clustering system."""
        
        try:
            # Initialize sparse matrix for large-scale operations
            import scipy.sparse as sp
            
            # Pre-allocate sparse adjacency matrix
            max_tasks = self.max_concurrent_tasks
            self.sparse_adjacency_matrix = sp.lil_matrix((max_tasks, max_tasks), dtype=np.float32)
            
            # Task clustering configuration
            self.clustering_config = {
                'algorithm': 'spectral',  # spectral clustering for quantum-inspired approach
                'max_clusters': max_tasks // self.cluster_size_limit,
                'similarity_threshold': 0.7,
                'rebalance_frequency': 100  # tasks processed between rebalancing
            }
            
            logger.info(f"Clustering system initialized: max_clusters={self.clustering_config['max_clusters']}")
            
        except ImportError:
            logger.warning("SciPy not available, disabling clustering optimization")
            self.enable_clustering = False
        except Exception as e:
            logger.warning(f"Failed to initialize clustering system: {e}")
            self.enable_clustering = False
    
    def _cluster_tasks_by_affinity(self, tasks: List[QuantumTask]) -> Dict[int, List[str]]:
        """
        Cluster tasks based on quantum affinity and computational requirements.
        
        Uses spectral clustering on quantum interference patterns.
        """
        
        if not self.enable_clustering or len(tasks) <= self.cluster_size_limit:
            # Single cluster for small task sets
            return {0: [task.task_id for task in tasks]}
        
        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create feature matrix from task properties
            features = []
            task_ids = []
            
            for task in tasks:
                feature_vector = [
                    task.priority.value,
                    task.energy_requirement,
                    task.spin,
                    task.superposition_weight,
                    len(task.entangled_tasks),
                    float(task.is_coherent())
                ]
                
                # Add physics-aware features
                if hasattr(task, 'momentum'):
                    feature_vector.extend(task.momentum)
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
                
                features.append(feature_vector)
                task_ids.append(task.task_id)
            
            features_array = np.array(features)
            
            # Calculate affinity matrix using cosine similarity
            affinity_matrix = cosine_similarity(features_array)
            
            # Apply quantum interference effects
            interference = self.task_graph.compute_interference_pattern()
            for i, task_id in enumerate(task_ids):
                if task_id in interference:
                    # Boost similarity for constructive interference
                    interference_factor = 1.0 + (1.0 - interference[task_id]) * 0.2
                    affinity_matrix[i] *= interference_factor
            
            # Determine optimal number of clusters
            n_tasks = len(tasks)
            n_clusters = min(
                max(1, n_tasks // self.cluster_size_limit),
                self.clustering_config['max_clusters']
            )
            
            # Perform spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            
            cluster_labels = clustering.fit_predict(affinity_matrix)
            
            # Group tasks by cluster
            clusters = {}
            for task_id, cluster_id in zip(task_ids, cluster_labels):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(task_id)
            
            self.execution_stats['cluster_rebalance_count'] += 1
            
            logger.debug(f"Clustered {n_tasks} tasks into {len(clusters)} clusters")
            return clusters
            
        except ImportError:
            logger.warning("sklearn not available, using simple clustering")
            return self._simple_task_clustering(tasks)
        except Exception as e:
            logger.error(f"Task clustering failed: {e}")
            return self._simple_task_clustering(tasks)
    
    def _simple_task_clustering(self, tasks: List[QuantumTask]) -> Dict[int, List[str]]:
        """Simple clustering based on priority and energy requirements."""
        
        clusters = {}
        cluster_id = 0
        current_cluster = []
        
        # Sort tasks by priority and energy
        sorted_tasks = sorted(tasks, key=lambda t: (t.priority.value, t.energy_requirement))
        
        for task in sorted_tasks:
            current_cluster.append(task.task_id)
            
            if len(current_cluster) >= self.cluster_size_limit:
                clusters[cluster_id] = current_cluster
                cluster_id += 1
                current_cluster = []
        
        # Add remaining tasks
        if current_cluster:
            clusters[cluster_id] = current_cluster
        
        return clusters
    
    def _adaptive_worker_scaling(self, current_load: int, queue_size: int) -> None:
        """
        Dynamically scale worker pool based on current load.
        
        Uses quantum-inspired scaling factors based on system state.
        """
        
        if not self.adaptive_scaling or self.adaptive_worker_pool is None:
            return
        
        try:
            # Calculate load factor
            load_factor = current_load / max(1, self.max_concurrent_tasks)
            queue_factor = queue_size / max(1, self.max_workers * 10)
            
            # Quantum-inspired scaling decision
            scaling_probability = 1.0 / (1.0 + np.exp(-5 * (load_factor + queue_factor - 0.7)))
            
            if scaling_probability > 0.8:
                # Scale up workers
                new_max_workers = min(
                    self.max_workers * 2,
                    self.max_concurrent_tasks // 10
                )
                
                if new_max_workers > self.max_workers:
                    # Create new executor with more workers
                    old_executor = self.executor
                    self.executor = ThreadPoolExecutor(max_workers=new_max_workers)
                    
                    # Update worker count
                    self.max_workers = new_max_workers
                    self.execution_stats['worker_scaling_events'] += 1
                    
                    logger.info(f"Scaled up workers to {new_max_workers}")
                    
                    # Schedule cleanup of old executor
                    def cleanup_old_executor():
                        time.sleep(1)
                        old_executor.shutdown(wait=False)
                    
                    import threading
                    threading.Thread(target=cleanup_old_executor, daemon=True).start()
            
            elif scaling_probability < 0.2 and self.max_workers > 2:
                # Scale down workers if load is low
                new_max_workers = max(2, self.max_workers // 2)
                
                if new_max_workers < self.max_workers:
                    # Create new executor with fewer workers
                    old_executor = self.executor
                    self.executor = ThreadPoolExecutor(max_workers=new_max_workers)
                    
                    self.max_workers = new_max_workers
                    self.execution_stats['worker_scaling_events'] += 1
                    
                    logger.info(f"Scaled down workers to {new_max_workers}")
                    
                    # Schedule cleanup
                    def cleanup_old_executor():
                        time.sleep(1)
                        old_executor.shutdown(wait=False)
                    
                    import threading
                    threading.Thread(target=cleanup_old_executor, daemon=True).start()
        
        except Exception as e:
            logger.error(f"Adaptive worker scaling failed: {e}")
    
    def _optimize_task_execution_order(self, ready_tasks: List[QuantumTask]) -> List[QuantumTask]:
        """
        Optimize task execution order using quantum annealing-inspired algorithm.
        
        Enhanced for large-scale optimization with hierarchical clustering.
        """
        
        if len(ready_tasks) <= 10:
            # Use existing simple optimization for small task sets
            return self._quantum_anneal_schedule(ready_tasks)
        
        # For large task sets, use hierarchical optimization
        try:
            # Step 1: Cluster tasks by affinity
            clusters = self._cluster_tasks_by_affinity(ready_tasks)
            
            # Step 2: Optimize within each cluster
            optimized_tasks = []
            cluster_priorities = {}
            
            for cluster_id, task_ids in clusters.items():
                cluster_tasks = [t for t in ready_tasks if t.task_id in task_ids]
                
                # Optimize cluster internally
                optimized_cluster = self._quantum_anneal_schedule(cluster_tasks)
                optimized_tasks.extend(optimized_cluster)
                
                # Calculate cluster priority (average of task priorities)
                if cluster_tasks:
                    avg_priority = np.mean([t.priority.value for t in cluster_tasks])
                    cluster_priorities[cluster_id] = avg_priority
            
            # Step 3: Order clusters by priority
            sorted_cluster_ids = sorted(cluster_priorities.keys(), 
                                      key=lambda cid: cluster_priorities[cid])
            
            # Step 4: Reorder tasks by cluster priority
            final_order = []
            for cluster_id in sorted_cluster_ids:
                cluster_task_ids = clusters[cluster_id]
                cluster_tasks = [t for t in optimized_tasks if t.task_id in cluster_task_ids]
                final_order.extend(cluster_tasks)
            
            logger.debug(f"Optimized execution order for {len(ready_tasks)} tasks using {len(clusters)} clusters")
            return final_order
            
        except Exception as e:
            logger.error(f"Large-scale optimization failed, falling back to simple annealing: {e}")
            return self._quantum_anneal_schedule(ready_tasks[:100])  # Limit to prevent memory issues
        
    def submit_task(
        self,
        task_id: str,
        name: str,
        operation: callable,
        *args,
        priority: TaskPriority = TaskPriority.EXCITED_2,
        dependencies: List[str] = None,
        energy_requirement: float = 1.0,
        **kwargs
    ) -> QuantumTask:
        """Submit task to quantum scheduler."""
        
        task = QuantumTask(
            task_id=task_id,
            name=name,
            operation=operation,
            args=args,
            kwargs=kwargs,
            priority=priority,
            energy_requirement=energy_requirement
        )
        
        self.task_graph.add_task(task)
        
        # Add dependencies (entanglements)
        if dependencies:
            for dep_id in dependencies:
                self.task_graph.entangle_tasks(dep_id, task_id)
        
        logger.info(f"Submitted quantum task: {task_id} ({name})")
        return task
    
    def quantum_anneal_schedule(self) -> List[QuantumTask]:
        """
        Use quantum annealing to find optimal task execution order.
        
        Minimizes total energy while respecting constraints.
        Enhanced with large-scale optimization capabilities.
        """
        ready_tasks = self.task_graph.get_ready_tasks()
        if not ready_tasks:
            return []
        
        # Use enhanced optimization for large task sets
        if len(ready_tasks) > 50:
            return self._optimize_task_execution_order(ready_tasks)
        
        # Use traditional annealing for smaller sets
        return self._quantum_anneal_schedule(ready_tasks)
    
    def _quantum_anneal_schedule(self, ready_tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Traditional quantum annealing for small to medium task sets."""
        
        # Current solution (random initial state)
        current_order = ready_tasks.copy()
        np.random.shuffle(current_order)
        current_energy = self._compute_total_energy(current_order)
        
        best_order = current_order.copy()
        best_energy = current_energy
        
        # Quantum annealing process
        for step in range(self.quantum_annealing_steps):
            # Cooling schedule
            temp = self.temperature * (1 - step / self.quantum_annealing_steps)
            
            # Quantum fluctuation: swap two random tasks
            new_order = current_order.copy()
            if len(new_order) > 1:
                i, j = np.random.choice(len(new_order), 2, replace=False)
                new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_energy = self._compute_total_energy(new_order)
            
            # Quantum tunneling probability
            if new_energy < current_energy or np.random.random() < np.exp(-(new_energy - current_energy) / max(temp, 1e-10)):
                current_order = new_order
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_order = new_order.copy()
                    best_energy = new_energy
        
        logger.debug(f"Quantum annealing found schedule with energy: {best_energy:.3f}")
        return best_order
    
    def _compute_total_energy(self, task_order: List[QuantumTask]) -> float:
        """Compute total energy of task execution order."""
        total_energy = 0.0
        
        # Base energy from task requirements
        for task in task_order:
            total_energy += task.energy_requirement * (1 + task.priority.value * 0.1)
        
        # Interference effects
        interference = self.task_graph.compute_interference_pattern()
        for task in task_order:
            total_energy += interference.get(task.task_id, 0) * 0.1
        
        # Momentum conservation penalty (prefer balanced workload)
        total_momentum = sum(task.momentum for task in task_order)
        momentum_penalty = np.linalg.norm(total_momentum) * 0.01
        total_energy += momentum_penalty
        
        return total_energy
    
    def execute_quantum_schedule(self) -> Dict[str, Any]:
        """
        Execute tasks using quantum-optimized scheduling with large-scale capabilities.
        
        Enhanced with adaptive scaling, load balancing, and hierarchical execution.
        Returns dictionary of task results and execution statistics.
        """
        results = {}
        statistics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'quantum_efficiency': 0.0,
            'max_concurrent_tasks': 0,
            'clusters_processed': 0,
            'scaling_events': 0
        }
        
        start_time = time.time()
        max_concurrent_reached = 0
        
        while True:
            # Get quantum-optimized task order
            scheduled_tasks = self.quantum_anneal_schedule()
            
            if not scheduled_tasks:
                # Check if any tasks are still running
                if not self.running_tasks:
                    break
                time.sleep(0.1)
                continue
            
            # Adaptive worker scaling based on workload
            current_load = len(self.running_tasks)
            queue_size = len(scheduled_tasks)
            max_concurrent_reached = max(max_concurrent_reached, current_load)
            
            self._adaptive_worker_scaling(current_load, queue_size)
            
            # Determine batch size for concurrent execution
            available_workers = self.max_workers - current_load
            batch_size = min(available_workers, len(scheduled_tasks), self.max_concurrent_tasks - current_load)
            
            # Execute tasks in batches with quantum coherence
            futures = {}
            tasks_to_execute = scheduled_tasks[:batch_size]
            
            for task in tasks_to_execute:
                if task.task_id not in self.running_tasks:
                    # Collapse wavefunction before execution
                    task.collapse_wavefunction()
                    
                    # Choose execution strategy based on task properties
                    executor = self._select_optimal_executor(task)
                    
                    # Submit for execution
                    future = executor.submit(self._execute_task, task)
                    futures[future] = task
                    self.running_tasks[task.task_id] = future
                    
                    statistics['total_tasks'] += 1
            
            # Process completed tasks
            if futures:
                # Use timeout to prevent blocking on long-running tasks
                completed_futures = []
                
                for future in as_completed(futures, timeout=1.0):
                    completed_futures.append(future)
                    task = futures[future]
                    
                    try:
                        result = future.result(timeout=0.1)
                        task.result = result
                        task.state = "executed"
                        results[task.task_id] = result
                        statistics['successful_tasks'] += 1
                        
                        # Update execution stats
                        self.execution_stats['tasks_processed'] += 1
                        if task.execution_time:
                            self.execution_stats['total_execution_time'] += task.execution_time
                        
                        logger.debug(f"Quantum task completed: {task.task_id}")
                        
                    except Exception as e:
                        task.state = "failed"
                        results[task.task_id] = str(e)
                        statistics['failed_tasks'] += 1
                        
                        logger.error(f"Quantum task failed: {task.task_id}, Error: {e}")
                    
                    finally:
                        # Remove from running tasks
                        if task.task_id in self.running_tasks:
                            del self.running_tasks[task.task_id]
                
                # Handle any remaining futures that didn't complete in time
                for future, task in futures.items():
                    if future not in [f for f in completed_futures]:
                        # Task is still running, will be handled in next iteration
                        pass
        
        # Compute final statistics
        total_time = time.time() - start_time
        statistics['total_execution_time'] = total_time
        statistics['max_concurrent_tasks'] = max_concurrent_reached
        statistics['scaling_events'] = self.execution_stats['worker_scaling_events']
        statistics['clusters_processed'] = self.execution_stats['cluster_rebalance_count']
        
        if statistics['total_tasks'] > 0:
            statistics['quantum_efficiency'] = statistics['successful_tasks'] / statistics['total_tasks']
            
            # Calculate average execution efficiency
            if self.execution_stats['tasks_processed'] > 0:
                avg_task_time = self.execution_stats['total_execution_time'] / self.execution_stats['tasks_processed']
                self.execution_stats['quantum_efficiency'] = min(1.0, 1.0 / (1.0 + avg_task_time))
        
        # Update global quantum state
        self._update_quantum_state()
        
        # Add enhanced statistics
        enhanced_stats = statistics.copy()
        enhanced_stats.update({
            'performance_metrics': self.execution_stats,
            'adaptive_scaling_enabled': self.adaptive_scaling,
            'clustering_enabled': self.enable_clustering,
            'max_workers_used': self.max_workers
        })
        
        return {'results': results, 'statistics': enhanced_stats}
    
    def _select_optimal_executor(self, task: QuantumTask) -> Any:
        """
        Select optimal executor for task based on characteristics.
        
        Uses quantum-inspired decision making for executor selection.
        """
        
        if not self.adaptive_scaling or self.adaptive_worker_pool is None:
            return self.executor
        
        # Analyze task characteristics
        is_cpu_intensive = task.energy_requirement > 5.0
        is_io_bound = task.energy_requirement < 1.0
        has_high_priority = task.priority.value <= 1
        
        # Quantum decision matrix
        thread_affinity = 0.5
        process_affinity = 0.5
        
        # Adjust based on task properties
        if is_cpu_intensive:
            process_affinity += 0.3
        if is_io_bound:
            thread_affinity += 0.3
        if has_high_priority:
            # High priority tasks prefer dedicated resources
            process_affinity += 0.2
        
        # Normalize probabilities
        total_affinity = thread_affinity + process_affinity
        thread_prob = thread_affinity / total_affinity
        
        # Make quantum decision
        if np.random.random() < thread_prob:
            return self.adaptive_worker_pool['thread_pool']
        else:
            return self.adaptive_worker_pool.get('process_pool', self.executor)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status for monitoring."""
        
        ready_tasks = self.task_graph.get_ready_tasks()
        
        status = {
            'quantum_state': {
                'global_phase': self.global_phase,
                'entanglement_entropy': self.entanglement_entropy,
                'coherent_tasks': len([t for t in ready_tasks if t.is_coherent()])
            },
            'execution_state': {
                'running_tasks': len(self.running_tasks),
                'ready_tasks': len(ready_tasks),
                'max_workers': self.max_workers,
                'total_tasks_in_graph': self.task_graph.graph.number_of_nodes()
            },
            'performance_metrics': self.execution_stats.copy(),
            'optimization_features': {
                'adaptive_scaling': self.adaptive_scaling,
                'clustering_enabled': self.enable_clustering,
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'cluster_size_limit': self.cluster_size_limit if self.enable_clustering else None
            }
        }
        
        # Add cluster information if enabled
        if self.enable_clustering and self.task_clusters:
            status['clustering_info'] = {
                'active_clusters': len(self.task_clusters),
                'cluster_sizes': [len(tasks) for tasks in self.task_clusters.values()],
                'average_cluster_size': np.mean([len(tasks) for tasks in self.task_clusters.values()]) if self.task_clusters else 0
            }
        
        return status
    
    def _execute_task(self, task: QuantumTask) -> Any:
        """Execute individual task with timing."""
        execution_start = time.time()
        
        try:
            # Apply physics-informed optimization if available
            if self.physics_operator and hasattr(task, 'physics_constraints'):
                result = self._physics_informed_execution(task)
            else:
                result = task.operation(*task.args, **task.kwargs)
            
            task.execution_time = time.time() - execution_start
            return result
            
        except Exception as e:
            task.execution_time = time.time() - execution_start
            raise e
    
    def _physics_informed_execution(self, task: QuantumTask) -> Any:
        """Execute task with physics-informed optimizations."""
        # Could apply conservation laws, symmetries, etc.
        # For now, just execute normally
        return task.operation(*task.args, **task.kwargs)
    
    def _update_quantum_state(self) -> None:
        """Update global quantum state of scheduler."""
        # Update global phase
        self.global_phase = (self.global_phase + np.pi/4) % (2 * np.pi)
        
        # Compute entanglement entropy
        n_entangled_pairs = len(self.task_graph.entangled_pairs)
        if n_entangled_pairs > 0:
            coupling_strengths = list(self.task_graph.entangled_pairs.values())
            self.entanglement_entropy = -sum(
                c * np.log(c + 1e-10) for c in coupling_strengths if c > 0
            )
    
    def get_quantum_metrics(self) -> Dict[str, float]:
        """Get current quantum state metrics."""
        return {
            'global_phase': self.global_phase,
            'entanglement_entropy': self.entanglement_entropy,
            'coherent_tasks': len([
                t for t in self.task_graph.graph.nodes()
                if self.task_graph.graph.nodes[t]['task'].is_coherent()
            ]),
            'total_tasks': len(self.task_graph.graph.nodes())
        }
    
    def shutdown(self) -> None:
        """Shutdown quantum scheduler gracefully."""
        logger.info("Shutting down quantum scheduler...")
        self.executor.shutdown(wait=True)