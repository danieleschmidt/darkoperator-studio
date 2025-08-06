"""
Quantum-inspired optimization algorithms for high-performance task planning.

Implements quantum computing concepts like superposition, entanglement, and 
quantum annealing for massive scale optimization problems.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import itertools

from ..planning.quantum_scheduler import QuantumTask, TaskPriority
from .caching import AdaptiveCache


logger = logging.getLogger(__name__)


@dataclass
class QuantumOptimizationConfig:
    """Configuration for quantum optimization algorithms."""
    
    # Quantum annealing parameters
    annealing_steps: int = 1000
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_schedule: str = 'exponential'  # 'linear', 'exponential', 'logarithmic'
    
    # Quantum state parameters
    max_qubits: int = 20  # Maximum qubits for quantum state representation
    coherence_time: float = 100.0  # Quantum coherence time (arbitrary units)
    entanglement_strength: float = 1.0
    
    # Optimization parameters
    population_size: int = 100
    max_generations: int = 1000
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Performance parameters
    max_workers: int = mp.cpu_count()
    use_gpu_acceleration: bool = True
    batch_size: int = 32
    cache_size: int = 10000
    
    # Physics-inspired parameters
    hamiltonian_terms: List[str] = field(default_factory=lambda: ['kinetic', 'potential', 'interaction'])
    energy_penalty_weight: float = 1.0
    conservation_penalty_weight: float = 10.0


class QuantumState:
    """Represents quantum state for optimization problems."""
    
    def __init__(self, n_qubits: int, device: torch.device = None):
        self.n_qubits = n_qubits
        self.device = device or torch.device('cpu')
        
        # Initialize in equal superposition
        self.amplitudes = torch.ones(2**n_qubits, dtype=torch.complex64, device=self.device)
        self.amplitudes /= torch.norm(self.amplitudes)
        
        # Entanglement matrix
        self.entanglement_matrix = torch.eye(n_qubits, device=self.device)
        
    def measure(self) -> int:
        """Measure quantum state, collapsing to classical state."""
        probabilities = torch.abs(self.amplitudes) ** 2
        probabilities = probabilities.cpu().numpy()
        
        # Quantum measurement
        measured_state = np.random.choice(len(probabilities), p=probabilities)
        
        # Collapse wavefunction
        self.amplitudes = torch.zeros_like(self.amplitudes)
        self.amplitudes[measured_state] = 1.0
        
        return measured_state
    
    def apply_rotation(self, qubit_index: int, theta: float, phi: float = 0.0):
        """Apply rotation gate to specific qubit."""
        if qubit_index >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_index} exceeds {self.n_qubits} qubits")
        
        # Pauli rotation matrices
        cos_half = torch.cos(torch.tensor(theta / 2, device=self.device))
        sin_half = torch.sin(torch.tensor(theta / 2, device=self.device))
        exp_phi = torch.exp(1j * torch.tensor(phi, device=self.device))
        
        # Build rotation operator for full state space
        n_states = 2 ** self.n_qubits
        rotation_op = torch.eye(n_states, dtype=torch.complex64, device=self.device)
        
        # Apply rotation to relevant amplitudes
        for state in range(n_states):
            if (state >> qubit_index) & 1:  # Qubit is in |1⟩ state
                flipped_state = state ^ (1 << qubit_index)  # Flip the qubit
                
                # Apply rotation
                amp_0 = self.amplitudes[flipped_state]
                amp_1 = self.amplitudes[state]
                
                new_amp_0 = cos_half * amp_0 - 1j * sin_half * exp_phi * amp_1
                new_amp_1 = cos_half * amp_1 - 1j * sin_half * exp_phi.conj() * amp_0
                
                self.amplitudes[flipped_state] = new_amp_0
                self.amplitudes[state] = new_amp_1
    
    def apply_entanglement(self, qubit1: int, qubit2: int, strength: float = 1.0):
        """Create entanglement between two qubits."""
        if max(qubit1, qubit2) >= self.n_qubits:
            raise ValueError("Qubit indices exceed system size")
        
        # Update entanglement matrix
        self.entanglement_matrix[qubit1, qubit2] = strength
        self.entanglement_matrix[qubit2, qubit1] = strength
        
        # Apply controlled-phase gate
        n_states = 2 ** self.n_qubits
        for state in range(n_states):
            bit1 = (state >> qubit1) & 1
            bit2 = (state >> qubit2) & 1
            
            if bit1 == 1 and bit2 == 1:
                phase_factor = torch.exp(1j * torch.tensor(strength, device=self.device))
                self.amplitudes[state] *= phase_factor
    
    def get_expectation_value(self, observable: torch.Tensor) -> float:
        """Compute expectation value of observable."""
        expectation = torch.real(torch.vdot(self.amplitudes, observable @ self.amplitudes))
        return expectation.item()
    
    def get_entropy(self) -> float:
        """Compute von Neumann entropy of quantum state."""
        probabilities = torch.abs(self.amplitudes) ** 2
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
        return entropy.item()


class QuantumAnnealer:
    """Quantum annealing optimizer for combinatorial problems."""
    
    def __init__(self, config: QuantumOptimizationConfig):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu_acceleration and torch.cuda.is_available() else 'cpu')
        self.cache = AdaptiveCache(max_size=config.cache_size)
        
        logger.info(f"Initialized quantum annealer on device: {self.device}")
    
    def optimize_task_schedule(
        self, 
        tasks: List[QuantumTask],
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize task schedule using quantum annealing.
        
        Args:
            tasks: List of quantum tasks to schedule
            constraints: Optimization constraints
            
        Returns:
            Optimization results with schedule and metadata
        """
        if not tasks:
            return {'optimal_schedule': [], 'energy': 0.0, 'optimization_time': 0.0}
        
        start_time = time.time()
        
        # Encode problem as QUBO (Quadratic Unconstrained Binary Optimization)
        qubo_matrix = self._create_qubo_matrix(tasks, constraints)
        
        # Initialize quantum state
        n_qubits = min(len(tasks), self.config.max_qubits)
        quantum_state = QuantumState(n_qubits, self.device)
        
        # Quantum annealing process
        best_schedule, best_energy = self._quantum_anneal(quantum_state, qubo_matrix, tasks)
        
        optimization_time = time.time() - start_time
        
        results = {
            'optimal_schedule': best_schedule,
            'energy': best_energy,
            'optimization_time': optimization_time,
            'quantum_entropy': quantum_state.get_entropy(),
            'annealing_steps': self.config.annealing_steps,
            'convergence_info': {
                'temperature_schedule': self._get_temperature_schedule(),
                'final_state_probabilities': torch.abs(quantum_state.amplitudes) ** 2
            }
        }
        
        logger.info(f"Quantum annealing completed: {len(tasks)} tasks, "
                   f"energy={best_energy:.6f}, time={optimization_time:.3f}s")
        
        return results
    
    def _create_qubo_matrix(
        self, 
        tasks: List[QuantumTask], 
        constraints: Dict[str, Any] = None
    ) -> torch.Tensor:
        """Create QUBO matrix for task scheduling problem."""
        n_tasks = len(tasks)
        n_positions = min(n_tasks, self.config.max_qubits)  # Time positions
        
        # QUBO matrix size: n_tasks * n_positions (each task can be at each position)
        qubo_size = n_tasks * n_positions
        Q = torch.zeros(qubo_size, qubo_size, device=self.device)
        
        # Objective: minimize total execution time + energy
        for i, task in enumerate(tasks):
            for t in range(n_positions):
                idx = i * n_positions + t
                
                # Time penalty (later = higher cost)
                time_penalty = t * 0.1
                
                # Energy penalty
                energy_penalty = task.energy_requirement * self.config.energy_penalty_weight
                
                # Diagonal term (linear cost)
                Q[idx, idx] = time_penalty + energy_penalty
        
        # Constraints: each task assigned to exactly one time slot
        for i in range(n_tasks):
            for t1 in range(n_positions):
                for t2 in range(t1 + 1, n_positions):
                    idx1 = i * n_positions + t1
                    idx2 = i * n_positions + t2
                    
                    # Penalty for assigning task to multiple slots
                    Q[idx1, idx2] += 100.0  # Large penalty
                    Q[idx2, idx1] += 100.0
        
        # Dependency constraints
        for i, task in enumerate(tasks):
            for dep_task_id in task.entangled_tasks:
                # Find dependency task index
                dep_idx = None
                for j, other_task in enumerate(tasks):
                    if other_task.task_id == dep_task_id:
                        dep_idx = j
                        break
                
                if dep_idx is not None:
                    # Dependency constraint: dep_task must come before task
                    for t1 in range(n_positions):
                        for t2 in range(t1):  # t2 < t1
                            task_idx = i * n_positions + t1
                            dep_task_idx = dep_idx * n_positions + t2
                            
                            # Reward for correct dependency order
                            Q[task_idx, dep_task_idx] -= 50.0
                            Q[dep_task_idx, task_idx] -= 50.0
        
        return Q
    
    def _quantum_anneal(
        self, 
        quantum_state: QuantumState, 
        qubo_matrix: torch.Tensor,
        tasks: List[QuantumTask]
    ) -> Tuple[List[str], float]:
        """Perform quantum annealing optimization."""
        
        best_energy = float('inf')
        best_schedule = []
        
        # Temperature schedule
        temperatures = self._get_temperature_schedule()
        
        for step, temperature in enumerate(temperatures):
            # Quantum evolution step
            self._quantum_evolution_step(quantum_state, qubo_matrix, temperature)
            
            # Periodic measurements and updates
            if step % 100 == 0 or step == len(temperatures) - 1:
                # Measure current state
                measured_state = quantum_state.measure()
                schedule, energy = self._decode_solution(measured_state, tasks, qubo_matrix)
                
                if energy < best_energy:
                    best_energy = energy
                    best_schedule = schedule
                
                # Reinitialize state for continued annealing
                if step < len(temperatures) - 1:
                    quantum_state = QuantumState(quantum_state.n_qubits, self.device)
                    # Add bias toward current best solution
                    self._bias_toward_solution(quantum_state, best_schedule, tasks)
        
        return best_schedule, best_energy
    
    def _quantum_evolution_step(
        self, 
        quantum_state: QuantumState, 
        qubo_matrix: torch.Tensor,
        temperature: float
    ):
        """Single step of quantum evolution during annealing."""
        
        # Apply random quantum gates (simulated thermal fluctuations)
        for _ in range(5):  # Multiple gates per step
            qubit = np.random.randint(quantum_state.n_qubits)
            
            # Rotation angle based on temperature
            theta = np.random.normal(0, temperature * 0.1)
            phi = np.random.uniform(0, 2 * np.pi)
            
            quantum_state.apply_rotation(qubit, theta, phi)
        
        # Apply entanglement based on QUBO interactions
        if temperature > 1.0:  # High temperature = more entanglement
            for _ in range(2):
                qubit1, qubit2 = np.random.choice(quantum_state.n_qubits, 2, replace=False)
                strength = np.random.exponential(temperature * 0.1)
                quantum_state.apply_entanglement(qubit1, qubit2, strength)
    
    def _get_temperature_schedule(self) -> np.ndarray:
        """Generate temperature schedule for annealing."""
        steps = self.config.annealing_steps
        T_initial = self.config.initial_temperature
        T_final = self.config.final_temperature
        
        if self.config.cooling_schedule == 'exponential':
            alpha = (T_final / T_initial) ** (1.0 / steps)
            temperatures = T_initial * (alpha ** np.arange(steps))
        elif self.config.cooling_schedule == 'linear':
            temperatures = np.linspace(T_initial, T_final, steps)
        elif self.config.cooling_schedule == 'logarithmic':
            temperatures = T_initial / (1 + np.log(1 + np.arange(steps)))
        else:
            temperatures = np.linspace(T_initial, T_final, steps)
        
        return temperatures
    
    def _decode_solution(
        self, 
        measured_state: int, 
        tasks: List[QuantumTask],
        qubo_matrix: torch.Tensor
    ) -> Tuple[List[str], float]:
        """Decode quantum measurement into task schedule."""
        
        n_tasks = len(tasks)
        n_positions = min(n_tasks, self.config.max_qubits)
        
        # Convert measured state to binary assignment
        binary_solution = []
        for i in range(qubo_matrix.shape[0]):
            bit = (measured_state >> i) & 1
            binary_solution.append(bit)
        
        # Decode schedule from binary solution
        schedule = [None] * n_positions
        task_assigned = set()
        
        for i, task in enumerate(tasks):
            for t in range(n_positions):
                idx = i * n_positions + t
                if idx < len(binary_solution) and binary_solution[idx] == 1:
                    if schedule[t] is None and task.task_id not in task_assigned:
                        schedule[t] = task.task_id
                        task_assigned.add(task.task_id)
                        break
        
        # Fill empty slots and handle unassigned tasks
        remaining_tasks = [task.task_id for task in tasks if task.task_id not in task_assigned]
        for t in range(n_positions):
            if schedule[t] is None and remaining_tasks:
                schedule[t] = remaining_tasks.pop(0)
        
        # Remove None values and add remaining tasks
        schedule = [task_id for task_id in schedule if task_id is not None]
        schedule.extend(remaining_tasks)
        
        # Compute solution energy
        energy = self._compute_solution_energy(schedule, tasks, qubo_matrix)
        
        return schedule, energy
    
    def _compute_solution_energy(
        self, 
        schedule: List[str], 
        tasks: List[QuantumTask],
        qubo_matrix: torch.Tensor
    ) -> float:
        """Compute energy of a solution."""
        
        # Create binary vector from schedule
        task_to_idx = {task.task_id: i for i, task in enumerate(tasks)}
        n_positions = min(len(tasks), self.config.max_qubits)
        
        x = torch.zeros(len(tasks) * n_positions, device=self.device)
        
        for t, task_id in enumerate(schedule[:n_positions]):
            if task_id in task_to_idx:
                i = task_to_idx[task_id]
                idx = i * n_positions + t
                if idx < len(x):
                    x[idx] = 1.0
        
        # Energy = x^T Q x
        energy = torch.dot(x, qubo_matrix @ x).item()
        return energy
    
    def _bias_toward_solution(
        self, 
        quantum_state: QuantumState, 
        solution: List[str],
        tasks: List[QuantumTask]
    ):
        """Bias quantum state toward a known good solution."""
        
        # This would implement a more sophisticated biasing mechanism
        # For now, apply small rotations toward the solution state
        task_to_idx = {task.task_id: i for i, task in enumerate(tasks)}
        
        for t, task_id in enumerate(solution[:quantum_state.n_qubits]):
            if task_id in task_to_idx and t < quantum_state.n_qubits:
                # Small rotation toward |1⟩ for this qubit
                quantum_state.apply_rotation(t, np.pi/8)


class ParallelQuantumOptimizer:
    """Parallel quantum optimization using multiple quantum annealers."""
    
    def __init__(self, config: QuantumOptimizationConfig):
        self.config = config
        self.n_workers = config.max_workers
        
    def optimize_parallel(
        self, 
        tasks: List[QuantumTask],
        constraints: Dict[str, Any] = None,
        n_runs: int = None
    ) -> Dict[str, Any]:
        """
        Run multiple quantum annealing instances in parallel.
        
        Args:
            tasks: Tasks to optimize
            constraints: Optimization constraints
            n_runs: Number of parallel runs (defaults to n_workers)
            
        Returns:
            Best optimization result across all runs
        """
        
        if n_runs is None:
            n_runs = self.n_workers
        
        start_time = time.time()
        
        # Create multiple annealer instances
        optimize_func = partial(
            self._single_optimization_run,
            tasks=tasks,
            constraints=constraints
        )
        
        best_result = None
        best_energy = float('inf')
        
        # Use ThreadPoolExecutor for I/O bound quantum simulation
        # Use ProcessPoolExecutor for CPU-intensive tasks
        executor_class = ThreadPoolExecutor if len(tasks) < 100 else ProcessPoolExecutor
        
        with executor_class(max_workers=self.n_workers) as executor:
            # Submit all optimization runs
            future_to_run = {
                executor.submit(optimize_func, run_id): run_id 
                for run_id in range(n_runs)
            }
            
            # Collect results as they complete
            completed_runs = 0
            for future in as_completed(future_to_run):
                run_id = future_to_run[future]
                
                try:
                    result = future.result()
                    completed_runs += 1
                    
                    # Track best result
                    if result['energy'] < best_energy:
                        best_energy = result['energy']
                        best_result = result
                        best_result['best_run_id'] = run_id
                    
                    logger.debug(f"Completed quantum run {run_id}: energy={result['energy']:.6f}")
                    
                except Exception as e:
                    logger.error(f"Quantum optimization run {run_id} failed: {e}")
        
        total_time = time.time() - start_time
        
        if best_result is None:
            # Fallback result if all runs failed
            best_result = {
                'optimal_schedule': [task.task_id for task in tasks],
                'energy': float('inf'),
                'optimization_time': total_time
            }
        
        # Add parallel execution metadata
        best_result.update({
            'parallel_runs': completed_runs,
            'total_optimization_time': total_time,
            'speedup_factor': completed_runs,  # Theoretical speedup
            'runs_completed': completed_runs,
            'runs_requested': n_runs
        })
        
        logger.info(f"Parallel quantum optimization completed: {completed_runs} runs, "
                   f"best energy={best_energy:.6f}, total time={total_time:.3f}s")
        
        return best_result
    
    def _single_optimization_run(
        self, 
        run_id: int,
        tasks: List[QuantumTask],
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Single optimization run for parallel execution."""
        
        # Create annealer with slight parameter variations for diversity
        config = self._create_varied_config(run_id)
        annealer = QuantumAnnealer(config)
        
        try:
            result = annealer.optimize_task_schedule(tasks, constraints)
            result['run_id'] = run_id
            return result
        except Exception as e:
            logger.error(f"Quantum annealing run {run_id} failed: {e}")
            raise
    
    def _create_varied_config(self, run_id: int) -> QuantumOptimizationConfig:
        """Create slightly varied configuration for diversity."""
        
        config = QuantumOptimizationConfig(
            annealing_steps=self.config.annealing_steps,
            initial_temperature=self.config.initial_temperature * (0.8 + 0.4 * np.random.random()),
            final_temperature=self.config.final_temperature * (0.5 + np.random.random()),
            cooling_schedule=np.random.choice(['exponential', 'linear', 'logarithmic']),
            max_qubits=self.config.max_qubits,
            coherence_time=self.config.coherence_time * (0.8 + 0.4 * np.random.random()),
            entanglement_strength=self.config.entanglement_strength * (0.5 + np.random.random()),
            population_size=self.config.population_size,
            max_generations=self.config.max_generations,
            mutation_rate=self.config.mutation_rate * (0.5 + np.random.random()),
            crossover_rate=self.config.crossover_rate,
            max_workers=1,  # Single worker per annealer
            use_gpu_acceleration=self.config.use_gpu_acceleration,
            batch_size=self.config.batch_size,
            cache_size=self.config.cache_size // self.config.max_workers,  # Distribute cache
            hamiltonian_terms=self.config.hamiltonian_terms,
            energy_penalty_weight=self.config.energy_penalty_weight * (0.8 + 0.4 * np.random.random()),
            conservation_penalty_weight=self.config.conservation_penalty_weight
        )
        
        return config


class QuantumGeneticOptimizer:
    """Hybrid quantum-genetic algorithm for large-scale optimization."""
    
    def __init__(self, config: QuantumOptimizationConfig):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu_acceleration and torch.cuda.is_available() else 'cpu')
        
        # Population of quantum states
        self.population = []
        self.fitness_cache = {}
        
    def optimize_genetic_quantum(
        self,
        tasks: List[QuantumTask],
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize using hybrid quantum-genetic algorithm.
        
        Args:
            tasks: Tasks to optimize  
            constraints: Constraints
            
        Returns:
            Optimization results
        """
        
        start_time = time.time()
        
        # Initialize population of quantum states
        self._initialize_population(len(tasks))
        
        best_fitness = float('inf')
        best_solution = None
        fitness_history = []
        
        for generation in range(self.config.max_generations):
            # Evaluate population fitness
            fitness_scores = self._evaluate_population_fitness(tasks, constraints)
            
            # Track best solution
            gen_best_idx = np.argmin(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_solution = self._decode_quantum_state(
                    self.population[gen_best_idx], tasks
                )
            
            fitness_history.append(gen_best_fitness)
            
            # Early stopping
            if generation > 50 and np.std(fitness_history[-20:]) < 1e-6:
                logger.info(f"Early convergence at generation {generation}")
                break
            
            # Selection, crossover, mutation
            self._evolve_population(fitness_scores)
            
            if generation % 100 == 0:
                logger.debug(f"Generation {generation}: best fitness = {gen_best_fitness:.6f}")
        
        optimization_time = time.time() - start_time
        
        results = {
            'optimal_schedule': best_solution,
            'energy': best_fitness,
            'optimization_time': optimization_time,
            'generations': len(fitness_history),
            'fitness_history': fitness_history,
            'convergence_generation': len(fitness_history),
            'final_population_diversity': self._compute_population_diversity()
        }
        
        logger.info(f"Quantum-genetic optimization completed: {len(fitness_history)} generations, "
                   f"best fitness={best_fitness:.6f}, time={optimization_time:.3f}s")
        
        return results
    
    def _initialize_population(self, n_tasks: int):
        """Initialize population of quantum states."""
        n_qubits = min(n_tasks, self.config.max_qubits)
        
        self.population = []
        for _ in range(self.config.population_size):
            quantum_state = QuantumState(n_qubits, self.device)
            
            # Apply random initialization
            for qubit in range(n_qubits):
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                quantum_state.apply_rotation(qubit, theta, phi)
            
            # Add random entanglement
            for _ in range(n_qubits // 2):
                q1, q2 = np.random.choice(n_qubits, 2, replace=False)
                strength = np.random.exponential(self.config.entanglement_strength)
                quantum_state.apply_entanglement(q1, q2, strength)
            
            self.population.append(quantum_state)
    
    def _evaluate_population_fitness(
        self, 
        tasks: List[QuantumTask],
        constraints: Dict[str, Any] = None
    ) -> List[float]:
        """Evaluate fitness of all quantum states in population."""
        
        fitness_scores = []
        
        for i, quantum_state in enumerate(self.population):
            # Use caching to avoid re-computation
            state_hash = self._hash_quantum_state(quantum_state)
            
            if state_hash in self.fitness_cache:
                fitness = self.fitness_cache[state_hash]
            else:
                # Measure quantum state and decode solution
                measured_state = quantum_state.measure()
                solution = self._decode_quantum_measurement(measured_state, tasks)
                
                # Compute fitness (lower is better)
                fitness = self._compute_solution_fitness(solution, tasks, constraints)
                self.fitness_cache[state_hash] = fitness
            
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _evolve_population(self, fitness_scores: List[float]):
        """Evolve population through selection, crossover, and mutation."""
        
        # Selection: tournament selection
        new_population = []
        
        while len(new_population) < self.config.population_size:
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                child = self._quantum_crossover(parent1, parent2)
            else:
                child = self._tournament_selection(fitness_scores)
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                self._quantum_mutation(child)
            
            new_population.append(child)
        
        self.population = new_population
    
    def _tournament_selection(self, fitness_scores: List[float]) -> QuantumState:
        """Tournament selection for quantum states."""
        tournament_size = min(5, len(self.population))
        tournament_indices = np.random.choice(
            len(self.population), tournament_size, replace=False
        )
        
        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
        
        # Return a copy of the quantum state
        selected = self.population[best_idx]
        new_state = QuantumState(selected.n_qubits, self.device)
        new_state.amplitudes = selected.amplitudes.clone()
        new_state.entanglement_matrix = selected.entanglement_matrix.clone()
        
        return new_state
    
    def _quantum_crossover(self, parent1: QuantumState, parent2: QuantumState) -> QuantumState:
        """Quantum crossover operation."""
        # Create child by mixing quantum amplitudes
        child = QuantumState(parent1.n_qubits, self.device)
        
        # Amplitude mixing
        mixing_ratio = np.random.random()
        child.amplitudes = (mixing_ratio * parent1.amplitudes + 
                           (1 - mixing_ratio) * parent2.amplitudes)
        child.amplitudes /= torch.norm(child.amplitudes)
        
        # Entanglement matrix mixing
        child.entanglement_matrix = (mixing_ratio * parent1.entanglement_matrix +
                                   (1 - mixing_ratio) * parent2.entanglement_matrix)
        
        return child
    
    def _quantum_mutation(self, quantum_state: QuantumState):
        """Quantum mutation operation."""
        n_mutations = np.random.poisson(1) + 1
        
        for _ in range(n_mutations):
            mutation_type = np.random.choice(['rotation', 'entanglement', 'decoherence'])
            
            if mutation_type == 'rotation':
                qubit = np.random.randint(quantum_state.n_qubits)
                theta = np.random.normal(0, np.pi/4)
                phi = np.random.uniform(0, 2*np.pi)
                quantum_state.apply_rotation(qubit, theta, phi)
            
            elif mutation_type == 'entanglement':
                if quantum_state.n_qubits > 1:
                    q1, q2 = np.random.choice(quantum_state.n_qubits, 2, replace=False)
                    strength = np.random.normal(0, self.config.entanglement_strength)
                    quantum_state.apply_entanglement(q1, q2, strength)
            
            elif mutation_type == 'decoherence':
                # Add quantum decoherence (noise)
                noise_strength = 0.01
                noise = torch.randn_like(quantum_state.amplitudes) * noise_strength
                quantum_state.amplitudes += noise
                quantum_state.amplitudes /= torch.norm(quantum_state.amplitudes)
    
    def _decode_quantum_state(self, quantum_state: QuantumState, tasks: List[QuantumTask]) -> List[str]:
        """Decode quantum state into task schedule."""
        measured_state = quantum_state.measure()
        return self._decode_quantum_measurement(measured_state, tasks)
    
    def _decode_quantum_measurement(self, measured_state: int, tasks: List[QuantumTask]) -> List[str]:
        """Decode quantum measurement into task schedule."""
        n_tasks = len(tasks)
        
        # Simple decoding: use measured state as permutation seed
        np.random.seed(measured_state % 2**16)  # Limit seed size
        permutation = np.random.permutation(n_tasks)
        
        return [tasks[i].task_id for i in permutation]
    
    def _compute_solution_fitness(
        self,
        solution: List[str], 
        tasks: List[QuantumTask],
        constraints: Dict[str, Any] = None
    ) -> float:
        """Compute fitness of a solution (lower is better)."""
        
        task_dict = {task.task_id: task for task in tasks}
        
        # Base fitness: total energy and time
        total_energy = 0.0
        total_time = 0.0
        
        for i, task_id in enumerate(solution):
            if task_id in task_dict:
                task = task_dict[task_id]
                total_energy += task.energy_requirement
                total_time += i * 0.1  # Position penalty
        
        fitness = total_energy + total_time
        
        # Dependency constraint penalties
        dependency_penalty = 0.0
        for i, task_id in enumerate(solution):
            if task_id in task_dict:
                task = task_dict[task_id]
                for dep_id in task.entangled_tasks:
                    if dep_id in solution:
                        dep_pos = solution.index(dep_id)
                        if dep_pos > i:  # Dependency violation
                            dependency_penalty += 100.0
        
        # Physics constraint penalties
        physics_penalty = 0.0
        if constraints:
            max_energy = constraints.get('max_energy', float('inf'))
            if total_energy > max_energy:
                physics_penalty += (total_energy - max_energy) * 10.0
        
        return fitness + dependency_penalty + physics_penalty
    
    def _hash_quantum_state(self, quantum_state: QuantumState) -> str:
        """Create hash of quantum state for caching."""
        # Simple hash based on amplitudes (rounded for numerical stability)
        rounded_amps = torch.round(quantum_state.amplitudes * 1000) / 1000
        hash_string = str(rounded_amps.cpu().numpy().tobytes())
        return str(hash(hash_string))
    
    def _compute_population_diversity(self) -> float:
        """Compute diversity measure for population."""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Compute quantum fidelity as distance measure
                state1 = self.population[i]
                state2 = self.population[j]
                
                fidelity = torch.abs(torch.vdot(state1.amplitudes, state2.amplitudes)) ** 2
                distance = 1.0 - fidelity.item()
                
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0