"""
Advanced Benchmarking System for DarkOperator Studio.

Provides comprehensive performance analysis, physics validation,
and scalability testing with statistical analysis and reporting.
"""

import time
import numpy as np
import logging
import json
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import statistics
import concurrent.futures
from abc import ABC, abstractmethod
import psutil
import gc

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class BenchmarkType(Enum):
    """Types of benchmarks available."""
    PERFORMANCE = "performance"
    PHYSICS = "physics"
    SCALABILITY = "scalability"
    ACCURACY = "accuracy"
    QUANTUM = "quantum"
    DISTRIBUTED = "distributed"
    MODEL = "model"
    MEMORY = "memory"


class BenchmarkStatus(Enum):
    """Benchmark execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark execution."""
    
    benchmark_name: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus
    
    # Performance metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Physics metrics
    physics_score: float = 0.0
    conservation_violation: float = 0.0
    symmetry_preservation: float = 0.0
    
    # Statistical metrics
    mean_value: float = 0.0
    std_deviation: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    percentile_95: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    start_time: float = 0.0
    end_time: float = 0.0
    error_message: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'benchmark_name': self.benchmark_name,
            'benchmark_type': self.benchmark_type.value,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'gpu_usage_percent': self.gpu_usage_percent,
            'physics_score': self.physics_score,
            'conservation_violation': self.conservation_violation,
            'symmetry_preservation': self.symmetry_preservation,
            'mean_value': self.mean_value,
            'std_deviation': self.std_deviation,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'percentile_95': self.percentile_95,
            'custom_metrics': self.custom_metrics,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'error_message': self.error_message,
            'config': self.config
        }


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, name: str, benchmark_type: BenchmarkType, config: Dict[str, Any] = None):
        self.name = name
        self.benchmark_type = benchmark_type
        self.config = config or {}
        self.result = BenchmarkResult(name, benchmark_type)
        
    @abstractmethod
    def setup(self) -> None:
        """Setup benchmark environment and data."""
        pass
    
    @abstractmethod
    def run_benchmark(self) -> Any:
        """Execute the benchmark and return results."""
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Clean up benchmark resources."""
        pass
    
    def validate_results(self, results: Any) -> bool:
        """Validate benchmark results."""
        return results is not None
    
    def compute_statistics(self, data: List[float]) -> None:
        """Compute statistical metrics from data."""
        if not data:
            return
        
        self.result.mean_value = statistics.mean(data)
        self.result.std_deviation = statistics.stdev(data) if len(data) > 1 else 0.0
        self.result.min_value = min(data)
        self.result.max_value = max(data)
        self.result.percentile_95 = np.percentile(data, 95)
    
    def execute(self, num_runs: int = 1) -> BenchmarkResult:
        """Execute benchmark with performance monitoring."""
        
        self.result.status = BenchmarkStatus.RUNNING
        self.result.start_time = time.time()
        self.result.config = self.config.copy()
        
        try:
            # Setup
            self.setup()
            
            # Monitor system resources
            process = psutil.Process()
            
            # Run benchmark multiple times
            execution_times = []
            memory_usages = []
            results_data = []
            
            for run in range(num_runs):
                # Clear memory before each run
                gc.collect()
                
                # Measure memory before
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Execute benchmark
                run_start = time.time()
                result = self.run_benchmark()
                run_end = time.time()
                
                # Measure memory after
                mem_after = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Record metrics
                execution_times.append(run_end - run_start)
                memory_usages.append(mem_after - mem_before)
                results_data.append(result)
                
                # Validate results
                if not self.validate_results(result):
                    raise ValueError(f"Benchmark validation failed on run {run + 1}")
            
            # Compute aggregate statistics
            self.compute_statistics(execution_times)
            self.result.execution_time = statistics.mean(execution_times)
            self.result.memory_usage_mb = statistics.mean(memory_usages)
            
            # CPU usage (approximate)
            self.result.cpu_usage_percent = process.cpu_percent()
            
            # GPU usage (if available)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.result.gpu_usage_percent = self._get_gpu_usage()
            
            # Teardown
            self.teardown()
            
            self.result.status = BenchmarkStatus.COMPLETED
            self.result.end_time = time.time()
            
        except Exception as e:
            self.result.status = BenchmarkStatus.FAILED
            self.result.error_message = str(e)
            self.result.end_time = time.time()
            
            # Log full traceback
            logger.error(f"Benchmark {self.name} failed: {e}")
            logger.error(traceback.format_exc())
            
            try:
                self.teardown()
            except Exception as teardown_error:
                logger.error(f"Teardown failed: {teardown_error}")
        
        return self.result
    
    def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        try:
            # Simplified GPU usage estimation
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                return gpu_memory * 100.0
            return 0.0
        except Exception:
            return 0.0


class BenchmarkSuite:
    """Collection of benchmarks to run together."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.benchmarks: List[BaseBenchmark] = []
        self.results: List[BenchmarkResult] = []
        
    def add_benchmark(self, benchmark: BaseBenchmark) -> None:
        """Add benchmark to suite."""
        self.benchmarks.append(benchmark)
    
    def run_all(self, num_runs: int = 1, parallel: bool = False) -> List[BenchmarkResult]:
        """Run all benchmarks in suite."""
        
        logger.info(f"Starting benchmark suite: {self.name} ({len(self.benchmarks)} benchmarks)")
        
        if parallel:
            return self._run_parallel(num_runs)
        else:
            return self._run_sequential(num_runs)
    
    def _run_sequential(self, num_runs: int) -> List[BenchmarkResult]:
        """Run benchmarks sequentially."""
        
        results = []
        for i, benchmark in enumerate(self.benchmarks):
            logger.info(f"Running benchmark {i+1}/{len(self.benchmarks)}: {benchmark.name}")
            
            result = benchmark.execute(num_runs)
            results.append(result)
            
            logger.info(f"Completed {benchmark.name}: {result.status.value}")
        
        self.results = results
        return results
    
    def _run_parallel(self, num_runs: int) -> List[BenchmarkResult]:
        """Run benchmarks in parallel."""
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_benchmark = {
                executor.submit(benchmark.execute, num_runs): benchmark 
                for benchmark in self.benchmarks
            }
            
            for future in concurrent.futures.as_completed(future_to_benchmark):
                benchmark = future_to_benchmark[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {benchmark.name}: {result.status.value}")
                except Exception as e:
                    logger.error(f"Benchmark {benchmark.name} failed: {e}")
        
        # Sort results by benchmark order
        benchmark_names = [b.name for b in self.benchmarks]
        results.sort(key=lambda r: benchmark_names.index(r.benchmark_name))
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark suite summary."""
        
        if not self.results:
            return {'error': 'No results available'}
        
        total_benchmarks = len(self.results)
        completed = sum(1 for r in self.results if r.status == BenchmarkStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == BenchmarkStatus.FAILED)
        
        # Aggregate metrics
        total_execution_time = sum(r.execution_time for r in self.results)
        avg_memory_usage = statistics.mean([r.memory_usage_mb for r in self.results if r.memory_usage_mb > 0])
        avg_physics_score = statistics.mean([r.physics_score for r in self.results if r.physics_score > 0])
        
        return {
            'suite_name': self.name,
            'total_benchmarks': total_benchmarks,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_benchmarks if total_benchmarks > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_memory_usage_mb': avg_memory_usage,
            'average_physics_score': avg_physics_score,
            'results': [r.to_dict() for r in self.results]
        }


class BenchmarkRunner:
    """
    Main benchmark runner with advanced analysis and reporting capabilities.
    
    Features:
    - Automated benchmark discovery and execution
    - Statistical analysis and performance profiling
    - Physics validation and correctness checking
    - Scalability analysis and regression detection
    - Comprehensive reporting and visualization
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.suites: Dict[str, BenchmarkSuite] = {}
        self.global_results: List[BenchmarkResult] = []
        
        # Analysis tools
        self.statistical_analyzer = StatisticalAnalyzer()
        self.regression_detector = RegressionDetector()
        self.performance_profiler = PerformanceProfiler()
        
        logger.info(f"BenchmarkRunner initialized, output: {self.output_dir}")
    
    def register_suite(self, suite: BenchmarkSuite) -> None:
        """Register benchmark suite."""
        self.suites[suite.name] = suite
        logger.info(f"Registered benchmark suite: {suite.name}")
    
    def run_suite(self, suite_name: str, num_runs: int = 1, parallel: bool = False) -> Dict[str, Any]:
        """Run specific benchmark suite."""
        
        if suite_name not in self.suites:
            raise ValueError(f"Unknown benchmark suite: {suite_name}")
        
        suite = self.suites[suite_name]
        results = suite.run_all(num_runs, parallel)
        
        # Add to global results
        self.global_results.extend(results)
        
        # Generate summary
        summary = suite.get_summary()
        
        # Save results
        self._save_results(suite_name, summary)
        
        # Generate report
        self._generate_report(suite_name, summary)
        
        return summary
    
    def run_all_suites(self, num_runs: int = 1, parallel: bool = False) -> Dict[str, Any]:
        """Run all registered benchmark suites."""
        
        all_summaries = {}
        
        for suite_name in self.suites:
            logger.info(f"Running benchmark suite: {suite_name}")
            summary = self.run_suite(suite_name, num_runs, parallel)
            all_summaries[suite_name] = summary
        
        # Generate comprehensive report
        comprehensive_summary = self._generate_comprehensive_summary(all_summaries)
        self._generate_comprehensive_report(comprehensive_summary)
        
        return comprehensive_summary
    
    def _save_results(self, suite_name: str, summary: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        
        results_file = self.output_dir / f"{suite_name}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved results: {results_file}")
    
    def _generate_report(self, suite_name: str, summary: Dict[str, Any]) -> None:
        """Generate detailed benchmark report."""
        
        report_file = self.output_dir / f"{suite_name}_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Benchmark Report: {suite_name}\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Benchmarks**: {summary['total_benchmarks']}\n")
            f.write(f"- **Completed**: {summary['completed']}\n")
            f.write(f"- **Failed**: {summary['failed']}\n")
            f.write(f"- **Success Rate**: {summary['success_rate']:.1%}\n")
            f.write(f"- **Total Execution Time**: {summary['total_execution_time']:.2f}s\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for result in summary['results']:
                f.write(f"### {result['benchmark_name']}\n\n")
                f.write(f"- **Type**: {result['benchmark_type']}\n")
                f.write(f"- **Status**: {result['status']}\n")
                f.write(f"- **Execution Time**: {result['execution_time']:.4f}s\n")
                f.write(f"- **Memory Usage**: {result['memory_usage_mb']:.2f} MB\n")
                
                if result['physics_score'] > 0:
                    f.write(f"- **Physics Score**: {result['physics_score']:.4f}\n")
                
                if result['custom_metrics']:
                    f.write("- **Custom Metrics**:\n")
                    for metric, value in result['custom_metrics'].items():
                        f.write(f"  - {metric}: {value}\n")
                
                if result['error_message']:
                    f.write(f"- **Error**: {result['error_message']}\n")
                
                f.write("\n")
        
        logger.info(f"Generated report: {report_file}")
    
    def _generate_comprehensive_summary(self, all_summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary across all suites."""
        
        total_benchmarks = sum(s['total_benchmarks'] for s in all_summaries.values())
        total_completed = sum(s['completed'] for s in all_summaries.values())
        total_failed = sum(s['failed'] for s in all_summaries.values())
        total_execution_time = sum(s['total_execution_time'] for s in all_summaries.values())
        
        # Performance analysis
        all_execution_times = []
        all_memory_usage = []
        all_physics_scores = []
        
        for summary in all_summaries.values():
            for result in summary['results']:
                if result['status'] == 'completed':
                    all_execution_times.append(result['execution_time'])
                    if result['memory_usage_mb'] > 0:
                        all_memory_usage.append(result['memory_usage_mb'])
                    if result['physics_score'] > 0:
                        all_physics_scores.append(result['physics_score'])
        
        performance_stats = {}
        if all_execution_times:
            performance_stats['execution_time'] = {
                'mean': statistics.mean(all_execution_times),
                'median': statistics.median(all_execution_times),
                'std': statistics.stdev(all_execution_times) if len(all_execution_times) > 1 else 0,
                'min': min(all_execution_times),
                'max': max(all_execution_times)
            }
        
        if all_memory_usage:
            performance_stats['memory_usage'] = {
                'mean': statistics.mean(all_memory_usage),
                'median': statistics.median(all_memory_usage),
                'std': statistics.stdev(all_memory_usage) if len(all_memory_usage) > 1 else 0,
                'min': min(all_memory_usage),
                'max': max(all_memory_usage)
            }
        
        if all_physics_scores:
            performance_stats['physics_scores'] = {
                'mean': statistics.mean(all_physics_scores),
                'median': statistics.median(all_physics_scores),
                'std': statistics.stdev(all_physics_scores) if len(all_physics_scores) > 1 else 0,
                'min': min(all_physics_scores),
                'max': max(all_physics_scores)
            }
        
        return {
            'timestamp': time.time(),
            'total_suites': len(all_summaries),
            'total_benchmarks': total_benchmarks,
            'total_completed': total_completed,
            'total_failed': total_failed,
            'overall_success_rate': total_completed / total_benchmarks if total_benchmarks > 0 else 0,
            'total_execution_time': total_execution_time,
            'performance_statistics': performance_stats,
            'suite_summaries': all_summaries
        }
    
    def _generate_comprehensive_report(self, summary: Dict[str, Any]) -> None:
        """Generate comprehensive benchmark report."""
        
        report_file = self.output_dir / "comprehensive_benchmark_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# DarkOperator Studio - Comprehensive Benchmark Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Benchmark Suites**: {summary['total_suites']}\n")
            f.write(f"- **Total Benchmarks**: {summary['total_benchmarks']}\n")
            f.write(f"- **Overall Success Rate**: {summary['overall_success_rate']:.1%}\n")
            f.write(f"- **Total Execution Time**: {summary['total_execution_time']:.2f}s\n\n")
            
            # Performance Overview
            if 'performance_statistics' in summary:
                f.write("## Performance Overview\n\n")
                
                perf_stats = summary['performance_statistics']
                
                if 'execution_time' in perf_stats:
                    et = perf_stats['execution_time']
                    f.write(f"**Execution Time Statistics:**\n")
                    f.write(f"- Mean: {et['mean']:.4f}s\n")
                    f.write(f"- Median: {et['median']:.4f}s\n")
                    f.write(f"- Standard Deviation: {et['std']:.4f}s\n")
                    f.write(f"- Range: {et['min']:.4f}s - {et['max']:.4f}s\n\n")
                
                if 'memory_usage' in perf_stats:
                    mem = perf_stats['memory_usage']
                    f.write(f"**Memory Usage Statistics:**\n")
                    f.write(f"- Mean: {mem['mean']:.2f} MB\n")
                    f.write(f"- Median: {mem['median']:.2f} MB\n")
                    f.write(f"- Standard Deviation: {mem['std']:.2f} MB\n")
                    f.write(f"- Range: {mem['min']:.2f} MB - {mem['max']:.2f} MB\n\n")
                
                if 'physics_scores' in perf_stats:
                    phys = perf_stats['physics_scores']
                    f.write(f"**Physics Validation Scores:**\n")
                    f.write(f"- Mean: {phys['mean']:.4f}\n")
                    f.write(f"- Median: {phys['median']:.4f}\n")
                    f.write(f"- Standard Deviation: {phys['std']:.4f}\n")
                    f.write(f"- Range: {phys['min']:.4f} - {phys['max']:.4f}\n\n")
            
            # Suite Details
            f.write("## Benchmark Suite Details\n\n")
            
            for suite_name, suite_summary in summary['suite_summaries'].items():
                f.write(f"### {suite_name}\n\n")
                f.write(f"- **Benchmarks**: {suite_summary['total_benchmarks']}\n")
                f.write(f"- **Success Rate**: {suite_summary['success_rate']:.1%}\n")
                f.write(f"- **Execution Time**: {suite_summary['total_execution_time']:.2f}s\n\n")
        
        logger.info(f"Generated comprehensive report: {report_file}")
    
    def compare_results(self, baseline_file: str, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with baseline for regression detection."""
        
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
            
            comparison = self.regression_detector.compare_results(baseline, current_results)
            
            # Save comparison results
            comparison_file = self.output_dir / "regression_analysis.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare results: {e}")
            return {'error': str(e)}


class StatisticalAnalyzer:
    """Statistical analysis utilities for benchmark results."""
    
    def analyze_distribution(self, data: List[float]) -> Dict[str, Any]:
        """Analyze data distribution and provide statistics."""
        
        if not data:
            return {}
        
        return {
            'count': len(data),
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'mode': statistics.mode(data) if len(set(data)) < len(data) else None,
            'std_dev': statistics.stdev(data) if len(data) > 1 else 0,
            'variance': statistics.variance(data) if len(data) > 1 else 0,
            'min': min(data),
            'max': max(data),
            'range': max(data) - min(data),
            'percentiles': {
                '25': np.percentile(data, 25),
                '50': np.percentile(data, 50),
                '75': np.percentile(data, 75),
                '90': np.percentile(data, 90),
                '95': np.percentile(data, 95),
                '99': np.percentile(data, 99)
            }
        }


class RegressionDetector:
    """Detects performance regressions between benchmark runs."""
    
    def compare_results(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results against baseline for regressions."""
        
        regressions = []
        improvements = []
        
        # Compare execution times
        baseline_times = self._extract_execution_times(baseline)
        current_times = self._extract_execution_times(current)
        
        for benchmark_name in set(baseline_times.keys()) & set(current_times.keys()):
            baseline_time = baseline_times[benchmark_name]
            current_time = current_times[benchmark_name]
            
            # Calculate percentage change
            pct_change = ((current_time - baseline_time) / baseline_time) * 100
            
            if pct_change > 10:  # More than 10% slower
                regressions.append({
                    'benchmark': benchmark_name,
                    'baseline_time': baseline_time,
                    'current_time': current_time,
                    'pct_change': pct_change,
                    'type': 'performance_regression'
                })
            elif pct_change < -10:  # More than 10% faster
                improvements.append({
                    'benchmark': benchmark_name,
                    'baseline_time': baseline_time,
                    'current_time': current_time,
                    'pct_change': pct_change,
                    'type': 'performance_improvement'
                })
        
        return {
            'regressions': regressions,
            'improvements': improvements,
            'total_benchmarks_compared': len(set(baseline_times.keys()) & set(current_times.keys())),
            'regression_count': len(regressions),
            'improvement_count': len(improvements)
        }
    
    def _extract_execution_times(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract execution times from results."""
        
        times = {}
        
        if 'suite_summaries' in results:
            for suite_name, suite_data in results['suite_summaries'].items():
                for result in suite_data.get('results', []):
                    if result['status'] == 'completed':
                        times[result['benchmark_name']] = result['execution_time']
        
        return times


class PerformanceProfiler:
    """Advanced performance profiling and analysis."""
    
    def profile_memory_usage(self, benchmark_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage during benchmark execution."""
        
        import tracemalloc
        
        # Start tracing
        tracemalloc.start()
        
        try:
            # Execute benchmark
            result = benchmark_func(*args, **kwargs)
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            
            return {
                'result': result,
                'current_memory_mb': current / (1024 * 1024),
                'peak_memory_mb': peak / (1024 * 1024)
            }
            
        finally:
            tracemalloc.stop()
    
    def profile_cpu_usage(self, benchmark_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile CPU usage during benchmark execution."""
        
        process = psutil.Process()
        
        # Get initial CPU time
        cpu_times_start = process.cpu_times()
        
        # Execute benchmark
        start_time = time.time()
        result = benchmark_func(*args, **kwargs)
        end_time = time.time()
        
        # Get final CPU time
        cpu_times_end = process.cpu_times()
        
        # Calculate CPU usage
        user_time = cpu_times_end.user - cpu_times_start.user
        system_time = cpu_times_end.system - cpu_times_start.system
        total_cpu_time = user_time + system_time
        wall_time = end_time - start_time
        
        cpu_usage_percent = (total_cpu_time / wall_time) * 100 if wall_time > 0 else 0
        
        return {
            'result': result,
            'cpu_usage_percent': cpu_usage_percent,
            'user_time': user_time,
            'system_time': system_time,
            'wall_time': wall_time
        }