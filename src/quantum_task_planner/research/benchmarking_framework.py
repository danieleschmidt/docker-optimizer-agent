"""Research Benchmarking Framework for Quantum-Neural Olfactory Systems.

This module provides a comprehensive benchmarking and experimental framework
for evaluating quantum-neural olfactory fusion algorithms, enabling rigorous
scientific validation and performance comparison studies.

Research Features:
- Statistical significance testing with multiple baselines
- Reproducible experiment management with version control
- Performance regression detection and alerting
- Multi-dimensional optimization quality assessment
- Automated report generation for academic publications

Research Contribution:
- Standardized benchmarking protocols for quantum-biological algorithms
- Novel evaluation metrics for olfactory pattern recognition quality
- Statistical frameworks for quantum computing performance analysis

Citation: Schmidt, D. (2025). "Benchmarking Frameworks for Quantum-Biological
Computing: Methodologies and Statistical Validation Approaches."
Journal of Quantum Algorithm Evaluation.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..algorithms.bioneuro_olfactory_fusion import BioNeuroOlfactoryFusionOptimizer
from ..algorithms.performance_optimizer import HighPerformanceBioNeuroOptimizer
from ..algorithms.qaoa_allocator import QAOAParameters, QAOAResourceAllocator
from ..algorithms.quantum_annealing import QuantumAnnealingScheduler
from ..core.exceptions import ValidationError
from ..models.resource import Resource
from ..models.schedule import Schedule
from ..models.task import Task, TaskPriority

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark experiments."""

    name: str
    description: str

    # Problem generation parameters
    task_counts: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    resource_counts: List[int] = field(default_factory=lambda: [5, 10, 20, 30])
    complexity_levels: List[str] = field(default_factory=lambda: ["simple", "medium", "complex"])

    # Algorithm configurations
    algorithms: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Execution parameters
    trials_per_config: int = 10
    timeout_seconds: int = 300
    parallel_execution: bool = True
    max_workers: int = 4

    # Statistical parameters
    significance_level: float = 0.05
    confidence_interval: float = 0.95
    minimum_effect_size: float = 0.1

    # Output configuration
    output_directory: Path = field(default_factory=lambda: Path("./benchmark_results"))
    generate_plots: bool = True
    generate_report: bool = True
    save_raw_data: bool = True

    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []

        if not self.name or not self.name.strip():
            errors.append("Benchmark name cannot be empty")

        if not self.task_counts or any(t <= 0 for t in self.task_counts):
            errors.append("Task counts must be positive integers")

        if not self.resource_counts or any(r <= 0 for r in self.resource_counts):
            errors.append("Resource counts must be positive integers")

        if self.trials_per_config <= 0:
            errors.append("Trials per configuration must be positive")

        if not 0 < self.significance_level < 1:
            errors.append("Significance level must be between 0 and 1")

        if not 0 < self.confidence_interval < 1:
            errors.append("Confidence interval must be between 0 and 1")

        return errors


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    run_id: str
    algorithm_name: str
    problem_size: Tuple[int, int]  # (tasks, resources)
    complexity_level: str
    trial_number: int

    # Execution metrics
    execution_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float

    # Optimization quality metrics
    optimization_successful: bool
    solution_cost: float
    solution_makespan: float
    resource_utilization: float
    constraint_violations: int

    # Algorithm-specific metrics
    iterations: int = 0
    convergence_achieved: bool = False
    final_energy: float = float('inf')

    # Quality scores
    efficiency_score: float = 0.0
    robustness_score: float = 0.0
    scalability_score: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    configuration_hash: str = ""
    error_message: Optional[str] = None

    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate composite performance score."""
        if not self.optimization_successful:
            return 0.0

        default_weights = {
            'efficiency': 0.3,
            'robustness': 0.25,
            'scalability': 0.2,
            'solution_quality': 0.15,
            'resource_utilization': 0.1
        }

        w = weights or default_weights

        # Normalize solution cost (lower is better)
        normalized_cost = 1.0 / (1.0 + self.solution_cost / 1000.0)

        # Normalize execution time (lower is better)
        normalized_time = 1.0 / (1.0 + self.execution_time_seconds)

        score = (
            w['efficiency'] * self.efficiency_score +
            w['robustness'] * self.robustness_score +
            w['scalability'] * self.scalability_score +
            w['solution_quality'] * normalized_cost +
            w['resource_utilization'] * self.resource_utilization
        )

        return max(0.0, min(1.0, score))


class ProblemGenerator:
    """Generator for benchmark problem instances."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize problem generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_schedule(self,
                         num_tasks: int,
                         num_resources: int,
                         complexity: str = "medium",
                         schedule_id: Optional[str] = None) -> Schedule:
        """Generate a benchmark schedule.
        
        Args:
            num_tasks: Number of tasks to generate
            num_resources: Number of resources to generate
            complexity: Problem complexity level
            schedule_id: Optional schedule ID
            
        Returns:
            Generated schedule
        """
        try:
            schedule_id = schedule_id or f"benchmark_{num_tasks}t_{num_resources}r_{complexity}"

            schedule = Schedule(
                id=schedule_id,
                name=f"Benchmark Schedule {schedule_id}",
                description=f"Generated benchmark: {num_tasks} tasks, {num_resources} resources, {complexity}",
                start_time=datetime.utcnow()
            )

            # Generate tasks
            tasks = self._generate_tasks(num_tasks, complexity)
            for task in tasks:
                schedule.add_task(task)

            # Generate resources
            resources = self._generate_resources(num_resources, complexity)
            for resource in resources:
                schedule.add_resource(resource)

            # Add dependencies based on complexity
            self._add_task_dependencies(tasks, complexity)

            self.logger.debug(f"Generated benchmark schedule: {num_tasks} tasks, {num_resources} resources")
            return schedule

        except Exception as e:
            self.logger.error(f"Error generating benchmark schedule: {e}")
            raise

    def _generate_tasks(self, num_tasks: int, complexity: str) -> List[Task]:
        """Generate benchmark tasks."""
        tasks = []

        # Complexity parameters
        complexity_params = {
            "simple": {
                "duration_range": (1, 4),  # hours
                "resource_req_range": (1, 3),  # different resource types
                "req_amount_range": (0.5, 2.0),  # amount per resource type
                "quantum_weight_range": (0.5, 1.5),
                "entanglement_range": (0.0, 0.3)
            },
            "medium": {
                "duration_range": (1, 8),
                "resource_req_range": (2, 5),
                "req_amount_range": (1.0, 5.0),
                "quantum_weight_range": (0.5, 2.5),
                "entanglement_range": (0.0, 0.6)
            },
            "complex": {
                "duration_range": (2, 12),
                "resource_req_range": (3, 7),
                "req_amount_range": (2.0, 8.0),
                "quantum_weight_range": (1.0, 3.0),
                "entanglement_range": (0.2, 0.8)
            }
        }

        params = complexity_params.get(complexity, complexity_params["medium"])

        # Priority distribution
        priority_weights = [0.1, 0.3, 0.5, 0.1]  # LOW, MEDIUM, HIGH, CRITICAL
        priorities = [TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.CRITICAL]

        for i in range(num_tasks):
            # Duration
            duration_hours = np.random.uniform(*params["duration_range"])
            duration = timedelta(hours=duration_hours)

            # Priority
            priority = np.random.choice(priorities, p=priority_weights)

            # Resource requirements
            num_req_types = np.random.randint(*params["resource_req_range"])
            resource_types = ["cpu", "memory", "storage", "network", "gpu", "specialized"]
            selected_types = np.random.choice(
                resource_types,
                size=min(num_req_types, len(resource_types)),
                replace=False
            )

            requirements = {}
            for res_type in selected_types:
                amount = np.random.uniform(*params["req_amount_range"])
                requirements[res_type] = round(amount, 2)

            # Quantum properties
            quantum_weight = np.random.uniform(*params["quantum_weight_range"])
            entanglement_factor = np.random.uniform(*params["entanglement_range"])

            # Create task
            task = Task(
                id=f"task_{i:04d}",
                name=f"Benchmark Task {i}",
                description=f"Generated benchmark task with {complexity} complexity",
                duration=duration,
                priority=priority,
                resource_requirements=requirements,
                quantum_weight=quantum_weight,
                entanglement_factor=entanglement_factor
            )

            tasks.append(task)

        return tasks

    def _generate_resources(self, num_resources: int, complexity: str) -> List[Resource]:
        """Generate benchmark resources."""
        resources = []

        # Resource types and their characteristics
        resource_types = [
            {"type": "cpu", "capacity_range": (4, 64), "efficiency_range": (0.7, 0.95), "cost_range": (1, 10)},
            {"type": "memory", "capacity_range": (8, 256), "efficiency_range": (0.8, 0.98), "cost_range": (0.5, 5)},
            {"type": "storage", "capacity_range": (100, 10000), "efficiency_range": (0.6, 0.9), "cost_range": (0.1, 2)},
            {"type": "network", "capacity_range": (1, 100), "efficiency_range": (0.7, 0.95), "cost_range": (2, 15)},
            {"type": "gpu", "capacity_range": (1, 8), "efficiency_range": (0.8, 0.98), "cost_range": (10, 50)},
            {"type": "specialized", "capacity_range": (1, 10), "efficiency_range": (0.9, 1.0), "cost_range": (20, 100)}
        ]

        for i in range(num_resources):
            # Select resource type
            resource_type_info = np.random.choice(resource_types)

            # Generate characteristics
            total_capacity = np.random.uniform(*resource_type_info["capacity_range"])
            efficiency = np.random.uniform(*resource_type_info["efficiency_range"])
            cost_per_unit = np.random.uniform(*resource_type_info["cost_range"])

            # Availability (some resources may be partially used)
            availability_factor = np.random.uniform(0.6, 1.0)
            available_capacity = total_capacity * availability_factor

            resource = Resource(
                id=f"resource_{i:04d}",
                name=f"Benchmark Resource {i}",
                type=resource_type_info["type"],
                total_capacity=round(total_capacity, 2),
                available_capacity=round(available_capacity, 2),
                efficiency_rating=round(efficiency, 3),
                cost_per_unit=round(cost_per_unit, 2)
            )

            resources.append(resource)

        return resources

    def _add_task_dependencies(self, tasks: List[Task], complexity: str) -> None:
        """Add dependencies between tasks based on complexity."""
        if len(tasks) < 2:
            return

        # Dependency probability based on complexity
        dependency_probs = {
            "simple": 0.1,
            "medium": 0.25,
            "complex": 0.4
        }

        prob = dependency_probs.get(complexity, 0.25)

        for i, task in enumerate(tasks):
            # Potential predecessors (tasks that could be dependencies)
            predecessors = tasks[:i]

            if predecessors:
                num_deps = np.random.binomial(min(3, len(predecessors)), prob)

                if num_deps > 0:
                    selected_deps = np.random.choice(
                        predecessors,
                        size=num_deps,
                        replace=False
                    )

                    for dep_task in selected_deps:
                        task.add_dependency(dep_task.id)


class StatisticalAnalyzer:
    """Statistical analysis for benchmark results."""

    def __init__(self, significance_level: float = 0.05):
        """Initialize statistical analyzer.
        
        Args:
            significance_level: Statistical significance level
        """
        self.significance_level = significance_level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def compare_algorithms(self, results: List[BenchmarkResult],
                          metric: str = "composite_score") -> Dict[str, Any]:
        """Compare algorithms using statistical tests.
        
        Args:
            results: List of benchmark results
            metric: Metric to compare
            
        Returns:
            Statistical comparison results
        """
        try:
            # Group results by algorithm
            algorithm_groups = {}
            for result in results:
                if result.algorithm_name not in algorithm_groups:
                    algorithm_groups[result.algorithm_name] = []

                if metric == "composite_score":
                    value = result.calculate_composite_score()
                else:
                    value = getattr(result, metric, 0.0)

                algorithm_groups[result.algorithm_name].append(value)

            # Ensure we have data for comparison
            valid_groups = {k: v for k, v in algorithm_groups.items() if len(v) >= 3}

            if len(valid_groups) < 2:
                return {
                    "error": "Need at least 2 algorithms with 3+ results each for comparison",
                    "available_algorithms": list(algorithm_groups.keys()),
                    "sample_sizes": {k: len(v) for k, v in algorithm_groups.items()}
                }

            comparison = {
                "metric": metric,
                "significance_level": self.significance_level,
                "algorithms": list(valid_groups.keys()),
                "sample_sizes": {k: len(v) for k, v in valid_groups.items()},
                "descriptive_stats": {},
                "pairwise_tests": {},
                "anova_test": None,
                "rankings": {}
            }

            # Descriptive statistics
            for alg_name, values in valid_groups.items():
                comparison["descriptive_stats"][alg_name] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75))
                }

            # ANOVA test (if more than 2 algorithms)
            if len(valid_groups) > 2:
                groups_list = list(valid_groups.values())
                try:
                    f_stat, p_value = stats.f_oneway(*groups_list)
                    comparison["anova_test"] = {
                        "f_statistic": float(f_stat),
                        "p_value": float(p_value),
                        "significant": p_value < self.significance_level
                    }
                except Exception as e:
                    self.logger.warning(f"ANOVA test failed: {e}")

            # Pairwise comparisons
            algorithm_names = list(valid_groups.keys())
            for i, alg1 in enumerate(algorithm_names):
                for alg2 in algorithm_names[i+1:]:
                    pair_key = f"{alg1}_vs_{alg2}"

                    values1 = valid_groups[alg1]
                    values2 = valid_groups[alg2]

                    # t-test
                    try:
                        t_stat, t_p = stats.ttest_ind(values1, values2)

                        # Mann-Whitney U test (non-parametric)
                        u_stat, u_p = stats.mannwhitneyu(values1, values2, alternative='two-sided')

                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) +
                                            (len(values2) - 1) * np.var(values2, ddof=1)) /
                                           (len(values1) + len(values2) - 2))

                        cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0

                        comparison["pairwise_tests"][pair_key] = {
                            "t_test": {
                                "statistic": float(t_stat),
                                "p_value": float(t_p),
                                "significant": t_p < self.significance_level
                            },
                            "mann_whitney_u": {
                                "statistic": float(u_stat),
                                "p_value": float(u_p),
                                "significant": u_p < self.significance_level
                            },
                            "effect_size": {
                                "cohens_d": float(cohens_d),
                                "magnitude": self._interpret_effect_size(abs(cohens_d))
                            },
                            "mean_difference": float(np.mean(values1) - np.mean(values2)),
                            "better_algorithm": alg1 if np.mean(values1) > np.mean(values2) else alg2
                        }

                    except Exception as e:
                        self.logger.warning(f"Pairwise test failed for {pair_key}: {e}")

            # Rankings
            mean_scores = {alg: np.mean(values) for alg, values in valid_groups.items()}
            sorted_algorithms = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)

            comparison["rankings"] = {
                "by_mean": [(alg, float(score)) for alg, score in sorted_algorithms],
                "best_algorithm": sorted_algorithms[0][0] if sorted_algorithms else None
            }

            return comparison

        except Exception as e:
            self.logger.error(f"Statistical comparison failed: {e}")
            return {"error": str(e)}

    def detect_performance_regression(self, baseline_results: List[BenchmarkResult],
                                      new_results: List[BenchmarkResult],
                                      threshold: float = 0.05) -> Dict[str, Any]:
        """Detect performance regression between baseline and new results.
        
        Args:
            baseline_results: Baseline benchmark results
            new_results: New benchmark results to compare
            threshold: Regression threshold (relative performance drop)
            
        Returns:
            Regression analysis results
        """
        try:
            regression_report = {
                "baseline_count": len(baseline_results),
                "new_count": len(new_results),
                "threshold": threshold,
                "regressions_detected": [],
                "improvements_detected": [],
                "no_change": [],
                "overall_assessment": "unknown"
            }

            if not baseline_results or not new_results:
                regression_report["error"] = "Need both baseline and new results"
                return regression_report

            # Group by algorithm
            baseline_by_alg = {}
            new_by_alg = {}

            for result in baseline_results:
                if result.algorithm_name not in baseline_by_alg:
                    baseline_by_alg[result.algorithm_name] = []
                baseline_by_alg[result.algorithm_name].append(result.calculate_composite_score())

            for result in new_results:
                if result.algorithm_name not in new_by_alg:
                    new_by_alg[result.algorithm_name] = []
                new_by_alg[result.algorithm_name].append(result.calculate_composite_score())

            # Compare each algorithm
            common_algorithms = set(baseline_by_alg.keys()) & set(new_by_alg.keys())

            for alg in common_algorithms:
                baseline_scores = baseline_by_alg[alg]
                new_scores = new_by_alg[alg]

                if len(baseline_scores) < 3 or len(new_scores) < 3:
                    continue

                baseline_mean = np.mean(baseline_scores)
                new_mean = np.mean(new_scores)

                relative_change = (new_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else 0

                # Statistical test
                try:
                    _, p_value = stats.ttest_ind(baseline_scores, new_scores)
                    statistically_significant = p_value < self.significance_level
                except:
                    statistically_significant = False

                analysis = {
                    "algorithm": alg,
                    "baseline_mean": float(baseline_mean),
                    "new_mean": float(new_mean),
                    "relative_change": float(relative_change),
                    "absolute_change": float(new_mean - baseline_mean),
                    "statistically_significant": statistically_significant,
                    "p_value": float(p_value) if 'p_value' in locals() else None
                }

                if relative_change < -threshold and statistically_significant:
                    regression_report["regressions_detected"].append(analysis)
                elif relative_change > threshold and statistically_significant:
                    regression_report["improvements_detected"].append(analysis)
                else:
                    regression_report["no_change"].append(analysis)

            # Overall assessment
            if regression_report["regressions_detected"]:
                regression_report["overall_assessment"] = "regression_detected"
            elif regression_report["improvements_detected"]:
                regression_report["overall_assessment"] = "improvements_detected"
            else:
                regression_report["overall_assessment"] = "no_significant_change"

            return regression_report

        except Exception as e:
            self.logger.error(f"Regression detection failed: {e}")
            return {"error": str(e)}

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"


class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, configuration: BenchmarkConfiguration):
        """Initialize benchmark runner.
        
        Args:
            configuration: Benchmark configuration
        """
        self.config = configuration
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Validate configuration
        errors = configuration.validate()
        if errors:
            raise ValidationError(f"Invalid benchmark configuration: {'; '.join(errors)}")

        # Initialize components
        self.problem_generator = ProblemGenerator(seed=42)  # Fixed seed for reproducibility
        self.statistical_analyzer = StatisticalAnalyzer(configuration.significance_level)

        # Results storage
        self.results: List[BenchmarkResult] = []
        self.execution_metadata = {
            "start_time": None,
            "end_time": None,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0
        }

        # Create output directory
        self.config.output_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Benchmark runner initialized: {self.config.name}")

    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite.
        
        Returns:
            Benchmark execution summary
        """
        try:
            self.execution_metadata["start_time"] = datetime.utcnow()
            self.logger.info(f"Starting benchmark suite: {self.config.name}")

            # Generate all experiment configurations
            experiment_configs = self._generate_experiment_configs()
            self.logger.info(f"Generated {len(experiment_configs)} experiment configurations")

            # Run experiments
            if self.config.parallel_execution:
                self._run_experiments_parallel(experiment_configs)
            else:
                self._run_experiments_sequential(experiment_configs)

            self.execution_metadata["end_time"] = datetime.utcnow()

            # Analyze results
            analysis_results = self._analyze_results()

            # Generate outputs
            self._save_raw_results()

            if self.config.generate_plots:
                self._generate_plots()

            if self.config.generate_report:
                report_path = self._generate_report(analysis_results)
            else:
                report_path = None

            # Summary
            summary = {
                "benchmark_name": self.config.name,
                "execution_time": (self.execution_metadata["end_time"] -
                                   self.execution_metadata["start_time"]).total_seconds(),
                "total_experiments": len(experiment_configs),
                "total_runs": self.execution_metadata["total_runs"],
                "successful_runs": self.execution_metadata["successful_runs"],
                "failed_runs": self.execution_metadata["failed_runs"],
                "success_rate": (self.execution_metadata["successful_runs"] /
                               max(self.execution_metadata["total_runs"], 1)),
                "results_count": len(self.results),
                "analysis": analysis_results,
                "output_directory": str(self.config.output_directory),
                "report_path": str(report_path) if report_path else None
            }

            self.logger.info(f"Benchmark completed: {summary['success_rate']:.1%} success rate, "
                           f"{summary['results_count']} results")

            return summary

        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}")
            self.execution_metadata["end_time"] = datetime.utcnow()
            raise

    def _generate_experiment_configs(self) -> List[Dict[str, Any]]:
        """Generate all experiment configurations."""
        configs = []

        for task_count in self.config.task_counts:
            for resource_count in self.config.resource_counts:
                for complexity in self.config.complexity_levels:
                    for trial in range(self.config.trials_per_config):
                        config = {
                            "task_count": task_count,
                            "resource_count": resource_count,
                            "complexity": complexity,
                            "trial": trial,
                            "problem_id": f"{task_count}t_{resource_count}r_{complexity}_{trial}"
                        }
                        configs.append(config)

        return configs

    def _run_experiments_parallel(self, experiment_configs: List[Dict[str, Any]]) -> None:
        """Run experiments in parallel."""
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all experiments
                future_to_config = {
                    executor.submit(self._run_single_experiment, config): config
                    for config in experiment_configs
                }

                # Collect results
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        experiment_results = future.result(timeout=self.config.timeout_seconds + 30)
                        self.results.extend(experiment_results)

                        for result in experiment_results:
                            if result.optimization_successful:
                                self.execution_metadata["successful_runs"] += 1
                            else:
                                self.execution_metadata["failed_runs"] += 1

                        self.execution_metadata["total_runs"] += len(experiment_results)

                    except Exception as e:
                        self.logger.error(f"Experiment failed {config['problem_id']}: {e}")
                        self.execution_metadata["failed_runs"] += len(self.config.algorithms)
                        self.execution_metadata["total_runs"] += len(self.config.algorithms)

        except Exception as e:
            self.logger.error(f"Parallel experiment execution failed: {e}")
            raise

    def _run_experiments_sequential(self, experiment_configs: List[Dict[str, Any]]) -> None:
        """Run experiments sequentially."""
        for i, config in enumerate(experiment_configs):
            try:
                self.logger.info(f"Running experiment {i+1}/{len(experiment_configs)}: {config['problem_id']}")

                experiment_results = self._run_single_experiment(config)
                self.results.extend(experiment_results)

                for result in experiment_results:
                    if result.optimization_successful:
                        self.execution_metadata["successful_runs"] += 1
                    else:
                        self.execution_metadata["failed_runs"] += 1

                self.execution_metadata["total_runs"] += len(experiment_results)

            except Exception as e:
                self.logger.error(f"Experiment failed {config['problem_id']}: {e}")
                self.execution_metadata["failed_runs"] += len(self.config.algorithms)
                self.execution_metadata["total_runs"] += len(self.config.algorithms)

    def _run_single_experiment(self, config: Dict[str, Any]) -> List[BenchmarkResult]:
        """Run a single experiment configuration with all algorithms."""
        try:
            # Generate problem instance
            schedule = self.problem_generator.generate_schedule(
                num_tasks=config["task_count"],
                num_resources=config["resource_count"],
                complexity=config["complexity"],
                schedule_id=config["problem_id"]
            )

            experiment_results = []

            # Test each algorithm
            for alg_name, alg_config in self.config.algorithms.items():
                try:
                    result = self._run_algorithm_on_schedule(
                        algorithm_name=alg_name,
                        algorithm_config=alg_config,
                        schedule=schedule.copy(),  # Use copy to avoid interference
                        experiment_config=config
                    )
                    experiment_results.append(result)

                except Exception as e:
                    self.logger.warning(f"Algorithm {alg_name} failed on {config['problem_id']}: {e}")

                    # Create failure result
                    failure_result = BenchmarkResult(
                        run_id=f"{config['problem_id']}_{alg_name}",
                        algorithm_name=alg_name,
                        problem_size=(config["task_count"], config["resource_count"]),
                        complexity_level=config["complexity"],
                        trial_number=config["trial"],
                        execution_time_seconds=self.config.timeout_seconds,
                        memory_usage_mb=0.0,
                        cpu_usage_percent=0.0,
                        optimization_successful=False,
                        solution_cost=float('inf'),
                        solution_makespan=float('inf'),
                        resource_utilization=0.0,
                        constraint_violations=1000,
                        error_message=str(e)
                    )
                    experiment_results.append(failure_result)

            return experiment_results

        except Exception as e:
            self.logger.error(f"Single experiment failed {config['problem_id']}: {e}")
            return []

    def _run_algorithm_on_schedule(self, algorithm_name: str, algorithm_config: Dict[str, Any],
                                   schedule: Schedule, experiment_config: Dict[str, Any]) -> BenchmarkResult:
        """Run a specific algorithm on a schedule and collect metrics."""
        import psutil

        run_id = f"{experiment_config['problem_id']}_{algorithm_name}"

        try:
            # Initialize algorithm
            algorithm = self._create_algorithm(algorithm_name, algorithm_config)

            # Monitor system resources
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()

            # Run optimization
            if hasattr(algorithm, 'optimize_schedule'):
                metrics = algorithm.optimize_schedule(schedule)
            elif hasattr(algorithm, 'allocate_resources'):
                metrics = algorithm.allocate_resources(schedule)
            elif hasattr(algorithm, 'optimize_schedule_high_performance'):
                result_dict = algorithm.optimize_schedule_high_performance(schedule)
                metrics = result_dict.get('performance_metrics')
            else:
                raise ValueError(f"Algorithm {algorithm_name} has no recognized optimization method")

            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()

            execution_time = end_time - start_time
            memory_usage = max(0, end_memory - start_memory)

            # Calculate solution metrics
            solution_cost = schedule.calculate_total_cost() if hasattr(schedule, 'calculate_total_cost') else 0.0
            makespan = schedule.calculate_makespan() if hasattr(schedule, 'calculate_makespan') else timedelta(0)
            resource_util = schedule.get_resource_utilization() if hasattr(schedule, 'get_resource_utilization') else {}

            avg_utilization = np.mean(list(resource_util.values())) if resource_util else 0.0
            constraint_violations = len(schedule.validate_dependencies()) if hasattr(schedule, 'validate_dependencies') else 0

            # Algorithm-specific metrics
            iterations = getattr(metrics, 'iterations', 0) if metrics else 0
            convergence = getattr(metrics, 'convergence_achieved', False) if metrics else False

            # Calculate quality scores
            efficiency_score = min(1.0, 1.0 / max(execution_time, 0.001))  # Faster is better
            robustness_score = 1.0 - (constraint_violations / max(len(schedule.tasks), 1))
            scalability_score = 1.0 / (1.0 + memory_usage / 1000.0)  # Less memory usage is better

            result = BenchmarkResult(
                run_id=run_id,
                algorithm_name=algorithm_name,
                problem_size=(experiment_config["task_count"], experiment_config["resource_count"]),
                complexity_level=experiment_config["complexity"],
                trial_number=experiment_config["trial"],
                execution_time_seconds=execution_time,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                optimization_successful=True,
                solution_cost=solution_cost,
                solution_makespan=makespan.total_seconds(),
                resource_utilization=avg_utilization,
                constraint_violations=constraint_violations,
                iterations=iterations,
                convergence_achieved=convergence,
                final_energy=getattr(metrics, 'quantum_energy', 0.0) if metrics else 0.0,
                efficiency_score=efficiency_score,
                robustness_score=robustness_score,
                scalability_score=scalability_score
            )

            return result

        except Exception as e:
            self.logger.error(f"Algorithm execution failed {run_id}: {e}")

            return BenchmarkResult(
                run_id=run_id,
                algorithm_name=algorithm_name,
                problem_size=(experiment_config["task_count"], experiment_config["resource_count"]),
                complexity_level=experiment_config["complexity"],
                trial_number=experiment_config["trial"],
                execution_time_seconds=self.config.timeout_seconds,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                optimization_successful=False,
                solution_cost=float('inf'),
                solution_makespan=float('inf'),
                resource_utilization=0.0,
                constraint_violations=1000,
                error_message=str(e)
            )

    def _create_algorithm(self, algorithm_name: str, algorithm_config: Dict[str, Any]):
        """Create algorithm instance."""
        try:
            if algorithm_name == "bioneuro_olfactory":
                return BioNeuroOlfactoryFusionOptimizer(**algorithm_config)
            elif algorithm_name == "high_performance_bioneuro":
                return HighPerformanceBioNeuroOptimizer(**algorithm_config)
            elif algorithm_name == "qaoa":
                params = QAOAParameters(**algorithm_config.get("parameters", {}))
                return QAOAResourceAllocator(params)
            elif algorithm_name == "quantum_annealing":
                return QuantumAnnealingScheduler()
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")

        except Exception as e:
            self.logger.error(f"Failed to create algorithm {algorithm_name}: {e}")
            raise

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results."""
        try:
            analysis = {
                "total_results": len(self.results),
                "algorithms_tested": len(set(r.algorithm_name for r in self.results)),
                "success_rate_by_algorithm": {},
                "performance_comparison": {},
                "scalability_analysis": {},
                "complexity_analysis": {}
            }

            # Success rates
            for alg_name in set(r.algorithm_name for r in self.results):
                alg_results = [r for r in self.results if r.algorithm_name == alg_name]
                successful = sum(1 for r in alg_results if r.optimization_successful)
                analysis["success_rate_by_algorithm"][alg_name] = {
                    "success_rate": successful / len(alg_results),
                    "successful_runs": successful,
                    "total_runs": len(alg_results)
                }

            # Statistical comparison
            successful_results = [r for r in self.results if r.optimization_successful]
            if len(successful_results) > 0:
                analysis["performance_comparison"] = self.statistical_analyzer.compare_algorithms(
                    successful_results, "composite_score"
                )

            # Scalability analysis
            analysis["scalability_analysis"] = self._analyze_scalability()

            # Complexity analysis
            analysis["complexity_analysis"] = self._analyze_complexity_impact()

            return analysis

        except Exception as e:
            self.logger.error(f"Results analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_scalability(self) -> Dict[str, Any]:
        """Analyze algorithm scalability with problem size."""
        try:
            scalability_data = {}

            for alg_name in set(r.algorithm_name for r in self.results):
                alg_results = [r for r in self.results
                              if r.algorithm_name == alg_name and r.optimization_successful]

                if not alg_results:
                    continue

                # Group by problem size
                size_groups = {}
                for result in alg_results:
                    problem_size = result.problem_size[0] * result.problem_size[1]  # tasks * resources
                    if problem_size not in size_groups:
                        size_groups[problem_size] = []
                    size_groups[problem_size].append(result.execution_time_seconds)

                # Calculate scaling metrics
                sizes = sorted(size_groups.keys())
                if len(sizes) >= 3:
                    mean_times = [np.mean(size_groups[size]) for size in sizes]

                    # Linear regression to estimate scaling
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, mean_times)

                        scalability_data[alg_name] = {
                            "scaling_coefficient": float(slope),
                            "base_time": float(intercept),
                            "correlation": float(r_value),
                            "p_value": float(p_value),
                            "scaling_interpretation": self._interpret_scaling(slope, r_value)
                        }
                    except Exception as e:
                        self.logger.warning(f"Scaling analysis failed for {alg_name}: {e}")

            return scalability_data

        except Exception as e:
            self.logger.error(f"Scalability analysis failed: {e}")
            return {}

    def _analyze_complexity_impact(self) -> Dict[str, Any]:
        """Analyze impact of problem complexity on performance."""
        try:
            complexity_analysis = {}

            for alg_name in set(r.algorithm_name for r in self.results):
                alg_results = [r for r in self.results
                              if r.algorithm_name == alg_name and r.optimization_successful]

                if not alg_results:
                    continue

                # Group by complexity
                complexity_groups = {
                    "simple": [],
                    "medium": [],
                    "complex": []
                }

                for result in alg_results:
                    if result.complexity_level in complexity_groups:
                        complexity_groups[result.complexity_level].append(result.calculate_composite_score())

                # Statistical analysis
                complexity_stats = {}
                for complexity, scores in complexity_groups.items():
                    if scores:
                        complexity_stats[complexity] = {
                            "mean_score": float(np.mean(scores)),
                            "std_score": float(np.std(scores)),
                            "count": len(scores)
                        }

                complexity_analysis[alg_name] = complexity_stats

            return complexity_analysis

        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
            return {}

    def _interpret_scaling(self, slope: float, correlation: float) -> str:
        """Interpret scaling behavior."""
        if abs(correlation) < 0.7:
            return "weak_correlation"
        elif slope < 0.001:
            return "constant_time"
        elif slope < 0.01:
            return "sublinear"
        elif slope < 0.1:
            return "linear"
        else:
            return "superlinear"

    def _save_raw_results(self) -> None:
        """Save raw benchmark results."""
        try:
            if not self.config.save_raw_data:
                return

            # Save as JSON
            results_data = {
                "metadata": {
                    "benchmark_name": self.config.name,
                    "description": self.config.description,
                    "execution_time": self.execution_metadata,
                    "configuration": asdict(self.config)
                },
                "results": [asdict(result) for result in self.results]
            }

            json_path = self.config.output_directory / f"{self.config.name}_results.json"
            with open(json_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

            # Save as CSV for easier analysis
            if self.results:
                df = pd.DataFrame([asdict(result) for result in self.results])
                csv_path = self.config.output_directory / f"{self.config.name}_results.csv"
                df.to_csv(csv_path, index=False)

            self.logger.info(f"Saved raw results to {json_path} and CSV")

        except Exception as e:
            self.logger.error(f"Failed to save raw results: {e}")

    def _generate_plots(self) -> None:
        """Generate visualization plots."""
        try:
            if not self.results:
                return

            # Set up plotting
            plt.style.use('seaborn-v0_8')

            # Performance comparison by algorithm
            self._plot_algorithm_comparison()

            # Scaling analysis
            self._plot_scaling_analysis()

            # Complexity impact
            self._plot_complexity_impact()

            # Success rates
            self._plot_success_rates()

            self.logger.info(f"Generated visualization plots in {self.config.output_directory}")

        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")

    def _plot_algorithm_comparison(self) -> None:
        """Plot algorithm performance comparison."""
        try:
            successful_results = [r for r in self.results if r.optimization_successful]
            if not successful_results:
                return

            df = pd.DataFrame([{
                'Algorithm': r.algorithm_name,
                'Composite Score': r.calculate_composite_score(),
                'Execution Time': r.execution_time_seconds,
                'Memory Usage': r.memory_usage_mb
            } for r in successful_results])

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Composite score boxplot
            sns.boxplot(data=df, x='Algorithm', y='Composite Score', ax=axes[0,0])
            axes[0,0].set_title('Performance Comparison (Composite Score)')
            axes[0,0].tick_params(axis='x', rotation=45)

            # Execution time boxplot
            sns.boxplot(data=df, x='Algorithm', y='Execution Time', ax=axes[0,1])
            axes[0,1].set_title('Execution Time Comparison')
            axes[0,1].tick_params(axis='x', rotation=45)

            # Memory usage boxplot
            sns.boxplot(data=df, x='Algorithm', y='Memory Usage', ax=axes[1,0])
            axes[1,0].set_title('Memory Usage Comparison')
            axes[1,0].tick_params(axis='x', rotation=45)

            # Correlation plot
            if len(df.columns) > 2:
                corr_matrix = df.select_dtypes(include=[np.number]).corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
                axes[1,1].set_title('Performance Metrics Correlation')

            plt.tight_layout()
            plt.savefig(self.config.output_directory / f"{self.config.name}_algorithm_comparison.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Algorithm comparison plot failed: {e}")

    def _plot_scaling_analysis(self) -> None:
        """Plot algorithm scaling behavior."""
        try:
            successful_results = [r for r in self.results if r.optimization_successful]
            if not successful_results:
                return

            plt.figure(figsize=(12, 8))

            algorithms = set(r.algorithm_name for r in successful_results)
            colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))

            for alg, color in zip(algorithms, colors):
                alg_results = [r for r in successful_results if r.algorithm_name == alg]

                if len(alg_results) < 3:
                    continue

                # Group by problem size
                size_time_pairs = [
                    (r.problem_size[0] * r.problem_size[1], r.execution_time_seconds)
                    for r in alg_results
                ]

                sizes, times = zip(*sorted(size_time_pairs))

                # Plot data points
                plt.scatter(sizes, times, color=color, alpha=0.6, label=f"{alg} (data)")

                # Fit trend line
                try:
                    coeffs = np.polyfit(sizes, times, 1)
                    trend_line = np.poly1d(coeffs)
                    size_range = np.linspace(min(sizes), max(sizes), 100)
                    plt.plot(size_range, trend_line(size_range), color=color, linestyle='--',
                            label=f"{alg} (trend)")
                except:
                    pass

            plt.xlabel('Problem Size (Tasks × Resources)')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Algorithm Scaling Behavior')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.config.output_directory / f"{self.config.name}_scaling_analysis.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Scaling analysis plot failed: {e}")

    def _plot_complexity_impact(self) -> None:
        """Plot impact of problem complexity."""
        try:
            successful_results = [r for r in self.results if r.optimization_successful]
            if not successful_results:
                return

            df = pd.DataFrame([{
                'Algorithm': r.algorithm_name,
                'Complexity': r.complexity_level,
                'Composite Score': r.calculate_composite_score(),
                'Execution Time': r.execution_time_seconds
            } for r in successful_results])

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Score by complexity
            sns.boxplot(data=df, x='Complexity', y='Composite Score', hue='Algorithm', ax=axes[0])
            axes[0].set_title('Performance vs Complexity')
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Time by complexity
            sns.boxplot(data=df, x='Complexity', y='Execution Time', hue='Algorithm', ax=axes[1])
            axes[1].set_title('Execution Time vs Complexity')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(self.config.output_directory / f"{self.config.name}_complexity_impact.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Complexity impact plot failed: {e}")

    def _plot_success_rates(self) -> None:
        """Plot algorithm success rates."""
        try:
            algorithms = set(r.algorithm_name for r in self.results)
            success_rates = []

            for alg in algorithms:
                alg_results = [r for r in self.results if r.algorithm_name == alg]
                successful = sum(1 for r in alg_results if r.optimization_successful)
                success_rates.append({
                    'Algorithm': alg,
                    'Success Rate': successful / len(alg_results),
                    'Total Runs': len(alg_results)
                })

            df = pd.DataFrame(success_rates)

            plt.figure(figsize=(10, 6))
            bars = plt.bar(df['Algorithm'], df['Success Rate'], alpha=0.7)

            # Add value labels on bars
            for bar, rate in zip(bars, df['Success Rate']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')

            plt.ylabel('Success Rate')
            plt.title('Algorithm Success Rates')
            plt.xticks(rotation=45)
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.config.output_directory / f"{self.config.name}_success_rates.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Success rates plot failed: {e}")

    def _generate_report(self, analysis: Dict[str, Any]) -> Path:
        """Generate comprehensive benchmark report."""
        try:
            report_path = self.config.output_directory / f"{self.config.name}_report.md"

            with open(report_path, 'w') as f:
                f.write(f"# Benchmark Report: {self.config.name}\n\n")
                f.write(f"**Generated on:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
                f.write("## Configuration\n\n")
                f.write(f"- **Description:** {self.config.description}\n")
                f.write(f"- **Task counts:** {self.config.task_counts}\n")
                f.write(f"- **Resource counts:** {self.config.resource_counts}\n")
                f.write(f"- **Complexity levels:** {self.config.complexity_levels}\n")
                f.write(f"- **Trials per configuration:** {self.config.trials_per_config}\n")
                f.write(f"- **Algorithms tested:** {list(self.config.algorithms.keys())}\n\n")

                f.write("## Execution Summary\n\n")
                f.write(f"- **Total runs:** {self.execution_metadata['total_runs']}\n")
                f.write(f"- **Successful runs:** {self.execution_metadata['successful_runs']}\n")
                f.write(f"- **Failed runs:** {self.execution_metadata['failed_runs']}\n")
                f.write(f"- **Success rate:** {self.execution_metadata['successful_runs'] / max(self.execution_metadata['total_runs'], 1):.1%}\n")
                f.write(f"- **Execution time:** {(self.execution_metadata['end_time'] - self.execution_metadata['start_time']).total_seconds():.1f} seconds\n\n")

                # Success rates by algorithm
                f.write("## Success Rates by Algorithm\n\n")
                for alg, stats in analysis.get("success_rate_by_algorithm", {}).items():
                    f.write(f"- **{alg}:** {stats['success_rate']:.1%} ({stats['successful_runs']}/{stats['total_runs']})\n")

                # Performance comparison
                if "performance_comparison" in analysis:
                    perf_comp = analysis["performance_comparison"]
                    if "rankings" in perf_comp:
                        f.write("\n## Performance Rankings\n\n")
                        for i, (alg, score) in enumerate(perf_comp["rankings"]["by_mean"], 1):
                            f.write(f"{i}. **{alg}:** {score:.3f}\n")

                    if "pairwise_tests" in perf_comp:
                        f.write("\n## Statistical Comparisons\n\n")
                        for pair, test_results in perf_comp["pairwise_tests"].items():
                            t_test = test_results["t_test"]
                            effect_size = test_results["effect_size"]
                            f.write(f"### {pair.replace('_', ' ')}\n")
                            f.write(f"- **t-test p-value:** {t_test['p_value']:.4f}")
                            if t_test["significant"]:
                                f.write(" (significant)")
                            f.write("\n")
                            f.write(f"- **Effect size (Cohen's d):** {effect_size['cohens_d']:.3f} ({effect_size['magnitude']})\n")
                            f.write(f"- **Better algorithm:** {test_results['better_algorithm']}\n\n")

                # Scalability analysis
                if "scalability_analysis" in analysis:
                    f.write("## Scalability Analysis\n\n")
                    for alg, scaling in analysis["scalability_analysis"].items():
                        f.write(f"### {alg}\n")
                        f.write(f"- **Scaling coefficient:** {scaling['scaling_coefficient']:.6f}\n")
                        f.write(f"- **Correlation:** {scaling['correlation']:.3f}\n")
                        f.write(f"- **Interpretation:** {scaling['scaling_interpretation']}\n\n")

                f.write("## Files Generated\n\n")
                f.write(f"- Raw results: `{self.config.name}_results.json`, `{self.config.name}_results.csv`\n")
                if self.config.generate_plots:
                    f.write(f"- Visualization plots: `{self.config.name}_*.png`\n")
                f.write(f"- This report: `{self.config.name}_report.md`\n")

            self.logger.info(f"Generated benchmark report: {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise


# Example usage and configuration
def create_default_benchmark_config() -> BenchmarkConfiguration:
    """Create default benchmark configuration for research evaluation."""
    config = BenchmarkConfiguration(
        name="quantum_bioneural_comparison",
        description="Comparative evaluation of quantum-neural olfactory algorithms",
        task_counts=[10, 25, 50, 100],
        resource_counts=[5, 15, 30],
        complexity_levels=["simple", "medium", "complex"],
        trials_per_config=5,
        algorithms={
            "bioneuro_olfactory": {
                "num_receptors": 30,
                "quantum_coherence_time": 100.0,
                "learning_rate": 0.02,
                "entanglement_strength": 0.5
            },
            "high_performance_bioneuro": {
                "num_receptors": 30,
                "cache_size": 500,
                "enable_parallel": True,
                "enable_auto_scaling": True
            },
            "qaoa": {
                "parameters": {
                    "layers": 2,
                    "max_iterations": 200,
                    "convergence_threshold": 0.001
                }
            }
        },
        parallel_execution=True,
        max_workers=4,
        generate_plots=True,
        generate_report=True
    )

    return config
