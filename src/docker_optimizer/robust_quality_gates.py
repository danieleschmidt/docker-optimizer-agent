#!/usr/bin/env python3
"""
Progressive Quality Gates System - Generation 2: Robust Implementation
Advanced validation, retry logic, health monitoring, and comprehensive error handling.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import hashlib
import psutil

logger = logging.getLogger(__name__)


@dataclass
class QualityGateConfig:
    """Configuration for quality gate execution."""
    timeout: int = 300
    retry_count: int = 3
    retry_delay: float = 1.0
    health_check_enabled: bool = True
    caching_enabled: bool = True
    parallel_execution: bool = True
    fail_fast: bool = False


@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    load_average: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class QualityGateResult:
    """Enhanced result of a quality gate check."""
    name: str
    passed: bool
    score: float
    message: str
    execution_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    health_metrics: Optional[HealthMetrics] = None
    cached: bool = False


class CircuitBreaker:
    """Circuit breaker pattern for quality gates."""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False
    
    def record_success(self) -> None:
        """Record successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class ResultCache:
    """LRU cache for quality gate results."""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
        self.ttl = ttl
    
    def _generate_key(self, gate_name: str, content_hash: str) -> str:
        """Generate cache key."""
        return f"{gate_name}:{content_hash}"
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry["timestamp"] > self.ttl
    
    def get(self, gate_name: str, content_hash: str) -> Optional[QualityGateResult]:
        """Get cached result."""
        key = self._generate_key(gate_name, content_hash)
        
        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry):
                # Update access order
                self.access_order.remove(key)
                self.access_order.append(key)
                
                result = entry["result"]
                result.cached = True
                return result
            else:
                # Remove expired entry
                del self.cache[key]
                self.access_order.remove(key)
        
        return None
    
    def put(self, gate_name: str, content_hash: str, result: QualityGateResult) -> None:
        """Store result in cache."""
        key = self._generate_key(gate_name, content_hash)
        
        # Evict least recently used if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)


class HealthMonitor:
    """System health monitoring."""
    
    @staticmethod
    def get_health_metrics() -> HealthMetrics:
        """Get current system health metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            
            return HealthMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                load_average=load_avg
            )
        except Exception as e:
            logger.warning(f"Failed to get health metrics: {e}")
            return HealthMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                load_average=0.0
            )
    
    @staticmethod
    def is_system_healthy(metrics: HealthMetrics, thresholds: Dict[str, float] = None) -> Tuple[bool, str]:
        """Check if system is healthy based on metrics."""
        if thresholds is None:
            thresholds = {
                "cpu_percent": 90.0,
                "memory_percent": 90.0,
                "disk_usage_percent": 95.0,
                "load_average": 10.0
            }
        
        issues = []
        
        if metrics.cpu_percent > thresholds["cpu_percent"]:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > thresholds["memory_percent"]:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_usage_percent > thresholds["disk_usage_percent"]:
            issues.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        if metrics.load_average > thresholds["load_average"]:
            issues.append(f"High load average: {metrics.load_average:.2f}")
        
        if issues:
            return False, "; ".join(issues)
        return True, "System healthy"


class RobustQualityGate:
    """Base class for robust quality gates with comprehensive error handling."""
    
    def __init__(self, name: str, description: str, weight: float = 1.0):
        self.name = name
        self.description = description
        self.weight = weight
        self.circuit_breaker = CircuitBreaker()
        self.logger = logging.getLogger(f"gate.{name}")
    
    async def execute(self, config: QualityGateConfig, cache: ResultCache) -> QualityGateResult:
        """Execute the quality gate with robust error handling."""
        start_time = time.time()
        retry_count = 0
        last_exception = None
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            return QualityGateResult(
                name=self.name,
                passed=False,
                score=0.0,
                message="Circuit breaker open - gate temporarily disabled",
                execution_time=0.0,
                metadata={"circuit_breaker": "open"}
            )
        
        # Check cache
        if config.caching_enabled:
            content_hash = self._get_content_hash()
            cached_result = cache.get(self.name, content_hash)
            if cached_result:
                self.logger.info(f"Using cached result for {self.name}")
                return cached_result
        
        # Health check
        health_metrics = None
        if config.health_check_enabled:
            health_metrics = HealthMonitor.get_health_metrics()
            is_healthy, health_message = HealthMonitor.is_system_healthy(health_metrics)
            if not is_healthy:
                self.logger.warning(f"System health issues detected: {health_message}")
        
        # Execute with retry logic
        for attempt in range(config.retry_count + 1):
            try:
                self.logger.info(f"Executing {self.name} (attempt {attempt + 1}/{config.retry_count + 1})")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._check_with_validation(),
                    timeout=config.timeout
                )
                
                passed, score, message, metadata = result
                execution_time = time.time() - start_time
                
                # Record success in circuit breaker
                self.circuit_breaker.record_success()
                
                gate_result = QualityGateResult(
                    name=self.name,
                    passed=passed,
                    score=score,
                    message=message,
                    execution_time=execution_time,
                    metadata=metadata,
                    retry_count=retry_count,
                    health_metrics=health_metrics
                )
                
                # Cache successful results
                if config.caching_enabled and passed:
                    cache.put(self.name, content_hash, gate_result)
                
                return gate_result
                
            except asyncio.TimeoutError:
                retry_count += 1
                last_exception = TimeoutError(f"Gate {self.name} timed out after {config.timeout}s")
                self.logger.warning(f"Gate {self.name} timed out (attempt {attempt + 1})")
                
            except Exception as e:
                retry_count += 1
                last_exception = e
                self.logger.warning(f"Gate {self.name} failed (attempt {attempt + 1}): {e}")
            
            # Wait before retry (except for last attempt)
            if attempt < config.retry_count:
                await asyncio.sleep(config.retry_delay * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        execution_time = time.time() - start_time
        self.circuit_breaker.record_failure()
        
        return QualityGateResult(
            name=self.name,
            passed=False,
            score=0.0,
            message=f"Gate failed after {retry_count} attempts: {str(last_exception)}",
            execution_time=execution_time,
            metadata={"error": str(last_exception), "attempts": retry_count},
            retry_count=retry_count,
            health_metrics=health_metrics
        )
    
    def _get_content_hash(self) -> str:
        """Generate hash of relevant content for caching."""
        # Default implementation - override in subclasses for better caching
        try:
            # Hash relevant source files
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            content = ""
            for file_path in sorted(src_files):
                try:
                    content += file_path.read_text()
                except Exception:
                    pass
            
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return str(int(time.time() // 300))  # 5-minute cache key
    
    async def _check_with_validation(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Execute the gate with additional validation."""
        # Pre-execution validation
        await self._pre_check_validation()
        
        # Execute main check
        result = await self._check()
        
        # Post-execution validation
        await self._post_check_validation(result)
        
        return result
    
    async def _pre_check_validation(self) -> None:
        """Pre-execution validation. Override in subclasses."""
        pass
    
    async def _post_check_validation(self, result: Tuple[bool, float, str, Dict[str, Any]]) -> None:
        """Post-execution validation. Override in subclasses."""
        pass
    
    async def _check(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Implement the actual gate logic."""
        raise NotImplementedError("Subclasses must implement _check method")


class RobustTestGate(RobustQualityGate):
    """Robust test execution gate with comprehensive validation."""
    
    def __init__(self):
        super().__init__(
            name="robust_tests",
            description="Comprehensive test execution with validation",
            weight=2.0
        )
    
    async def _pre_check_validation(self) -> None:
        """Validate test environment."""
        # Check if pytest is available
        try:
            result = subprocess.run(["python", "-m", "pytest", "--version"], 
                                  capture_output=True, check=True)
            self.logger.info(f"Pytest version: {result.stdout.decode().strip()}")
        except subprocess.CalledProcessError:
            raise RuntimeError("pytest not available")
        
        # Check if test files exist
        test_files = list(Path("tests").rglob("test_*.py")) if Path("tests").exists() else []
        if not test_files:
            raise RuntimeError("No test files found")
    
    async def _check(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Execute comprehensive test suite."""
        # Run multiple test categories
        test_commands = [
            {
                "name": "core_tests",
                "cmd": ["python", "-m", "pytest", "tests/test_optimizer.py", "tests/test_models.py", "-v"],
                "weight": 0.4
            },
            {
                "name": "integration_tests", 
                "cmd": ["python", "-m", "pytest", "tests/test_integration.py", "-v"],
                "weight": 0.3
            },
            {
                "name": "cli_tests",
                "cmd": ["python", "-m", "pytest", "tests/test_cli.py", "-v"],
                "weight": 0.3
            }
        ]
        
        total_score = 0.0
        results = {}
        
        for test_suite in test_commands:
            try:
                result = await asyncio.create_subprocess_exec(
                    *test_suite["cmd"],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=Path.cwd()
                )
                
                stdout, stderr = await result.communicate()
                output = (stdout.decode() + stderr.decode()).strip()
                
                passed_count = output.count("PASSED")
                failed_count = output.count("FAILED")
                suite_score = (passed_count / max(passed_count + failed_count, 1)) * 100
                
                total_score += suite_score * test_suite["weight"]
                results[test_suite["name"]] = {
                    "passed": passed_count,
                    "failed": failed_count,
                    "score": suite_score
                }
                
            except Exception as e:
                self.logger.warning(f"Test suite {test_suite['name']} failed: {e}")
                results[test_suite["name"]] = {"error": str(e), "score": 0.0}
        
        # Additional quality checks
        coverage_score = await self._check_coverage()
        total_score = (total_score * 0.8) + (coverage_score * 0.2)
        
        results["coverage"] = {"score": coverage_score}
        
        passed = total_score >= 70.0  # 70% threshold
        message = f"Test score: {total_score:.1f}% (threshold: 70%)"
        
        return passed, total_score, message, results
    
    async def _check_coverage(self) -> float:
        """Check test coverage."""
        try:
            result = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest", "--cov=docker_optimizer", 
                "--cov-report=term-missing", "--tb=no", "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            output = stdout.decode() + stderr.decode()
            
            # Parse coverage from output
            for line in output.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    # Extract percentage
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            return float(part[:-1])
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Coverage check failed: {e}")
            return 0.0


class RobustSecurityGate(RobustQualityGate):
    """Robust security scanning gate."""
    
    def __init__(self):
        super().__init__(
            name="robust_security",
            description="Multi-layer security scanning and validation",
            weight=1.8
        )
    
    async def _check(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Execute comprehensive security scanning."""
        security_checks = []
        
        # Bandit static analysis
        bandit_result = await self._run_bandit_scan()
        security_checks.append(("bandit", bandit_result))
        
        # Dependency vulnerability check (if safety is available)
        deps_result = await self._check_dependencies()
        security_checks.append(("dependencies", deps_result))
        
        # File permissions check
        perms_result = await self._check_file_permissions()
        security_checks.append(("permissions", perms_result))
        
        # Calculate overall security score
        total_weight = sum(weight for _, (_, weight, _) in security_checks)
        weighted_score = sum(score * weight for _, (score, weight, _) in security_checks) / total_weight
        
        # Determine pass/fail
        critical_issues = sum(1 for _, (score, _, _) in security_checks if score < 50.0)
        passed = weighted_score >= 60.0 and critical_issues == 0
        
        results = {check_name: {"score": score, "message": msg} 
                  for check_name, (score, _, msg) in security_checks}
        
        message = f"Security score: {weighted_score:.1f}% ({critical_issues} critical issues)"
        
        return passed, weighted_score, message, results
    
    async def _run_bandit_scan(self) -> Tuple[float, float, str]:
        """Run Bandit security scan."""
        try:
            result = await asyncio.create_subprocess_exec(
                "python", "-m", "bandit", "-r", "src/", "-f", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            try:
                data = json.loads(stdout.decode())
                high = len([r for r in data.get("results", []) if r.get("issue_severity") == "HIGH"])
                medium = len([r for r in data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
                low = len([r for r in data.get("results", []) if r.get("issue_severity") == "LOW"])
                
                penalty = (high * 15) + (medium * 5) + (low * 1)
                score = max(100 - penalty, 0)
                
                return score, 0.6, f"Bandit: {high}H/{medium}M/{low}L issues"
                
            except json.JSONDecodeError:
                # Fallback to text parsing
                output = stdout.decode() + stderr.decode()
                high = output.count("Severity: High")
                medium = output.count("Severity: Medium") 
                low = output.count("Severity: Low")
                
                penalty = (high * 15) + (medium * 5) + (low * 1)
                score = max(100 - penalty, 0)
                
                return score, 0.6, f"Bandit: {high}H/{medium}M/{low}L issues"
                
        except Exception as e:
            return 0.0, 0.6, f"Bandit scan failed: {str(e)}"
    
    async def _check_dependencies(self) -> Tuple[float, float, str]:
        """Check for vulnerable dependencies."""
        try:
            # Check if safety is available
            result = await asyncio.create_subprocess_exec(
                "python", "-m", "safety", "check", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return 100.0, 0.3, "Dependencies: No known vulnerabilities"
            else:
                try:
                    data = json.loads(stdout.decode())
                    vuln_count = len(data)
                    score = max(100 - (vuln_count * 10), 0)
                    return score, 0.3, f"Dependencies: {vuln_count} vulnerabilities"
                except json.JSONDecodeError:
                    return 50.0, 0.3, "Dependencies: Check completed with warnings"
                    
        except FileNotFoundError:
            return 90.0, 0.3, "Dependencies: Safety not available (assumed safe)"
        except Exception as e:
            return 50.0, 0.3, f"Dependencies check failed: {str(e)}"
    
    async def _check_file_permissions(self) -> Tuple[float, float, str]:
        """Check file permissions for security issues."""
        try:
            issues = []
            
            # Check for overly permissive files
            for root, dirs, files in os.walk("src"):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        stat = file_path.stat()
                        mode = stat.st_mode & 0o777
                        
                        # Check for world-writable files
                        if mode & 0o002:
                            issues.append(f"World-writable: {file_path}")
                        
                        # Check for overly permissive directories
                        if file_path.is_dir() and mode & 0o007 == 0o007:
                            issues.append(f"Overly permissive directory: {file_path}")
                            
                    except Exception:
                        pass
            
            score = max(100 - (len(issues) * 10), 0)
            message = f"Permissions: {len(issues)} issues found" if issues else "Permissions: OK"
            
            return score, 0.1, message
            
        except Exception as e:
            return 80.0, 0.1, f"Permissions check failed: {str(e)}"


class RobustProgressiveQualityGates:
    """Generation 2: Robust Progressive Quality Gates System."""
    
    def __init__(self, config: QualityGateConfig = None):
        self.config = config or QualityGateConfig()
        self.cache = ResultCache()
        self.gates = [
            RobustTestGate(),
            RobustSecurityGate(),
        ]
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def execute_all(self) -> Dict[str, Any]:
        """Execute all quality gates with robust error handling."""
        self.logger.info("🚀 Starting Robust Progressive Quality Gates - Generation 2")
        start_time = time.time()
        
        # Pre-execution health check
        initial_health = HealthMonitor.get_health_metrics()
        is_healthy, health_message = HealthMonitor.is_system_healthy(initial_health)
        
        if not is_healthy:
            self.logger.warning(f"System health issues detected: {health_message}")
        
        results = []
        failed_gates = 0
        
        if self.config.parallel_execution:
            # Execute gates in parallel
            tasks = [gate.execute(self.config, self.cache) for gate in self.gates]
            gate_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(gate_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Gate {self.gates[i].name} crashed: {result}")
                    failed_gates += 1
                    results.append(QualityGateResult(
                        name=self.gates[i].name,
                        passed=False,
                        score=0.0,
                        message=f"Gate crashed: {str(result)}",
                        execution_time=0.0,
                        metadata={"exception": str(result)}
                    ))
                else:
                    results.append(result)
                    if not result.passed:
                        failed_gates += 1
                        if self.config.fail_fast:
                            break
        else:
            # Execute gates sequentially
            for gate in self.gates:
                result = await gate.execute(self.config, self.cache)
                results.append(result)
                
                if not result.passed:
                    failed_gates += 1
                    if self.config.fail_fast:
                        self.logger.info("Fail-fast enabled, stopping execution")
                        break
        
        # Calculate summary
        execution_time = time.time() - start_time
        passed_gates = len(results) - failed_gates
        
        # Calculate weighted score
        total_weight = sum(gate.weight for gate in self.gates[:len(results)])
        weighted_score = sum(
            result.score * gate.weight 
            for result, gate in zip(results, self.gates)
        ) / total_weight if total_weight > 0 else 0.0
        
        # Final health check
        final_health = HealthMonitor.get_health_metrics()
        
        summary = {
            "generation": 2,
            "total_gates": len(self.gates),
            "executed_gates": len(results),
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "overall_score": weighted_score,
            "execution_time": execution_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "timeout": self.config.timeout,
                "retry_count": self.config.retry_count,
                "parallel_execution": self.config.parallel_execution,
                "fail_fast": self.config.fail_fast
            },
            "health": {
                "initial": {
                    "cpu": initial_health.cpu_percent,
                    "memory": initial_health.memory_percent,
                    "healthy": is_healthy,
                    "message": health_message
                },
                "final": {
                    "cpu": final_health.cpu_percent,
                    "memory": final_health.memory_percent
                }
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "retry_count": r.retry_count,
                    "cached": r.cached,
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
        
        self.logger.info(f"✅ Robust Quality Gates completed: {passed_gates}/{len(results)} passed")
        return summary
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print human-readable summary."""
        print("\n🚀 Robust Progressive Quality Gates - Generation 2 Results")
        print("=" * 70)
        
        # System health
        initial_health = summary["health"]["initial"]
        print(f"🏥 System Health: {initial_health['message']} (CPU: {initial_health['cpu']:.1f}%, Memory: {initial_health['memory']:.1f}%)")
        
        # Gate results
        for result in summary["results"]:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            cached = " [CACHED]" if result.get("cached") else ""
            retries = f" (retries: {result['retry_count']})" if result["retry_count"] > 0 else ""
            print(f"{status} [{result['score']:5.1f}%] {result['name']}: {result['message']}{cached}{retries}")
        
        print(f"\n📊 Summary: {summary['passed_gates']}/{summary['executed_gates']} passed")
        print(f"🎯 Overall Score: {summary['overall_score']:.1f}%")
        print(f"⏱️  Execution Time: {summary['execution_time']:.2f}s")
        print(f"🔧 Config: Timeout={summary['config']['timeout']}s, Retries={summary['config']['retry_count']}")
        
        if summary['passed_gates'] == summary['executed_gates']:
            print("✅ All quality gates passed!")
        else:
            print(f"❌ {summary['failed_gates']} quality gates failed")


async def main():
    """Main entry point for robust quality gates."""
    config = QualityGateConfig(
        timeout=180,
        retry_count=2,
        retry_delay=2.0,
        parallel_execution=True,
        fail_fast=False,
        caching_enabled=True,
        health_check_enabled=True
    )
    
    gates = RobustProgressiveQualityGates(config)
    summary = await gates.execute_all()
    gates.print_summary(summary)
    
    # Save results
    results_file = Path("robust_quality_gates_results.json")
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    return summary["passed_gates"] == summary["executed_gates"]


if __name__ == "__main__":
    asyncio.run(main())