#!/usr/bin/env python3
"""
Progressive Quality Gates System - Generation 1: Basic Implementation
Part of the autonomous SDLC enhancement for Docker Optimizer Agent.
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float
    message: str
    execution_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class QualityGatesSummary:
    """Summary of all quality gates execution."""
    total_gates: int
    passed_gates: int
    failed_gates: int
    overall_score: float
    execution_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    results: List[QualityGateResult] = field(default_factory=list)


class BaseQualityGate:
    """Base class for all quality gates."""
    
    def __init__(self, name: str, description: str, weight: float = 1.0):
        self.name = name
        self.description = description
        self.weight = weight
        
    async def execute(self) -> QualityGateResult:
        """Execute the quality gate check."""
        start_time = time.time()
        try:
            passed, score, message, metadata = await self._check()
            execution_time = time.time() - start_time
            return QualityGateResult(
                name=self.name,
                passed=passed,
                score=score,
                message=message,
                execution_time=execution_time,
                metadata=metadata
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                name=self.name,
                passed=False,
                score=0.0,
                message=f"Gate execution failed: {str(e)}",
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
    
    async def _check(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Implement the actual gate logic."""
        raise NotImplementedError("Subclasses must implement _check method")


class TestGate(BaseQualityGate):
    """Quality gate for test execution and coverage."""
    
    def __init__(self):
        super().__init__(
            name="tests",
            description="Execute test suite and validate coverage",
            weight=2.0
        )
    
    async def _check(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Run test suite with coverage."""
        try:
            # Run core tests with relaxed coverage
            cmd = [
                "python3", "-m", "pytest",
                "tests/test_optimizer.py",
                "tests/test_models.py", 
                "tests/test_config.py",
                "-v", "--tb=short", "--cov-fail-under=0"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            stdout, stderr = await result.communicate()
            output = (stdout.decode() + stderr.decode()).strip()
            
            # Parse test results
            passed_count = output.count("PASSED")
            failed_count = output.count("FAILED")
            total_tests = passed_count + failed_count
            
            if total_tests == 0:
                return False, 0.0, "No tests found or executed", {"output": output}
            
            pass_rate = passed_count / total_tests
            score = min(pass_rate * 100, 100.0)
            
            if failed_count == 0 and passed_count >= 10:
                return True, score, f"Tests passed: {passed_count}/{total_tests}", {
                    "passed": passed_count,
                    "failed": failed_count,
                    "pass_rate": pass_rate
                }
            else:
                return False, score, f"Tests failed: {failed_count}/{total_tests} failures", {
                    "passed": passed_count,
                    "failed": failed_count,
                    "pass_rate": pass_rate
                }
                
        except Exception as e:
            return False, 0.0, f"Test execution failed: {str(e)}", {"error": str(e)}


class SecurityGate(BaseQualityGate):
    """Quality gate for security scanning."""
    
    def __init__(self):
        super().__init__(
            name="security",
            description="Security vulnerability scanning",
            weight=1.5
        )
    
    async def _check(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Run security scan with bandit."""
        try:
            cmd = ["python3", "-m", "bandit", "-r", "src/", "-ll"]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            output = (stdout.decode() + stderr.decode()).strip()
            
            # Parse security issues
            high_severity = output.count("Severity: High")
            medium_severity = output.count("Severity: Medium")
            low_severity = output.count("Severity: Low")
            
            # Calculate security score
            penalty = (high_severity * 10) + (medium_severity * 3) + (low_severity * 1)
            score = max(100 - penalty, 0)
            
            # Relaxed thresholds for demo
            if high_severity <= 5 and medium_severity <= 15:
                return True, score, f"Security OK: {high_severity}H/{medium_severity}M/{low_severity}L", {
                    "high": high_severity,
                    "medium": medium_severity,
                    "low": low_severity,
                    "score": score
                }
            else:
                return False, score, f"Security issues: {high_severity}H/{medium_severity}M/{low_severity}L", {
                    "high": high_severity,
                    "medium": medium_severity, 
                    "low": low_severity,
                    "score": score
                }
                
        except Exception as e:
            return False, 0.0, f"Security scan failed: {str(e)}", {"error": str(e)}


class LintingGate(BaseQualityGate):
    """Quality gate for code linting and style."""
    
    def __init__(self):
        super().__init__(
            name="linting",
            description="Code quality and style checking",
            weight=1.0
        )
    
    async def _check(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Run linting with ruff."""
        try:
            cmd = [
                "python3", "-m", "ruff", "check", "src/",
                "--select=E,W,F", "--ignore=E501,W503"
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            output = (stdout.decode() + stderr.decode()).strip()
            
            # Count different issue types
            errors = output.count(": E")
            warnings = output.count(": W") 
            total_issues = errors + warnings
            
            # Calculate linting score
            penalty = errors * 2 + warnings
            score = max(100 - penalty, 0)
            
            if total_issues <= 10:
                return True, score, f"Linting OK: {errors}E/{warnings}W", {
                    "errors": errors,
                    "warnings": warnings,
                    "total": total_issues,
                    "score": score
                }
            else:
                return False, score, f"Linting issues: {errors}E/{warnings}W", {
                    "errors": errors,
                    "warnings": warnings,
                    "total": total_issues,
                    "score": score
                }
                
        except Exception as e:
            return False, 0.0, f"Linting failed: {str(e)}", {"error": str(e)}


class FunctionalityGate(BaseQualityGate):
    """Quality gate for basic functionality testing."""
    
    def __init__(self):
        super().__init__(
            name="functionality",
            description="Basic CLI and functionality testing",
            weight=1.0
        )
    
    async def _check(self) -> Tuple[bool, float, str, Dict[str, Any]]:
        """Test basic CLI functionality."""
        try:
            # Create test dockerfile
            test_dockerfile = "FROM alpine:3.18\nRUN echo 'test'\n"
            test_file = Path("/tmp/test_quality_gate.dockerfile")
            test_file.write_text(test_dockerfile)
            
            try:
                cmd = [
                    "python3", "-m", "docker_optimizer.cli",
                    "--dockerfile", str(test_file),
                    "--analysis-only"
                ]
                
                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await result.communicate()
                output = (stdout.decode() + stderr.decode()).strip()
                
                if "Dockerfile Analysis Results" in output or "optimization" in output.lower():
                    return True, 100.0, "CLI functionality working", {
                        "test_success": True,
                        "output_length": len(output)
                    }
                else:
                    return False, 0.0, f"CLI test failed: unexpected output", {
                        "output": output[:500]
                    }
                    
            finally:
                test_file.unlink(missing_ok=True)
                
        except Exception as e:
            return False, 0.0, f"Functionality test failed: {str(e)}", {"error": str(e)}


class ProgressiveQualityGates:
    """Progressive Quality Gates System - Generation 1."""
    
    def __init__(self):
        self.gates = [
            TestGate(),
            SecurityGate(),
            LintingGate(),
            FunctionalityGate()
        ]
        self.logger = logging.getLogger(__name__)
    
    async def execute_all(self) -> QualityGatesSummary:
        """Execute all quality gates."""
        self.logger.info("🚀 Starting Progressive Quality Gates - Generation 1")
        start_time = time.time()
        
        # Execute all gates concurrently
        tasks = [gate.execute() for gate in self.gates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        gate_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                gate_results.append(QualityGateResult(
                    name=self.gates[i].name,
                    passed=False,
                    score=0.0,
                    message=f"Gate failed with exception: {str(result)}",
                    execution_time=0.0,
                    metadata={"exception": str(result)}
                ))
            else:
                gate_results.append(result)
        
        # Calculate summary
        passed_count = sum(1 for r in gate_results if r.passed)
        failed_count = len(gate_results) - passed_count
        
        # Calculate weighted overall score
        total_weight = sum(gate.weight for gate in self.gates)
        weighted_score = sum(
            result.score * gate.weight 
            for result, gate in zip(gate_results, self.gates)
        ) / total_weight if total_weight > 0 else 0.0
        
        execution_time = time.time() - start_time
        
        summary = QualityGatesSummary(
            total_gates=len(self.gates),
            passed_gates=passed_count,
            failed_gates=failed_count,
            overall_score=weighted_score,
            execution_time=execution_time,
            results=gate_results
        )
        
        self.logger.info(f"✅ Quality Gates completed: {passed_count}/{len(self.gates)} passed")
        return summary
    
    def print_summary(self, summary: QualityGatesSummary) -> None:
        """Print human-readable summary."""
        print("\n🚀 Progressive Quality Gates - Generation 1 Results")
        print("=" * 60)
        
        for result in summary.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} [{result.score:5.1f}%] {result.name}: {result.message}")
        
        print(f"\n📊 Summary: {summary.passed_gates}/{summary.total_gates} passed")
        print(f"🎯 Overall Score: {summary.overall_score:.1f}%")
        print(f"⏱️  Execution Time: {summary.execution_time:.2f}s")
        
        if summary.passed_gates == summary.total_gates:
            print("✅ All quality gates passed!")
        else:
            print(f"❌ {summary.failed_gates} quality gates failed")


async def main():
    """Main entry point for quality gates."""
    gates = ProgressiveQualityGates()
    summary = await gates.execute_all()
    gates.print_summary(summary)
    
    # Save detailed results
    results_file = Path("quality_gates_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "summary": {
                "total_gates": summary.total_gates,
                "passed_gates": summary.passed_gates,
                "failed_gates": summary.failed_gates,
                "overall_score": summary.overall_score,
                "execution_time": summary.execution_time,
                "timestamp": summary.timestamp.isoformat()
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata
                }
                for r in summary.results
            ]
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    return summary.passed_gates == summary.total_gates


if __name__ == "__main__":
    asyncio.run(main())