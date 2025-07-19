"""Docker Optimizer Agent - LLM suggests minimal, secure Dockerfiles."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .external_security import ExternalSecurityScanner
from .models import (
    BuildStage,
    MultiStageOptimization,
    OptimizationResult,
    SecurityScore,
    VulnerabilityReport,
)
from .multistage import MultiStageOptimizer
from .optimizer import DockerfileOptimizer

__all__ = [
    "DockerfileOptimizer",
    "OptimizationResult",
    "MultiStageOptimizer",
    "MultiStageOptimization",
    "BuildStage",
    "ExternalSecurityScanner",
    "VulnerabilityReport",
    "SecurityScore"
]
