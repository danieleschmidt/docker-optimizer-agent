"""Docker Optimizer Agent - LLM suggests minimal, secure Dockerfiles."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .external_security import ExternalSecurityScanner
from .models import (
    BuildStage,
    MultiStageOptimization,
    OptimizationResult,
    OptimizationSuggestion,
    SecurityScore,
    SuggestionContext,
    VulnerabilityReport,
)
from .multistage import MultiStageOptimizer
from .optimizer import DockerfileOptimizer
from .realtime_suggestions import ProjectType, RealtimeSuggestionEngine

__all__ = [
    "DockerfileOptimizer",
    "OptimizationResult",
    "OptimizationSuggestion",
    "MultiStageOptimizer",
    "MultiStageOptimization",
    "BuildStage",
    "ExternalSecurityScanner",
    "VulnerabilityReport",
    "SecurityScore",
    "SuggestionContext",
    "RealtimeSuggestionEngine",
    "ProjectType",
]
