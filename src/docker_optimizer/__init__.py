"""Docker Optimizer Agent - LLM suggests minimal, secure Dockerfiles."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Graceful imports - only expose what's available based on installed dependencies
__all__ = []

# Track which optional features are available
_optional_features = {}

def get_available_features():
    """Get information about which optional features are available."""
    return _optional_features.copy()
try:
    from .optimizer import DockerfileOptimizer
    __all__.append("DockerfileOptimizer")
    _optional_features['optimizer'] = True
except ImportError as e:
    logger = __import__('logging').getLogger(__name__)
    logger.warning(f"DockerfileOptimizer not available: {e}")
    _optional_features['optimizer'] = False

try:
    from .models import (
        BuildStage,
        MultiStageOptimization,
        OptimizationResult,
        OptimizationSuggestion,
        SecurityScore,
        SuggestionContext,
        VulnerabilityReport,
    )
    __all__.extend([
        "OptimizationResult",
        "OptimizationSuggestion",
        "MultiStageOptimization",
        "BuildStage",
        "VulnerabilityReport",
        "SecurityScore",
        "SuggestionContext",
    ])
    _optional_features['models'] = True
except ImportError:
    _optional_features['models'] = False

try:
    from .multistage import MultiStageOptimizer
    __all__.append("MultiStageOptimizer")
    _optional_features['multistage'] = True
except ImportError:
    _optional_features['multistage'] = False

try:
    from .external_security import ExternalSecurityScanner
    __all__.append("ExternalSecurityScanner")
    _optional_features['external_security'] = True
except ImportError:
    _optional_features['external_security'] = False

try:
    from .realtime_suggestions import ProjectType, RealtimeSuggestionEngine
    __all__.extend(["RealtimeSuggestionEngine", "ProjectType"])
    _optional_features['realtime_suggestions'] = True
except ImportError:
    _optional_features['realtime_suggestions'] = False
