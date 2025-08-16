"""Quantum-Inspired Task Planner - Intelligent task scheduling with quantum algorithms."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Graceful imports - only expose what's available
__all__ = []

# Track available features
_features = {}

try:
    from .core.planner import QuantumTaskPlanner
    __all__.append("QuantumTaskPlanner")
    _features['planner'] = True
except ImportError:
    _features['planner'] = False

try:
    from .algorithms.qaoa_allocator import QAOAResourceAllocator
    from .algorithms.quantum_annealing import QuantumAnnealingScheduler
    from .algorithms.vqe_dependencies import VQEDependencyResolver
    __all__.extend(["QuantumAnnealingScheduler", "QAOAResourceAllocator", "VQEDependencyResolver"])
    _features['algorithms'] = True
except ImportError:
    _features['algorithms'] = False

try:
    from .models.resource import Resource, ResourceType
    from .models.schedule import Schedule, ScheduleOptimization
    from .models.task import Task, TaskPriority, TaskStatus
    __all__.extend(["Task", "TaskStatus", "TaskPriority", "Resource", "ResourceType", "Schedule", "ScheduleOptimization"])
    _features['models'] = True
except ImportError:
    _features['models'] = False

def get_features():
    """Get available feature information."""
    return _features.copy()
