"""Quantum-inspired algorithms for task planning."""

from .qaoa_allocator import QAOAParameters, QAOAResourceAllocator
from .quantum_annealing import QuantumAnnealingScheduler
from .vqe_dependencies import VQEDependencyResolver

__all__ = [
    "QuantumAnnealingScheduler",
    "QAOAResourceAllocator",
    "QAOAParameters",
    "VQEDependencyResolver"
]
