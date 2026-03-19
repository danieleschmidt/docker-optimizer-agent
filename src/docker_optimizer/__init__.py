"""docker-optimizer-agent — Dockerfile static analyser."""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"

from .optimizer import Report, analyze, analyze_file
from .parser import ParsedDockerfile, parse
from .rules import Finding, Severity, run_all

__all__ = [
    "analyze",
    "analyze_file",
    "parse",
    "run_all",
    "Report",
    "ParsedDockerfile",
    "Finding",
    "Severity",
]
