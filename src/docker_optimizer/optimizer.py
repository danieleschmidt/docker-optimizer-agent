"""High-level API — parse + run rules and return a structured report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .parser import ParsedDockerfile, parse
from .rules import Finding, Severity, run_all


@dataclass
class Report:
    """Analysis report for a Dockerfile."""

    path: Optional[str]
    parsed: ParsedDockerfile
    findings: List[Finding] = field(default_factory=list)

    @property
    def critical(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def info(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == Severity.INFO]

    @property
    def has_issues(self) -> bool:
        return bool(self.findings)

    @property
    def score(self) -> int:
        """0-100 health score (100 = no issues).

        Scoring weights:
          critical: -20 each
          warning:  -10 each
          info:     -3  each
        """
        deductions = len(self.critical) * 20 + len(self.warnings) * 10 + len(self.info) * 3
        return max(0, 100 - deductions)


def analyze(content: str, path: Optional[str] = None) -> Report:
    """Parse *content* and return a populated Report."""
    parsed = parse(content)
    findings = run_all(parsed)
    return Report(path=path, parsed=parsed, findings=findings)


def analyze_file(dockerfile_path: str | Path) -> Report:
    """Read *dockerfile_path* from disk and return a populated Report."""
    p = Path(dockerfile_path)
    content = p.read_text(encoding="utf-8")
    return analyze(content, path=str(p))
