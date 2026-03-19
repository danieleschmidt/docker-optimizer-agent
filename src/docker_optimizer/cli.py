"""CLI entry-point: docker-opt analyze <Dockerfile>"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from .optimizer import analyze_file
from .rules import Severity

console = Console()

SEVERITY_COLORS = {
    Severity.CRITICAL: "bold red",
    Severity.WARNING:  "bold yellow",
    Severity.INFO:     "cyan",
}

SEVERITY_ICONS = {
    Severity.CRITICAL: "🔴",
    Severity.WARNING:  "🟡",
    Severity.INFO:     "🔵",
}


def _score_color(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


@click.group()
@click.version_option(package_name="docker-optimizer-agent")
def main() -> None:
    """docker-opt — Dockerfile static analyser and optimization advisor."""


@main.command()
@click.argument("dockerfile", default="Dockerfile", type=click.Path(exists=True))
@click.option("--json", "output_json", is_flag=True, help="Output findings as JSON.")
@click.option(
    "--min-severity",
    type=click.Choice(["info", "warning", "critical"], case_sensitive=False),
    default="info",
    help="Only show findings at or above this severity.",
)
@click.option("--quiet", "-q", is_flag=True, help="Exit silently; only set exit code.")
def analyze(dockerfile: str, output_json: bool, min_severity: str, quiet: bool) -> None:
    """Analyze DOCKERFILE and report optimization suggestions.

    DOCKERFILE defaults to 'Dockerfile' in the current directory.
    """
    report = analyze_file(dockerfile)

    # Filter by severity
    order = [Severity.INFO, Severity.WARNING, Severity.CRITICAL]
    min_idx = order.index(Severity(min_severity))
    findings = [f for f in report.findings if order.index(f.severity) >= min_idx]

    if output_json:
        import json
        data = {
            "file": str(report.path),
            "score": report.score,
            "base_image": report.parsed.base_image,
            "total_instructions": len(report.parsed.instructions),
            "findings": [
                {
                    "rule_id": f.rule_id,
                    "severity": f.severity.value,
                    "line_no": f.line_no,
                    "title": f.title,
                    "detail": f.detail,
                    "suggestion": f.suggestion,
                }
                for f in findings
            ],
        }
        click.echo(json.dumps(data, indent=2))
        sys.exit(1 if report.critical else 0)

    if quiet:
        sys.exit(1 if report.critical else 0)

    # ── Header
    score = report.score
    score_str = Text(f"  Health score: {score}/100  ", style=_score_color(score))
    console.print()
    console.print(Panel(score_str, title=f"[bold]docker-opt[/bold] › {dockerfile}", expand=False))

    # ── Summary counts
    console.print(
        f"  [bold red]{len(report.critical)} critical[/bold red]  "
        f"[bold yellow]{len(report.warnings)} warning[/bold yellow]  "
        f"[cyan]{len(report.info)} info[/cyan]  "
        f"│  base image: [bold]{report.parsed.base_image or 'unknown'}[/bold]  "
        f"│  instructions: {len(report.parsed.instructions)}"
    )
    console.print()

    if not findings:
        console.print("[green]✅  No issues found — looks good![/green]\n")
        return

    # ── Findings table
    table = Table(box=box.SIMPLE_HEAD, show_footer=False, expand=True)
    table.add_column("Rule", style="dim", width=9)
    table.add_column("Sev", width=10)
    table.add_column("Line", width=5, justify="right")
    table.add_column("Title")

    for f in findings:
        icon = SEVERITY_ICONS[f.severity]
        color = SEVERITY_COLORS[f.severity]
        line_str = str(f.line_no) if f.line_no else "—"
        table.add_row(
            f.rule_id,
            Text(f"{icon} {f.severity.value}", style=color),
            line_str,
            f.title,
        )

    console.print(table)

    # ── Detail blocks
    for f in findings:
        color = SEVERITY_COLORS[f.severity]
        icon = SEVERITY_ICONS[f.severity]
        line_info = f"  line {f.line_no}" if f.line_no else ""
        console.rule(
            f"[{color}]{icon} {f.rule_id} — {f.title}[/{color}]{line_info}",
            style="dim",
        )
        console.print(f"  [dim]{f.detail}[/dim]")
        console.print(f"  [green]▶  {f.suggestion}[/green]")
        console.print()

    sys.exit(1 if report.critical else 0)
