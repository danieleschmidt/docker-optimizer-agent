"""Command-line interface for Docker Optimizer Agent."""

import sys
from pathlib import Path
from typing import Optional

import click

from .external_security import ExternalSecurityScanner
from .models import OptimizationResult
from .multistage import MultiStageOptimizer
from .optimizer import DockerfileOptimizer


@click.command()
@click.option(
    "--dockerfile",
    "-f",
    type=click.Path(exists=True, readable=True),
    default="Dockerfile",
    help="Path to the Dockerfile to optimize (default: ./Dockerfile)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for optimized Dockerfile (default: stdout)",
)
@click.option(
    "--analysis-only",
    is_flag=True,
    help="Only analyze the Dockerfile without generating optimizations",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json", "yaml"]),
    default="text",
    help="Output format (default: text)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--multistage",
    is_flag=True,
    help="Generate multi-stage build optimization"
)
@click.option(
    "--security-scan",
    is_flag=True,
    help="Perform external security vulnerability scan"
)
def main(
    dockerfile: str,
    output: Optional[str],
    analysis_only: bool,
    format: str,
    verbose: bool,
    multistage: bool,
    security_scan: bool,
) -> None:
    """Docker Optimizer Agent - Optimize Dockerfiles for security and size.

    This tool analyzes Dockerfiles and suggests optimizations for:
    - Security improvements (non-root users, specific versions)
    - Size reduction (multi-stage builds, layer optimization)
    - Best practices (cleanup commands, efficient package installation)
    """
    try:
        optimizer = DockerfileOptimizer()
        multistage_optimizer = MultiStageOptimizer()
        security_scanner = ExternalSecurityScanner()

        # Read Dockerfile
        dockerfile_path = Path(dockerfile)
        if not dockerfile_path.exists():
            click.echo(f"Error: Dockerfile not found at {dockerfile_path}", err=True)
            sys.exit(1)

        dockerfile_content = dockerfile_path.read_text(encoding="utf-8")

        if analysis_only:
            # Only analyze, don't optimize
            analysis = optimizer.analyze_dockerfile(dockerfile_content)
            _output_analysis(analysis, format, verbose)
        elif multistage:
            # Multi-stage optimization
            result = multistage_optimizer.generate_multistage_dockerfile(dockerfile_content)
            _output_multistage_result(result, output, format, verbose)
        elif security_scan:
            # External security vulnerability scan
            vulnerability_report = security_scanner.scan_dockerfile_for_vulnerabilities(dockerfile_content)
            security_score = security_scanner.calculate_security_score(vulnerability_report)
            suggestions = security_scanner.suggest_security_improvements(vulnerability_report)
            _output_security_scan_result(vulnerability_report, security_score, suggestions, output, format, verbose)
        else:
            # Full optimization
            result = optimizer.optimize_dockerfile(dockerfile_content)
            _output_result(result, output, format, verbose)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def _output_analysis(analysis, format: str, verbose: bool) -> None:
    """Output analysis results."""
    if format == "json":
        import json

        click.echo(json.dumps(analysis.dict(), indent=2))
    elif format == "yaml":
        import yaml

        click.echo(yaml.dump(analysis.dict(), default_flow_style=False))
    else:
        # Text format
        click.echo("🔍 Dockerfile Analysis Results")
        click.echo("=" * 40)
        click.echo(f"Base Image: {analysis.base_image}")
        click.echo(f"Total Layers: {analysis.total_layers}")

        if analysis.estimated_size:
            click.echo(f"Estimated Size: {analysis.estimated_size}")

        if analysis.has_security_issues:
            click.echo("\n🚨 Security Issues Found:")
            for i, issue in enumerate(analysis.security_issues, 1):
                click.echo(f"  {i}. {issue}")
        else:
            click.echo("\n✅ No security issues found")

        if analysis.has_optimization_opportunities:
            click.echo("\n🎯 Optimization Opportunities:")
            for i, opp in enumerate(analysis.optimization_opportunities, 1):
                click.echo(f"  {i}. {opp}")
        else:
            click.echo("\n✅ No obvious optimization opportunities")


def _output_result(
    result: OptimizationResult, output_path: Optional[str], format: str, verbose: bool
) -> None:
    """Output optimization results."""
    if format == "json":
        import json

        output_content = json.dumps(result.dict(), indent=2)
    elif format == "yaml":
        import yaml

        output_content = yaml.dump(result.dict(), default_flow_style=False)
    else:
        # Text format - show summary and optimized Dockerfile
        summary_lines = [
            "🚀 Docker Optimization Results",
            "=" * 40,
            f"Original Size: {result.original_size}",
            f"Optimized Size: {result.optimized_size}",
            f"Explanation: {result.explanation}",
        ]

        if result.has_security_improvements:
            summary_lines.append(
                f"\n🔒 Security Fixes Applied: {len(result.security_fixes)}"
            )
            if verbose:
                for fix in result.security_fixes:
                    summary_lines.append(f"  • {fix.description} ({fix.severity})")

        if result.has_layer_optimizations:
            summary_lines.append(
                f"\n⚡ Layer Optimizations: {len(result.layer_optimizations)}"
            )
            if verbose:
                for opt in result.layer_optimizations:
                    summary_lines.append(f"  • {opt.reasoning}")

        summary_lines.extend(
            ["\n📄 Optimized Dockerfile:", "-" * 30, result.optimized_dockerfile]
        )

        output_content = "\n".join(summary_lines)

    # Output to file or stdout
    if output_path:
        Path(output_path).write_text(output_content, encoding="utf-8")
        click.echo(f"✅ Optimized Dockerfile written to {output_path}")
    else:
        click.echo(output_content)


def _output_multistage_result(result, output_path: Optional[str], format: str, verbose: bool) -> None:
    """Output multi-stage optimization results."""
    if format == "json":
        import json
        output_content = json.dumps(result.dict(), indent=2)
    elif format == "yaml":
        import yaml
        output_content = yaml.dump(result.dict(), default_flow_style=False)
    else:
        # Text format
        summary_lines = [
            "🚀 Multi-Stage Build Optimization Results",
            "=" * 45,
            f"Estimated Size Reduction: {result.estimated_size_reduction}MB",
            f"Security Improvements: {result.security_improvements}",
            f"Number of Stages: {len(result.stages)}",
            f"Explanation: {result.explanation}",
        ]

        if result.has_multiple_stages:
            summary_lines.append("\n📋 Build Stages:")
            for i, stage in enumerate(result.stages, 1):
                summary_lines.append(f"  {i}. {stage.name} ({stage.purpose}) - {stage.base_image}")

        summary_lines.extend([
            "\n📄 Optimized Multi-Stage Dockerfile:",
            "-" * 40,
            result.optimized_dockerfile
        ])

        output_content = "\n".join(summary_lines)

    # Output to file or stdout
    if output_path:
        Path(output_path).write_text(output_content, encoding="utf-8")
        click.echo(f"✅ Multi-stage Dockerfile written to {output_path}")
    else:
        click.echo(output_content)


def _output_security_scan_result(vulnerability_report, security_score, suggestions, output_path: Optional[str], format: str, verbose: bool) -> None:
    """Output security scan results."""
    if format == "json":
        import json
        output_data = {
            "vulnerability_report": vulnerability_report.dict(),
            "security_score": security_score.dict(),
            "suggestions": suggestions
        }
        output_content = json.dumps(output_data, indent=2)
    elif format == "yaml":
        import yaml
        output_data = {
            "vulnerability_report": vulnerability_report.dict(),
            "security_score": security_score.dict(),
            "suggestions": suggestions
        }
        output_content = yaml.dump(output_data, default_flow_style=False)
    else:
        # Text format
        summary_lines = [
            "🔒 Security Vulnerability Scan Results",
            "=" * 42,
            f"Security Score: {security_score.score}/100 (Grade: {security_score.grade})",
            f"Total Vulnerabilities: {vulnerability_report.total_vulnerabilities}",
        ]

        if vulnerability_report.total_vulnerabilities > 0:
            summary_lines.extend([
                f"  Critical: {vulnerability_report.critical_count}",
                f"  High: {vulnerability_report.high_count}",
                f"  Medium: {vulnerability_report.medium_count}",
                f"  Low: {vulnerability_report.low_count}",
            ])

        summary_lines.append(f"\nAnalysis: {security_score.analysis}")

        if vulnerability_report.cve_details and verbose:
            summary_lines.append("\n🚨 Top Vulnerabilities:")
            for i, cve in enumerate(vulnerability_report.cve_details[:5], 1):
                summary_lines.append(f"  {i}. {cve.cve_id} ({cve.severity}) - {cve.package}")
                if cve.fixed_version:
                    summary_lines.append(f"     Fix: Update to {cve.fixed_version}")

        if suggestions:
            summary_lines.append("\n💡 Security Recommendations:")
            for i, suggestion in enumerate(suggestions, 1):
                summary_lines.append(f"  {i}. {suggestion}")

        if security_score.recommendations:
            summary_lines.append("\n📋 General Recommendations:")
            for i, rec in enumerate(security_score.recommendations, 1):
                summary_lines.append(f"  {i}. {rec}")

        output_content = "\n".join(summary_lines)

    # Output to file or stdout
    if output_path:
        Path(output_path).write_text(output_content, encoding="utf-8")
        click.echo(f"✅ Security scan results written to {output_path}")
    else:
        click.echo(output_content)


if __name__ == "__main__":
    main()
