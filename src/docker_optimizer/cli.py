"""Command-line interface for Docker Optimizer Agent."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List

import click

from .external_security import ExternalSecurityScanner
from .models import OptimizationResult
from .multistage import MultiStageOptimizer
from .optimizer import DockerfileOptimizer
from .performance import PerformanceOptimizer


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
@click.option(
    "--performance",
    is_flag=True,
    help="Enable performance optimizations (caching, parallel processing)"
)
@click.option(
    "--batch",
    multiple=True,
    help="Process multiple Dockerfiles (can be specified multiple times)"
)
@click.option(
    "--performance-report",
    is_flag=True,
    help="Show performance metrics after optimization"
)
def main(
    dockerfile: str,
    output: Optional[str],
    analysis_only: bool,
    format: str,
    verbose: bool,
    multistage: bool,
    security_scan: bool,
    performance: bool,
    batch: tuple,
    performance_report: bool,
) -> None:
    """Docker Optimizer Agent - Optimize Dockerfiles for security and size.

    This tool analyzes Dockerfiles and suggests optimizations for:
    - Security improvements (non-root users, specific versions)
    - Size reduction (multi-stage builds, layer optimization)
    - Best practices (cleanup commands, efficient package installation)
    - Performance optimizations (caching, parallel processing)
    """
    try:
        optimizer = DockerfileOptimizer()
        multistage_optimizer = MultiStageOptimizer()
        security_scanner = ExternalSecurityScanner()
        
        # Initialize performance optimizer if requested
        perf_optimizer = PerformanceOptimizer() if performance else None

        # Handle batch processing
        if batch:
            dockerfiles_to_process = list(batch)
        else:
            dockerfiles_to_process = [dockerfile]

        # Process single or multiple Dockerfiles
        if len(dockerfiles_to_process) == 1:
            # Single Dockerfile processing
            dockerfile_path = Path(dockerfiles_to_process[0])
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
            elif performance and perf_optimizer:
                # Performance-optimized processing
                result = perf_optimizer.optimize_with_performance(dockerfile_content)
                _output_result(result, output, format, verbose)
                
                if performance_report:
                    _output_performance_report(perf_optimizer.get_performance_report(), format)
            else:
                # Full optimization
                result = optimizer.optimize_dockerfile(dockerfile_content)
                _output_result(result, output, format, verbose)
        else:
            # Batch processing
            if performance and perf_optimizer:
                # Use async batch processing for performance
                asyncio.run(_process_batch_with_performance(
                    dockerfiles_to_process, perf_optimizer, output, format, verbose, performance_report
                ))
            else:
                # Regular batch processing
                _process_batch_regular(dockerfiles_to_process, optimizer, output, format, verbose)

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


async def _process_batch_with_performance(
    dockerfiles: List[str], 
    perf_optimizer: PerformanceOptimizer, 
    output_path: Optional[str], 
    format: str, 
    verbose: bool, 
    show_performance_report: bool
) -> None:
    """Process multiple Dockerfiles with performance optimization."""
    # Read all Dockerfile contents
    dockerfile_contents = []
    valid_paths = []
    
    for dockerfile_path in dockerfiles:
        path = Path(dockerfile_path)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            dockerfile_contents.append(content)
            valid_paths.append(dockerfile_path)
        else:
            click.echo(f"Warning: Dockerfile not found at {dockerfile_path}", err=True)
    
    if not dockerfile_contents:
        click.echo("Error: No valid Dockerfiles found for batch processing", err=True)
        sys.exit(1)
    
    # Process batch with performance optimization
    results = await perf_optimizer.optimize_multiple_with_performance(dockerfile_contents)
    
    # Output results
    for i, (dockerfile_path, result) in enumerate(zip(valid_paths, results)):
        click.echo(f"\n{'='*50}")
        click.echo(f"Results for: {dockerfile_path}")
        click.echo(f"{'='*50}")
        
        # Generate output path for this Dockerfile
        batch_output_path = None
        if output_path:
            base_path = Path(output_path)
            batch_output_path = str(base_path.parent / f"{base_path.stem}_{i+1}{base_path.suffix}")
        
        _output_result(result, batch_output_path, format, verbose)
    
    # Show performance report if requested
    if show_performance_report:
        click.echo(f"\n{'='*50}")
        click.echo("Performance Report")
        click.echo(f"{'='*50}")
        _output_performance_report(perf_optimizer.get_performance_report(), format)


def _process_batch_regular(
    dockerfiles: List[str], 
    optimizer: DockerfileOptimizer, 
    output_path: Optional[str], 
    format: str, 
    verbose: bool
) -> None:
    """Process multiple Dockerfiles with regular optimization."""
    for i, dockerfile_path in enumerate(dockerfiles):
        path = Path(dockerfile_path)
        if not path.exists():
            click.echo(f"Warning: Dockerfile not found at {dockerfile_path}", err=True)
            continue
        
        dockerfile_content = path.read_text(encoding="utf-8")
        result = optimizer.optimize_dockerfile(dockerfile_content)
        
        click.echo(f"\n{'='*50}")
        click.echo(f"Results for: {dockerfile_path}")
        click.echo(f"{'='*50}")
        
        # Generate output path for this Dockerfile
        batch_output_path = None
        if output_path:
            base_path = Path(output_path)
            batch_output_path = str(base_path.parent / f"{base_path.stem}_{i+1}{base_path.suffix}")
        
        _output_result(result, batch_output_path, format, verbose)


def _output_performance_report(performance_report: dict, format: str) -> None:
    """Output performance metrics report."""
    if format == "json":
        import json
        click.echo(json.dumps(performance_report, indent=2))
    elif format == "yaml":
        import yaml
        click.echo(yaml.dump(performance_report, default_flow_style=False))
    else:
        # Text format
        click.echo("⚡ Performance Metrics:")
        click.echo("-" * 25)
        click.echo(f"Processing Time: {performance_report['processing_time']:.2f}s")
        click.echo(f"Memory Usage: {performance_report['memory_usage_mb']:.1f}MB")
        click.echo(f"Dockerfiles Processed: {performance_report['dockerfiles_processed']}")
        click.echo(f"Cache Hits: {performance_report['cache_hits']}")
        click.echo(f"Cache Misses: {performance_report['cache_misses']}")
        click.echo(f"Cache Hit Ratio: {performance_report['cache_hit_ratio']:.1%}")
        click.echo(f"Cache Size: {performance_report['cache_size']}/{performance_report['cache_max_size']}")


if __name__ == "__main__":
    main()
