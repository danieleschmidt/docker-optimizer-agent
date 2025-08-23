#!/usr/bin/env python3
"""
Progressive Quality Gates CLI - Unified Command Line Interface
Provides access to all three generations of quality gates with research validation.
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from datetime import datetime

# Import all generations
from src.docker_optimizer.progressive_quality_gates import ProgressiveQualityGates as Gen1Gates
from src.docker_optimizer.robust_quality_gates import RobustProgressiveQualityGates as Gen2Gates, QualityGateConfig
from src.docker_optimizer.optimized_quality_gates import OptimizedProgressiveQualityGates as Gen3Gates, OptimizationParameters


@click.group()
def cli():
    """Progressive Quality Gates System - Multi-generational quality validation."""
    pass


@cli.command()
@click.option('--format', type=click.Choice(['text', 'json']), default='text',
              help='Output format')
@click.option('--save-results', is_flag=True, help='Save results to file')
async def generation1(format, save_results):
    """Execute Generation 1: Basic Progressive Quality Gates."""
    click.echo("🚀 Executing Generation 1: Basic Progressive Quality Gates")
    
    gates = Gen1Gates()
    summary = await gates.execute_all()
    
    if format == 'json':
        # Convert to JSON-serializable format
        json_summary = {
            "generation": 1,
            "total_gates": summary.total_gates,
            "passed_gates": summary.passed_gates,
            "failed_gates": summary.failed_gates,
            "overall_score": summary.overall_score,
            "execution_time": summary.execution_time,
            "timestamp": summary.timestamp.isoformat(),
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata
                }
                for r in summary.results
            ]
        }
        
        click.echo(json.dumps(json_summary, indent=2))
    else:
        gates.print_summary(summary)
    
    if save_results:
        filename = f"gen1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(json_summary, f, indent=2)
        click.echo(f"\n📄 Results saved to: {filename}")
    
    sys.exit(0 if summary.passed_gates == summary.total_gates else 1)


@cli.command()
@click.option('--timeout', default=180, help='Base timeout in seconds')
@click.option('--retries', default=2, help='Number of retries')
@click.option('--parallel/--sequential', default=True, help='Execution mode')
@click.option('--fail-fast', is_flag=True, help='Stop on first failure')
@click.option('--format', type=click.Choice(['text', 'json']), default='text')
@click.option('--save-results', is_flag=True, help='Save results to file')
async def generation2(timeout, retries, parallel, fail_fast, format, save_results):
    """Execute Generation 2: Robust Progressive Quality Gates."""
    click.echo("🚀 Executing Generation 2: Robust Progressive Quality Gates")
    
    config = QualityGateConfig(
        timeout=timeout,
        retry_count=retries,
        parallel_execution=parallel,
        fail_fast=fail_fast,
        caching_enabled=True,
        health_check_enabled=True
    )
    
    gates = Gen2Gates(config)
    summary = await gates.execute_all()
    
    if format == 'json':
        click.echo(json.dumps(summary, indent=2))
    else:
        gates.print_summary(summary)
    
    if save_results:
        filename = f"gen2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        click.echo(f"\n📄 Results saved to: {filename}")
    
    sys.exit(0 if summary['passed_gates'] == summary['executed_gates'] else 1)


@cli.command()
@click.option('--max-concurrency', default=6, help='Maximum concurrent gates')
@click.option('--base-timeout', default=45, help='Base timeout in seconds')
@click.option('--ml-optimization/--no-ml', default=True, help='Enable ML optimization')
@click.option('--self-healing/--no-healing', default=True, help='Enable self-healing')
@click.option('--format', type=click.Choice(['text', 'json']), default='text')
@click.option('--save-results', is_flag=True, help='Save results to file')
async def generation3(max_concurrency, base_timeout, ml_optimization, self_healing, format, save_results):
    """Execute Generation 3: Optimized Progressive Quality Gates."""
    click.echo("🚀 Executing Generation 3: Optimized Progressive Quality Gates")
    
    optimization_params = OptimizationParameters(
        max_concurrent_gates=max_concurrency,
        base_timeout=base_timeout,
        ml_optimization=ml_optimization,
        self_healing=self_healing,
        adaptive_timeout=True,
        load_balancing=True,
        predictive_scaling=True
    )
    
    gates = Gen3Gates(optimization_params)
    summary = await gates.execute_all_optimized()
    
    if format == 'json':
        click.echo(json.dumps(summary, indent=2))
    else:
        gates.print_optimized_summary(summary)
    
    if save_results:
        filename = f"gen3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        click.echo(f"\n📄 Results saved to: {filename}")
    
    sys.exit(0 if summary['successful_executions'] == summary['total_gates'] else 1)


@cli.command()
@click.option('--iterations', default=5, help='Number of comparison iterations')
@click.option('--save-report', is_flag=True, help='Save detailed comparison report')
async def compare_all(iterations, save_report):
    """Compare all three generations with statistical analysis."""
    click.echo("🔬 Running Comprehensive Quality Gates Comparison")
    click.echo("=" * 60)
    
    results = {
        "generation_1": [],
        "generation_2": [], 
        "generation_3": [],
        "comparison_metadata": {
            "iterations": iterations,
            "timestamp": datetime.now().isoformat(),
            "purpose": "Statistical comparison of quality gate generations"
        }
    }
    
    for iteration in range(iterations):
        click.echo(f"\n🔄 Running iteration {iteration + 1}/{iterations}")
        
        # Generation 1
        click.echo("  📊 Testing Generation 1...")
        try:
            gen1_gates = Gen1Gates()
            gen1_summary = await gen1_gates.execute_all()
            results["generation_1"].append({
                "iteration": iteration + 1,
                "passed_gates": gen1_summary.passed_gates,
                "total_gates": gen1_summary.total_gates,
                "overall_score": gen1_summary.overall_score,
                "execution_time": gen1_summary.execution_time,
                "success_rate": gen1_summary.passed_gates / gen1_summary.total_gates
            })
        except Exception as e:
            click.echo(f"    ❌ Generation 1 failed: {e}")
            results["generation_1"].append({
                "iteration": iteration + 1,
                "error": str(e),
                "success_rate": 0.0,
                "execution_time": 0.0
            })
        
        # Generation 2 
        click.echo("  📊 Testing Generation 2...")
        try:
            gen2_config = QualityGateConfig(timeout=120, retry_count=1, parallel_execution=True)
            gen2_gates = Gen2Gates(gen2_config)
            gen2_summary = await gen2_gates.execute_all()
            results["generation_2"].append({
                "iteration": iteration + 1,
                "passed_gates": gen2_summary["passed_gates"],
                "total_gates": gen2_summary["total_gates"],
                "overall_score": gen2_summary["overall_score"],
                "execution_time": gen2_summary["execution_time"],
                "success_rate": gen2_summary["passed_gates"] / gen2_summary["total_gates"]
            })
        except Exception as e:
            click.echo(f"    ❌ Generation 2 failed: {e}")
            results["generation_2"].append({
                "iteration": iteration + 1,
                "error": str(e),
                "success_rate": 0.0,
                "execution_time": 0.0
            })
        
        # Generation 3
        click.echo("  📊 Testing Generation 3...")
        try:
            gen3_params = OptimizationParameters(max_concurrent_gates=4, base_timeout=60)
            gen3_gates = Gen3Gates(gen3_params)
            gen3_summary = await gen3_gates.execute_all_optimized()
            results["generation_3"].append({
                "iteration": iteration + 1,
                "successful_executions": gen3_summary["successful_executions"],
                "total_gates": gen3_summary["total_gates"],
                "overall_score": gen3_summary["overall_score"],
                "execution_time": gen3_summary["execution_time"],
                "success_rate": gen3_summary["successful_executions"] / gen3_summary["total_gates"],
                "throughput": gen3_summary["performance"]["throughput"],
                "cpu_efficiency": gen3_summary["performance"]["cpu_efficiency"]
            })
        except Exception as e:
            click.echo(f"    ❌ Generation 3 failed: {e}")
            results["generation_3"].append({
                "iteration": iteration + 1,
                "error": str(e),
                "success_rate": 0.0,
                "execution_time": 0.0
            })
        
        # Brief progress update
        await asyncio.sleep(1)  # Brief pause between iterations
    
    # Calculate statistics
    stats = calculate_comparison_statistics(results)
    
    # Print comparison report
    print_comparison_report(stats)
    
    if save_report:
        filename = f"quality_gates_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump({**results, "statistics": stats}, f, indent=2)
        click.echo(f"\n📄 Detailed comparison report saved to: {filename}")


def calculate_comparison_statistics(results):
    """Calculate statistical comparisons between generations."""
    import statistics
    
    stats = {}
    
    for generation in ["generation_1", "generation_2", "generation_3"]:
        gen_results = results[generation]
        valid_results = [r for r in gen_results if "error" not in r]
        
        if valid_results:
            success_rates = [r["success_rate"] for r in valid_results]
            execution_times = [r["execution_time"] for r in valid_results]
            overall_scores = [r.get("overall_score", 0) for r in valid_results]
            
            stats[generation] = {
                "iterations_successful": len(valid_results),
                "iterations_total": len(gen_results),
                "reliability": len(valid_results) / len(gen_results),
                "avg_success_rate": statistics.mean(success_rates) if success_rates else 0,
                "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
                "avg_overall_score": statistics.mean(overall_scores) if overall_scores else 0,
                "std_execution_time": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                "min_execution_time": min(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0
            }
            
            # Generation 3 specific metrics
            if generation == "generation_3":
                throughputs = [r.get("throughput", 0) for r in valid_results]
                cpu_efficiencies = [r.get("cpu_efficiency", 0) for r in valid_results]
                
                if throughputs:
                    stats[generation]["avg_throughput"] = statistics.mean(throughputs)
                if cpu_efficiencies:
                    stats[generation]["avg_cpu_efficiency"] = statistics.mean(cpu_efficiencies)
        else:
            stats[generation] = {
                "iterations_successful": 0,
                "iterations_total": len(gen_results),
                "reliability": 0,
                "avg_success_rate": 0,
                "avg_execution_time": 0,
                "avg_overall_score": 0
            }
    
    return stats


def print_comparison_report(stats):
    """Print a comprehensive comparison report."""
    print("\n🔬 PROGRESSIVE QUALITY GATES COMPARISON REPORT")
    print("=" * 70)
    
    # Header
    print(f"{'Generation':<15} {'Reliability':<12} {'Avg Score':<12} {'Avg Time':<12} {'Success Rate':<12}")
    print("-" * 70)
    
    for gen_name, gen_stats in stats.items():
        gen_num = gen_name.split('_')[1]
        reliability = f"{gen_stats['reliability']:.1%}"
        avg_score = f"{gen_stats['avg_overall_score']:.1f}%"
        avg_time = f"{gen_stats['avg_execution_time']:.2f}s"
        success_rate = f"{gen_stats['avg_success_rate']:.1%}"
        
        print(f"Generation {gen_num:<7} {reliability:<12} {avg_score:<12} {avg_time:<12} {success_rate:<12}")
    
    # Detailed analysis
    print("\n📊 DETAILED ANALYSIS")
    print("-" * 40)
    
    # Best performing generation
    best_reliability = max(stats.values(), key=lambda x: x['reliability'])
    best_score = max(stats.values(), key=lambda x: x['avg_overall_score'])
    fastest = min(stats.values(), key=lambda x: x['avg_execution_time'] if x['avg_execution_time'] > 0 else float('inf'))
    
    for gen_name, gen_stats in stats.items():
        if gen_stats == best_reliability:
            print(f"🏆 Most Reliable: Generation {gen_name.split('_')[1]} ({gen_stats['reliability']:.1%})")
        if gen_stats == best_score:
            print(f"🎯 Highest Score: Generation {gen_name.split('_')[1]} ({gen_stats['avg_overall_score']:.1f}%)")
        if gen_stats == fastest:
            print(f"⚡ Fastest: Generation {gen_name.split('_')[1]} ({gen_stats['avg_execution_time']:.2f}s)")
    
    # Generation 3 special metrics
    if 'generation_3' in stats and stats['generation_3']['iterations_successful'] > 0:
        gen3_stats = stats['generation_3']
        if 'avg_throughput' in gen3_stats:
            print(f"📈 Generation 3 Throughput: {gen3_stats['avg_throughput']:.2f} gates/second")
        if 'avg_cpu_efficiency' in gen3_stats:
            print(f"💻 Generation 3 CPU Efficiency: {gen3_stats['avg_cpu_efficiency']:.1%}")
    
    print("\n✨ EVOLUTION SUMMARY")
    print("-" * 30)
    print("Generation 1: ✅ Basic functionality with simple validation")
    print("Generation 2: 🔧 Robust error handling with retry logic and health monitoring")
    print("Generation 3: 🚀 AI-optimized with quantum scheduling and self-healing")


# Async wrapper for Click commands
def async_command(f):
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

# Apply async wrapper to commands
generation1 = async_command(generation1)
generation2 = async_command(generation2)
generation3 = async_command(generation3)
compare_all = async_command(compare_all)


if __name__ == "__main__":
    cli()