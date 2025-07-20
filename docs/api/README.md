# API Documentation

This directory contains comprehensive API documentation for the Docker Optimizer Agent.

## Core Modules

### DockerfileOptimizer (`docker_optimizer.optimizer`)
The main optimization engine that analyzes and optimizes Dockerfiles.

#### Class: `DockerfileOptimizer`

**Methods:**

- `analyze_dockerfile(content: str) -> Dict[str, Any]`
  - Analyzes a Dockerfile and returns optimization opportunities
  - **Parameters:**
    - `content`: The Dockerfile content as a string
  - **Returns:** Dictionary containing analysis results
  - **Example:**
    ```python
    optimizer = DockerfileOptimizer()
    analysis = optimizer.analyze_dockerfile("FROM ubuntu:latest\nRUN apt-get update")
    ```

- `optimize_dockerfile(content: str) -> OptimizationResult`
  - Performs full optimization of a Dockerfile
  - **Parameters:**
    - `content`: The Dockerfile content as a string
  - **Returns:** `OptimizationResult` object with optimization details
  - **Example:**
    ```python
    result = optimizer.optimize_dockerfile(dockerfile_content)
    print(f"Size reduction: {result.original_size} → {result.optimized_size}")
    ```

### MultiStageOptimizer (`docker_optimizer.multistage`)
Specialized optimizer for creating multi-stage builds.

#### Class: `MultiStageOptimizer`

**Methods:**

- `generate_multistage_dockerfile(content: str) -> MultiStageResult`
  - Converts single-stage Dockerfile to optimized multi-stage build
  - **Parameters:**
    - `content`: Original Dockerfile content
  - **Returns:** `MultiStageResult` with optimized multi-stage Dockerfile
  - **Example:**
    ```python
    multistage = MultiStageOptimizer()
    result = multistage.generate_multistage_dockerfile(content)
    print(result.optimized_dockerfile)
    ```

- `identify_build_dependencies(content: str) -> List[str]`
  - Identifies build-time dependencies that can be separated
  - **Parameters:**
    - `content`: Dockerfile content
  - **Returns:** List of build dependency commands
  - **Example:**
    ```python
    build_deps = multistage.identify_build_dependencies(content)
    ```

### SecurityAnalyzer (`docker_optimizer.security`)
Analyzes Dockerfiles for security vulnerabilities and best practices.

#### Class: `SecurityAnalyzer`

**Methods:**

- `analyze_security(content: str) -> List[SecurityFix]`
  - Performs security analysis of Dockerfile
  - **Parameters:**
    - `content`: Dockerfile content
  - **Returns:** List of `SecurityFix` objects
  - **Example:**
    ```python
    analyzer = SecurityAnalyzer()
    fixes = analyzer.analyze_security(content)
    for fix in fixes:
        print(f"{fix.severity}: {fix.description}")
    ```

### ExternalSecurityScanner (`docker_optimizer.external_security`)
Integration with external security scanning tools like Trivy.

#### Class: `ExternalSecurityScanner`

**Methods:**

- `scan_dockerfile_for_vulnerabilities(content: str) -> VulnerabilityReport`
  - Scans Dockerfile using external tools for vulnerabilities
  - **Parameters:**
    - `content`: Dockerfile content
  - **Returns:** `VulnerabilityReport` with vulnerability details
  - **Example:**
    ```python
    scanner = ExternalSecurityScanner()
    report = scanner.scan_dockerfile_for_vulnerabilities(content)
    print(f"Total vulnerabilities: {report.total_vulnerabilities}")
    ```

- `calculate_security_score(report: VulnerabilityReport) -> SecurityScore`
  - Calculates a security score (A-F grade) based on vulnerabilities
  - **Parameters:**
    - `report`: VulnerabilityReport from scanning
  - **Returns:** `SecurityScore` with letter grade and analysis
  - **Example:**
    ```python
    score = scanner.calculate_security_score(report)
    print(f"Security grade: {score.grade}")
    ```

### PerformanceOptimizer (`docker_optimizer.performance`)
Provides performance optimization with caching and parallel processing.

#### Class: `PerformanceOptimizer`

**Methods:**

- `optimize_with_performance(content: str) -> OptimizationResult`
  - Optimizes Dockerfile with performance enhancements
  - **Parameters:**
    - `content`: Dockerfile content
  - **Returns:** `OptimizationResult` with caching and performance metrics
  - **Example:**
    ```python
    perf_optimizer = PerformanceOptimizer()
    result = perf_optimizer.optimize_with_performance(content)
    ```

- `optimize_multiple_with_performance(contents: List[str]) -> List[OptimizationResult]`
  - Processes multiple Dockerfiles in parallel with caching
  - **Parameters:**
    - `contents`: List of Dockerfile contents
  - **Returns:** List of `OptimizationResult` objects
  - **Example:**
    ```python
    results = await perf_optimizer.optimize_multiple_with_performance([
        dockerfile1_content,
        dockerfile2_content
    ])
    ```

## Data Models

### OptimizationResult
Main result object containing optimization details.

**Attributes:**
- `original_size: str` - Estimated original image size
- `optimized_size: str` - Estimated optimized image size  
- `explanation: str` - Human-readable explanation of optimizations
- `optimized_dockerfile: str` - The optimized Dockerfile content
- `security_fixes: List[SecurityFix]` - Applied security fixes

### SecurityFix
Represents a security issue and its fix.

**Attributes:**
- `description: str` - Description of the security issue
- `severity: str` - Severity level (LOW, MEDIUM, HIGH, CRITICAL)
- `fix_suggestion: str` - Suggested fix for the issue

### VulnerabilityReport
Contains vulnerability scan results.

**Attributes:**
- `total_vulnerabilities: int` - Total number of vulnerabilities
- `critical_count: int` - Number of critical vulnerabilities
- `high_count: int` - Number of high severity vulnerabilities
- `medium_count: int` - Number of medium severity vulnerabilities
- `low_count: int` - Number of low severity vulnerabilities
- `cve_details: List[CVEDetail]` - Detailed CVE information

### SecurityScore
Security scoring result.

**Attributes:**
- `grade: str` - Letter grade (A, B, C, D, F)
- `score: int` - Numeric score (0-100)
- `analysis: str` - Analysis explanation
- `recommendations: List[str]` - Security recommendations

## CLI Module (`docker_optimizer.cli`)

The command-line interface provides easy access to all optimization features.

### Usage Examples

**Basic optimization:**
```bash
docker-optimizer --dockerfile Dockerfile --output Dockerfile.optimized
```

**Security scanning:**
```bash
docker-optimizer --dockerfile Dockerfile --security-scan --format json
```

**Performance optimization:**
```bash
docker-optimizer --dockerfile Dockerfile --performance --performance-report
```

**Batch processing:**
```bash
docker-optimizer --batch Dockerfile1 --batch Dockerfile2 --performance
```

### Command Line Arguments

- `--dockerfile, -f`: Path to Dockerfile (default: ./Dockerfile)
- `--output, -o`: Output path for optimized Dockerfile
- `--analysis-only`: Only analyze without optimizing
- `--format`: Output format (text, json, yaml)
- `--verbose, -v`: Enable verbose output
- `--multistage`: Generate multi-stage build optimization
- `--security-scan`: Perform external security vulnerability scan
- `--performance`: Enable performance optimizations
- `--batch`: Process multiple Dockerfiles
- `--performance-report`: Show performance metrics

## Integration Examples

### Python API Usage

```python
from docker_optimizer import DockerfileOptimizer, MultiStageOptimizer
from docker_optimizer.external_security import ExternalSecurityScanner

# Basic optimization
optimizer = DockerfileOptimizer()
with open('Dockerfile', 'r') as f:
    content = f.read()

result = optimizer.optimize_dockerfile(content)
print(f"Optimization: {result.explanation}")

# Multi-stage optimization
multistage = MultiStageOptimizer()
ms_result = multistage.generate_multistage_dockerfile(content)
print(ms_result.optimized_dockerfile)

# Security scanning
scanner = ExternalSecurityScanner()
vuln_report = scanner.scan_dockerfile_for_vulnerabilities(content)
security_score = scanner.calculate_security_score(vuln_report)
print(f"Security grade: {security_score.grade}")
```

### Error Handling

```python
try:
    result = optimizer.optimize_dockerfile(invalid_content)
except ValueError as e:
    print(f"Invalid Dockerfile: {e}")
except Exception as e:
    print(f"Optimization failed: {e}")
```

### Async Usage

```python
import asyncio
from docker_optimizer.performance import PerformanceOptimizer

async def optimize_multiple():
    perf_optimizer = PerformanceOptimizer()
    dockerfiles = [content1, content2, content3]
    
    results = await perf_optimizer.optimize_multiple_with_performance(dockerfiles)
    
    for i, result in enumerate(results):
        print(f"Dockerfile {i+1}: {result.optimized_size}")

# Run async optimization
asyncio.run(optimize_multiple())
```

## Configuration

### Environment Variables

- `DOCKER_OPTIMIZER_CACHE_SIZE`: Maximum cache size (default: 1000)
- `DOCKER_OPTIMIZER_CACHE_TTL`: Cache TTL in seconds (default: 3600)
- `DOCKER_OPTIMIZER_MAX_WORKERS`: Maximum worker threads (default: 4)

### Performance Tuning

```python
from docker_optimizer.performance import PerformanceOptimizer

# Custom configuration
optimizer = PerformanceOptimizer(
    max_workers=8,
    cache_size=2000,
    cache_ttl=7200
)
```

## Error Codes

- `0`: Success
- `1`: General application error
- `2`: File not found or invalid arguments
- `3`: Optimization failed
- `4`: Security scan failed

## Dependencies

### Required
- `click>=8.0.0`: CLI framework
- `dockerfile-parse>=2.0.0`: Dockerfile parsing
- `requests>=2.28.0`: HTTP requests
- `pyyaml>=6.0`: YAML support
- `rich>=13.0.0`: Rich text output
- `pydantic>=2.0.0`: Data validation
- `psutil>=5.9.0`: System metrics

### Optional
- `trivy`: External security scanning
- `bandit`: Security linting
- `safety`: Dependency vulnerability scanning