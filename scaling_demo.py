#!/usr/bin/env python3
"""Demonstrate scaling and advanced features."""

import asyncio
import tempfile
from pathlib import Path

async def demo_high_throughput_processing():
    """Demonstrate high-throughput processing capabilities."""
    # Create test dockerfiles
    test_dockerfiles = []
    for i in range(50):
        content = f"""FROM python:3.11-slim
WORKDIR /app
COPY requirements_{i}.txt .
RUN pip install --no-cache-dir -r requirements_{i}.txt
COPY app_{i}.py .
EXPOSE {8000 + i}
CMD ["python", "app_{i}.py"]"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dockerfile', delete=False) as f:
            f.write(content)
            test_dockerfiles.append(f.name)
    
    return test_dockerfiles

def demo_advanced_features():
    """Demonstrate advanced optimization features."""
    print("🔬 Advanced Docker Optimizer Features Demo")
    print("=" * 50)
    
    # Test multi-stage optimization
    multistage_dockerfile = """FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine AS runtime  
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]"""
    
    from docker_optimizer.multistage import MultiStageOptimizer
    from docker_optimizer.optimizer import DockerfileOptimizer
    
    # Basic optimization
    optimizer = DockerfileOptimizer()
    result = optimizer.optimize_dockerfile(multistage_dockerfile)
    
    print("✅ Multi-stage Dockerfile optimized")
    print(f"   Layers optimized: {len(result.layer_optimizations)}")
    print(f"   Security fixes: {len(result.security_fixes)}")
    
    # Test language detection
    try:
        from docker_optimizer.language_optimizer import analyze_project_language
        
        # Simulate project structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "package.json").write_text('{"name": "test-app"}')
            (temp_path / "app.js").write_text('console.log("Hello World");')
            
            language_analysis = analyze_project_language(temp_path)
            print(f"✅ Language detection: {language_analysis.get('language', 'unknown')}")
            
    except Exception as e:
        print(f"⚠️ Language detection: {e}")
    
    # Test security features
    try:
        from docker_optimizer.external_security import ExternalSecurityScanner
        
        scanner = ExternalSecurityScanner()
        vuln_report = scanner.scan_dockerfile_for_vulnerabilities(multistage_dockerfile)
        security_score = scanner.calculate_security_score(vuln_report)
        
        print(f"✅ Security scan completed")
        print(f"   Security grade: {security_score.grade}")
        print(f"   Total vulnerabilities: {vuln_report.total_vulnerabilities}")
        
    except Exception as e:
        print(f"⚠️ Security scanning: {e}")
    
    # Test configuration management
    try:
        from docker_optimizer.config import Config
        
        config = Config()
        print(f"✅ Configuration loaded")
        print(f"   Cache settings: {config.cache_enabled}")
        print(f"   Default optimization: {config.optimization_level}")
        
    except Exception as e:
        print(f"⚠️ Configuration: {e}")

async def main():
    """Main demo function."""
    # Sync demo
    demo_advanced_features()
    
    print(f"\n🚀 High-Throughput Processing Demo")
    print("=" * 40)
    
    try:
        dockerfiles = await demo_high_throughput_processing()
        print(f"✅ Created {len(dockerfiles)} test Dockerfiles")
        print(f"   Ready for high-throughput processing")
        
        # Cleanup
        for dockerfile in dockerfiles:
            Path(dockerfile).unlink(missing_ok=True)
            
    except Exception as e:
        print(f"⚠️ High-throughput demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())