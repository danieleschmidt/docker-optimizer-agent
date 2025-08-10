"""Tests for AI-powered optimization engine."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from src.docker_optimizer.ai_optimization_engine import (
    AIOptimizationEngine,
    AIOptimizationRequest,
    OptimizationStrategy,
)
from src.docker_optimizer.models import OptimizationSuggestion


class TestAIOptimizationEngine:
    """Test suite for AI optimization engine."""

    @pytest.fixture
    def ai_engine(self):
        """Create AI optimization engine instance."""
        return AIOptimizationEngine(
            enable_llm_integration=True,
            enable_research_mode=False,
            cache_enabled=True
        )

    @pytest.fixture
    def sample_dockerfile(self):
        """Sample Dockerfile for testing."""
        return """FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3 python3-pip curl

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app

USER root

CMD ["python3", "app.py"]"""

    @pytest.fixture
    def ai_request(self, sample_dockerfile):
        """Create AI optimization request."""
        return AIOptimizationRequest(
            dockerfile_content=sample_dockerfile,
            strategy=OptimizationStrategy.BALANCED,
            target_environment="production",
            security_requirements=["non-root", "version-pinning"],
            performance_requirements=["minimal-size"],
            compliance_frameworks=["HIPAA"]
        )

    @pytest.mark.asyncio
    async def test_ai_optimization_basic(self, ai_engine, ai_request):
        """Test basic AI optimization functionality."""
        result = await ai_engine.optimize_dockerfile_with_ai(ai_request)
        
        # Verify result structure
        assert result.optimized_dockerfile is not None
        assert len(result.explanations) > 0
        assert len(result.security_improvements) > 0
        assert result.metrics.confidence_score > 0
        assert result.metrics.processing_time > 0

    @pytest.mark.asyncio
    async def test_pattern_optimization(self, ai_engine):
        """Test pattern-based optimization."""
        dockerfile = """FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y python3"""
        
        optimized = await ai_engine._apply_pattern_optimization(
            dockerfile, OptimizationStrategy.BALANCED
        )
        
        # Should combine RUN commands
        assert "RUN apt-get update && apt-get install" in optimized

    @pytest.mark.asyncio
    async def test_context_optimization_security(self, ai_engine, ai_request):
        """Test context-aware optimization for security."""
        optimized = await ai_engine._apply_context_optimization(
            ai_request.dockerfile_content, ai_request
        )
        
        # Should add non-root user
        assert "appuser" in optimized
        assert "USER appuser" in optimized

    @pytest.mark.asyncio
    async def test_context_optimization_compliance(self, ai_engine, ai_request):
        """Test context-aware optimization for compliance."""
        optimized = await ai_engine._apply_context_optimization(
            ai_request.dockerfile_content, ai_request
        )
        
        # Should add HIPAA health check
        assert "HEALTHCHECK" in optimized
        assert "health" in optimized.lower()

    @pytest.mark.asyncio
    async def test_nodejs_multistage_generation(self, ai_engine):
        """Test Node.js multi-stage Dockerfile generation."""
        dockerfile = """FROM node:18
COPY package.json .
RUN npm install
COPY . .
CMD ["node", "app.js"]"""
        
        optimized = await ai_engine._apply_llm_optimization(dockerfile, Mock())
        
        # Should generate multi-stage build
        assert "FROM node" in optimized
        assert "AS builder" in optimized
        assert "AS production" in optimized

    @pytest.mark.asyncio
    async def test_python_multistage_generation(self, ai_engine):
        """Test Python multi-stage Dockerfile generation."""
        dockerfile = """FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]"""
        
        optimized = await ai_engine._apply_llm_optimization(dockerfile, Mock())
        
        # Should generate multi-stage build
        assert "FROM python" in optimized
        assert "AS builder" in optimized
        assert "AS production" in optimized
        assert "venv" in optimized

    def test_optimization_recommendations(self, ai_engine):
        """Test AI-powered optimization recommendations."""
        analysis = {
            "has_root_user": True,
            "layer_count": 15,
            "base_image": "ubuntu:latest"
        }
        
        suggestions = ai_engine.get_optimization_recommendations(analysis)
        
        # Should generate relevant suggestions
        assert len(suggestions) > 0
        security_suggestions = [s for s in suggestions if s.type == "security"]
        performance_suggestions = [s for s in suggestions if s.type == "performance"]
        size_suggestions = [s for s in suggestions if s.type == "size"]
        
        assert len(security_suggestions) > 0
        assert len(performance_suggestions) > 0
        assert len(size_suggestions) > 0

    @pytest.mark.asyncio
    async def test_benchmark_optimization_impact(self, ai_engine, sample_dockerfile):
        """Test optimization impact benchmarking."""
        optimized_dockerfile = """FROM alpine:3.18
RUN apk add --no-cache python3 py3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
RUN adduser -D appuser
USER appuser
CMD ["python3", "app.py"]"""
        
        benchmark = await ai_engine.benchmark_optimization_impact(
            sample_dockerfile, optimized_dockerfile
        )
        
        # Should return meaningful metrics
        assert "estimated_size_reduction" in benchmark
        assert "estimated_build_time_improvement" in benchmark
        assert "security_score_improvement" in benchmark
        assert benchmark["estimated_size_reduction"] > 0

    @pytest.mark.asyncio
    async def test_aggressive_strategy(self, ai_engine, sample_dockerfile):
        """Test aggressive optimization strategy."""
        request = AIOptimizationRequest(
            dockerfile_content=sample_dockerfile,
            strategy=OptimizationStrategy.AGGRESSIVE,
            target_environment="production"
        )
        
        result = await ai_engine.optimize_dockerfile_with_ai(request)
        
        # Aggressive strategy should apply more optimizations
        assert result.metrics.confidence_score > 0.8
        assert len(result.explanations) >= 2

    @pytest.mark.asyncio
    async def test_conservative_strategy(self, ai_engine, sample_dockerfile):
        """Test conservative optimization strategy."""
        request = AIOptimizationRequest(
            dockerfile_content=sample_dockerfile,
            strategy=OptimizationStrategy.CONSERVATIVE,
            target_environment="production"
        )
        
        result = await ai_engine.optimize_dockerfile_with_ai(request)
        
        # Conservative strategy should be more cautious
        assert result.metrics.confidence_score > 0.9

    @pytest.mark.asyncio
    async def test_research_mode(self, ai_engine, sample_dockerfile):
        """Test research mode functionality."""
        ai_engine.enable_research_mode = True
        
        request = AIOptimizationRequest(
            dockerfile_content=sample_dockerfile,
            strategy=OptimizationStrategy.RESEARCH,
            target_environment="research"
        )
        
        result = await ai_engine.optimize_dockerfile_with_ai(request)
        
        # Research mode should provide detailed analysis
        assert result.metrics.confidence_score > 0
        assert len(result.alternative_approaches) > 0

    def test_generate_explanations(self, ai_engine):
        """Test explanation generation."""
        original = "FROM ubuntu:latest\nRUN apt-get update"
        optimized = """FROM alpine:3.18
RUN apk add --no-cache python3
USER appuser
HEALTHCHECK --interval=30s CMD curl -f http://localhost/health"""
        
        explanations = ai_engine._generate_explanations(original, optimized)
        
        assert len(explanations) > 0
        assert any("non-root user" in exp.lower() for exp in explanations)
        assert any("health check" in exp.lower() for exp in explanations)
        assert any("alpine" in exp.lower() for exp in explanations)

    def test_extract_security_improvements(self, ai_engine):
        """Test security improvement extraction."""
        original = "FROM ubuntu:latest\nUSER root"
        optimized = "FROM ubuntu:22.04\nUSER appuser\nHEALTHCHECK CMD curl -f http://localhost/health"
        
        improvements = ai_engine._extract_security_improvements(original, optimized)
        
        assert len(improvements) > 0
        assert any("latest" in imp.lower() for imp in improvements)
        assert any("non-root" in imp.lower() for imp in improvements)

    def test_extract_performance_improvements(self, ai_engine):
        """Test performance improvement extraction."""
        original = """FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y pip"""
        
        optimized = """FROM alpine:3.18 AS builder
RUN apk add --no-cache python3 py3-pip
FROM alpine:3.18 AS production
COPY --from=builder /usr/bin/python3 /usr/bin/python3"""
        
        improvements = ai_engine._extract_performance_improvements(original, optimized)
        
        assert len(improvements) > 0
        assert any("layer" in imp.lower() or "alpine" in imp.lower() or "multi-stage" in imp.lower() 
                  for imp in improvements)

    @pytest.mark.asyncio
    async def test_error_handling(self, ai_engine):
        """Test error handling in AI optimization."""
        # Test with invalid Dockerfile content
        request = AIOptimizationRequest(
            dockerfile_content="",  # Empty content
            strategy=OptimizationStrategy.BALANCED,
            target_environment="production"
        )
        
        # Should handle gracefully
        result = await ai_engine.optimize_dockerfile_with_ai(request)
        assert result is not None

    def test_cache_functionality(self):
        """Test caching functionality."""
        cached_engine = AIOptimizationEngine(cache_enabled=True)
        non_cached_engine = AIOptimizationEngine(cache_enabled=False)
        
        assert cached_engine.cache_enabled
        assert not non_cached_engine.cache_enabled

    def test_llm_integration_toggle(self):
        """Test LLM integration toggle."""
        llm_engine = AIOptimizationEngine(enable_llm_integration=True)
        no_llm_engine = AIOptimizationEngine(enable_llm_integration=False)
        
        assert llm_engine.enable_llm_integration
        assert not no_llm_engine.enable_llm_integration


class TestOptimizationStrategy:
    """Test optimization strategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert OptimizationStrategy.AGGRESSIVE == "aggressive"
        assert OptimizationStrategy.BALANCED == "balanced"
        assert OptimizationStrategy.CONSERVATIVE == "conservative"
        assert OptimizationStrategy.RESEARCH == "research"

    def test_strategy_creation(self):
        """Test strategy creation from string."""
        strategy = OptimizationStrategy("balanced")
        assert strategy == OptimizationStrategy.BALANCED