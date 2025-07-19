"""Test cases for multi-stage build optimization."""

from docker_optimizer.models import BuildStage, MultiStageOptimization
from docker_optimizer.multistage import MultiStageOptimizer


class TestMultiStageOptimizer:
    """Test cases for MultiStageOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = MultiStageOptimizer()

    def test_optimizer_initialization(self):
        """Test that multi-stage optimizer initializes correctly."""
        assert isinstance(self.optimizer, MultiStageOptimizer)

    def test_identify_build_dependencies(self):
        """Test identification of build-only dependencies."""
        dockerfile_content = """
FROM python:3.11
RUN apt-get update && apt-get install -y gcc build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"""

        build_deps = self.optimizer.identify_build_dependencies(dockerfile_content)

        assert len(build_deps) > 0
        assert any("gcc" in dep for dep in build_deps)
        assert any("build-essential" in dep for dep in build_deps)

    def test_identify_runtime_dependencies(self):
        """Test identification of runtime dependencies."""
        dockerfile_content = """
FROM python:3.11
RUN apt-get update && apt-get install -y curl nginx
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8080
CMD ["python", "app.py"]
"""

        runtime_deps = self.optimizer.identify_runtime_dependencies(dockerfile_content)

        assert len(runtime_deps) > 0
        assert any("curl" in dep for dep in runtime_deps)

    def test_analyze_multistage_opportunity(self):
        """Test analysis of multi-stage build opportunities."""
        dockerfile_content = """
FROM python:3.11
RUN apt-get update && apt-get install -y gcc build-essential make
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
RUN python setup.py build
CMD ["python", "app.py"]
"""

        opportunity = self.optimizer.analyze_multistage_opportunity(dockerfile_content)

        assert opportunity.has_build_dependencies
        assert opportunity.recommended
        assert len(opportunity.build_dependencies) > 0
        assert any("Size reduction" in benefit for benefit in opportunity.benefits)

    def test_generate_multistage_dockerfile(self):
        """Test generation of optimized multi-stage Dockerfile."""
        dockerfile_content = """
FROM python:3.11
RUN apt-get update && apt-get install -y gcc build-essential
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"""

        result = self.optimizer.generate_multistage_dockerfile(dockerfile_content)

        assert isinstance(result, MultiStageOptimization)
        assert "FROM python:3.11" in result.optimized_dockerfile
        assert "AS builder" in result.optimized_dockerfile
        assert "FROM python:3.11-slim" in result.optimized_dockerfile
        assert result.estimated_size_reduction > 0

    def test_extract_build_stages(self):
        """Test extraction of build stages from existing multi-stage Dockerfile."""
        dockerfile_content = """
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
"""

        stages = self.optimizer.extract_build_stages(dockerfile_content)

        assert len(stages) == 2
        assert stages[0].name == "builder"
        assert stages[0].base_image == "node:18"
        assert stages[1].base_image == "nginx:alpine"

    def test_optimize_existing_multistage(self):
        """Test optimization of existing multi-stage Dockerfile."""
        dockerfile_content = """
FROM ubuntu:latest AS builder
RUN apt-get update && apt-get install -y gcc
COPY . /app
WORKDIR /app
RUN make build

FROM ubuntu:latest
COPY --from=builder /app/output /app/
CMD ["/app/main"]
"""

        result = self.optimizer.optimize_existing_multistage(dockerfile_content)

        assert "ubuntu:22.04-slim" in result.optimized_dockerfile
        assert result.security_improvements > 0
        assert result.size_reduction > 0

    def test_suggest_optimal_base_images(self):
        """Test suggestion of optimal base images for different stages."""
        build_stage = BuildStage(
            name="builder",
            base_image="ubuntu:latest",
            commands=["apt-get install -y gcc", "make build"],
            purpose="build"
        )

        runtime_stage = BuildStage(
            name="runtime",
            base_image="ubuntu:latest",
            commands=["COPY --from=builder /app/output /app/"],
            purpose="runtime"
        )

        suggestions = self.optimizer.suggest_optimal_base_images([build_stage, runtime_stage])

        assert len(suggestions) == 2
        assert "slim" in suggestions[1]["base_image"]  # Runtime should use slim image

    def test_calculate_multistage_benefits(self):
        """Test calculation of multi-stage build benefits."""
        original_size = "500MB"
        build_deps = ["gcc", "build-essential", "make"]

        benefits = self.optimizer.calculate_multistage_benefits(original_size, build_deps)

        assert benefits["size_reduction_mb"] > 0
        assert benefits["security_improvement"] is True
        assert benefits["build_cache_efficiency"] is True
        assert len(benefits["removed_dependencies"]) > 0

    def test_node_js_multistage_pattern(self):
        """Test specific optimization for Node.js applications."""
        dockerfile_content = """
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
"""

        result = self.optimizer.generate_multistage_dockerfile(dockerfile_content)

        assert "node:18-alpine" in result.optimized_dockerfile
        assert "npm ci --only=production" in result.optimized_dockerfile
        assert "--from=builder" in result.optimized_dockerfile

    def test_python_multistage_pattern(self):
        """Test specific optimization for Python applications."""
        dockerfile_content = """
FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
RUN python setup.py install
CMD ["python", "app.py"]
"""

        result = self.optimizer.generate_multistage_dockerfile(dockerfile_content)

        assert "python:3.11-slim" in result.optimized_dockerfile
        assert "pip install --no-cache-dir" in result.optimized_dockerfile
