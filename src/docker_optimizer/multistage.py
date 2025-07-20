"""Multi-stage build optimization engine."""

import re
from typing import Dict, List, Union

from .models import BuildStage, MultiStageOpportunity, MultiStageOptimization
from .parser import DockerfileParser


class MultiStageOptimizer:
    """Optimizer for multi-stage Docker builds."""

    def __init__(self) -> None:
        """Initialize the multi-stage optimizer."""
        self.parser = DockerfileParser()

        # Common build dependencies that should be in build stage only
        self.build_only_packages = {
            # C/C++ build tools
            "gcc", "g++", "build-essential", "make", "cmake", "autoconf", "automake",
            # Development headers
            "libc6-dev", "linux-headers", "kernel-headers",
            # Language-specific build tools
            "python3-dev", "python-dev", "nodejs-dev", "ruby-dev",
            # Package managers for building
            "yarn", "npm" # npm can be runtime too, but often build-only
        }

        # Runtime dependencies that should stay in final stage
        self.runtime_packages = {
            "curl", "wget", "ca-certificates", "openssl",
            "nginx", "apache2", "postgresql-client", "mysql-client",
            "redis-tools", "git" # git can be runtime for some apps
        }

        # Base image patterns for different purposes
        self.base_image_patterns = {
            "build": {
                "python": ["python:{version}", "python:{version}-slim"],
                "node": ["node:{version}", "node:{version}-alpine"],
                "golang": ["golang:{version}", "golang:{version}-alpine"],
                "rust": ["rust:{version}", "rust:{version}-slim"],
                "java": ["openjdk:{version}", "openjdk:{version}-slim"]
            },
            "runtime": {
                "python": ["python:{version}-slim", "python:{version}-alpine"],
                "node": ["node:{version}-alpine", "node:{version}-slim"],
                "golang": ["alpine:latest", "gcr.io/distroless/static"],
                "rust": ["debian:stable-slim", "alpine:latest"],
                "java": ["openjdk:{version}-jre-slim", "eclipse-temurin:{version}-jre-alpine"]
            }
        }

    def identify_build_dependencies(self, dockerfile_content: str) -> List[str]:
        """Identify dependencies that are only needed during build."""
        build_deps = []

        # Parse RUN commands and look for build tools
        instructions = self.parser.parse(dockerfile_content)
        for instruction in instructions:
            if instruction['instruction'] == 'RUN':
                command = instruction['value']

                # Check for build tools in package installation commands
                for package in self.build_only_packages:
                    if package in command.lower():
                        build_deps.append(package)

                # Check for compilation patterns
                if any(pattern in command.lower() for pattern in
                       ['make', 'cmake', 'gcc', 'g++', 'pip install', 'npm install', 'yarn install']):
                    # This looks like a build command
                    if 'make' in command:
                        build_deps.append('make')
                    if 'cmake' in command:
                        build_deps.append('cmake')

        return list(set(build_deps))  # Remove duplicates

    def identify_runtime_dependencies(self, dockerfile_content: str) -> List[str]:
        """Identify dependencies needed at runtime."""
        runtime_deps = []

        instructions = self.parser.parse(dockerfile_content)
        for instruction in instructions:
            if instruction['instruction'] == 'RUN':
                command = instruction['value']

                # Check for runtime tools
                for package in self.runtime_packages:
                    if package in command.lower():
                        runtime_deps.append(package)

        return list(set(runtime_deps))

    def analyze_multistage_opportunity(self, dockerfile_content: str) -> MultiStageOpportunity:
        """Analyze if Dockerfile would benefit from multi-stage build."""
        build_deps = self.identify_build_dependencies(dockerfile_content)
        runtime_deps = self.identify_runtime_dependencies(dockerfile_content)

        has_build_deps = len(build_deps) > 0

        # Check for compilation patterns
        has_compilation = any(pattern in dockerfile_content.lower() for pattern in [
            'make', 'cmake', 'gcc', 'g++', 'pip install', 'npm install', 'yarn install',
            'mvn', 'gradle', 'cargo build', 'go build'
        ])

        recommended = has_build_deps or has_compilation

        benefits = []
        if has_build_deps:
            benefits.append("Size reduction by removing build dependencies")
        if has_compilation:
            benefits.append("Improved security by excluding build tools")
            benefits.append("Better layer caching for builds")

        # Estimate size reduction
        estimated_reduction = "50-200MB"
        if len(build_deps) > 5:
            estimated_reduction = "100-300MB"
        elif len(build_deps) < 2:
            estimated_reduction = "20-100MB"

        # Complexity score (1-10, where 1 is easy)
        complexity = 3  # Base complexity
        if 'pip install' in dockerfile_content and 'requirements.txt' in dockerfile_content:
            complexity = 2  # Python is easy
        elif 'npm install' in dockerfile_content:
            complexity = 2  # Node.js is easy
        elif 'maven' in dockerfile_content or 'gradle' in dockerfile_content:
            complexity = 4  # Java is more complex

        return MultiStageOpportunity(
            recommended=recommended,
            has_build_dependencies=has_build_deps,
            build_dependencies=build_deps,
            runtime_dependencies=runtime_deps,
            benefits=benefits,
            estimated_size_reduction=estimated_reduction,
            complexity_score=complexity
        )

    def generate_multistage_dockerfile(self, dockerfile_content: str) -> MultiStageOptimization:
        """Generate optimized multi-stage Dockerfile."""
        opportunity = self.analyze_multistage_opportunity(dockerfile_content)

        if not opportunity.recommended:
            # Return minimal optimization
            return MultiStageOptimization(
                original_dockerfile=dockerfile_content,
                optimized_dockerfile=dockerfile_content,
                stages=[],
                estimated_size_reduction=0,
                size_reduction=0,
                explanation="Multi-stage build not recommended for this Dockerfile"
            )

        # Detect language/framework
        language = self._detect_language(dockerfile_content)

        if language == "python":
            return self._generate_python_multistage(dockerfile_content)
        elif language == "node":
            return self._generate_node_multistage(dockerfile_content)
        elif language == "golang":
            return self._generate_golang_multistage(dockerfile_content)
        else:
            return self._generate_generic_multistage(dockerfile_content)

    def extract_build_stages(self, dockerfile_content: str) -> List[BuildStage]:
        """Extract existing build stages from multi-stage Dockerfile."""
        stages: List[BuildStage] = []
        instructions = self.parser.parse(dockerfile_content)

        current_stage = None
        stage_commands: List[str] = []

        for instruction in instructions:
            if instruction['instruction'] == 'FROM':
                # Save previous stage if exists
                if current_stage:
                    stages.append(BuildStage(
                        name=current_stage.get('name', f'stage{len(stages)}'),
                        base_image=current_stage['base_image'],
                        commands=stage_commands,
                        purpose='build' if len(stages) == 0 else 'runtime',
                        dependencies=[]
                    ))

                # Parse new stage
                from_value = instruction['value']
                if ' AS ' in from_value.upper():
                    parts = from_value.split()
                    base_image = parts[0]
                    stage_name = parts[2]  # After "AS"
                else:
                    base_image = from_value
                    stage_name = f'stage{len(stages)}'

                current_stage = {'name': stage_name, 'base_image': base_image}
                stage_commands = []
            else:
                stage_commands.append(f"{instruction['instruction']} {instruction['value']}")

        # Add final stage
        if current_stage:
            stages.append(BuildStage(
                name=current_stage.get('name', f'stage{len(stages)}'),
                base_image=current_stage['base_image'],
                commands=stage_commands,
                purpose='runtime',
                dependencies=[]
            ))

        return stages

    def optimize_existing_multistage(self, dockerfile_content: str) -> MultiStageOptimization:
        """Optimize an existing multi-stage Dockerfile."""
        stages = self.extract_build_stages(dockerfile_content)

        optimized_lines = []
        security_improvements = 0
        size_reduction = 0

        for i, stage in enumerate(stages):
            # Optimize base image
            optimized_base = self._optimize_base_image(stage.base_image)
            if optimized_base != stage.base_image:
                security_improvements += 1
                size_reduction += 50  # Estimate

            if stage.name and stage.name != f'stage{i}':
                optimized_lines.append(f"FROM {optimized_base} AS {stage.name}")
            else:
                optimized_lines.append(f"FROM {optimized_base}")

            # Add stage commands
            optimized_lines.extend(stage.commands)
            optimized_lines.append("")  # Empty line between stages

        return MultiStageOptimization(
            original_dockerfile=dockerfile_content,
            optimized_dockerfile="\n".join(optimized_lines).strip(),
            stages=stages,
            estimated_size_reduction=size_reduction,
            security_improvements=security_improvements,
            size_reduction=size_reduction,
            explanation=f"Optimized {len(stages)} stages with improved base images"
        )

    def suggest_optimal_base_images(self, stages: List[BuildStage]) -> List[Dict[str, str]]:
        """Suggest optimal base images for build stages."""
        suggestions = []

        for stage in stages:
            suggestion = {"stage": stage.name, "current": stage.base_image}

            if stage.is_build_stage:
                # Build stage can use full images
                suggestion["base_image"] = stage.base_image
                suggestion["reasoning"] = "Build stage - current image is appropriate"
            else:
                # Runtime stage should use minimal images
                optimized = self._optimize_base_image(stage.base_image)
                suggestion["base_image"] = optimized
                suggestion["reasoning"] = "Runtime stage - use minimal image"

            suggestions.append(suggestion)

        return suggestions

    def calculate_multistage_benefits(self, original_size: str, build_deps: List[str]) -> Dict[str, Union[int, bool, List[str]]]:
        """Calculate benefits of multi-stage build."""
        # Estimate size reduction based on build dependencies
        size_reduction_mb = len(build_deps) * 30  # Rough estimate: 30MB per build tool

        return {
            "size_reduction_mb": size_reduction_mb,
            "security_improvement": len(build_deps) > 0,
            "build_cache_efficiency": True,
            "removed_dependencies": build_deps
        }

    def _detect_language(self, dockerfile_content: str) -> str:
        """Detect the primary language/framework."""
        content_lower = dockerfile_content.lower()

        if 'python' in content_lower or 'pip install' in content_lower:
            return "python"
        elif 'node' in content_lower or 'npm install' in content_lower or 'yarn install' in content_lower:
            return "node"
        elif 'golang' in content_lower or 'go build' in content_lower:
            return "golang"
        elif 'java' in content_lower or 'mvn' in content_lower or 'gradle' in content_lower:
            return "java"
        else:
            return "generic"

    def _generate_python_multistage(self, dockerfile_content: str) -> MultiStageOptimization:
        """Generate Python-specific multi-stage Dockerfile."""

        # Extract version from original
        version_match = re.search(r'python:(\d+\.\d+)', dockerfile_content.lower())
        version = version_match.group(1) if version_match else "3.11"

        optimized_dockerfile = f"""# Build stage
FROM python:{version} AS builder
WORKDIR /app

# Install build dependencies and packages
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:{version}-slim
WORKDIR /app

# Copy only the installed packages
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

CMD ["python", "app.py"]"""

        build_stage = BuildStage(
            name="builder",
            base_image=f"python:{version}",
            commands=["WORKDIR /app", "COPY requirements.txt .", "RUN pip install --no-cache-dir --user -r requirements.txt"],
            purpose="build",
            dependencies=["pip"]
        )

        runtime_stage = BuildStage(
            name="runtime",
            base_image=f"python:{version}-slim",
            commands=["WORKDIR /app", "COPY --from=builder /root/.local /root/.local", "COPY . .", "ENV PATH=/root/.local/bin:$PATH"],
            purpose="runtime",
            dependencies=[]
        )

        return MultiStageOptimization(
            original_dockerfile=dockerfile_content,
            optimized_dockerfile=optimized_dockerfile,
            stages=[build_stage, runtime_stage],
            estimated_size_reduction=150,
            security_improvements=2,
            size_reduction=150,
            explanation="Created Python multi-stage build with user packages and slim runtime image"
        )

    def _generate_node_multistage(self, dockerfile_content: str) -> MultiStageOptimization:
        """Generate Node.js-specific multi-stage Dockerfile."""

        # Extract version from original
        version_match = re.search(r'node:(\d+)', dockerfile_content.lower())
        version = version_match.group(1) if version_match else "18"

        optimized_dockerfile = f"""# Build stage
FROM node:{version}-alpine AS builder
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (including dev dependencies)
RUN npm ci --include=dev

# Copy source code and build
COPY . .
RUN npm run build

# Runtime stage
FROM node:{version}-alpine
WORKDIR /app

# Copy package files and install production dependencies only
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist

# Create non-root user
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001
USER nodejs

EXPOSE 3000
CMD ["npm", "start"]"""

        build_stage = BuildStage(
            name="builder",
            base_image=f"node:{version}-alpine",
            commands=["WORKDIR /app", "COPY package*.json ./", "RUN npm ci --include=dev", "COPY . .", "RUN npm run build"],
            purpose="build",
            dependencies=["npm", "dev-dependencies"]
        )

        runtime_stage = BuildStage(
            name="runtime",
            base_image=f"node:{version}-alpine",
            commands=["WORKDIR /app", "COPY package*.json ./", "RUN npm ci --only=production", "COPY --from=builder /app/dist ./dist"],
            purpose="runtime",
            dependencies=["npm"]
        )

        return MultiStageOptimization(
            original_dockerfile=dockerfile_content,
            optimized_dockerfile=optimized_dockerfile,
            stages=[build_stage, runtime_stage],
            estimated_size_reduction=200,
            security_improvements=3,
            size_reduction=200,
            explanation="Created Node.js multi-stage build separating dev dependencies from runtime"
        )

    def _generate_golang_multistage(self, dockerfile_content: str) -> MultiStageOptimization:
        """Generate Go-specific multi-stage Dockerfile."""

        version_match = re.search(r'golang:(\d+\.\d+)', dockerfile_content.lower())
        version = version_match.group(1) if version_match else "1.21"

        optimized_dockerfile = f"""# Build stage
FROM golang:{version}-alpine AS builder
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Runtime stage - minimal image
FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/main .

# Create non-root user
RUN adduser -D -s /bin/sh appuser
USER appuser

CMD ["./main"]"""

        build_stage = BuildStage(
            name="builder",
            base_image=f"golang:{version}-alpine",
            commands=["WORKDIR /app", "COPY go.mod go.sum ./", "RUN go mod download", "COPY . .", "RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ."],
            purpose="build",
            dependencies=["golang", "go-tools"]
        )

        runtime_stage = BuildStage(
            name="runtime",
            base_image="alpine:latest",
            commands=["RUN apk --no-cache add ca-certificates", "WORKDIR /root/", "COPY --from=builder /app/main ."],
            purpose="runtime",
            dependencies=["ca-certificates"]
        )

        return MultiStageOptimization(
            original_dockerfile=dockerfile_content,
            optimized_dockerfile=optimized_dockerfile,
            stages=[build_stage, runtime_stage],
            estimated_size_reduction=500,
            security_improvements=4,
            size_reduction=500,
            explanation="Created Go multi-stage build with minimal Alpine runtime (huge size reduction)"
        )

    def _generate_generic_multistage(self, dockerfile_content: str) -> MultiStageOptimization:
        """Generate generic multi-stage Dockerfile."""

        # Basic pattern: build stage + runtime stage
        instructions = self.parser.parse(dockerfile_content)
        base_image = "ubuntu:22.04"

        for instruction in instructions:
            if instruction['instruction'] == 'FROM':
                base_image = instruction['value']
                break

        # Use slim version for runtime
        runtime_image = self._optimize_base_image(base_image)

        optimized_dockerfile = f"""# Build stage
FROM {base_image} AS builder
WORKDIR /app

# Install build dependencies and build application
COPY . .
RUN make build || echo "No build step defined"

# Runtime stage
FROM {runtime_image}
WORKDIR /app

# Copy built artifacts from builder stage
COPY --from=builder /app/output /app/

# Create non-root user
RUN useradd -r -s /bin/false appuser
USER appuser

CMD ["./app"]"""

        build_stage = BuildStage(
            name="builder",
            base_image=base_image,
            commands=["WORKDIR /app", "COPY . .", "RUN make build || echo \"No build step defined\""],
            purpose="build",
            dependencies=["build-tools"]
        )

        runtime_stage = BuildStage(
            name="runtime",
            base_image=runtime_image,
            commands=["WORKDIR /app", "COPY --from=builder /app/output /app/"],
            purpose="runtime",
            dependencies=[]
        )

        return MultiStageOptimization(
            original_dockerfile=dockerfile_content,
            optimized_dockerfile=optimized_dockerfile,
            stages=[build_stage, runtime_stage],
            estimated_size_reduction=100,
            security_improvements=2,
            size_reduction=100,
            explanation="Created generic multi-stage build with build/runtime separation"
        )

    def _optimize_base_image(self, base_image: str) -> str:
        """Optimize base image for runtime use."""
        image_lower = base_image.lower()

        # Convert to slim/alpine variants
        if 'ubuntu:latest' in image_lower:
            return 'ubuntu:22.04-slim'
        elif 'ubuntu:' in image_lower and 'slim' not in image_lower:
            return base_image.replace('ubuntu:', 'ubuntu:') + '-slim'
        elif 'python:' in image_lower and 'slim' not in image_lower and 'alpine' not in image_lower:
            return base_image + '-slim'
        elif 'node:' in image_lower and 'alpine' not in image_lower:
            return base_image + '-alpine'

        return base_image
